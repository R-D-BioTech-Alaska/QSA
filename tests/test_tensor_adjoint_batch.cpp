#include "qubit/qtensor_adjoint_batch.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactAdjointGradientBatchConfig;
using qubit::ExactAdjointGradientBatchPlan;
using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QStateError;
using qubit::TensorNetworkConfig;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<ParameterizedOperation> operations() {
    return {
        {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
        {{OperationCode::Rz, 1U, 0U, 0.0, 0.0}, 1, -1},
        {{OperationCode::Cnot, 0U, 1U, 0.0, 0.0}, -1, -1},
        {{OperationCode::Rx, 2U, 0U, 0.0, 0.0}, 0, -1},
        {{OperationCode::Ry, 3U, 0U, 0.0, 0.0}, 2, -1},
        {{OperationCode::Cnot, 2U, 3U, 0.0, 0.0}, -1, -1},
        {{OperationCode::Rz, 4U, 0U, 0.0, 0.0}, 1, -1},
        {{OperationCode::Cnot, 4U, 5U, 0.0, 0.0}, -1, -1},
        {{OperationCode::Ry, 5U, 0U, 0.0, 0.0}, 2, -1},
    };
}

std::vector<PauliObservable> observables() {
    constexpr std::size_t qubits = 6U;
    std::vector<PauliObservable> result;

    PauliObservable first(qubits, PauliPropagationConfig{65'536U});
    const std::vector<PauliFactor> first_factors{
        {0U, PauliAxis::X},
        {1U, PauliAxis::Y},
    };
    first.add_term({0.7, -0.1}, first_factors);
    result.push_back(std::move(first));

    PauliObservable second(qubits, PauliPropagationConfig{65'536U});
    const PauliFactor z{2U, PauliAxis::Z};
    second.add_term({-0.4, 0.0}, std::span<const PauliFactor>(&z, 1U));
    second.add_term({0.125, 0.0});
    result.push_back(std::move(second));

    PauliObservable third(qubits, PauliPropagationConfig{65'536U});
    const std::vector<PauliFactor> third_factors{
        {3U, PauliAxis::X},
        {4U, PauliAxis::Z},
        {5U, PauliAxis::Y},
    };
    third.add_term({0.55, 0.05}, third_factors);
    result.push_back(std::move(third));

    PauliObservable identity(qubits);
    identity.add_term({0.75, -0.2});
    result.push_back(std::move(identity));

    return result;
}

std::vector<double> parameter_points(std::size_t point_count) {
    constexpr std::size_t parameter_count = 3U;
    std::vector<double> result(point_count * parameter_count);
    for (std::size_t point = 0U; point < point_count; ++point) {
        result[point * parameter_count + 0U] =
            -0.71 + 0.083 * static_cast<double>(point + 1U);
        result[point * parameter_count + 1U] =
            0.43 - 0.057 * static_cast<double>(point + 2U);
        result[point * parameter_count + 2U] =
            -0.19 + 0.041 * static_cast<double>(point + 3U);
    }
    return result;
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 6U;
    constexpr std::size_t point_count = 7U;
    const TensorNetworkConfig tensor{1U << 18U, 100'000U};
    const std::vector<ParameterizedOperation> templated = operations();
    const std::vector<PauliObservable> queries = observables();
    const std::vector<double> parameters = parameter_points(point_count);

    ExactAdjointGradientPlan serial(
        qubits,
        templated,
        queries,
        ExactAdjointGradientConfig{tensor, 1U});
    auto serial_workspace = serial.workspace();
    const std::size_t parameter_count = serial.parameter_count();
    const std::size_t observable_count = serial.observable_count();
    const std::size_t point_gradient_entries =
        observable_count * parameter_count;

    std::vector<QComplex> reference_values(point_count * observable_count);
    std::vector<QComplex> reference_gradients(
        point_count * point_gradient_entries);
    for (std::size_t point = 0U; point < point_count; ++point) {
        serial.value_and_gradient(
            std::span<const double>(
                parameters.data() + point * parameter_count,
                parameter_count),
            std::span<QComplex>(
                reference_values.data() + point * observable_count,
                observable_count),
            std::span<QComplex>(
                reference_gradients.data() + point * point_gradient_entries,
                point_gradient_entries),
            serial_workspace);
    }
    require(serial_workspace.rebind_count() == point_count,
            "serial adjoint reference rebind count changed");

    ExactAdjointGradientBatchPlan batch(
        qubits,
        templated,
        queries,
        ExactAdjointGradientBatchConfig{tensor, 4U});
    auto workspace = batch.workspace();
    require(batch.point_worker_count() == 4U,
            "adjoint batch did not honor its point worker count");
    require(batch.effective_point_worker_count(point_count) == 4U,
            "adjoint batch effective point worker count changed");
    require(batch.effective_point_worker_count(2U) == 2U,
            "adjoint batch did not clamp workers to point count");
    require(batch.stats().point.worker_count == 1U,
            "adjoint batch enabled nested term parallelism");

    std::vector<QComplex> batch_values(reference_values.size());
    std::vector<QComplex> batch_gradients(reference_gradients.size());
    batch.value_and_gradient_batch(
        parameters,
        point_count,
        batch_values,
        batch_gradients,
        workspace);
    require(batch_values == reference_values,
            "adjoint point batch values differ from repeated serial adjoint");
    require(batch_gradients == reference_gradients,
            "adjoint point batch gradients differ from repeated serial adjoint");
    require(workspace.rebind_count() == point_count,
            "adjoint point batch rebind count does not match evaluated points");
    require(workspace.point_capacity() == point_count,
            "adjoint point batch staged point capacity changed");

    const std::vector<QComplex> first_values = batch_values;
    const std::vector<QComplex> first_gradients = batch_gradients;
    batch.value_and_gradient_batch(
        parameters,
        point_count,
        batch_values,
        batch_gradients,
        workspace);
    require(batch_values == first_values,
            "adjoint point batch values are not deterministic");
    require(batch_gradients == first_gradients,
            "adjoint point batch gradients are not deterministic");
    require(workspace.rebind_count() == 2U * point_count,
            "adjoint point batch cumulative rebind count changed");

    {
        const std::vector<QComplex> preserved_values = batch_values;
        const std::vector<QComplex> preserved_gradients = batch_gradients;
        bool rejected = false;
        try {
            batch.value_and_gradient_batch(
                std::span<const double>(
                    parameters.data(), parameters.size() - 1U),
                point_count,
                batch_values,
                batch_gradients,
                workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint point batch accepted a malformed parameter span");
        require(batch_values == preserved_values,
                "malformed adjoint batch mutated caller values");
        require(batch_gradients == preserved_gradients,
                "malformed adjoint batch mutated caller gradients");
    }

    {
        std::vector<double> nonfinite = parameters;
        nonfinite[parameter_count + 1U] =
            std::numeric_limits<double>::quiet_NaN();
        const std::vector<QComplex> preserved_values = batch_values;
        const std::vector<QComplex> preserved_gradients = batch_gradients;
        bool rejected = false;
        try {
            batch.value_and_gradient_batch(
                nonfinite,
                point_count,
                batch_values,
                batch_gradients,
                workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint point batch accepted a nonfinite parameter");
        require(batch_values == preserved_values,
                "nonfinite adjoint batch mutated caller values");
        require(batch_gradients == preserved_gradients,
                "nonfinite adjoint batch mutated caller gradients");
    }

    {
        ExactAdjointGradientBatchPlan foreign(
            qubits,
            templated,
            queries,
            ExactAdjointGradientBatchConfig{tensor, 4U});
        auto foreign_workspace = foreign.workspace();
        bool rejected = false;
        try {
            batch.value_and_gradient_batch(
                parameters,
                point_count,
                batch_values,
                batch_gradients,
                foreign_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint point batch accepted a foreign workspace");
    }

    {
        bool rejected = false;
        try {
            static_cast<void>(ExactAdjointGradientBatchPlan(
                qubits,
                templated,
                queries,
                ExactAdjointGradientBatchConfig{tensor, 17U}));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint point batch accepted an unbounded worker count");
    }

    {
        std::vector<QComplex> empty_values;
        std::vector<QComplex> empty_gradients;
        const std::size_t rebinds = workspace.rebind_count();
        batch.value_and_gradient_batch(
            std::span<const double>(),
            0U,
            empty_values,
            empty_gradients,
            workspace);
        require(workspace.rebind_count() == rebinds,
                "empty adjoint point batch changed rebind count");
        require(batch.effective_point_worker_count(0U) == 0U,
                "empty adjoint batch reported active workers");
    }

    return 0;
}
