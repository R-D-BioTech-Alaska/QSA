#include "qubit/qtensor_adjoint.hpp"
#include "qubit/qtensor_rebind.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::ExactParameterShiftConfig;
using qubit::ExactParameterShiftPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::TensorNetworkConfig;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    QComplex actual,
    QComplex expected,
    double tolerance,
    const char* message) {
    if (!qubit::almost_equal(actual, expected, tolerance)) {
        throw std::runtime_error(message);
    }
}

std::vector<PauliObservable> observables(std::size_t qubits) {
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

std::vector<Operation> bind(
    const std::vector<ParameterizedOperation>& operations,
    const std::vector<double>& parameters) {
    std::vector<Operation> result;
    result.reserve(operations.size());
    for (const ParameterizedOperation& templated : operations) {
        Operation operation = templated.operation;
        if (templated.parameter_slot >= 0) {
            operation.parameter =
                parameters[static_cast<std::size_t>(templated.parameter_slot)];
        }
        result.push_back(operation);
    }
    return result;
}

std::vector<QComplex> values(
    std::size_t qubits,
    const std::vector<Operation>& operations,
    const std::vector<PauliObservable>& queries) {
    QRegister state(qubits);
    OperationPlan plan(operations, false);
    plan.execute(state);
    std::vector<QComplex> result(queries.size());
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        result[index] = queries[index].expectation(state);
    }
    return result;
}

std::vector<QComplex> parameter_shift_reference(
    std::size_t qubits,
    const std::vector<ParameterizedOperation>& operations,
    const std::vector<double>& parameters,
    const std::vector<PauliObservable>& queries,
    std::size_t parameter_count) {
    constexpr double shift = 1.57079632679489661923;
    std::vector<QComplex> gradient(queries.size() * parameter_count);
    std::vector<Operation> bound = bind(operations, parameters);
    for (std::size_t operation_index = 0U;
         operation_index < operations.size();
         ++operation_index) {
        if (operations[operation_index].parameter_slot < 0) {
            continue;
        }
        const std::size_t parameter =
            static_cast<std::size_t>(operations[operation_index].parameter_slot);
        const double original = bound[operation_index].parameter;
        bound[operation_index].parameter = original + shift;
        const std::vector<QComplex> plus = values(qubits, bound, queries);
        bound[operation_index].parameter = original - shift;
        const std::vector<QComplex> minus = values(qubits, bound, queries);
        bound[operation_index].parameter = original;
        for (std::size_t observable = 0U; observable < queries.size(); ++observable) {
            gradient[observable * parameter_count + parameter] +=
                (plus[observable] - minus[observable]) * 0.5;
        }
    }
    return gradient;
}

}  // namespace

int main() {
    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        PauliObservable z(1U);
        const PauliFactor factor{0U, PauliAxis::Z};
        z.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
        const std::vector<PauliObservable> queries{z};
        ExactAdjointGradientPlan plan(1U, operations, queries);
        auto workspace = plan.workspace();
        const std::vector<double> parameters{0.43};
        std::vector<QComplex> output(1U);
        std::vector<QComplex> gradient(1U);
        plan.value_and_gradient(parameters, output, gradient, workspace);
        require_close(output[0], {std::cos(parameters[0]), 0.0}, 2e-12,
                      "single-Ry adjoint value is incorrect");
        require_close(gradient[0], {-std::sin(parameters[0]), 0.0}, 2e-12,
                      "single-Ry adjoint gradient is incorrect");
        require(workspace.rebind_count() == 1U,
                "single-Ry adjoint used more than one circuit rebind");
        require(plan.stats().parameter_shift_equivalent_evaluations == 3U,
                "single-Ry parameter-shift equivalent count is incorrect");
        require(plan.worker_count() == 1U,
                "default adjoint worker count changed");
        require(plan.stats().balanced_peak_estimated_work ==
                    plan.stats().round_robin_peak_estimated_work,
                "single-lane adjoint balancing changed its work estimate");
    }

    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
            {{OperationCode::Ry, 1U, 0U, 0.0, 0.0}, 0, -1},
        };
        PauliObservable zz(2U);
        const std::vector<PauliFactor> factors{
            {0U, PauliAxis::Z},
            {1U, PauliAxis::Z},
        };
        zz.add_term({1.0, 0.0}, factors);
        const std::vector<PauliObservable> queries{zz};
        ExactAdjointGradientPlan plan(2U, operations, queries);
        auto workspace = plan.workspace();
        const std::vector<double> parameters{0.37};
        std::vector<QComplex> output(1U);
        std::vector<QComplex> gradient(1U);
        plan.value_and_gradient(parameters, output, gradient, workspace);
        require_close(
            output[0],
            {std::cos(parameters[0]) * std::cos(parameters[0]), 0.0},
            2e-12,
            "shared-parameter adjoint value is incorrect");
        require_close(
            gradient[0],
            {-std::sin(2.0 * parameters[0]), 0.0},
            2e-12,
            "shared-parameter adjoint did not sum occurrences");
        require(plan.parameterized_operation_count() == 2U,
                "shared-parameter adjoint occurrence count changed");
        require(plan.stats().parameter_shift_equivalent_evaluations == 5U,
                "shared-parameter equivalent evaluation count is incorrect");
    }

    {
        constexpr std::size_t qubits = 6U;
        const std::vector<PauliObservable> queries = observables(qubits);
        const std::vector<ParameterizedOperation> operations{
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
        const std::vector<double> parameters{0.31, -0.52, 0.73};
        const TensorNetworkConfig tensor{1U << 18U, 100'000U};
        ExactAdjointGradientPlan adjoint(
            qubits, operations, queries, ExactAdjointGradientConfig{tensor});
        auto workspace = adjoint.workspace();
        std::vector<QComplex> adjoint_values(queries.size());
        std::vector<QComplex> adjoint_gradient(
            queries.size() * parameters.size());
        adjoint.value_and_gradient(
            parameters, adjoint_values, adjoint_gradient, workspace);

        const std::vector<Operation> bound = bind(operations, parameters);
        const std::vector<QComplex> reference_values =
            values(qubits, bound, queries);
        const std::vector<QComplex> reference_gradient =
            parameter_shift_reference(
                qubits, operations, parameters, queries, parameters.size());
        for (std::size_t observable = 0U; observable < queries.size(); ++observable) {
            require_close(
                adjoint_values[observable], reference_values[observable], 7e-11,
                "adjoint value differs from QRegister");
            for (std::size_t parameter = 0U;
                 parameter < parameters.size();
                 ++parameter) {
                const std::size_t index = adjoint.gradient_index(observable, parameter);
                require_close(
                    adjoint_gradient[index], reference_gradient[index], 9e-11,
                    "adjoint gradient differs from exact parameter shift");
            }
        }
        require(adjoint.stats().tensor.peak_contraction_entries <= (1U << 18U),
                "adjoint gradient exceeded its tensor resource certificate");
        require(adjoint.stats().source_derivative_bindings > 0U,
                "adjoint gradient produced no source derivative bindings");
        require(workspace.rebind_count() == 1U,
                "adjoint gradient should rebind the circuit once per gradient call");

        ExactAdjointGradientPlan parallel(
            qubits,
            operations,
            queries,
            ExactAdjointGradientConfig{tensor, 4U});
        auto parallel_workspace = parallel.workspace();
        std::vector<QComplex> parallel_values(queries.size());
        std::vector<QComplex> parallel_gradient(
            queries.size() * parameters.size());
        parallel.value_and_gradient(
            parameters,
            parallel_values,
            parallel_gradient,
            parallel_workspace);
        require(parallel.worker_count() == 3U,
                "parallel adjoint worker count did not clamp to differentiated terms");
        const auto parallel_stats = parallel.stats();
        require(parallel_stats.worker_count == parallel.worker_count(),
                "parallel adjoint stats worker count changed");
        require(parallel_stats.estimated_work > 0U,
                "parallel adjoint produced no estimated work");
        require(parallel_stats.balanced_peak_estimated_work <=
                    parallel_stats.round_robin_peak_estimated_work,
                "parallel adjoint balanced schedule regressed estimated peak work");
        for (std::size_t index = 0U; index < parallel_values.size(); ++index) {
            require_close(
                parallel_values[index], adjoint_values[index], 2e-13,
                "parallel adjoint value differs from serial adjoint");
        }
        for (std::size_t index = 0U; index < parallel_gradient.size(); ++index) {
            require_close(
                parallel_gradient[index], adjoint_gradient[index], 2e-13,
                "parallel adjoint gradient differs from serial adjoint");
        }
        const std::vector<QComplex> first_parallel_values = parallel_values;
        const std::vector<QComplex> first_parallel_gradient = parallel_gradient;
        parallel.value_and_gradient(
            parameters,
            parallel_values,
            parallel_gradient,
            parallel_workspace);
        require(parallel_values == first_parallel_values,
                "parallel adjoint value reduction is not deterministic");
        require(parallel_gradient == first_parallel_gradient,
                "parallel adjoint gradient reduction is not deterministic");
        require(parallel_workspace.rebind_count() == 2U,
                "parallel adjoint rebind count changed");

        ExactParameterShiftPlan shift(
            qubits,
            operations,
            queries,
            ExactParameterShiftConfig{tensor, 1U});
        auto shift_workspace = shift.workspace();
        std::vector<QComplex> shift_values(queries.size());
        std::vector<QComplex> shift_gradient(
            queries.size() * parameters.size());
        shift.value_and_gradient(
            parameters, shift_values, shift_gradient, shift_workspace);
        for (std::size_t index = 0U; index < shift_gradient.size(); ++index) {
            require_close(
                adjoint_gradient[index], shift_gradient[index], 9e-11,
                "adjoint gradient differs from reusable parameter shift");
        }

        const std::vector<QComplex> preserved_values = adjoint_values;
        const std::vector<QComplex> preserved_gradient = adjoint_gradient;
        bool rejected = false;
        try {
            const std::vector<double> too_short{parameters[0], parameters[1]};
            adjoint.value_and_gradient(
                too_short, adjoint_values, adjoint_gradient, workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint gradient accepted too few parameters");
        require(adjoint_values == preserved_values,
                "rejected adjoint call mutated value outputs");
        require(adjoint_gradient == preserved_gradient,
                "rejected adjoint call mutated gradient outputs");

        ExactAdjointGradientPlan foreign(
            qubits, operations, queries, ExactAdjointGradientConfig{tensor});
        auto foreign_workspace = foreign.workspace();
        rejected = false;
        try {
            adjoint.value_and_gradient(
                parameters, adjoint_values, adjoint_gradient, foreign_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint gradient accepted a foreign workspace");
    }

    {
        const std::vector<ParameterizedOperation> invalid{
            {{OperationCode::H, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        PauliObservable z(1U);
        const PauliFactor factor{0U, PauliAxis::Z};
        z.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
        const std::vector<PauliObservable> queries{z};
        bool rejected = false;
        try {
            static_cast<void>(ExactAdjointGradientPlan(1U, invalid, queries));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint gradient accepted a parameterized H gate");
    }

    {
        const std::vector<ParameterizedOperation> invalid{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, 1},
        };
        PauliObservable z(1U);
        const PauliFactor factor{0U, PauliAxis::Z};
        z.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
        const std::vector<PauliObservable> queries{z};
        bool rejected = false;
        try {
            static_cast<void>(ExactAdjointGradientPlan(1U, invalid, queries));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint gradient accepted a stochastic sample slot");
    }

    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        PauliObservable z(1U);
        const PauliFactor factor{0U, PauliAxis::Z};
        z.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
        const std::vector<PauliObservable> queries{z};
        bool rejected = false;
        try {
            static_cast<void>(ExactAdjointGradientPlan(
                1U,
                operations,
                queries,
                ExactAdjointGradientConfig{{}, 33U}));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "adjoint gradient accepted an unbounded worker count");
    }

    return 0;
}
