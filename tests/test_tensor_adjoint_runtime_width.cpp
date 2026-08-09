#include "qubit/qtensor_adjoint.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;
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

    PauliObservable first(qubits);
    first.add_term(
        {0.7, -0.1},
        std::vector<PauliFactor>{{0U, PauliAxis::X}, {1U, PauliAxis::Y}});
    first.add_term(
        {-0.25, 0.05},
        std::vector<PauliFactor>{{2U, PauliAxis::Z}, {3U, PauliAxis::X}});
    first.add_term(
        {0.125, 0.0},
        std::vector<PauliFactor>{{4U, PauliAxis::Y}});
    result.push_back(std::move(first));

    PauliObservable second(qubits);
    second.add_term(
        {-0.4, 0.0},
        std::vector<PauliFactor>{{1U, PauliAxis::Z}});
    second.add_term(
        {0.33, -0.02},
        std::vector<PauliFactor>{{3U, PauliAxis::Y}, {5U, PauliAxis::Z}});
    result.push_back(std::move(second));

    PauliObservable third(qubits);
    third.add_term(
        {0.55, 0.05},
        std::vector<PauliFactor>{{0U, PauliAxis::Z}, {4U, PauliAxis::X}});
    third.add_term(
        {-0.2, 0.01},
        std::vector<PauliFactor>{{2U, PauliAxis::Y}, {5U, PauliAxis::X}});
    result.push_back(std::move(third));

    return result;
}

void compare_exact(
    const std::vector<QComplex>& actual,
    const std::vector<QComplex>& expected,
    const char* message) {
    require(actual.size() == expected.size(), "runtime-width result shape changed");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require_close(actual[index], expected[index], 2e-13, message);
    }
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 6U;
    const TensorNetworkConfig tensor{1U << 18U, 100'000U};
    const auto templated = operations();
    const auto queries = observables();
    const std::vector<double> parameters{0.31, -0.52, 0.73};

    ExactAdjointGradientPlan shared(
        qubits,
        templated,
        queries,
        ExactAdjointGradientConfig{tensor, 4U});
    require(shared.worker_count() == 4U,
            "runtime-width test did not produce four adjoint workers");

    auto serial_workspace = shared.workspace(1U);
    auto parallel_workspace = shared.workspace(shared.worker_count());
    require(serial_workspace.execution_worker_count() == 1U,
            "serial runtime-width workspace has the wrong worker count");
    require(parallel_workspace.execution_worker_count() == shared.worker_count(),
            "parallel runtime-width workspace has the wrong worker count");

    const std::size_t serial_bytes = serial_workspace.estimated_bytes();
    const std::size_t parallel_bytes = parallel_workspace.estimated_bytes();

    std::vector<QComplex> serial_values(shared.observable_count());
    std::vector<QComplex> serial_gradients(
        shared.observable_count() * shared.parameter_count());
    std::vector<QComplex> parallel_values(shared.observable_count());
    std::vector<QComplex> parallel_gradients(serial_gradients.size());

    shared.value_and_gradient(
        parameters, serial_values, serial_gradients, serial_workspace);
    shared.value_and_gradient(
        parameters, parallel_values, parallel_gradients, parallel_workspace);

    require(serial_workspace.estimated_bytes() == serial_bytes,
            "serial runtime-width workspace grew during exact execution");
    require(parallel_workspace.estimated_bytes() == parallel_bytes,
            "parallel runtime-width workspace grew during exact execution");
    require(serial_workspace.rebind_count() == 1U,
            "serial runtime-width workspace rebind count changed");
    require(parallel_workspace.rebind_count() == 1U,
            "parallel runtime-width workspace rebind count changed");

    ExactAdjointGradientPlan serial_reference(
        qubits,
        templated,
        queries,
        ExactAdjointGradientConfig{tensor, 1U});
    auto serial_reference_workspace = serial_reference.workspace();
    std::vector<QComplex> reference_values(serial_reference.observable_count());
    std::vector<QComplex> reference_gradients(
        serial_reference.observable_count() * serial_reference.parameter_count());
    serial_reference.value_and_gradient(
        parameters,
        reference_values,
        reference_gradients,
        serial_reference_workspace);

    require(serial_values == reference_values,
            "one-lane workspace changed serial adjoint reduction order");
    require(serial_gradients == reference_gradients,
            "one-lane workspace changed serial adjoint gradient reduction order");
    compare_exact(
        parallel_values,
        reference_values,
        "parallel runtime-width value differs from serial reference");
    compare_exact(
        parallel_gradients,
        reference_gradients,
        "parallel runtime-width gradient differs from serial reference");

    ExactAdjointGradientPlan parallel_reference(
        qubits,
        templated,
        queries,
        ExactAdjointGradientConfig{tensor, 4U});
    auto parallel_reference_workspace = parallel_reference.workspace();
    std::vector<QComplex> parallel_reference_values(
        parallel_reference.observable_count());
    std::vector<QComplex> parallel_reference_gradients(
        parallel_reference.observable_count() * parallel_reference.parameter_count());
    parallel_reference.value_and_gradient(
        parameters,
        parallel_reference_values,
        parallel_reference_gradients,
        parallel_reference_workspace);
    require(parallel_values == parallel_reference_values,
            "runtime-width parallel values differ from an independent parallel plan");
    require(parallel_gradients == parallel_reference_gradients,
            "runtime-width parallel gradients differ from an independent parallel plan");

    const std::vector<double> second_parameters{-0.19, 0.44, -0.61};
    shared.value_and_gradient(
        second_parameters, serial_values, serial_gradients, serial_workspace);
    shared.value_and_gradient(
        second_parameters, parallel_values, parallel_gradients, parallel_workspace);
    require(serial_workspace.estimated_bytes() == serial_bytes,
            "serial runtime-width workspace grew on a later parameter point");
    require(parallel_workspace.estimated_bytes() == parallel_bytes,
            "parallel runtime-width workspace grew on a later parameter point");
    require(serial_workspace.rebind_count() == 2U,
            "serial runtime-width workspace did not retain reuse state");
    require(parallel_workspace.rebind_count() == 2U,
            "parallel runtime-width workspace did not retain reuse state");

    bool rejected = false;
    try {
        static_cast<void>(shared.workspace(0U));
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adjoint plan accepted a zero-worker workspace");

    rejected = false;
    try {
        static_cast<void>(shared.workspace(shared.worker_count() + 1U));
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adjoint plan accepted a workspace wider than its compiled plan");

    return 0;
}
