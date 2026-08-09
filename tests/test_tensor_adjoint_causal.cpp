#include "qubit/qtensor_adjoint_causal.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace qubit;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(QComplex actual, QComplex expected, double tolerance, const char* message) {
    if (!almost_equal(actual, expected, tolerance)) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] PauliObservable pauli(std::size_t qubits, QubitId qubit, PauliAxis axis) {
    PauliObservable observable(qubits);
    const PauliFactor factor{qubit, axis};
    observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
    return observable;
}

[[nodiscard]] std::vector<Operation> bind(
    const std::vector<ParameterizedOperation>& operations,
    std::span<const double> parameters) {
    std::vector<Operation> result;
    result.reserve(operations.size());
    for (const ParameterizedOperation& templated : operations) {
        Operation operation = templated.operation;
        if (templated.parameter_slot >= 0) {
            operation.parameter = parameters[static_cast<std::size_t>(templated.parameter_slot)];
        }
        result.push_back(operation);
    }
    return result;
}

[[nodiscard]] std::vector<QComplex> direct_values(
    std::size_t qubits,
    const std::vector<ParameterizedOperation>& operations,
    std::span<const double> parameters,
    const std::vector<PauliObservable>& observables) {
    QRegister state(qubits);
    const std::vector<Operation> concrete = bind(operations, parameters);
    OperationPlan plan(concrete, false);
    plan.execute(state);
    std::vector<QComplex> result(observables.size());
    for (std::size_t index = 0U; index < observables.size(); ++index) {
        result[index] = observables[index].expectation(state);
    }
    return result;
}

[[nodiscard]] std::vector<QComplex> shift_gradient(
    std::size_t qubits,
    const std::vector<ParameterizedOperation>& operations,
    std::span<const double> parameters,
    const std::vector<PauliObservable>& observables) {
    constexpr double shift = 1.57079632679489661923;
    std::vector<QComplex> result(observables.size() * parameters.size());
    std::vector<double> shifted(parameters.begin(), parameters.end());
    for (std::size_t parameter = 0U; parameter < parameters.size(); ++parameter) {
        shifted.assign(parameters.begin(), parameters.end());
        shifted[parameter] += shift;
        const auto plus = direct_values(qubits, operations, shifted, observables);
        shifted[parameter] -= 2.0 * shift;
        const auto minus = direct_values(qubits, operations, shifted, observables);
        for (std::size_t observable = 0U; observable < observables.size(); ++observable) {
            result[observable * parameters.size() + parameter] =
                (plus[observable] - minus[observable]) * 0.5;
        }
    }
    return result;
}

}  // namespace

int main() {
    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        const std::vector<PauliObservable> observables{pauli(1U, 0U, PauliAxis::Z)};
        ExactCausalAdjointGradientPlan plan(1U, operations, observables);
        require(plan.stats().dynamic_term_count == 1U,
                "Ry/Z term was not classified dynamic");
        const std::vector<double> parameters{0.43};
        auto workspace = plan.workspace();
        std::vector<QComplex> values(1U);
        std::vector<QComplex> gradients(1U);
        plan.value_and_gradient(parameters, values, gradients, workspace);
        require_close(values[0], {std::cos(parameters[0]), 0.0}, 2e-12,
                      "Ry/Z value is incorrect");
        require_close(gradients[0], {-std::sin(parameters[0]), 0.0}, 2e-12,
                      "Ry/Z gradient is incorrect");
    }

    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Rz, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        const std::vector<PauliObservable> observables{pauli(1U, 0U, PauliAxis::Z)};
        ExactCausalAdjointGradientPlan plan(1U, operations, observables);
        require(plan.stats().dynamic_term_count == 1U,
                "support classifier must conservatively retain an on-qubit Rz/Z term");
        const std::vector<double> parameters{1.37};
        auto workspace = plan.workspace();
        std::vector<QComplex> values(1U);
        std::vector<QComplex> gradients(1U);
        plan.value_and_gradient(parameters, values, gradients, workspace);
        require_close(values[0], {1.0, 0.0}, 2e-12, "Rz/Z value changed");
        require_close(gradients[0], {}, 2e-12, "Rz/Z gradient is not zero");
    }

    {
        constexpr std::size_t qubits = 4U;
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
            {{OperationCode::H, 2U, 0U, 0.0, 0.0}, -1, -1},
        };
        PauliObservable mixed(qubits);
        const PauliFactor z0{0U, PauliAxis::Z};
        const PauliFactor z3{3U, PauliAxis::Z};
        mixed.add_term({0.75, 0.0}, std::span<const PauliFactor>(&z0, 1U));
        mixed.add_term({-0.25, 0.0}, std::span<const PauliFactor>(&z3, 1U));
        PauliObservable cached(qubits);
        cached.add_term({0.5, 0.0}, std::span<const PauliFactor>(&z3, 1U));
        cached.add_term({0.125, 0.0});
        const std::vector<PauliObservable> observables{mixed, cached};

        ExactCausalAdjointGradientConfig config;
        config.adjoint = ExactAdjointGradientConfig{
            TensorNetworkConfig{1U << 16U, 100'000U}, 4U};
        ExactCausalAdjointGradientPlan causal(qubits, operations, observables, config);
        const auto stats = causal.stats();
        require(stats.dynamic_term_count == 1U && stats.static_term_count == 3U,
                "support classifier did not isolate disconnected static terms");
        require(causal.worker_count() == 1U,
                "causal adjoint did not clamp to one dynamic job");

        const std::vector<double> parameters{0.71};
        auto causal_workspace = causal.workspace();
        std::vector<QComplex> causal_values(observables.size());
        std::vector<QComplex> causal_gradients(observables.size());
        causal.value_and_gradient(
            parameters, causal_values, causal_gradients, causal_workspace);

        const auto reference_values = direct_values(qubits, operations, parameters, observables);
        const auto reference_gradients = shift_gradient(qubits, operations, parameters, observables);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            require_close(causal_values[index], reference_values[index], 4e-12,
                          "causal value differs from QRegister");
            require_close(causal_gradients[index], reference_gradients[index], 4e-12,
                          "causal gradient differs from parameter shift");
        }
        require_close(causal_gradients[1], {}, 0.0,
                      "disconnected static observable gradient is not exactly zero");

        ExactAdjointGradientPlan legacy(
            qubits, operations, observables,
            ExactAdjointGradientConfig{TensorNetworkConfig{1U << 16U, 100'000U}, 4U});
        auto legacy_workspace = legacy.workspace();
        std::vector<QComplex> legacy_values(observables.size());
        std::vector<QComplex> legacy_gradients(observables.size());
        legacy.value_and_gradient(
            parameters, legacy_values, legacy_gradients, legacy_workspace);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            require_close(causal_values[index], legacy_values[index], 2e-13,
                          "causal value differs from legacy adjoint");
            require_close(causal_gradients[index], legacy_gradients[index], 2e-13,
                          "causal gradient differs from legacy adjoint");
        }

        const auto preserved_values = causal_values;
        const auto preserved_gradients = causal_gradients;
        bool rejected = false;
        try {
            const std::vector<double> invalid{std::numeric_limits<double>::quiet_NaN()};
            causal.value_and_gradient(
                invalid, causal_values, causal_gradients, causal_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "causal adjoint accepted a nonfinite parameter");
        require(causal_values == preserved_values && causal_gradients == preserved_gradients,
                "rejected causal adjoint call mutated outputs");

        ExactCausalAdjointGradientPlan foreign(qubits, operations, observables, config);
        auto foreign_workspace = foreign.workspace();
        rejected = false;
        try {
            causal.value_and_gradient(
                parameters, causal_values, causal_gradients, foreign_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "causal adjoint accepted a foreign workspace");
    }

    {
        constexpr std::size_t qubits = 3U;
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
            {{OperationCode::Cnot, 0U, 1U, 0.0, 0.0}, -1, -1},
        };
        const std::vector<PauliObservable> observables{
            pauli(qubits, 1U, PauliAxis::Z),
            pauli(qubits, 2U, PauliAxis::Z),
        };
        ExactCausalAdjointGradientPlan plan(qubits, operations, observables);
        require(plan.stats().dynamic_term_count == 1U && plan.stats().static_term_count == 1U,
                "backward CNOT support cone classification changed");
        const std::vector<double> parameters{-0.62};
        auto workspace = plan.workspace();
        std::vector<QComplex> values(observables.size());
        std::vector<QComplex> gradients(observables.size());
        plan.value_and_gradient(parameters, values, gradients, workspace);
        const auto reference_values = direct_values(qubits, operations, parameters, observables);
        const auto reference_gradients = shift_gradient(qubits, operations, parameters, observables);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            require_close(values[index], reference_values[index], 4e-12,
                          "CNOT causal value differs from QRegister");
            require_close(gradients[index], reference_gradients[index], 4e-12,
                          "CNOT causal gradient differs from parameter shift");
        }
    }

    return 0;
}
