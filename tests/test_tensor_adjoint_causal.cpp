#include "qubit/qtensor_adjoint_causal.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::ExactCausalAdjointGradientConfig;
using qubit::ExactCausalAdjointGradientPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
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

[[nodiscard]] std::vector<Operation> bind(
    const std::vector<ParameterizedOperation>& operations,
    std::span<const double> parameters) {
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

[[nodiscard]] std::vector<QComplex> values(
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

[[nodiscard]] std::vector<QComplex> parameter_shift(
    std::size_t qubits,
    const std::vector<ParameterizedOperation>& operations,
    std::span<const double> parameters,
    const std::vector<PauliObservable>& observables,
    std::size_t parameter_count) {
    constexpr double shift = 1.57079632679489661923;
    std::vector<QComplex> result(observables.size() * parameter_count);
    std::vector<double> shifted(parameters.begin(), parameters.end());
    for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
        shifted.assign(parameters.begin(), parameters.end());
        shifted[parameter] += shift;
        const std::vector<QComplex> plus =
            values(qubits, operations, shifted, observables);
        shifted[parameter] -= 2.0 * shift;
        const std::vector<QComplex> minus =
            values(qubits, operations, shifted, observables);
        for (std::size_t observable = 0U;
             observable < observables.size();
             ++observable) {
            result[observable * parameter_count + parameter] =
                (plus[observable] - minus[observable]) * 0.5;
        }
    }
    return result;
}

[[nodiscard]] PauliObservable single_pauli(
    std::size_t qubits,
    qubit::QubitId qubit,
    PauliAxis axis) {
    PauliObservable observable(qubits);
    const PauliFactor factor{qubit, axis};
    observable.add_term(
        {1.0, 0.0},
        std::span<const PauliFactor>(&factor, 1U));
    return observable;
}

}  // namespace

int main() {
    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        const std::vector<PauliObservable> observables{
            single_pauli(1U, 0U, PauliAxis::Z),
        };
        ExactCausalAdjointGradientPlan plan(1U, operations, observables);
        const auto stats = plan.stats();
        require(stats.dynamic_term_count == 1U,
                "Ry/Z causal classifier did not retain the dynamic term");
        require(stats.static_term_count == 0U,
                "Ry/Z causal classifier produced a false static term");
        require(stats.classification_fallback_term_count == 0U,
                "Ry/Z causal classifier unexpectedly fell back");

        const std::vector<double> parameters{0.43};
        auto workspace = plan.workspace();
        std::vector<QComplex> output(1U);
        std::vector<QComplex> gradient(1U);
        plan.value_and_gradient(parameters, output, gradient, workspace);
        require_close(output[0], {std::cos(parameters[0]), 0.0}, 2e-12,
                      "causal Ry/Z value is incorrect");
        require_close(gradient[0], {-std::sin(parameters[0]), 0.0}, 2e-12,
                      "causal Ry/Z gradient is incorrect");
        require(workspace.rebind_count() == 1U,
                "dynamic causal adjoint did not use one exact rebind");
    }

    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Rz, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        const std::vector<PauliObservable> observables{
            single_pauli(1U, 0U, PauliAxis::Z),
        };
        ExactCausalAdjointGradientPlan plan(1U, operations, observables);
        const auto stats = plan.stats();
        require(stats.dynamic_term_count == 0U,
                "Rz/Z commutation was not certified static");
        require(stats.static_term_count == 1U,
                "Rz/Z static term was not cached");
        require(plan.worker_count() == 1U,
                "static-only causal adjoint allocated dynamic workers");

        auto workspace = plan.workspace();
        const std::vector<double> parameters{1.37};
        std::vector<QComplex> output(1U);
        std::vector<QComplex> gradient(1U, {99.0, 99.0});
        plan.value_and_gradient(parameters, output, gradient, workspace);
        require_close(output[0], {1.0, 0.0}, 0.0,
                      "Rz/Z cached value changed");
        require_close(gradient[0], {}, 0.0,
                      "Rz/Z cached gradient is not exactly zero");
        require(workspace.rebind_count() == 0U,
                "static-only causal adjoint performed a tensor rebind");
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

        PauliObservable static_only(qubits);
        static_only.add_term({0.5, 0.0}, std::span<const PauliFactor>(&z3, 1U));
        static_only.add_term({0.125, 0.0});

        const std::vector<PauliObservable> observables{mixed, static_only};
        ExactCausalAdjointGradientConfig config;
        config.adjoint = ExactAdjointGradientConfig{
            TensorNetworkConfig{1U << 16U, 100'000U},
            4U,
        };
        ExactCausalAdjointGradientPlan causal(
            qubits, operations, observables, config);
        const auto stats = causal.stats();
        require(stats.dynamic_term_count == 1U,
                "mixed causal workload did not isolate one dynamic term");
        require(stats.static_term_count == 3U,
                "mixed causal workload did not cache three static terms");
        require(stats.dynamic_observable_count == 1U,
                "mixed causal workload dynamic observable count changed");
        require(stats.static_contribution_observable_count == 2U,
                "mixed causal workload static contribution count changed");
        require(causal.worker_count() == 1U,
                "causal adjoint did not clamp workers to one dynamic job");

        const std::vector<double> parameters{0.71};
        auto workspace = causal.workspace();
        std::vector<QComplex> causal_values(observables.size());
        std::vector<QComplex> causal_gradients(observables.size());
        causal.value_and_gradient(
            parameters, causal_values, causal_gradients, workspace);

        const std::vector<QComplex> reference_values =
            values(qubits, operations, parameters, observables);
        const std::vector<QComplex> reference_gradients =
            parameter_shift(
                qubits, operations, parameters, observables, parameters.size());
        for (std::size_t observable = 0U;
             observable < observables.size();
             ++observable) {
            require_close(
                causal_values[observable], reference_values[observable], 3e-12,
                "causal mixed value differs from QRegister");
            require_close(
                causal_gradients[observable], reference_gradients[observable], 4e-12,
                "causal mixed gradient differs from exact parameter shift");
        }
        require_close(causal_gradients[1], {}, 0.0,
                      "static observable gradient is not exactly zero");

        ExactAdjointGradientPlan legacy(
            qubits,
            operations,
            observables,
            ExactAdjointGradientConfig{
                TensorNetworkConfig{1U << 16U, 100'000U},
                4U,
            });
        auto legacy_workspace = legacy.workspace();
        std::vector<QComplex> legacy_values(observables.size());
        std::vector<QComplex> legacy_gradients(observables.size());
        legacy.value_and_gradient(
            parameters, legacy_values, legacy_gradients, legacy_workspace);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            require_close(
                causal_values[index], legacy_values[index], 2e-13,
                "causal value differs from legacy exact adjoint");
            require_close(
                causal_gradients[index], legacy_gradients[index], 2e-13,
                "causal gradient differs from legacy exact adjoint");
        }

        const std::vector<QComplex> preserved_values = causal_values;
        const std::vector<QComplex> preserved_gradients = causal_gradients;
        bool rejected = false;
        try {
            const std::vector<double> invalid{
                std::numeric_limits<double>::quiet_NaN(),
            };
            causal.value_and_gradient(
                invalid, causal_values, causal_gradients, workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "causal adjoint accepted a nonfinite parameter");
        require(causal_values == preserved_values,
                "rejected causal adjoint call mutated values");
        require(causal_gradients == preserved_gradients,
                "rejected causal adjoint call mutated gradients");

        ExactCausalAdjointGradientPlan foreign(
            qubits, operations, observables, config);
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
            single_pauli(qubits, 1U, PauliAxis::Z),
            single_pauli(qubits, 2U, PauliAxis::Z),
        };
        ExactCausalAdjointGradientPlan plan(qubits, operations, observables);
        const auto stats = plan.stats();
        require(stats.dynamic_term_count == 1U,
                "CNOT backward cone did not connect target observable to parameter");
        require(stats.static_term_count == 1U,
                "disconnected observable was not certified static");

        const std::vector<double> parameters{-0.62};
        auto workspace = plan.workspace();
        std::vector<QComplex> output(observables.size());
        std::vector<QComplex> gradient(observables.size());
        plan.value_and_gradient(parameters, output, gradient, workspace);
        const std::vector<QComplex> reference_values =
            values(qubits, operations, parameters, observables);
        const std::vector<QComplex> reference_gradient =
            parameter_shift(
                qubits, operations, parameters, observables, 1U);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            require_close(output[index], reference_values[index], 4e-12,
                          "CNOT causal value differs from QRegister");
            require_close(gradient[index], reference_gradient[index], 4e-12,
                          "CNOT causal gradient differs from parameter shift");
        }
    }

    return 0;
}
