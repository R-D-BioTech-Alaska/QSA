#include "qubit/qtensor_adjoint_causal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::ExactCausalAdjointGradientConfig;
using qubit::ExactCausalAdjointGradientPlan;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::TensorNetworkConfig;

template <typename Function>
[[nodiscard]] double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

[[nodiscard]] std::vector<ParameterizedOperation> parameterized_brickwork(
    std::size_t qubits,
    std::size_t layers,
    std::size_t parameter_count) {
    std::vector<ParameterizedOperation> operations;
    operations.reserve(layers * (qubits * 2U + qubits / 2U));
    std::size_t next_parameter = 0U;
    for (std::size_t layer = 0U; layer < layers; ++layer) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            ParameterizedOperation ry;
            ry.operation = {
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.007 * static_cast<double>((layer + 1U) * (qubit + 3U)),
                0.0,
            };
            if (next_parameter < parameter_count) {
                ry.parameter_slot = static_cast<std::int32_t>(next_parameter++);
            }
            operations.push_back(ry);

            ParameterizedOperation rz;
            rz.operation = {
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                -0.005 * static_cast<double>((layer + 2U) * (qubit + 1U)),
                0.0,
            };
            if (next_parameter < parameter_count) {
                rz.parameter_slot = static_cast<std::int32_t>(next_parameter++);
            }
            operations.push_back(rz);
        }
        const std::size_t start = layer & 1U;
        for (std::size_t qubit = start; qubit + 1U < qubits; qubit += 2U) {
            ParameterizedOperation cnot;
            cnot.operation = {
                OperationCode::Cnot,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
                0.0,
                0.0,
            };
            operations.push_back(cnot);
        }
    }
    if (next_parameter != parameter_count) {
        throw std::runtime_error("causal adjoint benchmark did not assign all parameters");
    }
    return operations;
}

[[nodiscard]] std::vector<PauliObservable> observables(
    std::size_t qubits,
    std::size_t count) {
    std::vector<PauliObservable> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        PauliObservable observable(qubits, PauliPropagationConfig{65'536U});
        const std::size_t first = (index * 7U + 3U) % qubits;
        const std::size_t second = (first + 1U + index % 3U) % qubits;
        const std::size_t third = (second + 2U) % qubits;
        std::vector<PauliFactor> factors{
            {static_cast<qubit::QubitId>(first), static_cast<PauliAxis>(1U + index % 3U)},
            {static_cast<qubit::QubitId>(second), static_cast<PauliAxis>(1U + (index + 1U) % 3U)},
        };
        if ((index & 1U) != 0U && third != first && third != second) {
            factors.push_back({
                static_cast<qubit::QubitId>(third),
                static_cast<PauliAxis>(1U + (index + 2U) % 3U),
            });
        }
        observable.add_term({1.0, 0.0}, factors);
        result.push_back(std::move(observable));
    }
    return result;
}

[[nodiscard]] double max_error(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    if (first.size() != second.size()) {
        throw std::runtime_error("causal adjoint comparison shapes differ");
    }
    double error = 0.0;
    for (std::size_t index = 0U; index < first.size(); ++index) {
        error = std::max(error, (first[index] - second[index]).magnitude());
    }
    return error;
}

void run_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t parameter_count,
    std::size_t repetitions) {
    const TensorNetworkConfig tensor{1U << 16U, 1'000'000U};
    const std::vector<ParameterizedOperation> operations =
        parameterized_brickwork(qubits, 5U, parameter_count);
    const std::vector<PauliObservable> queries = observables(qubits, query_count);
    std::vector<double> parameters(parameter_count);
    for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
        parameters[parameter] =
            -0.39 + 0.113 * static_cast<double>(parameter + 1U);
    }
    const std::size_t workers = std::min<std::size_t>(4U, query_count);

    std::unique_ptr<ExactAdjointGradientPlan> legacy;
    const double legacy_compile_ms = milliseconds([&] {
        legacy = std::make_unique<ExactAdjointGradientPlan>(
            qubits,
            operations,
            queries,
            ExactAdjointGradientConfig{tensor, workers});
    });
    auto legacy_workspace = legacy->workspace();
    std::vector<QComplex> legacy_values(query_count);
    std::vector<QComplex> legacy_gradients(query_count * parameter_count);
    std::vector<double> legacy_times;
    legacy_times.reserve(repetitions);
    for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
        legacy_times.push_back(milliseconds([&] {
            legacy->value_and_gradient(
                parameters,
                legacy_values,
                legacy_gradients,
                legacy_workspace);
        }));
    }

    std::unique_ptr<ExactCausalAdjointGradientPlan> causal;
    const double causal_compile_ms = milliseconds([&] {
        causal = std::make_unique<ExactCausalAdjointGradientPlan>(
            qubits,
            operations,
            queries,
            ExactCausalAdjointGradientConfig{
                ExactAdjointGradientConfig{tensor, workers},
                65'536U,
            });
    });
    auto causal_workspace = causal->workspace();
    std::vector<QComplex> causal_values(query_count);
    std::vector<QComplex> causal_gradients(query_count * parameter_count);
    std::vector<double> causal_times;
    causal_times.reserve(repetitions);
    for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
        causal_times.push_back(milliseconds([&] {
            causal->value_and_gradient(
                parameters,
                causal_values,
                causal_gradients,
                causal_workspace);
        }));
    }

    const double value_error = max_error(legacy_values, causal_values);
    const double gradient_error = max_error(legacy_gradients, causal_gradients);
    if (value_error > 5e-11 || gradient_error > 5e-11) {
        throw std::runtime_error("causal adjoint differs from legacy exact adjoint");
    }

    const double legacy_first = legacy_times.front();
    const double legacy_best = *std::min_element(legacy_times.begin(), legacy_times.end());
    const double causal_first = causal_times.front();
    const double causal_best = *std::min_element(causal_times.begin(), causal_times.end());
    const auto legacy_stats = legacy->stats();
    const auto causal_stats = causal->stats();
    const std::string prefix = "causal_adjoint" + std::to_string(qubits);

    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_parameters=" << parameter_count << '\n';
    std::cout << prefix << "_occurrences="
              << causal->parameterized_operation_count() << '\n';
    std::cout << prefix << "_classification_dynamic_terms="
              << causal_stats.dynamic_term_count << '\n';
    std::cout << prefix << "_classification_static_terms="
              << causal_stats.static_term_count << '\n';
    std::cout << prefix << "_classification_dynamic_observables="
              << causal_stats.dynamic_observable_count << '\n';
    std::cout << prefix << "_classification_static_contribution_observables="
              << causal_stats.static_contribution_observable_count << '\n';
    std::cout << prefix << "_classification_fallback_terms="
              << causal_stats.classification_fallback_term_count << '\n';
    std::cout << prefix << "_legacy_differentiated_terms="
              << legacy_stats.differentiated_term_count << '\n';
    std::cout << prefix << "_causal_differentiated_terms="
              << causal_stats.dynamic.differentiated_term_count << '\n';
    std::cout << prefix << "_legacy_compile_ms=" << legacy_compile_ms << '\n';
    std::cout << prefix << "_causal_compile_ms=" << causal_compile_ms << '\n';
    std::cout << prefix << "_legacy_first_ms=" << legacy_first << '\n';
    std::cout << prefix << "_legacy_best_ms=" << legacy_best << '\n';
    std::cout << prefix << "_causal_first_ms=" << causal_first << '\n';
    std::cout << prefix << "_causal_best_ms=" << causal_best << '\n';
    std::cout << prefix << "_first_speedup="
              << legacy_first / causal_first << '\n';
    std::cout << prefix << "_best_speedup="
              << legacy_best / causal_best << '\n';
    std::cout << prefix << "_legacy_setup_plus_first_ms="
              << legacy_compile_ms + legacy_first << '\n';
    std::cout << prefix << "_causal_setup_plus_first_ms="
              << causal_compile_ms + causal_first << '\n';
    std::cout << prefix << "_end_to_end_speedup="
              << (legacy_compile_ms + legacy_first) /
                     (causal_compile_ms + causal_first)
              << '\n';
    std::cout << prefix << "_value_error=" << value_error << '\n';
    std::cout << prefix << "_gradient_error=" << gradient_error << '\n';
    std::cout << prefix << "_legacy_plan_bytes="
              << legacy->estimated_bytes() << '\n';
    std::cout << prefix << "_causal_plan_bytes="
              << causal->estimated_bytes() << '\n';
    std::cout << prefix << "_legacy_workspace_bytes="
              << legacy_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_causal_workspace_bytes="
              << causal_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_legacy_rebind_count="
              << legacy_workspace.rebind_count() << '\n';
    std::cout << prefix << "_causal_rebind_count="
              << causal_workspace.rebind_count() << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    run_case(18U, 24U, 6U, 5U);
    run_case(100U, 8U, 4U, 5U);
    return 0;
}
