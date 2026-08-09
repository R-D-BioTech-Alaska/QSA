#include "qubit/qtensor_adjoint.hpp"
#include "qubit/qtensor_rebind.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::ExactParameterShiftConfig;
using qubit::ExactParameterShiftPlan;
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

std::vector<ParameterizedOperation> parameterized_brickwork(
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
    return operations;
}

std::vector<PauliObservable> observables(std::size_t qubits, std::size_t count) {
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
        if (index % 2U != 0U && third != first && third != second) {
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

void run_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t parameter_count) {
    const TensorNetworkConfig tensor{1U << 16U, 1'000'000U};
    const std::vector<ParameterizedOperation> operations =
        parameterized_brickwork(qubits, 5U, parameter_count);
    const std::vector<PauliObservable> queries = observables(qubits, query_count);
    std::vector<double> parameters(parameter_count);
    for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
        parameters[parameter] =
            -0.39 + 0.113 * static_cast<double>(parameter + 1U);
    }

    std::unique_ptr<ExactParameterShiftPlan> shift;
    const double shift_compile_ms = milliseconds([&] {
        shift = std::make_unique<ExactParameterShiftPlan>(
            qubits,
            operations,
            queries,
            ExactParameterShiftConfig{tensor, 1U});
    });
    auto shift_workspace = shift->workspace();
    std::vector<QComplex> shift_values(query_count);
    std::vector<QComplex> shift_gradients(query_count * parameter_count);
    const double shift_eval_ms = milliseconds([&] {
        shift->value_and_gradient(
            parameters, shift_values, shift_gradients, shift_workspace);
    });

    std::unique_ptr<ExactAdjointGradientPlan> adjoint;
    const double adjoint_compile_ms = milliseconds([&] {
        adjoint = std::make_unique<ExactAdjointGradientPlan>(
            qubits,
            operations,
            queries,
            ExactAdjointGradientConfig{tensor});
    });
    auto adjoint_workspace = adjoint->workspace();
    std::vector<QComplex> adjoint_values(query_count);
    std::vector<QComplex> adjoint_gradients(query_count * parameter_count);
    const double adjoint_eval_ms = milliseconds([&] {
        adjoint->value_and_gradient(
            parameters, adjoint_values, adjoint_gradients, adjoint_workspace);
    });

    double value_error = 0.0;
    double gradient_error = 0.0;
    for (std::size_t observable = 0U; observable < query_count; ++observable) {
        value_error = std::max(
            value_error,
            (shift_values[observable] - adjoint_values[observable]).magnitude());
    }
    for (std::size_t index = 0U; index < shift_gradients.size(); ++index) {
        gradient_error = std::max(
            gradient_error,
            (shift_gradients[index] - adjoint_gradients[index]).magnitude());
    }

    const auto stats = adjoint->stats();
    const std::string prefix = "adjoint" + std::to_string(qubits);
    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_parameters=" << parameter_count << '\n';
    std::cout << prefix << "_occurrences="
              << adjoint->parameterized_operation_count() << '\n';
    std::cout << prefix << "_shift_compile_ms=" << shift_compile_ms << '\n';
    std::cout << prefix << "_shift_eval_ms=" << shift_eval_ms << '\n';
    std::cout << prefix << "_shift_first_total_ms="
              << shift_compile_ms + shift_eval_ms << '\n';
    std::cout << prefix << "_adjoint_compile_ms=" << adjoint_compile_ms << '\n';
    std::cout << prefix << "_adjoint_eval_ms=" << adjoint_eval_ms << '\n';
    std::cout << prefix << "_adjoint_first_total_ms="
              << adjoint_compile_ms + adjoint_eval_ms << '\n';
    std::cout << prefix << "_eval_speedup=" << shift_eval_ms / adjoint_eval_ms << '\n';
    std::cout << prefix << "_first_total_speedup="
              << (shift_compile_ms + shift_eval_ms) /
                     (adjoint_compile_ms + adjoint_eval_ms)
              << '\n';
    std::cout << prefix << "_value_error=" << value_error << '\n';
    std::cout << prefix << "_gradient_error=" << gradient_error << '\n';
    std::cout << prefix << "_rebind_count="
              << adjoint_workspace.rebind_count() << '\n';
    std::cout << prefix << "_differentiated_terms="
              << stats.differentiated_term_count << '\n';
    std::cout << prefix << "_source_derivative_bindings="
              << stats.source_derivative_bindings << '\n';
    std::cout << prefix << "_parameter_shift_equivalent_evaluations="
              << stats.parameter_shift_equivalent_evaluations << '\n';
    std::cout << prefix << "_plan_bytes=" << adjoint->estimated_bytes() << '\n';
    std::cout << prefix << "_workspace_bytes="
              << adjoint_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_peak_entries="
              << stats.tensor.peak_contraction_entries << '\n';
    std::cout << prefix << "_peak_variables="
              << stats.tensor.peak_union_variables << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    run_case(18U, 24U, 6U);
    run_case(100U, 8U, 4U);
    return 0;
}
