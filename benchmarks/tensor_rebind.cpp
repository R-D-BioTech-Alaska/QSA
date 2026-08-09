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
using qubit::ExactParameterShiftConfig;
using qubit::ExactParameterShiftPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::TensorExpectationPlan;
using qubit::TensorExpectationRebindPlan;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

template <typename Function>
[[nodiscard]] double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

std::vector<Operation> brickwork(
    std::size_t qubits,
    std::size_t layers,
    double scale) {
    std::vector<Operation> operations;
    operations.reserve(layers * (qubits * 2U + qubits / 2U));
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                scale * 0.007 * static_cast<double>((layer + 1U) * (qubit + 3U)),
                0.0,
            });
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                -scale * 0.005 * static_cast<double>((layer + 2U) * (qubit + 1U)),
                0.0,
            });
        }
        const std::size_t start = layer & 1U;
        for (std::size_t qubit = start; qubit + 1U < qubits; qubit += 2U) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
                0.0,
                0.0,
            });
        }
    }
    return operations;
}

std::vector<ParameterizedOperation> parameterized_brickwork(
    std::size_t qubits,
    std::size_t layers,
    std::size_t parameter_count) {
    const std::vector<Operation> base = brickwork(qubits, layers, 1.0);
    std::vector<ParameterizedOperation> result;
    result.reserve(base.size());
    std::size_t next_parameter = 0U;
    for (const Operation& operation : base) {
        ParameterizedOperation templated;
        templated.operation = operation;
        if (next_parameter < parameter_count &&
            (operation.code == OperationCode::Rx ||
             operation.code == OperationCode::Ry ||
             operation.code == OperationCode::Rz)) {
            templated.parameter_slot = static_cast<std::int32_t>(next_parameter++);
        }
        result.push_back(templated);
    }
    return result;
}

std::vector<Operation> bind_parameterized(
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

std::vector<PauliObservable> observables(std::size_t qubits, std::size_t count) {
    std::vector<PauliObservable> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
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

void evaluate_fresh(
    std::size_t qubits,
    const std::vector<Operation>& operations,
    const std::vector<PauliObservable>& queries,
    TensorNetworkConfig config,
    std::vector<QComplex>& values) {
    const TensorNetworkCircuit circuit(qubits, operations, config);
    TensorExpectationPlan plan(circuit, queries);
    auto workspace = plan.workspace();
    plan.expectations(values, workspace);
}

void run_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t sweeps) {
    const std::vector<PauliObservable> queries = observables(qubits, query_count);
    const TensorNetworkCircuit initial(
        qubits,
        brickwork(qubits, 5U, 1.0));
    TensorExpectationRebindPlan reusable(initial, queries);
    auto reusable_workspace = reusable.workspace();
    std::vector<QComplex> reusable_values(query_count);
    std::vector<QComplex> fresh_values(query_count);

    double fresh_compile_ms = 0.0;
    double fresh_query_ms = 0.0;
    double rebind_ms = 0.0;
    double rebind_query_ms = 0.0;
    double max_error = 0.0;

    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        const double scale = 0.55 + 0.073 * static_cast<double>(sweep + 1U);
        const TensorNetworkCircuit circuit(
            qubits,
            brickwork(qubits, 5U, scale));

        std::unique_ptr<TensorExpectationPlan> fresh;
        fresh_compile_ms += milliseconds([&] {
            fresh = std::make_unique<TensorExpectationPlan>(circuit, queries);
        });
        auto fresh_workspace = fresh->workspace();
        fresh_query_ms += milliseconds([&] {
            fresh->expectations(fresh_values, fresh_workspace);
        });

        rebind_ms += milliseconds([&] {
            reusable.rebind(circuit);
        });
        rebind_query_ms += milliseconds([&] {
            reusable.expectations(reusable_values, reusable_workspace);
        });

        for (std::size_t index = 0; index < query_count; ++index) {
            max_error = std::max(
                max_error,
                (fresh_values[index] - reusable_values[index]).magnitude());
        }
    }

    const std::string prefix = "rebind" + std::to_string(qubits);
    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_sweeps=" << sweeps << '\n';
    std::cout << prefix << "_fresh_compile_total_ms=" << fresh_compile_ms << '\n';
    std::cout << prefix << "_rebind_total_ms=" << rebind_ms << '\n';
    std::cout << prefix << "_compile_speedup=" << fresh_compile_ms / rebind_ms << '\n';
    std::cout << prefix << "_fresh_query_total_ms=" << fresh_query_ms << '\n';
    std::cout << prefix << "_rebind_query_total_ms=" << rebind_query_ms << '\n';
    std::cout << prefix << "_fresh_total_ms=" << fresh_compile_ms + fresh_query_ms << '\n';
    std::cout << prefix << "_reused_total_ms=" << rebind_ms + rebind_query_ms << '\n';
    std::cout << prefix << "_total_speedup="
              << (fresh_compile_ms + fresh_query_ms) /
                     (rebind_ms + rebind_query_ms)
              << '\n';
    std::cout << prefix << "_max_error=" << max_error << '\n';
    std::cout << prefix << "_rebind_count=" << reusable.rebind_count() << '\n';
    std::cout << prefix << "_plan_bytes=" << reusable.estimated_bytes() << '\n';
    std::cout << prefix << "_workspace_bytes="
              << reusable_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_peak_entries="
              << reusable.stats().peak_contraction_entries << '\n';
    std::cout << prefix << "_peak_variables="
              << reusable.stats().peak_union_variables << '\n';
}

void run_gradient_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t parameter_count) {
    constexpr double shift = 1.57079632679489661923;
    const TensorNetworkConfig tensor{1U << 16U, 1'000'000U};
    const std::vector<PauliObservable> queries = observables(qubits, query_count);
    const std::vector<ParameterizedOperation> operations =
        parameterized_brickwork(qubits, 5U, parameter_count);
    std::vector<double> parameters(parameter_count);
    for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
        parameters[parameter] =
            -0.41 + 0.137 * static_cast<double>(parameter + 1U);
    }

    std::vector<QComplex> fresh_values(query_count);
    std::vector<QComplex> fresh_gradients(query_count * parameter_count);
    const double fresh_ms = milliseconds([&] {
        std::vector<Operation> bound = bind_parameterized(operations, parameters);
        evaluate_fresh(qubits, bound, queries, tensor, fresh_values);
        for (std::size_t operation_index = 0U;
             operation_index < operations.size();
             ++operation_index) {
            if (operations[operation_index].parameter_slot < 0) {
                continue;
            }
            const std::size_t parameter =
                static_cast<std::size_t>(operations[operation_index].parameter_slot);
            const double original = bound[operation_index].parameter;
            std::vector<QComplex> plus(query_count);
            std::vector<QComplex> minus(query_count);
            bound[operation_index].parameter = original + shift;
            evaluate_fresh(qubits, bound, queries, tensor, plus);
            bound[operation_index].parameter = original - shift;
            evaluate_fresh(qubits, bound, queries, tensor, minus);
            bound[operation_index].parameter = original;
            for (std::size_t observable = 0U; observable < query_count; ++observable) {
                fresh_gradients[observable * parameter_count + parameter] +=
                    (plus[observable] - minus[observable]) * 0.5;
            }
        }
    });

    std::unique_ptr<ExactParameterShiftPlan> serial;
    const double serial_compile_ms = milliseconds([&] {
        serial = std::make_unique<ExactParameterShiftPlan>(
            qubits,
            operations,
            queries,
            ExactParameterShiftConfig{tensor, 1U});
    });
    auto serial_workspace = serial->workspace();
    std::vector<QComplex> serial_values(query_count);
    std::vector<QComplex> serial_gradients(query_count * parameter_count);
    const double serial_eval_ms = milliseconds([&] {
        serial->value_and_gradient(
            parameters, serial_values, serial_gradients, serial_workspace);
    });

    const std::size_t parallel_workers = std::min<std::size_t>(4U, parameter_count);
    std::unique_ptr<ExactParameterShiftPlan> parallel;
    const double parallel_compile_ms = milliseconds([&] {
        parallel = std::make_unique<ExactParameterShiftPlan>(
            qubits,
            operations,
            queries,
            ExactParameterShiftConfig{tensor, parallel_workers});
    });
    auto parallel_workspace = parallel->workspace();
    std::vector<QComplex> parallel_values(query_count);
    std::vector<QComplex> parallel_gradients(query_count * parameter_count);
    const double parallel_eval_ms = milliseconds([&] {
        parallel->value_and_gradient(
            parameters, parallel_values, parallel_gradients, parallel_workspace);
    });

    double value_error = 0.0;
    double gradient_error = 0.0;
    double parallel_error = 0.0;
    for (std::size_t observable = 0U; observable < query_count; ++observable) {
        value_error = std::max(
            value_error,
            (fresh_values[observable] - serial_values[observable]).magnitude());
    }
    for (std::size_t index = 0U; index < fresh_gradients.size(); ++index) {
        gradient_error = std::max(
            gradient_error,
            (fresh_gradients[index] - serial_gradients[index]).magnitude());
        parallel_error = std::max(
            parallel_error,
            (serial_gradients[index] - parallel_gradients[index]).magnitude());
    }

    const double serial_total_ms = serial_compile_ms + serial_eval_ms;
    const double parallel_total_ms = parallel_compile_ms + parallel_eval_ms;
    const std::string prefix = "gradient" + std::to_string(qubits);
    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_parameters=" << parameter_count << '\n';
    std::cout << prefix << "_occurrences="
              << serial->parameterized_operation_count() << '\n';
    std::cout << prefix << "_fresh_total_ms=" << fresh_ms << '\n';
    std::cout << prefix << "_serial_compile_ms=" << serial_compile_ms << '\n';
    std::cout << prefix << "_serial_eval_ms=" << serial_eval_ms << '\n';
    std::cout << prefix << "_serial_first_total_ms=" << serial_total_ms << '\n';
    std::cout << prefix << "_serial_reuse_speedup=" << fresh_ms / serial_total_ms << '\n';
    std::cout << prefix << "_parallel_workers=" << parallel->worker_count() << '\n';
    std::cout << prefix << "_parallel_compile_ms=" << parallel_compile_ms << '\n';
    std::cout << prefix << "_parallel_eval_ms=" << parallel_eval_ms << '\n';
    std::cout << prefix << "_parallel_first_total_ms=" << parallel_total_ms << '\n';
    std::cout << prefix << "_parallel_eval_ratio="
              << serial_eval_ms / parallel_eval_ms << '\n';
    std::cout << prefix << "_value_error=" << value_error << '\n';
    std::cout << prefix << "_gradient_error=" << gradient_error << '\n';
    std::cout << prefix << "_parallel_error=" << parallel_error << '\n';
    std::cout << prefix << "_serial_rebind_count="
              << serial_workspace.rebind_count() << '\n';
    std::cout << prefix << "_parallel_rebind_count="
              << parallel_workspace.rebind_count() << '\n';
    std::cout << prefix << "_evaluations="
              << serial->stats().value_and_gradient_evaluations << '\n';
    std::cout << prefix << "_plan_bytes=" << serial->estimated_bytes() << '\n';
    std::cout << prefix << "_workspace_bytes="
              << serial_workspace.estimated_bytes() << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    run_case(18U, 24U, 12U);
    run_case(100U, 8U, 8U);
    run_gradient_case(18U, 24U, 6U);
    run_gradient_case(100U, 8U, 4U);
    return 0;
}
