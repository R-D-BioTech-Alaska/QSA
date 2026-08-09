#include "qubit/qestimator.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactEstimatorConfig;
using qubit::ExactEstimatorPlan;
using qubit::ExactEstimatorResult;
using qubit::ExactExecutionRoute;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QRegister;

template <typename Function>
[[nodiscard]] double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

[[nodiscard]] std::vector<Operation> brickwork(std::size_t qubits, std::size_t layers) {
    std::vector<Operation> operations;
    operations.reserve(layers * (qubits * 2U + qubits / 2U));
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.007 * static_cast<double>((layer + 1U) * (qubit + 3U)),
                0.0,
            });
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                -0.005 * static_cast<double>((layer + 2U) * (qubit + 1U)),
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

[[nodiscard]] std::vector<PauliObservable> observables(std::size_t qubits, std::size_t count) {
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

struct RouteCounts {
    std::size_t causal{0U};
    std::size_t tensor{0U};
    std::size_t register_state{0U};
};

[[nodiscard]] RouteCounts route_counts(std::span<const ExactEstimatorResult> results) {
    RouteCounts counts;
    for (const ExactEstimatorResult& result : results) {
        switch (result.route) {
            case ExactExecutionRoute::CausalPauli:
                ++counts.causal;
                break;
            case ExactExecutionRoute::TensorNetwork:
                ++counts.tensor;
                break;
            case ExactExecutionRoute::Register:
                ++counts.register_state;
                break;
        }
    }
    return counts;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    {
        constexpr std::size_t qubits = 18U;
        const std::vector<Operation> operations = brickwork(qubits, 5U);
        const std::vector<PauliObservable> queries = observables(qubits, 24U);

        QRegister state(qubits);
        const OperationPlan qregister_plan(operations, false);
        const double qregister_setup_ms = milliseconds([&] { qregister_plan.execute(state); });
        std::vector<QComplex> qregister_values(queries.size());
        const double qregister_query_ms = milliseconds([&] {
            for (std::size_t index = 0; index < queries.size(); ++index) {
                qregister_values[index] = queries[index].expectation(state);
            }
        });

        ExactEstimatorConfig serial_config;
        serial_config.tensor_worker_count = 1U;
        ExactEstimatorPlan serial_estimator(qubits, operations, serial_config);
        std::unique_ptr<qubit::ExactEstimatorBatchPlan> serial_batch;
        const double serial_compile_ms = milliseconds([&] {
            serial_batch = std::make_unique<qubit::ExactEstimatorBatchPlan>(
                serial_estimator.compile(queries));
        });
        auto serial_workspace = serial_batch->workspace();
        std::vector<ExactEstimatorResult> serial_results(queries.size());
        const double serial_query_ms = milliseconds([&] {
            serial_batch->estimate(serial_results, serial_workspace);
        });

        std::unique_ptr<ExactEstimatorPlan> estimator;
        const double estimator_circuit_compile_ms = milliseconds([&] {
            estimator = std::make_unique<ExactEstimatorPlan>(qubits, operations);
        });
        std::unique_ptr<qubit::ExactEstimatorBatchPlan> batch;
        const double estimator_observable_compile_ms = milliseconds([&] {
            batch = std::make_unique<qubit::ExactEstimatorBatchPlan>(
                estimator->compile(queries));
        });
        auto estimator_workspace = batch->workspace();
        std::vector<ExactEstimatorResult> results(queries.size());
        const double estimator_query_ms = milliseconds([&] {
            batch->estimate(results, estimator_workspace);
        });

        double max_error = 0.0;
        double serial_parallel_error = 0.0;
        std::size_t peak_terms = 0U;
        std::size_t peak_support = 0U;
        std::size_t visited_operations = 0U;
        for (std::size_t index = 0; index < queries.size(); ++index) {
            max_error = std::max(
                max_error,
                (qregister_values[index] - results[index].value).magnitude());
            serial_parallel_error = std::max(
                serial_parallel_error,
                (serial_results[index].value - results[index].value).magnitude());
            peak_terms = std::max(peak_terms, results[index].pauli_stats.peak_terms);
            peak_support = std::max(peak_support, results[index].pauli_stats.peak_support);
            visited_operations += results[index].pauli_stats.visited_operations;
        }
        const RouteCounts counts = route_counts(results);
        std::cout << "est18_qubits=" << qubits << '\n';
        std::cout << "est18_operations=" << operations.size() << '\n';
        std::cout << "est18_queries=" << queries.size() << '\n';
        std::cout << "est18_qregister_setup_ms=" << qregister_setup_ms << '\n';
        std::cout << "est18_qregister_query_ms=" << qregister_query_ms << '\n';
        std::cout << "est18_serial_tensor_workers=" << serial_batch->tensor_worker_count() << '\n';
        std::cout << "est18_serial_observable_compile_ms=" << serial_compile_ms << '\n';
        std::cout << "est18_serial_query_ms=" << serial_query_ms << '\n';
        std::cout << "est18_circuit_compile_ms=" << estimator_circuit_compile_ms << '\n';
        std::cout << "est18_observable_compile_ms=" << estimator_observable_compile_ms << '\n';
        std::cout << "est18_compile_parallel_ratio="
                  << serial_compile_ms / estimator_observable_compile_ms << '\n';
        std::cout << "est18_tensor_workers=" << batch->tensor_worker_count() << '\n';
        std::cout << "est18_query_ms=" << estimator_query_ms << '\n';
        std::cout << "est18_parallel_ratio=" << serial_query_ms / estimator_query_ms << '\n';
        std::cout << "est18_end_to_end_ratio="
                  << (qregister_setup_ms + qregister_query_ms) /
                         (estimator_circuit_compile_ms +
                          estimator_observable_compile_ms +
                          estimator_query_ms)
                  << '\n';
        std::cout << "est18_post_state_ratio=" << qregister_query_ms / estimator_query_ms << '\n';
        std::cout << "est18_max_error=" << max_error << '\n';
        std::cout << "est18_serial_parallel_error=" << serial_parallel_error << '\n';
        std::cout << "est18_causal_routes=" << counts.causal << '\n';
        std::cout << "est18_tensor_routes=" << counts.tensor << '\n';
        std::cout << "est18_register_routes=" << counts.register_state << '\n';
        std::cout << "est18_peak_pauli_terms=" << peak_terms << '\n';
        std::cout << "est18_peak_pauli_support=" << peak_support << '\n';
        std::cout << "est18_visited_operations=" << visited_operations << '\n';
        std::cout << "est18_circuit_plan_bytes=" << estimator->estimated_bytes() << '\n';
        std::cout << "est18_observable_plan_bytes=" << batch->estimated_bytes() << '\n';
        std::cout << "est18_workspace_bytes=" << estimator_workspace.estimated_bytes() << '\n';
    }

    {
        constexpr std::size_t qubits = 100U;
        const std::vector<Operation> operations = brickwork(qubits, 5U);
        const std::vector<PauliObservable> queries = observables(qubits, 8U);

        ExactEstimatorConfig serial_config;
        serial_config.tensor_worker_count = 1U;
        ExactEstimatorPlan serial_estimator(qubits, operations, serial_config);
        std::unique_ptr<qubit::ExactEstimatorBatchPlan> serial_batch;
        const double serial_compile_ms = milliseconds([&] {
            serial_batch = std::make_unique<qubit::ExactEstimatorBatchPlan>(
                serial_estimator.compile(queries));
        });
        auto serial_workspace = serial_batch->workspace();
        std::vector<ExactEstimatorResult> serial_results(queries.size());
        const double serial_query_ms = milliseconds([&] {
            serial_batch->estimate(serial_results, serial_workspace);
        });

        std::unique_ptr<ExactEstimatorPlan> estimator;
        const double circuit_compile_ms = milliseconds([&] {
            estimator = std::make_unique<ExactEstimatorPlan>(qubits, operations);
        });
        std::unique_ptr<qubit::ExactEstimatorBatchPlan> batch;
        const double observable_compile_ms = milliseconds([&] {
            batch = std::make_unique<qubit::ExactEstimatorBatchPlan>(
                estimator->compile(queries));
        });
        auto estimator_workspace = batch->workspace();
        std::vector<ExactEstimatorResult> results(queries.size());
        const double query_ms = milliseconds([&] {
            batch->estimate(results, estimator_workspace);
        });
        QComplex checksum{};
        double serial_parallel_error = 0.0;
        std::size_t peak_terms = 0U;
        std::size_t peak_support = 0U;
        std::size_t visited_operations = 0U;
        for (std::size_t index = 0; index < results.size(); ++index) {
            const ExactEstimatorResult& result = results[index];
            checksum += result.value;
            serial_parallel_error = std::max(
                serial_parallel_error,
                (serial_results[index].value - result.value).magnitude());
            peak_terms = std::max(peak_terms, result.pauli_stats.peak_terms);
            peak_support = std::max(peak_support, result.pauli_stats.peak_support);
            visited_operations += result.pauli_stats.visited_operations;
        }
        const RouteCounts counts = route_counts(results);
        std::cout << "est100_qubits=" << qubits << '\n';
        std::cout << "est100_operations=" << operations.size() << '\n';
        std::cout << "est100_queries=" << queries.size() << '\n';
        std::cout << "est100_serial_tensor_workers=" << serial_batch->tensor_worker_count() << '\n';
        std::cout << "est100_serial_observable_compile_ms=" << serial_compile_ms << '\n';
        std::cout << "est100_serial_query_ms=" << serial_query_ms << '\n';
        std::cout << "est100_circuit_compile_ms=" << circuit_compile_ms << '\n';
        std::cout << "est100_observable_compile_ms=" << observable_compile_ms << '\n';
        std::cout << "est100_compile_parallel_ratio="
                  << serial_compile_ms / observable_compile_ms << '\n';
        std::cout << "est100_tensor_workers=" << batch->tensor_worker_count() << '\n';
        std::cout << "est100_query_ms=" << query_ms << '\n';
        std::cout << "est100_parallel_ratio=" << serial_query_ms / query_ms << '\n';
        std::cout << "est100_serial_parallel_error=" << serial_parallel_error << '\n';
        std::cout << "est100_causal_routes=" << counts.causal << '\n';
        std::cout << "est100_tensor_routes=" << counts.tensor << '\n';
        std::cout << "est100_register_routes=" << counts.register_state << '\n';
        std::cout << "est100_peak_pauli_terms=" << peak_terms << '\n';
        std::cout << "est100_peak_pauli_support=" << peak_support << '\n';
        std::cout << "est100_visited_operations=" << visited_operations << '\n';
        std::cout << "est100_circuit_plan_bytes=" << estimator->estimated_bytes() << '\n';
        std::cout << "est100_observable_plan_bytes=" << batch->estimated_bytes() << '\n';
        std::cout << "est100_workspace_bytes=" << estimator_workspace.estimated_bytes() << '\n';
        std::cout << "est100_checksum_real=" << checksum.re << '\n';
        std::cout << "est100_checksum_imag=" << checksum.im << '\n';
    }

    return 0;
}
