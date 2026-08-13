#include "qubit/qfabric.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t qubits = 4096U;
    constexpr std::size_t block = 4U;
    constexpr std::size_t blocks = qubits / block;

    std::vector<Operation> operations;
    operations.reserve(blocks * 4U + 1U);
    for (std::size_t index = 0U; index < blocks; ++index) {
        const QubitId base = static_cast<QubitId>(index * block);
        operations.push_back({OperationCode::H, base});
        operations.push_back({OperationCode::Cnot, base, static_cast<QubitId>(base + 1U)});
        operations.push_back({OperationCode::Cnot, static_cast<QubitId>(base + 1U), static_cast<QubitId>(base + 2U)});
        operations.push_back({OperationCode::Cnot, static_cast<QubitId>(base + 2U), static_cast<QubitId>(base + 3U)});
    }
    operations.push_back({OperationCode::T, static_cast<QubitId>(qubits - block)});

    const std::vector<QubitId> query{
        0U, 1U, 2U, 3U,
        2048U, 2049U, 2050U, 2051U,
    };
    const std::vector<std::uint8_t> bits(query.size(), 0U);

    ExactComponentFabricConfig config;
    config.max_qubits = 8192U;
    config.max_components = 2048U;
    config.max_operations = 8192U;

    const auto full_setup_begin = std::chrono::steady_clock::now();
    const ExactComponentProbabilityPlan full_plan(qubits, operations, config);
    const auto full_setup_end = std::chrono::steady_clock::now();

    const auto pruned_setup_begin = std::chrono::steady_clock::now();
    const ExactComponentProbabilityPlan pruned_plan = ExactComponentProbabilityPlan::for_marginals(
        qubits, operations, query, config);
    const auto pruned_setup_end = std::chrono::steady_clock::now();

    const auto global_setup_begin = std::chrono::steady_clock::now();
    const ExactPreparedProbabilityPlan global = ExactPreparedProbabilityPlan::for_marginals(
        qubits, operations, config.broker);
    const auto global_setup_end = std::chrono::steady_clock::now();

    const auto pruned_query_begin = std::chrono::steady_clock::now();
    const double probability = pruned_plan.marginal_probability(query, bits);
    const auto pruned_query_end = std::chrono::steady_clock::now();
    const auto global_query_begin = std::chrono::steady_clock::now();
    const double global_probability = global.marginal_probability(query, bits).value;
    const auto global_query_end = std::chrono::steady_clock::now();

    std::size_t full_stabilizer_routes = 0U;
    std::size_t full_non_stabilizer_routes = 0U;
    for (const ExactComponentReceipt& receipt : full_plan.receipts()) {
        if (!receipt.prepared) {
            continue;
        }
        if (receipt.route == ExactExecutionRoute::Stabilizer) {
            ++full_stabilizer_routes;
        } else {
            ++full_non_stabilizer_routes;
        }
    }

    std::size_t pruned_stabilizer_routes = 0U;
    std::size_t pruned_non_stabilizer_routes = 0U;
    for (const ExactComponentReceipt& receipt : pruned_plan.receipts()) {
        if (!receipt.prepared) {
            continue;
        }
        if (receipt.route == ExactExecutionRoute::Stabilizer) {
            ++pruned_stabilizer_routes;
        } else {
            ++pruned_non_stabilizer_routes;
        }
    }

    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "fabric_qubits=" << qubits << '\n';
    std::cout << "fabric_global_dense_log2_amplitudes=" << qubits << '\n';
    std::cout << "fabric_operations=" << operations.size() << '\n';
    std::cout << "fabric_components=" << full_plan.stats().components << '\n';
    std::cout << "fabric_largest_component_qubits=" << full_plan.stats().largest_component_qubits << '\n';
    std::cout << "fabric_full_active_components=" << full_plan.stats().active_components << '\n';
    std::cout << "fabric_full_prepared_components=" << full_plan.stats().prepared_components << '\n';
    std::cout << "fabric_full_stabilizer_routes=" << full_stabilizer_routes << '\n';
    std::cout << "fabric_full_non_stabilizer_routes=" << full_non_stabilizer_routes << '\n';
    std::cout << "fabric_pruned_active_components=" << pruned_plan.stats().active_components << '\n';
    std::cout << "fabric_pruned_prepared_components=" << pruned_plan.stats().prepared_components << '\n';
    std::cout << "fabric_pruned_stabilizer_routes=" << pruned_stabilizer_routes << '\n';
    std::cout << "fabric_pruned_non_stabilizer_routes=" << pruned_non_stabilizer_routes << '\n';
    std::cout << "fabric_full_estimated_bytes=" << full_plan.stats().estimated_bytes << '\n';
    std::cout << "fabric_pruned_estimated_bytes=" << pruned_plan.stats().estimated_bytes << '\n';
    std::cout << "fabric_selected_probability=" << probability << '\n';
    std::cout << "fabric_global_probability=" << global_probability << '\n';
    std::cout << "fabric_global_route=" << exact_execution_route_name(global.prepared_route()) << '\n';
    std::cout << "fabric_global_estimated_bytes=" << global.estimated_bytes() << '\n';
    std::cout << "fabric_full_setup_ms=" << milliseconds(full_setup_begin, full_setup_end) << '\n';
    std::cout << "fabric_pruned_setup_ms=" << milliseconds(pruned_setup_begin, pruned_setup_end) << '\n';
    std::cout << "fabric_pruned_query_ms=" << milliseconds(pruned_query_begin, pruned_query_end) << '\n';
    std::cout << "fabric_global_setup_ms=" << milliseconds(global_setup_begin, global_setup_end) << '\n';
    std::cout << "fabric_global_query_ms=" << milliseconds(global_query_begin, global_query_end) << '\n';

    return full_plan.stats().components == blocks &&
                   full_plan.stats().largest_component_qubits == block &&
                   full_plan.stats().active_components == blocks &&
                   full_plan.stats().prepared_components == blocks &&
                   full_stabilizer_routes == blocks - 1U &&
                   full_non_stabilizer_routes == 1U &&
                   pruned_plan.stats().active_components == 2U &&
                   pruned_plan.stats().prepared_components == 2U &&
                   pruned_stabilizer_routes == 2U &&
                   pruned_non_stabilizer_routes == 0U &&
                   pruned_plan.stats().estimated_bytes < full_plan.stats().estimated_bytes &&
                   probability > 0.249999999999 && probability < 0.250000000001 &&
                   std::abs(probability - global_probability) < 1e-12
               ? 0
               : 1;
}
