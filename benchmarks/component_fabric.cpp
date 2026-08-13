#include "qubit/qfabric.hpp"

#include <chrono>
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

    ExactComponentFabricConfig config;
    config.max_qubits = 8192U;
    config.max_components = 2048U;
    config.max_operations = 8192U;

    const auto setup_begin = std::chrono::steady_clock::now();
    const ExactComponentProbabilityPlan plan(qubits, operations, config);
    const auto setup_end = std::chrono::steady_clock::now();

    const auto global_setup_begin = std::chrono::steady_clock::now();
    const ExactPreparedProbabilityPlan global = ExactPreparedProbabilityPlan::for_marginals(
        qubits, operations, config.broker);
    const auto global_setup_end = std::chrono::steady_clock::now();

    const std::vector<QubitId> query{
        0U, 1U, 2U, 3U,
        2048U, 2049U, 2050U, 2051U,
    };
    const std::vector<std::uint8_t> bits(query.size(), 0U);
    const auto query_begin = std::chrono::steady_clock::now();
    const double probability = plan.marginal_probability(query, bits);
    const auto query_end = std::chrono::steady_clock::now();
    const auto global_query_begin = std::chrono::steady_clock::now();
    const double global_probability = global.marginal_probability(query, bits).value;
    const auto global_query_end = std::chrono::steady_clock::now();

    std::size_t stabilizer_routes = 0U;
    std::size_t non_stabilizer_routes = 0U;
    for (const ExactComponentReceipt& receipt : plan.receipts()) {
        if (receipt.route == ExactExecutionRoute::Stabilizer) {
            ++stabilizer_routes;
        } else {
            ++non_stabilizer_routes;
        }
    }

    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "fabric_qubits=" << qubits << '\n';
    std::cout << "fabric_global_dense_log2_amplitudes=" << qubits << '\n';
    std::cout << "fabric_operations=" << operations.size() << '\n';
    std::cout << "fabric_components=" << plan.stats().components << '\n';
    std::cout << "fabric_largest_component_qubits=" << plan.stats().largest_component_qubits << '\n';
    std::cout << "fabric_prepared_components=" << plan.stats().prepared_components << '\n';
    std::cout << "fabric_stabilizer_routes=" << stabilizer_routes << '\n';
    std::cout << "fabric_non_stabilizer_routes=" << non_stabilizer_routes << '\n';
    std::cout << "fabric_estimated_bytes=" << plan.stats().estimated_bytes << '\n';
    std::cout << "fabric_selected_probability=" << probability << '\n';
    std::cout << "fabric_global_probability=" << global_probability << '\n';
    std::cout << "fabric_global_route=" << exact_execution_route_name(global.prepared_route()) << '\n';
    std::cout << "fabric_global_estimated_bytes=" << global.estimated_bytes() << '\n';
    std::cout << "fabric_setup_ms=" << milliseconds(setup_begin, setup_end) << '\n';
    std::cout << "fabric_query_ms=" << milliseconds(query_begin, query_end) << '\n';
    std::cout << "fabric_global_setup_ms=" << milliseconds(global_setup_begin, global_setup_end) << '\n';
    std::cout << "fabric_global_query_ms=" << milliseconds(global_query_begin, global_query_end) << '\n';

    return plan.stats().components == blocks &&
                   plan.stats().largest_component_qubits == block &&
                   plan.stats().prepared_components == blocks &&
                   stabilizer_routes == blocks - 1U &&
                   non_stabilizer_routes == 1U &&
                   probability > 0.249999999999 && probability < 0.250000000001 &&
                   std::abs(probability - global_probability) < 1e-12
               ? 0
               : 1;
}
