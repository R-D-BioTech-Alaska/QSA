#include "qubit/qcompiler.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t qubits = 10000U;
    constexpr std::size_t layers = 10U;

    std::vector<Operation> operations;
    operations.reserve(99998U);
    for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
        operations.push_back({OperationCode::H, 0U});
    }
    for (std::size_t layer = 0U; layer < layers; ++layer) {
        for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<QubitId>(qubit),
                static_cast<QubitId>(qubit + 1U),
            });
        }
    }

    const std::vector<QubitId> query{0U};
    const std::vector<std::uint8_t> bits{0U};

    const auto setup_start = std::chrono::steady_clock::now();
    ExactMarginalCompilerPlan compiler(qubits, operations, query);
    const auto setup_end = std::chrono::steady_clock::now();

    const auto index_start = std::chrono::steady_clock::now();
    ExactCausalOperationIndex index(qubits, operations);
    const auto index_end = std::chrono::steady_clock::now();
    const auto indexed_setup_start = std::chrono::steady_clock::now();
    ExactIndexedMarginalCompilerPlan indexed(index, query);
    const auto indexed_setup_end = std::chrono::steady_clock::now();

    constexpr std::size_t repeats = 1000U;
    double probability = 0.0;
    const auto query_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        probability = compiler.probability(bits);
    }
    const auto query_end = std::chrono::steady_clock::now();

    double indexed_probability = 0.0;
    const auto indexed_query_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        indexed_probability = indexed.probability(bits);
    }
    const auto indexed_query_end = std::chrono::steady_clock::now();

    const double setup_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
    const double index_build_ms =
        std::chrono::duration<double, std::milli>(index_end - index_start).count();
    const double indexed_setup_ms = std::chrono::duration<double, std::milli>(
        indexed_setup_end - indexed_setup_start).count();
    const double query_us =
        std::chrono::duration<double, std::micro>(query_end - query_start).count() /
        static_cast<double>(repeats);
    const double indexed_query_us =
        std::chrono::duration<double, std::micro>(indexed_query_end - indexed_query_start).count() /
        static_cast<double>(repeats);

    const ExactMarginalCompilerStats& stats = compiler.stats();
    const ExactMarginalCompilerStats& indexed_stats = indexed.stats();
    std::size_t stabilizer_routes = 0U;
    std::size_t register_routes = 0U;
    for (const ExactComponentReceipt& receipt : compiler.component_receipts()) {
        if (!receipt.prepared) {
            continue;
        }
        if (receipt.route == ExactExecutionRoute::Stabilizer) {
            ++stabilizer_routes;
        }
        if (receipt.route == ExactExecutionRoute::Register) {
            ++register_routes;
        }
    }

    std::cout << std::setprecision(17);
    std::cout << "compiler_original_qubits=" << stats.original_qubits << '\n';
    std::cout << "compiler_original_operations=" << stats.original_operations << '\n';
    std::cout << "compiler_causal_qubits=" << stats.causal_qubits << '\n';
    std::cout << "compiler_causal_operations=" << stats.causal_operations << '\n';
    std::cout << "compiler_removed_qubits=" << stats.removed_qubits << '\n';
    std::cout << "compiler_removed_operations=" << stats.removed_operations << '\n';
    std::cout << "compiler_components=" << stats.components << '\n';
    std::cout << "compiler_prepared_components=" << stats.prepared_components << '\n';
    std::cout << "compiler_stabilizer_routes=" << stabilizer_routes << '\n';
    std::cout << "compiler_register_routes=" << register_routes << '\n';
    std::cout << "compiler_estimated_bytes=" << stats.estimated_bytes << '\n';
    std::cout << "compiler_probability=" << probability << '\n';
    std::cout << "compiler_setup_ms=" << setup_ms << '\n';
    std::cout << "compiler_query_us=" << query_us << '\n';
    std::cout << "index_estimated_bytes=" << index.stats().estimated_bytes << '\n';
    std::cout << "index_build_ms=" << index_build_ms << '\n';
    std::cout << "indexed_causal_qubits=" << indexed_stats.causal_qubits << '\n';
    std::cout << "indexed_causal_operations=" << indexed_stats.causal_operations << '\n';
    std::cout << "indexed_estimated_bytes=" << indexed_stats.estimated_bytes << '\n';
    std::cout << "indexed_probability=" << indexed_probability << '\n';
    std::cout << "indexed_setup_ms=" << indexed_setup_ms << '\n';
    std::cout << "indexed_query_us=" << indexed_query_us << '\n';
    std::cout << "indexed_setup_speedup_over_one_shot=" << setup_ms / indexed_setup_ms << '\n';
    return 0;
}
