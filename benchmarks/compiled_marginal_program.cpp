#include "qubit/qprogram.hpp"

#include <chrono>
#include <cmath>
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

    ExactCompiledMarginalProgramConfig config;
    config.max_cached_plans = 128U;
    const auto program_start = std::chrono::steady_clock::now();
    ExactCompiledMarginalProgram program(qubits, operations, config);
    const auto program_end = std::chrono::steady_clock::now();

    const std::vector<QubitId> query{0U};
    const std::vector<std::uint8_t> zero{0U};
    const double warm_probability = program.probability(query, zero);
    const ExactIndexedMarginalCompilerPlan& warm_plan = program.prepare(query);

    constexpr std::size_t cached_repeats = 5000U;
    double cached_probability = 0.0;
    const auto cached_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < cached_repeats; ++repeat) {
        cached_probability = program.probability(query, zero);
    }
    const auto cached_end = std::chrono::steady_clock::now();
    const double cached_us =
        std::chrono::duration<double, std::micro>(cached_end - cached_start).count() /
        static_cast<double>(cached_repeats);

    constexpr std::size_t one_shot_repeats = 32U;
    double one_shot_probability = 0.0;
    const auto one_shot_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < one_shot_repeats; ++repeat) {
        ExactMarginalCompilerPlan plan(qubits, operations, query);
        one_shot_probability = plan.probability(zero);
    }
    const auto one_shot_end = std::chrono::steady_clock::now();
    const double one_shot_us =
        std::chrono::duration<double, std::micro>(one_shot_end - one_shot_start).count() /
        static_cast<double>(one_shot_repeats);

    constexpr std::size_t unique_queries = 32U;
    double indexed_unique_sum = 0.0;
    const auto indexed_unique_start = std::chrono::steady_clock::now();
    for (std::size_t q = 0U; q < unique_queries; ++q) {
        const std::vector<QubitId> support{static_cast<QubitId>(q)};
        indexed_unique_sum += program.probability(support, zero);
    }
    const auto indexed_unique_end = std::chrono::steady_clock::now();

    double one_shot_unique_sum = 0.0;
    const auto one_shot_unique_start = std::chrono::steady_clock::now();
    for (std::size_t q = 0U; q < unique_queries; ++q) {
        const std::vector<QubitId> support{static_cast<QubitId>(q)};
        ExactMarginalCompilerPlan plan(qubits, operations, support);
        one_shot_unique_sum += plan.probability(zero);
    }
    const auto one_shot_unique_end = std::chrono::steady_clock::now();

    const double indexed_unique_ms = std::chrono::duration<double, std::milli>(
        indexed_unique_end - indexed_unique_start).count();
    const double one_shot_unique_ms = std::chrono::duration<double, std::milli>(
        one_shot_unique_end - one_shot_unique_start).count();

    std::size_t stabilizer_routes = 0U;
    for (const ExactComponentReceipt& receipt : warm_plan.component_receipts()) {
        if (receipt.prepared && receipt.route == ExactExecutionRoute::Stabilizer) {
            ++stabilizer_routes;
        }
    }

    std::cout << std::setprecision(17);
    std::cout << "program_qubits=" << program.stats().qubits << '\n';
    std::cout << "program_operations=" << program.stats().operations << '\n';
    std::cout << "program_index_estimated_bytes=" << program.stats().index_estimated_bytes << '\n';
    std::cout << "program_build_ms="
              << std::chrono::duration<double, std::milli>(program_end - program_start).count()
              << '\n';
    std::cout << "warm_causal_qubits=" << warm_plan.stats().causal_qubits << '\n';
    std::cout << "warm_causal_operations=" << warm_plan.stats().causal_operations << '\n';
    std::cout << "warm_stabilizer_routes=" << stabilizer_routes << '\n';
    std::cout << "warm_probability=" << warm_probability << '\n';
    std::cout << "cached_probability=" << cached_probability << '\n';
    std::cout << "one_shot_probability=" << one_shot_probability << '\n';
    std::cout << "cached_query_us=" << cached_us << '\n';
    std::cout << "one_shot_compile_query_us=" << one_shot_us << '\n';
    std::cout << "cached_over_one_shot_speedup=" << one_shot_us / cached_us << '\n';
    std::cout << "unique_indexed_ms=" << indexed_unique_ms << '\n';
    std::cout << "unique_one_shot_ms=" << one_shot_unique_ms << '\n';
    std::cout << "unique_indexed_over_one_shot_speedup="
              << one_shot_unique_ms / indexed_unique_ms << '\n';
    std::cout << "unique_probability_sum_error="
              << std::abs(indexed_unique_sum - one_shot_unique_sum) << '\n';
    std::cout << "cached_plans=" << program.stats().cached_plans << '\n';
    std::cout << "cached_bytes=" << program.stats().cached_bytes << '\n';
    std::cout << "compile_count=" << program.stats().compile_count << '\n';
    std::cout << "cache_hits=" << program.stats().cache_hits << '\n';
    std::cout << "cache_misses=" << program.stats().cache_misses << '\n';
    return 0;
}
