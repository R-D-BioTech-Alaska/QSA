#include "qubit/qhpath_factor.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactSingleHPathAmplitudePlan;
    using qubit::ExactSingleHPathConfig;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    constexpr std::size_t qubits = 1536U;
    std::vector<Operation> operations;
    operations.reserve(3U * qubits);
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(qubit),
            static_cast<qubit::QubitId>(qubit + 1U),
        });
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::H,
            static_cast<qubit::QubitId>(qubit),
        });
    }

    ExactSingleHPathConfig config;
    config.factor.max_factor_entries = 64U;
    config.factor.max_factors = 10000U;
    config.factor.max_variables = 4096U;
    config.factor.max_compiled_index_entries = 1U << 20U;
    config.factor.reuse_workspace_slots = true;
    config.max_qubits = 4096U;
    config.max_operations = 10000U;
    config.max_h_defects = 4096U;

    const auto compile_begin = Clock::now();
    ExactSingleHPathAmplitudePlan plan(qubits, operations, config);
    const auto compile_end = Clock::now();
    std::array<std::uint8_t, qubits> zero_bits{};
    const auto query_begin = Clock::now();
    const auto result = plan.scaled_amplitude_bits(zero_bits);
    const auto query_end = Clock::now();

    ExactSingleHPathConfig hard_config = config;
    hard_config.factor.max_factor_entries = 1024U;
    hard_config.factor.max_variables = 64U;
    hard_config.max_qubits = 64U;
    hard_config.max_h_defects = 64U;
    std::vector<Operation> hard;
    constexpr std::size_t hard_qubits = 32U;
    hard.reserve(hard_qubits * hard_qubits / 2U + hard_qubits);
    for (std::size_t first = 0U; first < hard_qubits; ++first) {
        for (std::size_t second = first + 1U; second < hard_qubits; ++second) {
            hard.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(first),
                static_cast<qubit::QubitId>(second),
            });
        }
    }
    for (std::size_t qubit = 0U; qubit < hard_qubits; ++qubit) {
        hard.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    ExactSingleHPathAmplitudePlan hard_plan(hard_qubits, hard, hard_config);
    std::array<std::uint8_t, hard_qubits> hard_zero{};
    bool dense_width_rejected = false;
    const auto hard_begin = Clock::now();
    try {
        (void)hard_plan.scaled_amplitude_bits(hard_zero);
    } catch (const QStateError&) {
        dense_width_rejected = true;
    }
    const auto hard_end = Clock::now();

    const double compile_us =
        std::chrono::duration<double, std::micro>(compile_end - compile_begin).count();
    const double query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_begin).count();
    const double hard_ms =
        std::chrono::duration<double, std::milli>(hard_end - hard_begin).count();
    const double dense_to_peak_log2 = static_cast<double>(qubits) -
        std::log2(static_cast<double>(result.factor_stats.peak_factor_entries));

    std::cout << std::setprecision(17)
              << "carrier_qubits=" << qubits << '\n'
              << "carrier_operations=" << operations.size() << '\n'
              << "carrier_h_defects=" << result.h_defects << '\n'
              << "carrier_raw_branch_count_log2=" << result.h_defects << '\n'
              << "carrier_factor_variables=" << result.factor_variables << '\n'
              << "carrier_factor_count=" << result.factor_count << '\n'
              << "carrier_peak_union_variables="
              << result.factor_stats.peak_union_variables << '\n'
              << "carrier_peak_factor_entries="
              << result.factor_stats.peak_factor_entries << '\n'
              << "carrier_workspace_slots=" << result.factor_stats.workspace_slots << '\n'
              << "carrier_graph_estimated_bytes=" << result.graph_estimated_bytes << '\n'
              << "carrier_plan_estimated_bytes=" << result.plan_estimated_bytes << '\n'
              << "carrier_dense_to_peak_factor_ratio_log2=" << dense_to_peak_log2 << '\n'
              << "carrier_log2_probability_zero=" << result.log2_probability() << '\n'
              << "carrier_compile_us=" << compile_us << '\n'
              << "carrier_query_ms=" << query_ms << '\n'
              << "hard_qubits=" << hard_qubits << '\n'
              << "hard_cz_edges=" << (hard_qubits * (hard_qubits - 1U) / 2U) << '\n'
              << "hard_factor_entry_cap=" << hard_config.factor.max_factor_entries << '\n'
              << "hard_dense_width_rejected=" << (dense_width_rejected ? 1 : 0) << '\n'
              << "hard_rejection_ms=" << hard_ms << '\n'
              << "dense_state_materialized=0\n"
              << "explicit_branch_list_materialized=0\n";
    return 0;
}
