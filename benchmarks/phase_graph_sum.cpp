#include "qubit/qphase_sum.hpp"

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
    constexpr std::size_t protected_prefix = 8U;
    constexpr std::size_t phase_modes = qubits - protected_prefix;

    PhaseGraphConfig graph_config;
    graph_config.max_edges = 20000U;
    const auto graph_start = std::chrono::steady_clock::now();
    PhaseGraphState base(qubits, graph_config);
    for (const std::size_t offset : std::vector<std::size_t>{1U, 3U, 5U, 7U}) {
        for (std::size_t local = 0U; local < phase_modes; ++local) {
            const QubitId first = static_cast<QubitId>(protected_prefix + local);
            const QubitId second = static_cast<QubitId>(
                protected_prefix + (local + offset) % phase_modes);
            base.apply_controlled_phase(first, second, 0.003 * static_cast<double>(offset));
        }
    }
    for (std::size_t local = 0U; local < 32U; ++local) {
        const QubitId first = static_cast<QubitId>(protected_prefix + local);
        const QubitId second = static_cast<QubitId>(
            protected_prefix + (local + 11U) % phase_modes);
        base.apply_controlled_phase(first, second, 0.031);
    }
    const auto graph_end = std::chrono::steady_clock::now();

    PhaseGraphBranchSumConfig sum_config;
    sum_config.max_branches = 256U;
    sum_config.max_retained_estimated_bytes = 512U * 1024U * 1024U;
    sum_config.phase_graph = graph_config;

    const auto sum_start = std::chrono::steady_clock::now();
    ExactPhaseGraphBranchSum sum(std::move(base), sum_config);
    for (QubitId qubit = 0U; qubit < 8U; ++qubit) {
        sum.apply_h(qubit);
    }
    const auto sum_end = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> zeros(qubits, 0U);
    const auto query_start = std::chrono::steady_clock::now();
    const ScaledPhaseGraphAmplitude scaled = sum.scaled_amplitude_bits(zeros);
    const double log2_probability = sum.log2_probability_bits(zeros);
    const auto query_end = std::chrono::steady_clock::now();

    bool direct_underflow_rejected = false;
    try {
        (void)sum.amplitude_bits(zeros);
    } catch (const QStateError&) {
        direct_underflow_rejected = true;
    }

    bool ninth_h_rejected = false;
    try {
        sum.apply_h(8U);
    } catch (const QStateError&) {
        ninth_h_rejected = true;
    }

    const double dense_bytes_log2 = static_cast<double>(qubits) + 4.0;
    const double retained_bytes_log2 =
        std::log2(static_cast<double>(sum.stats().retained_estimated_bytes));
    const double dense_to_retained_ratio_log2 = dense_bytes_log2 - retained_bytes_log2;

    std::cout << std::setprecision(17);
    std::cout << "carrier_qubits=" << qubits << '\n';
    std::cout << "carrier_phase_edges_per_branch="
              << sum.branches().front().state.edge_count() << '\n';
    std::cout << "carrier_hadamard_defects=" << sum.stats().hadamard_defects << '\n';
    std::cout << "carrier_branches=" << sum.branch_count() << '\n';
    std::cout << "carrier_total_phase_edges=" << sum.stats().total_phase_edges << '\n';
    std::cout << "carrier_retained_estimated_bytes="
              << sum.stats().retained_estimated_bytes << '\n';
    std::cout << "carrier_dense_to_retained_ratio_log2="
              << dense_to_retained_ratio_log2 << '\n';
    std::cout << "carrier_scaled_mantissa_norm2=" << scaled.mantissa.norm2() << '\n';
    std::cout << "carrier_log2_amplitude_scale=" << scaled.log2_scale << '\n';
    std::cout << "carrier_log2_probability_zero=" << log2_probability << '\n';
    std::cout << "carrier_graph_build_ms="
              << std::chrono::duration<double, std::milli>(graph_end - graph_start).count() << '\n';
    std::cout << "carrier_branch_build_ms="
              << std::chrono::duration<double, std::milli>(sum_end - sum_start).count() << '\n';
    std::cout << "carrier_scaled_query_ms="
              << std::chrono::duration<double, std::milli>(query_end - query_start).count() << '\n';
    std::cout << "carrier_direct_amplitude_underflow_rejected="
              << (direct_underflow_rejected ? 1 : 0) << '\n';
    std::cout << "carrier_ninth_h_rejected=" << (ninth_h_rejected ? 1 : 0) << '\n';
    std::cout << "carrier_branch_count_after_rejection=" << sum.branch_count() << '\n';
    std::cout << "carrier_dense_state_materialized=0\n";
    return 0;
}
