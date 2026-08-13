#include "qubit/qsymbolic_phase.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactSymbolicPhaseConfig;
    using qubit::ExactSymbolicPhaseGraphSum;
    using qubit::QStateError;

    constexpr std::size_t qubits = 4096U;
    ExactSymbolicPhaseConfig config;
    config.max_live_branches = 1024U;
    config.max_intermediate_branches = 2048U;
    config.max_coefficient_terms = 4096U;
    config.max_symbol_terms = 1U << 20U;
    config.max_integer_bits = 32U;
    config.max_retained_estimated_bytes = 128U * 1024U * 1024U;

    ExactSymbolicPhaseGraphSum state(qubits, config);
    for (std::uint32_t symbol = 0U; symbol < 256U; ++symbol) {
        const double angle = -2.4 + 4.8 * static_cast<double>(symbol + 1U) / 257.0;
        state.bind_symbol(symbol, angle);
    }

    const auto begin = Clock::now();
    for (std::size_t pair = 0U; pair < 256U; ++pair) {
        const auto target = static_cast<qubit::QubitId>(pair);
        const auto symbol = static_cast<qubit::SymbolicPhaseId>(pair);
        state.apply_h(target);
        state.apply_rz_symbol(target, symbol);
        state.apply_rz_symbol(target, symbol, -1);
        if ((pair & 1U) != 0U) {
            const auto other = static_cast<qubit::QubitId>(512U + pair);
            state.apply_controlled_phase_symbol(target, other, symbol);
            state.apply_controlled_phase_symbol(target, other, symbol, -1);
        }
        state.apply_h(target);
    }
    const auto end = Clock::now();

    const auto stats = state.stats();
    std::array<std::uint8_t, qubits> zero_bits{};
    const auto query_begin = Clock::now();
    const auto exact_zero = state.exact_amplitude_bits(zero_bits);
    const double log2_probability = state.log2_probability_bits(zero_bits);
    const auto query_end = Clock::now();

    ExactSymbolicPhaseConfig hard_config = config;
    hard_config.max_live_branches = 1024U;
    hard_config.max_intermediate_branches = 2048U;
    hard_config.max_coefficient_terms = 4096U;
    ExactSymbolicPhaseGraphSum hard(64U, hard_config);
    for (std::uint32_t symbol = 0U; symbol < 6U; ++symbol) {
        hard.bind_symbol(symbol, 0.113 + 0.271 * static_cast<double>(symbol + 1U));
    }
    for (std::uint32_t defect = 0U; defect < 5U; ++defect) {
        const auto target = static_cast<qubit::QubitId>(defect);
        hard.apply_h(target);
        hard.apply_rz_symbol(target, defect);
        hard.apply_h(target);
    }
    const auto hard_before = hard.stats();
    bool sixth_rejected = false;
    try {
        hard.apply_h(5U);
    } catch (const QStateError&) {
        sixth_rejected = true;
    }

    bool direct_underflow_rejected = false;
    try {
        (void)state.amplitude_bits(zero_bits);
    } catch (const QStateError&) {
        direct_underflow_rejected = true;
    }

    const double evolution_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const double query_us = std::chrono::duration<double, std::micro>(query_end - query_begin).count();
    const double dense_log2_bytes = static_cast<double>(qubits) + 4.0;
    const double dense_to_retained_log2 = dense_log2_bytes -
        std::log2(static_cast<double>(stats.retained_estimated_bytes));

    std::int64_t exact_multiplicity = 0;
    std::size_t exact_symbol_terms = 0U;
    std::uint8_t exact_eighths = 0U;
    if (exact_zero.terms.size() == 1U) {
        const auto& term = *exact_zero.terms.begin();
        exact_eighths = term.first.eighths;
        exact_symbol_terms = term.first.terms.size();
        exact_multiplicity = term.second;
    }

    std::cout << std::setprecision(17)
              << "carrier_qubits=" << qubits << '\n'
              << "carrier_symbols=" << stats.symbols << '\n'
              << "carrier_hadamard_defects=" << stats.hadamard_defects << '\n'
              << "carrier_raw_branch_count_log2=512\n"
              << "carrier_live_branches=" << stats.live_branches << '\n'
              << "carrier_max_live_branches=" << stats.max_live_branches << '\n'
              << "carrier_coefficient_terms=" << stats.coefficient_terms << '\n'
              << "carrier_max_coefficient_terms=" << stats.max_coefficient_terms << '\n'
              << "carrier_max_integer_bits=" << stats.max_integer_bits << '\n'
              << "carrier_graph_merges=" << stats.graph_merges << '\n'
              << "carrier_exact_cancellations=" << stats.exact_cancellations << '\n'
              << "carrier_extracted_power_of_two_bits=" << stats.extracted_power_of_two_bits << '\n'
              << "carrier_sqrt2_denominator_power=" << stats.sqrt2_denominator_power << '\n'
              << "carrier_retained_estimated_bytes=" << stats.retained_estimated_bytes << '\n'
              << "carrier_dense_to_retained_ratio_log2=" << dense_to_retained_log2 << '\n'
              << "carrier_exact_zero_terms=" << exact_zero.terms.size() << '\n'
              << "carrier_exact_zero_eighths=" << static_cast<unsigned>(exact_eighths) << '\n'
              << "carrier_exact_zero_symbol_terms=" << exact_symbol_terms << '\n'
              << "carrier_exact_zero_multiplicity=" << exact_multiplicity << '\n'
              << "carrier_exact_zero_denominator=" << exact_zero.sqrt2_denominator_power << '\n'
              << "carrier_log2_probability_zero=" << log2_probability << '\n'
              << "carrier_evolution_ms=" << evolution_ms << '\n'
              << "carrier_query_us=" << query_us << '\n'
              << "carrier_direct_amplitude_underflow_rejected="
              << (direct_underflow_rejected ? 1 : 0) << '\n'
              << "irreducible_live_before_rejection=" << hard_before.live_branches << '\n'
              << "irreducible_hadamards_before_rejection=" << hard_before.hadamard_defects << '\n'
              << "irreducible_sixth_h_rejected=" << (sixth_rejected ? 1 : 0) << '\n'
              << "dense_state_materialized=0\n";
    return 0;
}
