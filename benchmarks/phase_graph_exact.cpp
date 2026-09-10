#include "qubit/qphase_exact.hpp"

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
    constexpr std::size_t repeated_h = 512U;

    ExactCompressedPhaseConfig config;
    config.max_live_branches = 1024U;
    config.max_intermediate_branches = 2048U;
    config.max_retained_estimated_bytes = 64U * 1024U * 1024U;

    const auto started = std::chrono::steady_clock::now();
    ExactCompressedPhaseGraphSum state(qubits, config);
    for (QubitId qubit = 0U; qubit < 64U; ++qubit) {
        state.apply_t(qubit);
    }
    for (std::size_t repeat = 0U; repeat < repeated_h; ++repeat) {
        state.apply_h(0U);
    }
    const auto evolved = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> zeros(qubits, 0U);
    const auto query_started = std::chrono::steady_clock::now();
    const ExactCyclotomicScaledAmplitude exact = state.exact_amplitude_bits(zeros);
    const double log2_probability = state.log2_probability_bits(zeros);
    const auto query_finished = std::chrono::steady_clock::now();

    bool direct_underflow_rejected = false;
    try {
        (void)state.amplitude_bits(zeros);
    } catch (const QStateError&) {
        direct_underflow_rejected = true;
    }

    ExactCompressedPhaseConfig growth_config;
    growth_config.max_live_branches = 1024U;
    growth_config.max_intermediate_branches = 2048U;
    ExactCompressedPhaseGraphSum growth(32U, growth_config);
    for (QubitId qubit = 0U; qubit < 10U; ++qubit) {
        growth.apply_h(qubit);
    }
    bool irreducible_rejected = false;
    try {
        growth.apply_h(10U);
    } catch (const QStateError&) {
        irreducible_rejected = true;
    }

    bool floating_rz_rejected = false;
    try {
        state.apply(Operation{OperationCode::Rz, 1U, 0U, 0.37});
    } catch (const QStateError&) {
        floating_rz_rejected = true;
    }

    const double dense_bytes_log2 = static_cast<double>(qubits) + 4.0;
    const double retained_bytes_log2 =
        std::log2(static_cast<double>(state.stats().retained_estimated_bytes));

    std::cout << std::setprecision(17);
    std::cout << "carrier_qubits=" << qubits << '\n';
    std::cout << "carrier_discrete_t_gates=64\n";
    std::cout << "carrier_hadamard_defects=" << state.stats().hadamard_defects << '\n';
    std::cout << "carrier_raw_branch_count_log2=" << repeated_h << '\n';
    std::cout << "carrier_live_branches=" << state.branch_count() << '\n';
    std::cout << "carrier_max_live_branches=" << state.stats().max_live_branches << '\n';
    std::cout << "carrier_merged_branches=" << state.stats().merged_branches << '\n';
    std::cout << "carrier_exact_cancellations=" << state.stats().exact_cancellations << '\n';
    std::cout << "carrier_extracted_power_of_two_bits="
              << state.stats().extracted_power_of_two_bits << '\n';
    std::cout << "carrier_scale_half_power=" << state.stats().scale_half_power << '\n';
    std::cout << "carrier_retained_estimated_bytes="
              << state.stats().retained_estimated_bytes << '\n';
    std::cout << "carrier_dense_to_retained_ratio_log2="
              << dense_bytes_log2 - retained_bytes_log2 << '\n';
    std::cout << "carrier_exact_coefficient_0=" << exact.coefficient.values[0] << '\n';
    std::cout << "carrier_exact_coefficient_1=" << exact.coefficient.values[1] << '\n';
    std::cout << "carrier_exact_coefficient_2=" << exact.coefficient.values[2] << '\n';
    std::cout << "carrier_exact_coefficient_3=" << exact.coefficient.values[3] << '\n';
    std::cout << "carrier_log2_probability_zero=" << log2_probability << '\n';
    std::cout << "carrier_evolution_ms="
              << std::chrono::duration<double, std::milli>(evolved - started).count() << '\n';
    std::cout << "carrier_query_us="
              << std::chrono::duration<double, std::micro>(query_finished - query_started).count()
              << '\n';
    std::cout << "carrier_direct_amplitude_underflow_rejected="
              << (direct_underflow_rejected ? 1 : 0) << '\n';
    std::cout << "irreducible_live_before_rejection=" << growth.branch_count() << '\n';
    std::cout << "irreducible_hadamards_before_rejection="
              << growth.stats().hadamard_defects << '\n';
    std::cout << "irreducible_next_h_rejected=" << (irreducible_rejected ? 1 : 0) << '\n';
    std::cout << "floating_rz_rejected=" << (floating_rz_rejected ? 1 : 0) << '\n';
    std::cout << "dense_state_materialized=0\n";
    return 0;
}
