#include "qubit/qmagic.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactLowMagicConfig;
    using qubit::ExactLowMagicStabilizerSum;
    using qubit::OperationCode;
    using qubit::QStateError;

    constexpr std::size_t qubits = 4096U;
    ExactLowMagicConfig config;
    config.max_live_branches = 256U;
    config.max_intermediate_branches = 512U;
    config.max_retained_estimated_bytes = 128U * 1024U * 1024U;
    config.max_expectation_word_ops = 100'000'000U;

    ExactLowMagicStabilizerSum state(qubits, config);
    const auto evolution_begin = Clock::now();
    for (std::size_t qubit = 0U; qubit < 256U; ++qubit) {
        state.apply({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    for (std::size_t qubit = 0U; qubit < 256U; ++qubit) {
        state.apply({
            OperationCode::Cnot,
            static_cast<qubit::QubitId>(qubit),
            static_cast<qubit::QubitId>(qubit + 256U),
        });
    }
    for (std::size_t defect = 0U; defect < 4U; ++defect) {
        state.apply({OperationCode::T, static_cast<qubit::QubitId>(defect)});
        state.apply({OperationCode::H, static_cast<qubit::QubitId>(defect)});
        state.apply({
            OperationCode::Cnot,
            static_cast<qubit::QubitId>(defect),
            static_cast<qubit::QubitId>(1024U + defect),
        });
    }
    const auto t4_ready = Clock::now();
    const auto t4_stats = state.stats();

    const auto query_begin = Clock::now();
    const double probability = state.probability_one(0U);
    const auto query_end = Clock::now();

    for (std::size_t defect = 4U; defect < 8U; ++defect) {
        state.apply({OperationCode::T, static_cast<qubit::QubitId>(defect)});
        state.apply({OperationCode::H, static_cast<qubit::QubitId>(defect)});
        state.apply({
            OperationCode::Cnot,
            static_cast<qubit::QubitId>(defect),
            static_cast<qubit::QubitId>(1024U + defect),
        });
    }
    const auto t8_ready = Clock::now();
    const auto t8_stats = state.stats();

    bool ninth_rejected = false;
    try {
        state.apply({OperationCode::T, 8U});
    } catch (const QStateError&) {
        ninth_rejected = true;
    }
    const auto after_rejection = state.stats();

    ExactLowMagicStabilizerSum reconverged(qubits, config);
    const auto recon_begin = Clock::now();
    for (std::size_t pair = 0U; pair < 256U; ++pair) {
        const auto qubit = static_cast<qubit::QubitId>(pair & 255U);
        reconverged.apply({OperationCode::T, qubit});
        reconverged.apply({OperationCode::Tdg, qubit});
    }
    const auto recon_end = Clock::now();
    const auto recon_stats = reconverged.stats();
    const double recon_norm = reconverged.norm_squared();

    const double t4_ms = std::chrono::duration<double, std::milli>(
        t4_ready - evolution_begin).count();
    const double query_ms = std::chrono::duration<double, std::milli>(
        query_end - query_begin).count();
    const double t8_increment_ms = std::chrono::duration<double, std::milli>(
        t8_ready - query_end).count();
    const double recon_ms = std::chrono::duration<double, std::milli>(
        recon_end - recon_begin).count();

    const double t4_naive_bytes =
        static_cast<double>(t4_stats.base_tableau_bytes) *
        static_cast<double>(t4_stats.live_branches);
    const double t8_naive_bytes =
        static_cast<double>(t8_stats.base_tableau_bytes) *
        static_cast<double>(t8_stats.live_branches);
    const double t4_ratio = t4_naive_bytes /
        static_cast<double>(t4_stats.retained_estimated_bytes);
    const double t8_ratio = t8_naive_bytes /
        static_cast<double>(t8_stats.retained_estimated_bytes);
    const double dense_log2_bytes = static_cast<double>(qubits) + 4.0;
    const double dense_to_shared_log2 = dense_log2_bytes -
        std::log2(static_cast<double>(t8_stats.retained_estimated_bytes));

    std::cout << std::setprecision(17)
              << "carrier_qubits=" << qubits << '\n'
              << "t4_live_branches=" << t4_stats.live_branches << '\n'
              << "t4_base_tableau_bytes=" << t4_stats.base_tableau_bytes << '\n'
              << "t4_retained_estimated_bytes=" << t4_stats.retained_estimated_bytes << '\n'
              << "t4_naive_full_tableau_branch_bytes=" << t4_naive_bytes << '\n'
              << "t4_naive_to_shared_ratio=" << t4_ratio << '\n'
              << "t4_probability_one_q0=" << probability << '\n'
              << "t4_evolution_ms=" << t4_ms << '\n'
              << "t4_query_ms=" << query_ms << '\n'
              << "t8_live_branches=" << t8_stats.live_branches << '\n'
              << "t8_base_tableau_bytes=" << t8_stats.base_tableau_bytes << '\n'
              << "t8_retained_estimated_bytes=" << t8_stats.retained_estimated_bytes << '\n'
              << "t8_naive_full_tableau_branch_bytes=" << t8_naive_bytes << '\n'
              << "t8_naive_to_shared_ratio=" << t8_ratio << '\n'
              << "t8_dense_to_shared_ratio_log2=" << dense_to_shared_log2 << '\n'
              << "t8_increment_ms=" << t8_increment_ms << '\n'
              << "ninth_t_rejected=" << (ninth_rejected ? 1 : 0) << '\n'
              << "post_rejection_live_branches=" << after_rejection.live_branches << '\n'
              << "post_rejection_t_defects=" << after_rejection.t_defects << '\n'
              << "recon_t_defects=" << recon_stats.t_defects << '\n'
              << "recon_raw_branch_count_log2=512\n"
              << "recon_live_branches=" << recon_stats.live_branches << '\n'
              << "recon_max_live_branches=" << recon_stats.max_live_branches << '\n'
              << "recon_merged_branches=" << recon_stats.merged_branches << '\n'
              << "recon_exact_cancellations=" << recon_stats.exact_cancellations << '\n'
              << "recon_extracted_power_of_two_bits="
              << recon_stats.extracted_power_of_two_bits << '\n'
              << "recon_scale_power_two=" << recon_stats.scale_power_two << '\n'
              << "recon_norm_squared=" << recon_norm << '\n'
              << "recon_evolution_ms=" << recon_ms << '\n'
              << "dense_state_materialized=0\n";
    return 0;
}
