#include "qubit/qadaptive_amplitude.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactAdaptiveAmplitudeConfig;
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    constexpr std::size_t qubits = 1024U;
    constexpr std::size_t rounds = 8U;
    std::vector<Operation> operations;
    operations.reserve(4U * qubits * rounds);
    for (std::size_t round = 0U; round < rounds; ++round) {
        for (std::size_t target = 0U; target < qubits; ++target) {
            operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(target)});
        }
        for (std::size_t target = 0U; target < qubits; ++target) {
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(target),
                0U,
                0.000173 * static_cast<double>(1U + round * qubits + target),
            });
            operations.push_back({
                ((round + target) & 1U) == 0U ? OperationCode::T : OperationCode::Sdg,
                static_cast<qubit::QubitId>(target),
            });
        }
        for (std::size_t target = 0U; target + 1U < qubits; ++target) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(target),
                static_cast<qubit::QubitId>(target + 1U),
            });
        }
    }

    ExactAdaptiveAmplitudeConfig config;
    config.advisor.max_qubits = 2048U;
    config.advisor.max_operations = 50000U;
    config.advisor.max_phase_h_defects = 10000U;
    config.advisor.max_hpath_events = 10000U;
    config.advisor.minimum_hpath_log2_margin = 8U;
    config.phase_graph.max_branches = 1U << 20U;
    config.phase_graph.max_retained_estimated_bytes = 512U * 1024U * 1024U;
    config.hpath.factor.max_variables = 10000U;
    config.hpath.factor.max_factors = 50000U;
    config.hpath.factor.max_factor_entries = 4096U;
    config.hpath.factor.max_compiled_index_entries = 8U * 1024U * 1024U;
    config.hpath.factor.reuse_workspace_slots = true;
    config.hpath.max_qubits = 2048U;
    config.hpath.max_operations = 50000U;
    config.hpath.max_h_events = 10000U;
    config.hpath.max_metadata_bytes = 64U * 1024U * 1024U;

    const auto prepare_begin = Clock::now();
    ExactAdaptiveAmplitudePlan plan(qubits, operations, config);
    const auto prepare_end = Clock::now();
    std::array<std::uint8_t, qubits> bits{};
    for (std::size_t index = 0U; index < bits.size(); ++index) {
        bits[index] = static_cast<std::uint8_t>((index * 13U + 5U) & 1U);
    }
    const auto query_begin = Clock::now();
    const double log_probability = plan.log2_probability_bits(bits);
    const auto query_end = Clock::now();

    const double prepare_ms =
        std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
    const double query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_begin).count();
    const auto& decision = plan.decision();

    std::cout << std::setprecision(17)
              << "adaptive_qubits=" << qubits << '\n'
              << "adaptive_rounds=" << rounds << '\n'
              << "adaptive_h_events=" << decision.hpath.h_events << '\n'
              << "adaptive_peak_union_variables=" << decision.hpath.peak_union_variables << '\n'
              << "adaptive_peak_factor_entries=" << decision.hpath.peak_factor_entries << '\n'
              << "adaptive_structural_log2_margin=" << decision.structural_log2_margin << '\n'
              << "adaptive_route=" << static_cast<unsigned>(plan.route()) << '\n'
              << "adaptive_log2_probability_probe=" << log_probability << '\n'
              << "adaptive_prepare_ms=" << prepare_ms << '\n'
              << "adaptive_query_ms=" << query_ms << '\n'
              << "state_expansion_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";
    return plan.route() == ExactAmplitudeRoute::HadamardPathFactor ? 0 : 1;
}
