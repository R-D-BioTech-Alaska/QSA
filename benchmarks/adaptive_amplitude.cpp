#include "qubit/qadaptive_amplitude.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::vector<qubit::Operation> large_spacetime() {
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
    return operations;
}

std::vector<qubit::Operation> comparable_chain(std::size_t qubits) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    for (std::size_t target = 0U; target < qubits; ++target) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(target),
            0U,
            0.031 * static_cast<double>(target + 1U),
        });
    }
    for (std::size_t target = 0U; target + 1U < qubits; ++target) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(target),
            static_cast<qubit::QubitId>(target + 1U),
        });
    }
    for (std::size_t target = 0U; target < qubits; ++target) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(target)});
    }
    return operations;
}

qubit::ExactAdaptiveAmplitudeConfig make_config() {
    qubit::ExactAdaptiveAmplitudeConfig config;
    config.advisor.max_qubits = 2048U;
    config.advisor.max_operations = 50000U;
    config.advisor.max_phase_h_defects = 10000U;
    config.advisor.max_phase_branches = 1U << 20U;
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
    return config;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::ExactHadamardPathAmplitudePlan;
    using qubit::ExactPhaseGraphBranchSum;

    auto config = make_config();
    const auto large_operations = large_spacetime();
    const auto prepare_begin = Clock::now();
    ExactAdaptiveAmplitudePlan large_plan(1024U, large_operations, config);
    const auto prepare_end = Clock::now();
    std::array<std::uint8_t, 1024U> large_bits{};
    for (std::size_t index = 0U; index < large_bits.size(); ++index) {
        large_bits[index] = static_cast<std::uint8_t>((index * 13U + 5U) & 1U);
    }
    const auto query_begin = Clock::now();
    const double large_log_probability = large_plan.log2_probability_bits(large_bits);
    const auto query_end = Clock::now();

    constexpr std::size_t compare_qubits = 12U;
    const auto compare_operations = comparable_chain(compare_qubits);
    ExactAdaptiveAmplitudePlan compare_plan(compare_qubits, compare_operations, config);
    ExactHadamardPathAmplitudePlan direct_hpath(
        compare_qubits, compare_operations, config.hpath);
    ExactPhaseGraphBranchSum direct_phase(compare_qubits, config.phase_graph);
    for (const auto& operation : compare_operations) {
        direct_phase.apply(operation);
    }
    std::array<std::uint8_t, compare_qubits> compare_bits{};
    for (std::size_t index = 0U; index < compare_bits.size(); ++index) {
        compare_bits[index] = static_cast<std::uint8_t>((index * 5U + 1U) & 1U);
    }
    const qubit::QComplex adaptive_value = compare_plan.amplitude_bits(compare_bits);
    const qubit::QComplex hpath_value = direct_hpath.amplitude_bits(compare_bits);
    const qubit::QComplex phase_value = direct_phase.amplitude_bits(compare_bits);
    const double hpath_error = std::hypot(
        adaptive_value.re - hpath_value.re,
        adaptive_value.im - hpath_value.im);
    const double phase_error = std::hypot(
        adaptive_value.re - phase_value.re,
        adaptive_value.im - phase_value.im);

    const double prepare_ms =
        std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
    const double query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_begin).count();
    const auto& large = large_plan.decision();
    const auto& compare = compare_plan.decision();

    std::cout << std::setprecision(17)
              << "adaptive_large_qubits=1024\n"
              << "adaptive_large_rounds=8\n"
              << "adaptive_large_h_events=" << large.hpath.h_events << '\n'
              << "adaptive_large_phase_eligible=" << (large.phase_graph_eligible ? 1 : 0) << '\n'
              << "adaptive_large_hpath_eligible=" << (large.hpath_eligible ? 1 : 0) << '\n'
              << "adaptive_large_peak_union_variables=" << large.hpath.peak_union_variables << '\n'
              << "adaptive_large_peak_factor_entries=" << large.hpath.peak_factor_entries << '\n'
              << "adaptive_large_route=" << static_cast<unsigned>(large_plan.route()) << '\n'
              << "adaptive_large_log2_probability_probe=" << large_log_probability << '\n'
              << "adaptive_large_prepare_ms=" << prepare_ms << '\n'
              << "adaptive_large_query_ms=" << query_ms << '\n'
              << "adaptive_compare_qubits=" << compare_qubits << '\n'
              << "adaptive_compare_phase_eligible=" << (compare.phase_graph_eligible ? 1 : 0) << '\n'
              << "adaptive_compare_hpath_eligible=" << (compare.hpath_eligible ? 1 : 0) << '\n'
              << "adaptive_compare_phase_branches=" << compare.phase_branch_envelope << '\n'
              << "adaptive_compare_peak_factor_entries=" << compare.hpath.peak_factor_entries << '\n'
              << "adaptive_compare_structural_log2_margin=" << compare.structural_log2_margin << '\n'
              << "adaptive_compare_route=" << static_cast<unsigned>(compare_plan.route()) << '\n'
              << "adaptive_compare_hpath_error=" << hpath_error << '\n'
              << "adaptive_compare_phase_error=" << phase_error << '\n'
              << "state_expansion_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return large_plan.route() == ExactAmplitudeRoute::HadamardPathFactor &&
            compare_plan.route() == ExactAmplitudeRoute::HadamardPathFactor &&
            hpath_error <= 1e-11 && phase_error <= 1e-11
        ? 0
        : 1;
}
