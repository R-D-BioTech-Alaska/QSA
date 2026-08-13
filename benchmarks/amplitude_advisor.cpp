#include "qubit/qamplitude_advisor.hpp"

#include <chrono>
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

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactAmplitudeAdvisorConfig;
    using qubit::ExactAmplitudeRepresentationAdvisor;
    using qubit::ExactAmplitudeRoute;

    ExactAmplitudeAdvisorConfig config;
    config.factor.max_variables = 10000U;
    config.factor.max_factors = 50000U;
    config.factor.max_factor_entries = 4096U;
    config.factor.max_compiled_index_entries = 8U * 1024U * 1024U;
    config.factor.reuse_workspace_slots = true;
    config.max_qubits = 2048U;
    config.max_operations = 50000U;
    config.max_phase_h_defects = 10000U;
    config.max_phase_branches = 1U << 20U;
    config.max_hpath_events = 10000U;
    config.minimum_hpath_log2_margin = 8U;

    const auto large_operations = large_spacetime();
    const auto large_begin = Clock::now();
    const auto large = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        1024U, large_operations, config);
    const auto large_end = Clock::now();

    constexpr std::size_t compare_qubits = 12U;
    const auto compare_operations = comparable_chain(compare_qubits);
    const auto compare = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        compare_qubits, compare_operations, config);

    const double large_ms =
        std::chrono::duration<double, std::milli>(large_end - large_begin).count();
    std::cout << std::setprecision(17)
              << "advisor_large_qubits=1024\n"
              << "advisor_large_rounds=8\n"
              << "advisor_large_operations=" << large_operations.size() << '\n'
              << "advisor_large_phase_h_defects=" << large.phase_h_defects << '\n'
              << "advisor_large_phase_branch_fits_size_t="
              << (large.phase_branch_envelope_fits_size_t ? 1 : 0) << '\n'
              << "advisor_large_phase_eligible=" << (large.phase_graph_eligible ? 1 : 0) << '\n'
              << "advisor_large_hpath_eligible=" << (large.hpath_eligible ? 1 : 0) << '\n'
              << "advisor_large_hpath_events=" << large.hpath.h_events << '\n'
              << "advisor_large_peak_union_variables=" << large.hpath.peak_union_variables << '\n'
              << "advisor_large_peak_factor_entries=" << large.hpath.peak_factor_entries << '\n'
              << "advisor_large_workspace_slots=" << large.hpath.workspace_slots << '\n'
              << "advisor_large_graph_bytes=" << large.hpath.graph_estimated_bytes << '\n'
              << "advisor_large_plan_bytes=" << large.hpath.plan_estimated_bytes << '\n'
              << "advisor_large_route=" << static_cast<unsigned>(large.route) << '\n'
              << "advisor_large_analysis_ms=" << large_ms << '\n'
              << "advisor_compare_qubits=" << compare_qubits << '\n'
              << "advisor_compare_phase_h_defects=" << compare.phase_h_defects << '\n'
              << "advisor_compare_phase_branches=" << compare.phase_branch_envelope << '\n'
              << "advisor_compare_phase_eligible=" << (compare.phase_graph_eligible ? 1 : 0) << '\n'
              << "advisor_compare_hpath_eligible=" << (compare.hpath_eligible ? 1 : 0) << '\n'
              << "advisor_compare_peak_factor_entries=" << compare.hpath.peak_factor_entries << '\n'
              << "advisor_compare_structural_log2_margin=" << compare.structural_log2_margin << '\n'
              << "advisor_compare_route=" << static_cast<unsigned>(compare.route) << '\n'
              << "state_expansion_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return large.route == ExactAmplitudeRoute::HadamardPathFactor &&
            compare.route == ExactAmplitudeRoute::HadamardPathFactor
        ? 0
        : 1;
}
