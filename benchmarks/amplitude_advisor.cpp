#include "qubit/qamplitude_advisor.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactAmplitudeAdvisorConfig;
    using qubit::ExactAmplitudeRepresentationAdvisor;
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

    ExactAmplitudeAdvisorConfig config;
    config.factor.max_variables = 10000U;
    config.factor.max_factors = 50000U;
    config.factor.max_factor_entries = 4096U;
    config.factor.max_compiled_index_entries = 8U * 1024U * 1024U;
    config.factor.reuse_workspace_slots = true;
    config.max_qubits = 2048U;
    config.max_operations = 50000U;
    config.max_phase_h_defects = 10000U;
    config.max_hpath_events = 10000U;
    config.minimum_hpath_log2_margin = 8U;

    const auto begin = Clock::now();
    const auto decision = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        qubits, operations, config);
    const auto end = Clock::now();

    const std::vector<Operation> phase_only{
        {OperationCode::H, 0U},
        {OperationCode::X, 0U},
        {OperationCode::T, 1U},
    };
    const auto phase = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        4U, phase_only, config);

    const double analysis_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    std::cout << std::setprecision(17)
              << "advisor_qubits=" << qubits << '\n'
              << "advisor_rounds=" << rounds << '\n'
              << "advisor_operations=" << operations.size() << '\n'
              << "advisor_phase_h_defects=" << decision.phase_h_defects << '\n'
              << "advisor_phase_branch_log2=" << decision.phase_h_defects << '\n'
              << "advisor_phase_branch_fits_size_t="
              << (decision.phase_branch_envelope_fits_size_t ? 1 : 0) << '\n'
              << "advisor_hpath_events=" << decision.hpath.h_events << '\n'
              << "advisor_hpath_active_qubits=" << decision.hpath.h_active_qubits << '\n'
              << "advisor_hpath_factor_variables=" << decision.hpath.factor_variables << '\n'
              << "advisor_hpath_factor_count=" << decision.hpath.factor_count << '\n'
              << "advisor_hpath_peak_union_variables=" << decision.hpath.peak_union_variables << '\n'
              << "advisor_hpath_peak_factor_entries=" << decision.hpath.peak_factor_entries << '\n'
              << "advisor_hpath_peak_log2_ceiling=" << decision.hpath.peak_factor_log2_ceiling << '\n'
              << "advisor_hpath_compiled_index_entries=" << decision.hpath.compiled_index_entries << '\n'
              << "advisor_hpath_workspace_slots=" << decision.hpath.workspace_slots << '\n'
              << "advisor_hpath_graph_bytes=" << decision.hpath.graph_estimated_bytes << '\n'
              << "advisor_hpath_plan_bytes=" << decision.hpath.plan_estimated_bytes << '\n'
              << "advisor_structural_log2_margin=" << decision.structural_log2_margin << '\n'
              << "advisor_selected_route=" << static_cast<unsigned>(decision.route) << '\n'
              << "advisor_analysis_ms=" << analysis_ms << '\n'
              << "phase_only_selected_route=" << static_cast<unsigned>(phase.route) << '\n'
              << "state_expansion_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";
    return decision.route == ExactAmplitudeRoute::HadamardPathFactor ? 0 : 1;
}
