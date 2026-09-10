#include "qubit/qdefect_advisor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactDefectAdvisorConfig;
    using qubit::ExactDefectQuery;
    using qubit::ExactDefectRepresentationAdvisor;
    using qubit::Operation;
    using qubit::OperationCode;

    ExactDefectAdvisorConfig config;
    config.max_qubits = 8192U;
    config.max_operations = 1U << 20U;
    config.max_phase_h_defects = 128U;
    config.max_magic_t_defects = 128U;

    const auto build_case = [](std::size_t h_count, std::size_t t_count) {
        std::vector<Operation> operations;
        operations.reserve(h_count + t_count + 128U);
        for (std::size_t index = 0U; index < h_count; ++index) {
            operations.push_back({
                OperationCode::H,
                static_cast<qubit::QubitId>(index % 4096U),
            });
        }
        for (std::size_t index = 0U; index < t_count; ++index) {
            operations.push_back({
                (index & 1U) == 0U ? OperationCode::T : OperationCode::Tdg,
                static_cast<qubit::QubitId>((index * 17U) % 4096U),
            });
        }
        for (std::size_t index = 0U; index < 128U; ++index) {
            const auto first = static_cast<qubit::QubitId>((index * 19U) % 4096U);
            const auto second = static_cast<qubit::QubitId>((first + 1U) % 4096U);
            operations.push_back({OperationCode::Cz, first, second});
        }
        return operations;
    };

    const auto phase_case = build_case(4U, 64U);
    const auto magic_case = build_case(64U, 4U);
    const auto tie_case = build_case(16U, 16U);

    const auto begin = Clock::now();
    const auto phase = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4096U, phase_case, ExactDefectQuery::StateCarrier, config);
    const auto magic = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4096U, magic_case, ExactDefectQuery::StateCarrier, config);
    const auto tie = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4096U, tie_case, ExactDefectQuery::StateCarrier, config);
    const auto end = Clock::now();

    std::vector<Operation> diagonal;
    diagonal.reserve(262144U);
    for (std::size_t index = 0U; index < 262144U; ++index) {
        diagonal.push_back({
            OperationCode::T,
            static_cast<qubit::QubitId>(index % 4096U),
        });
    }
    const auto large_begin = Clock::now();
    const auto large = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4096U, diagonal, ExactDefectQuery::StateCarrier, config);
    const auto large_end = Clock::now();

    const double triad_us = std::chrono::duration<double, std::micro>(end - begin).count();
    const double large_ms = std::chrono::duration<double, std::milli>(large_end - large_begin).count();

    std::cout << std::setprecision(17)
              << "advisor_qubits=4096\n"
              << "phase_case_route=" << static_cast<unsigned>(phase.route) << '\n'
              << "phase_case_h_defects=" << phase.phase_graph.defects << '\n'
              << "phase_case_t_defects=" << phase.low_magic.defects << '\n'
              << "phase_case_gap_log2=" << phase.branch_envelope_gap_log2 << '\n'
              << "phase_case_selected_envelope=" << phase.phase_graph.branch_envelope << '\n'
              << "magic_case_route=" << static_cast<unsigned>(magic.route) << '\n'
              << "magic_case_h_defects=" << magic.phase_graph.defects << '\n'
              << "magic_case_t_defects=" << magic.low_magic.defects << '\n'
              << "magic_case_gap_log2=" << magic.branch_envelope_gap_log2 << '\n'
              << "magic_case_selected_envelope=" << magic.low_magic.branch_envelope << '\n'
              << "tie_case_route=" << static_cast<unsigned>(tie.route) << '\n'
              << "tie_case_gap_log2=" << tie.branch_envelope_gap_log2 << '\n'
              << "triad_analysis_us=" << triad_us << '\n'
              << "large_operations=" << diagonal.size() << '\n'
              << "large_route=" << static_cast<unsigned>(large.route) << '\n'
              << "large_phase_h_defects=" << large.phase_graph.defects << '\n'
              << "large_magic_t_defects=" << large.low_magic.defects << '\n'
              << "large_magic_envelope_fits_size_t="
              << (large.low_magic.branch_envelope_fits_size_t ? 1 : 0) << '\n'
              << "large_analysis_ms=" << large_ms << '\n'
              << "state_expansion_materialized=0\n";
    return 0;
}
