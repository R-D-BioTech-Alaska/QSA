#include "qubit/qadaptive_basis_amplitude.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::vector<qubit::Operation> carrier(std::size_t qubits) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    operations.reserve(10U * qubits);
    for (std::size_t round = 0U; round < 2U; ++round) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.000137 * static_cast<double>(1U + qubit + round * qubits),
            });
            operations.push_back({
                ((qubit + round) & 1U) == 0U ? OperationCode::T : OperationCode::Sdg,
                static_cast<qubit::QubitId>(qubit),
            });
        }
        for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
            });
        }
    }
    return operations;
}

qubit::ExactAdaptiveAmplitudeConfig config() {
    qubit::ExactAdaptiveAmplitudeConfig result;
    result.advisor.max_qubits = 4096U;
    result.advisor.max_operations = 32768U;
    result.advisor.max_phase_h_defects = 20U;
    result.advisor.max_phase_branches = 1U << 20U;
    result.advisor.max_hpath_events = 8192U;
    result.advisor.minimum_hpath_log2_margin = 8U;
    result.hpath.factor.max_variables = 8192U;
    result.hpath.factor.max_factors = 100000U;
    result.hpath.factor.max_factor_entries = 4096U;
    result.hpath.factor.max_compiled_index_entries = 8U * 1024U * 1024U;
    result.hpath.factor.reuse_workspace_slots = true;
    result.hpath.max_qubits = 4096U;
    result.hpath.max_operations = 32768U;
    result.hpath.max_h_events = 8192U;
    result.hpath.max_metadata_bytes = 128U * 1024U * 1024U;
    result.phase_graph.max_branches = 1U << 20U;
    result.phase_graph.max_retained_estimated_bytes = 512U * 1024U * 1024U;
    return result;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactAdaptiveBasisAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::ExactPreparedBasisHadamardPathPlan;

    constexpr std::size_t qubits = 2048U;
    constexpr std::size_t query_count = 8U;
    std::vector<std::uint8_t> input(qubits, 0U);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        input[qubit] = static_cast<std::uint8_t>(((qubit * 29U + 7U) >> 3U) & 1U);
    }
    const auto operations = carrier(qubits);
    const auto settings = config();

    const auto adaptive_setup_begin = Clock::now();
    ExactAdaptiveBasisAmplitudePlan adaptive(input, operations, settings);
    auto adaptive_workspace = adaptive.workspace();
    const auto adaptive_setup_end = Clock::now();

    const auto direct_setup_begin = Clock::now();
    ExactPreparedBasisHadamardPathPlan direct(input, operations, settings.hpath);
    auto direct_workspace = direct.workspace();
    const auto direct_setup_end = Clock::now();

    std::vector<std::uint8_t> output(qubits, 0U);
    std::array<double, query_count> adaptive_logp{};
    double max_logp_error = 0.0;

    const auto adaptive_query_begin = Clock::now();
    for (std::size_t query = 0U; query < query_count; ++query) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            output[qubit] = static_cast<std::uint8_t>(
                ((qubit * (17U + 2U * query) + 11U + query) >> (query & 3U)) & 1U);
        }
        adaptive_logp[query] =
            adaptive.log2_probability_bits(output, adaptive_workspace);
    }
    const auto adaptive_query_end = Clock::now();

    const auto direct_query_begin = Clock::now();
    for (std::size_t query = 0U; query < query_count; ++query) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            output[qubit] = static_cast<std::uint8_t>(
                ((qubit * (17U + 2U * query) + 11U + query) >> (query & 3U)) & 1U);
        }
        const double direct_logp = direct.log2_probability_bits(output, direct_workspace);
        if (std::isfinite(adaptive_logp[query]) && std::isfinite(direct_logp)) {
            max_logp_error = std::max(
                max_logp_error, std::abs(adaptive_logp[query] - direct_logp));
        } else if (!(std::isinf(adaptive_logp[query]) && std::isinf(direct_logp))) {
            max_logp_error = std::numeric_limits<double>::infinity();
        }
    }
    const auto direct_query_end = Clock::now();

    const double adaptive_setup_ms =
        std::chrono::duration<double, std::milli>(adaptive_setup_end - adaptive_setup_begin).count();
    const double direct_setup_ms =
        std::chrono::duration<double, std::milli>(direct_setup_end - direct_setup_begin).count();
    const double adaptive_query_ms =
        std::chrono::duration<double, std::milli>(adaptive_query_end - adaptive_query_begin).count();
    const double direct_query_ms =
        std::chrono::duration<double, std::milli>(direct_query_end - direct_query_begin).count();

    const auto& decision = adaptive.decision();
    const auto& direct_stats = direct.stats();
    const double setup_overhead_ratio = adaptive_setup_ms / direct_setup_ms;
    const double query_overhead_ratio = adaptive_query_ms / direct_query_ms;

    std::cout << std::setprecision(17)
              << "adaptive_basis_qubits=" << qubits << '\n'
              << "adaptive_basis_operations=" << operations.size() << '\n'
              << "adaptive_basis_target_h_events=" << decision.target_h_events << '\n'
              << "adaptive_basis_preparation_h_events=" << decision.basis_preparation_h_events << '\n'
              << "adaptive_basis_phase_h_defects=" << decision.phase_h_defects << '\n'
              << "adaptive_basis_phase_envelope_fits_size_t="
              << (decision.phase_branch_envelope_fits_size_t ? 1 : 0) << '\n'
              << "adaptive_basis_phase_eligible=" << (decision.phase_graph_eligible ? 1 : 0) << '\n'
              << "adaptive_basis_hpath_eligible=" << (decision.hpath_eligible ? 1 : 0) << '\n'
              << "adaptive_basis_hpath_resource_rejected="
              << (decision.hpath_resource_rejected ? 1 : 0) << '\n'
              << "adaptive_basis_route=" << static_cast<unsigned>(decision.route) << '\n'
              << "adaptive_basis_decision_code=" << static_cast<unsigned>(decision.code) << '\n'
              << "adaptive_basis_hpath_variables=" << decision.hpath.factor_variables << '\n'
              << "adaptive_basis_hpath_peak_union_variables="
              << decision.hpath.factor.peak_union_variables << '\n'
              << "adaptive_basis_hpath_peak_factor_entries="
              << decision.hpath.factor.peak_factor_entries << '\n'
              << "adaptive_basis_hpath_plan_bytes=" << decision.hpath.plan_estimated_bytes << '\n'
              << "adaptive_basis_workspace_bytes=" << adaptive_workspace.estimated_bytes() << '\n'
              << "adaptive_basis_direct_variables=" << direct_stats.factor_variables << '\n'
              << "adaptive_basis_query_count=" << query_count << '\n'
              << "adaptive_basis_setup_ms=" << adaptive_setup_ms << '\n'
              << "adaptive_basis_direct_setup_ms=" << direct_setup_ms << '\n'
              << "adaptive_basis_setup_overhead_ratio=" << setup_overhead_ratio << '\n'
              << "adaptive_basis_query_ms=" << adaptive_query_ms << '\n'
              << "adaptive_basis_direct_query_ms=" << direct_query_ms << '\n'
              << "adaptive_basis_query_overhead_ratio=" << query_overhead_ratio << '\n'
              << "adaptive_basis_max_logp_error=" << max_logp_error << '\n'
              << "dense_state_materialized=0\n"
              << "explicit_phase_branch_list_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return decision.route == ExactAmplitudeRoute::HadamardPathFactor &&
            decision.target_h_events == 2U * qubits &&
            decision.basis_preparation_h_events == qubits &&
            decision.phase_h_defects == 3U * qubits &&
            !decision.phase_branch_envelope_fits_size_t &&
            !decision.phase_graph_eligible &&
            decision.hpath_eligible &&
            !decision.hpath_resource_rejected &&
            decision.hpath.factor_variables == qubits &&
            decision.hpath.factor_variables == direct_stats.factor_variables &&
            decision.hpath.factor.peak_union_variables <= 8U &&
            decision.hpath.factor.peak_factor_entries <= 256U &&
            adaptive_workspace.estimated_bytes() < 64U * 1024U * 1024U &&
            max_logp_error <= 1e-10 &&
            std::isfinite(setup_overhead_ratio) && setup_overhead_ratio > 0.0 &&
            std::isfinite(query_overhead_ratio) && query_overhead_ratio > 0.0
        ? 0
        : 1;
}
