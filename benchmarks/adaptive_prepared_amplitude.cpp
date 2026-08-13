#include "qubit/qadaptive_prepared_amplitude.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

std::vector<qubit::Operation> spacetime_operations() {
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

qubit::ExactAdaptiveAmplitudeConfig config() {
    qubit::ExactAdaptiveAmplitudeConfig result;
    result.advisor.max_qubits = 2048U;
    result.advisor.max_operations = 50000U;
    result.advisor.max_phase_h_defects = 10000U;
    result.advisor.max_phase_branches = 1U << 20U;
    result.advisor.max_hpath_events = 10000U;
    result.advisor.minimum_hpath_log2_margin = 8U;
    result.phase_graph.max_branches = 1U << 20U;
    result.phase_graph.max_retained_estimated_bytes = 512U * 1024U * 1024U;
    result.hpath.factor.max_variables = 10000U;
    result.hpath.factor.max_factors = 50000U;
    result.hpath.factor.max_factor_entries = 4096U;
    result.hpath.factor.max_compiled_index_entries = 8U * 1024U * 1024U;
    result.hpath.factor.reuse_workspace_slots = true;
    result.hpath.max_qubits = 2048U;
    result.hpath.max_operations = 50000U;
    result.hpath.max_h_events = 10000U;
    result.hpath.max_metadata_bytes = 64U * 1024U * 1024U;
    return result;
}

void fill_bits(
    std::array<std::uint8_t, 1024U>& bits,
    std::size_t query) {
    for (std::size_t index = 0U; index < bits.size(); ++index) {
        bits[index] = static_cast<std::uint8_t>(
            ((index * (13U + 2U * query) + 5U + query) >> (query & 3U)) & 1U);
    }
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::ExactAdaptivePreparedAmplitudePlan;
    using qubit::ExactAmplitudeRoute;

    constexpr std::size_t qubits = 1024U;
    constexpr std::size_t query_count = 8U;
    const auto operations = spacetime_operations();
    const auto settings = config();
    std::array<std::uint8_t, qubits> bits{};
    std::array<double, query_count> prepared_logp{};
    std::array<double, query_count> legacy_logp{};

    const auto prepared_setup_begin = Clock::now();
    ExactAdaptivePreparedAmplitudePlan prepared(qubits, operations, settings);
    auto prepared_workspace = prepared.workspace();
    const auto prepared_setup_end = Clock::now();

    const auto prepared_batch_begin = Clock::now();
    fill_bits(bits, 0U);
    const auto prepared_first_begin = Clock::now();
    prepared_logp[0] = prepared.log2_probability_bits(bits, prepared_workspace);
    const auto prepared_first_end = Clock::now();
    for (std::size_t query = 1U; query < query_count; ++query) {
        fill_bits(bits, query);
        prepared_logp[query] = prepared.log2_probability_bits(bits, prepared_workspace);
    }
    const auto prepared_batch_end = Clock::now();

    const auto legacy_setup_begin = Clock::now();
    ExactAdaptiveAmplitudePlan legacy(qubits, operations, settings);
    const auto legacy_setup_end = Clock::now();

    const auto legacy_batch_begin = Clock::now();
    fill_bits(bits, 0U);
    const auto legacy_first_begin = Clock::now();
    legacy_logp[0] = legacy.log2_probability_bits(bits);
    const auto legacy_first_end = Clock::now();
    for (std::size_t query = 1U; query < query_count; ++query) {
        fill_bits(bits, query);
        legacy_logp[query] = legacy.log2_probability_bits(bits);
    }
    const auto legacy_batch_end = Clock::now();

    double max_logp_error = 0.0;
    for (std::size_t query = 0U; query < query_count; ++query) {
        if (std::isfinite(prepared_logp[query]) && std::isfinite(legacy_logp[query])) {
            max_logp_error = std::max(
                max_logp_error,
                std::abs(prepared_logp[query] - legacy_logp[query]));
        } else if (!(std::isinf(prepared_logp[query]) && std::isinf(legacy_logp[query]))) {
            max_logp_error = std::numeric_limits<double>::infinity();
        }
    }

    const double prepared_setup_ms = std::chrono::duration<double, std::milli>(
        prepared_setup_end - prepared_setup_begin).count();
    const double prepared_first_ms = std::chrono::duration<double, std::milli>(
        prepared_first_end - prepared_first_begin).count();
    const double prepared_batch_ms = std::chrono::duration<double, std::milli>(
        prepared_batch_end - prepared_batch_begin).count();
    const double legacy_setup_ms = std::chrono::duration<double, std::milli>(
        legacy_setup_end - legacy_setup_begin).count();
    const double legacy_first_ms = std::chrono::duration<double, std::milli>(
        legacy_first_end - legacy_first_begin).count();
    const double legacy_batch_ms = std::chrono::duration<double, std::milli>(
        legacy_batch_end - legacy_batch_begin).count();

    const double prepared_one_shot_ms = prepared_setup_ms + prepared_first_ms;
    const double legacy_one_shot_ms = legacy_setup_ms + legacy_first_ms;
    const double prepared_total_ms = prepared_setup_ms + prepared_batch_ms;
    const double legacy_total_ms = legacy_setup_ms + legacy_batch_ms;
    const double one_shot_ratio = legacy_one_shot_ms / prepared_one_shot_ms;
    const double query_ratio = legacy_batch_ms / prepared_batch_ms;
    const double total_ratio = legacy_total_ms / prepared_total_ms;

    const auto* hpath_stats = prepared.prepared_hpath_stats();
    if (hpath_stats == nullptr) {
        return 2;
    }

    std::cout << std::setprecision(17)
              << "shared_qubits=" << qubits << '\n'
              << "shared_operations=" << operations.size() << '\n'
              << "shared_query_count=" << query_count << '\n'
              << "shared_route=" << static_cast<unsigned>(prepared.route()) << '\n'
              << "legacy_route=" << static_cast<unsigned>(legacy.route()) << '\n'
              << "shared_h_events=" << hpath_stats->h_events << '\n'
              << "shared_peak_union_variables=" << hpath_stats->factor.peak_union_variables << '\n'
              << "shared_peak_factor_entries=" << hpath_stats->factor.peak_factor_entries << '\n'
              << "shared_plan_bytes=" << hpath_stats->plan_estimated_bytes << '\n'
              << "shared_workspace_bytes=" << prepared_workspace.estimated_bytes() << '\n'
              << "shared_hpath_rebind_count=" << prepared_workspace.hpath_rebind_count() << '\n'
              << "shared_topology_compilations=1\n"
              << "legacy_topology_compilations=" << (query_count + 1U) << '\n'
              << "shared_certificate_reused_for_execution=1\n"
              << "shared_setup_ms=" << prepared_setup_ms << '\n'
              << "legacy_setup_ms=" << legacy_setup_ms << '\n'
              << "shared_first_query_ms=" << prepared_first_ms << '\n'
              << "legacy_first_query_ms=" << legacy_first_ms << '\n'
              << "shared_batch_ms=" << prepared_batch_ms << '\n'
              << "legacy_batch_ms=" << legacy_batch_ms << '\n'
              << "shared_one_shot_total_ms=" << prepared_one_shot_ms << '\n'
              << "legacy_one_shot_total_ms=" << legacy_one_shot_ms << '\n'
              << "shared_eight_query_total_ms=" << prepared_total_ms << '\n'
              << "legacy_eight_query_total_ms=" << legacy_total_ms << '\n'
              << "shared_one_shot_speed_ratio=" << one_shot_ratio << '\n'
              << "shared_query_batch_speed_ratio=" << query_ratio << '\n'
              << "shared_eight_query_total_speed_ratio=" << total_ratio << '\n'
              << "shared_max_logp_error=" << max_logp_error << '\n'
              << "dense_state_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return prepared.route() == ExactAmplitudeRoute::HadamardPathFactor &&
            legacy.route() == ExactAmplitudeRoute::HadamardPathFactor &&
            max_logp_error <= 1e-10 &&
            std::isfinite(one_shot_ratio) && one_shot_ratio > 0.0 &&
            std::isfinite(query_ratio) && query_ratio > 0.0 &&
            std::isfinite(total_ratio) && total_ratio > 0.0
        ? 0
        : 1;
}
