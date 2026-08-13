#include "qubit/qhpath_prepared.hpp"

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

qubit::ExactHadamardPathConfig config() {
    qubit::ExactHadamardPathConfig result;
    result.factor.max_variables = 10000U;
    result.factor.max_factors = 50000U;
    result.factor.max_factor_entries = 4096U;
    result.factor.max_compiled_index_entries = 8U * 1024U * 1024U;
    result.factor.reuse_workspace_slots = true;
    result.max_qubits = 2048U;
    result.max_operations = 50000U;
    result.max_h_events = 10000U;
    result.max_metadata_bytes = 64U * 1024U * 1024U;
    return result;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactHadamardPathAmplitudePlan;
    using qubit::ExactPreparedHadamardPathPlan;
    using qubit::ExactPreparedHadamardPathWorkspace;
    using qubit::QComplex;

    constexpr std::size_t qubits = 1024U;
    constexpr std::size_t query_count = 8U;
    constexpr std::size_t workspace_count = 16U;
    const auto operations = spacetime_operations();
    const auto settings = config();

    const auto prepare_begin = Clock::now();
    ExactPreparedHadamardPathPlan prepared(qubits, operations, settings);
    auto workspace = prepared.workspace();
    const auto prepare_end = Clock::now();

    ExactHadamardPathAmplitudePlan direct(qubits, operations, settings);
    std::array<std::uint8_t, qubits> bits{};
    std::array<double, query_count> prepared_logp{};
    double max_logp_error = 0.0;
    QComplex prepared_guard{};
    QComplex direct_guard{};

    const auto prepared_begin = Clock::now();
    for (std::size_t query = 0U; query < query_count; ++query) {
        for (std::size_t index = 0U; index < bits.size(); ++index) {
            bits[index] = static_cast<std::uint8_t>(
                ((index * (13U + 2U * query) + 5U + query) >> (query & 3U)) & 1U);
        }
        const auto value = prepared.scaled_amplitude_bits(bits, workspace);
        prepared_guard += value.mantissa;
        prepared_logp[query] = value.log2_probability();
    }
    const auto prepared_end = Clock::now();

    const auto direct_begin = Clock::now();
    for (std::size_t query = 0U; query < query_count; ++query) {
        for (std::size_t index = 0U; index < bits.size(); ++index) {
            bits[index] = static_cast<std::uint8_t>(
                ((index * (13U + 2U * query) + 5U + query) >> (query & 3U)) & 1U);
        }
        const auto direct_value = direct.scaled_amplitude_bits(bits);
        direct_guard += direct_value.mantissa;
        const double direct_logp = direct_value.log2_probability();
        if (std::isfinite(direct_logp) && std::isfinite(prepared_logp[query])) {
            max_logp_error = std::max(
                max_logp_error, std::abs(direct_logp - prepared_logp[query]));
        } else if (!(std::isinf(direct_logp) && std::isinf(prepared_logp[query]))) {
            max_logp_error = std::numeric_limits<double>::infinity();
        }
    }
    const auto direct_end = Clock::now();

    const double setup_ms =
        std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
    const double prepared_batch_ms =
        std::chrono::duration<double, std::milli>(prepared_end - prepared_begin).count();
    const double direct_batch_ms =
        std::chrono::duration<double, std::milli>(direct_end - direct_begin).count();
    const double prepared_total_ms = setup_ms + prepared_batch_ms;
    const double query_speed_ratio = direct_batch_ms / prepared_batch_ms;
    const double amortized_total_ratio = direct_batch_ms / prepared_total_ms;
    const double direct_per_query = direct_batch_ms / static_cast<double>(query_count);
    const double prepared_per_query = prepared_batch_ms / static_cast<double>(query_count);
    const double saved_per_query = direct_per_query - prepared_per_query;
    const double break_even_queries = saved_per_query > 0.0 ? setup_ms / saved_per_query : -1.0;

    const auto& stats = prepared.stats();
    const std::size_t compact_workspace_bytes = workspace.estimated_bytes();
    const std::size_t plan_copy_equivalent_bytes =
        compact_workspace_bytes + stats.plan_estimated_bytes;
    const double workspace_memory_ratio =
        static_cast<double>(plan_copy_equivalent_bytes) /
        static_cast<double>(compact_workspace_bytes);

    std::vector<ExactPreparedHadamardPathWorkspace> workspaces;
    workspaces.reserve(workspace_count);
    std::size_t compact_workspace_batch_bytes = 0U;
    for (std::size_t index = 0U; index < workspace_count; ++index) {
        workspaces.push_back(prepared.workspace());
        compact_workspace_batch_bytes += workspaces.back().estimated_bytes();
    }
    const std::size_t plan_copy_equivalent_batch_bytes =
        workspace_count * plan_copy_equivalent_bytes;

    std::cout << std::setprecision(17)
              << "prepared_qubits=" << qubits << '\n'
              << "prepared_operations=" << operations.size() << '\n'
              << "prepared_h_events=" << stats.h_events << '\n'
              << "prepared_factor_variables=" << stats.factor_variables << '\n'
              << "prepared_factor_count=" << stats.factor_count << '\n'
              << "prepared_peak_union_variables=" << stats.factor.peak_union_variables << '\n'
              << "prepared_peak_factor_entries=" << stats.factor.peak_factor_entries << '\n'
              << "prepared_output_bindings=" << stats.output_bindings << '\n'
              << "prepared_fixed_cz_pairs=" << stats.fixed_cz_pairs << '\n'
              << "prepared_metadata_bytes=" << stats.metadata_estimated_bytes << '\n'
              << "prepared_graph_bytes=" << stats.graph_estimated_bytes << '\n'
              << "prepared_plan_bytes=" << stats.plan_estimated_bytes << '\n'
              << "prepared_workspace_bytes=" << compact_workspace_bytes << '\n'
              << "prepared_plan_copy_equivalent_workspace_bytes=" << plan_copy_equivalent_bytes << '\n'
              << "prepared_workspace_memory_reduction_ratio=" << workspace_memory_ratio << '\n'
              << "prepared_workspace_count=" << workspace_count << '\n'
              << "prepared_compact_workspace_batch_bytes=" << compact_workspace_batch_bytes << '\n'
              << "prepared_plan_copy_equivalent_batch_bytes=" << plan_copy_equivalent_batch_bytes << '\n'
              << "prepared_shared_plan_serializes_queries=1\n"
              << "prepared_query_count=" << query_count << '\n'
              << "prepared_rebind_count=" << workspace.rebind_count() << '\n'
              << "prepared_setup_ms=" << setup_ms << '\n'
              << "prepared_batch_ms=" << prepared_batch_ms << '\n'
              << "direct_batch_ms=" << direct_batch_ms << '\n'
              << "prepared_query_speed_ratio=" << query_speed_ratio << '\n'
              << "prepared_amortized_total_ratio=" << amortized_total_ratio << '\n'
              << "prepared_break_even_queries=" << break_even_queries << '\n'
              << "prepared_max_logp_error=" << max_logp_error << '\n'
              << "prepared_guard_norm2=" << prepared_guard.norm2() << '\n'
              << "direct_guard_norm2=" << direct_guard.norm2() << '\n'
              << "dense_state_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return max_logp_error <= 1e-10 &&
            workspace_memory_ratio > 20.0 &&
            compact_workspace_batch_bytes < plan_copy_equivalent_batch_bytes &&
            std::isfinite(query_speed_ratio) && query_speed_ratio > 0.0 &&
            std::isfinite(amortized_total_ratio) && amortized_total_ratio > 0.0
        ? 0
        : 1;
}
