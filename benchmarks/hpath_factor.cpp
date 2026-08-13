#include "qubit/qhpath_factor.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct CarrierResult {
    qubit::ExactHadamardPathAmplitude result{};
    double compile_us{0.0};
    double query_ms{0.0};
};

CarrierResult spatial_chain() {
    using qubit::ExactHadamardPathAmplitudePlan;
    using qubit::ExactHadamardPathConfig;
    using qubit::Operation;
    using qubit::OperationCode;

    constexpr std::size_t qubits = 1536U;
    std::vector<Operation> operations;
    operations.reserve(2U * qubits);
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(qubit),
            static_cast<qubit::QubitId>(qubit + 1U),
        });
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }

    ExactHadamardPathConfig config;
    config.factor.max_factor_entries = 64U;
    config.factor.max_factors = 10000U;
    config.factor.max_variables = 4096U;
    config.factor.max_compiled_index_entries = 1U << 20U;
    config.factor.reuse_workspace_slots = true;
    config.max_qubits = 4096U;
    config.max_operations = 10000U;
    config.max_h_events = 4096U;

    const auto compile_begin = Clock::now();
    ExactHadamardPathAmplitudePlan plan(qubits, operations, config);
    const auto compile_end = Clock::now();
    std::array<std::uint8_t, qubits> bits{};
    const auto query_begin = Clock::now();
    auto result = plan.scaled_amplitude_bits(bits);
    const auto query_end = Clock::now();
    return CarrierResult{
        std::move(result),
        std::chrono::duration<double, std::micro>(compile_end - compile_begin).count(),
        std::chrono::duration<double, std::milli>(query_end - query_begin).count(),
    };
}

CarrierResult temporal_grid() {
    using qubit::ExactHadamardPathAmplitudePlan;
    using qubit::ExactHadamardPathConfig;
    using qubit::Operation;
    using qubit::OperationCode;

    constexpr std::size_t qubits = 6U;
    constexpr std::size_t rounds = 682U;
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
                0.00091 * static_cast<double>(1U + round * qubits + target),
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

    ExactHadamardPathConfig config;
    config.factor.max_factor_entries = 2048U;
    config.factor.max_factors = 50000U;
    config.factor.max_variables = 8192U;
    config.factor.max_compiled_index_entries = 4U * 1024U * 1024U;
    config.factor.reuse_workspace_slots = true;
    config.max_qubits = 64U;
    config.max_operations = 50000U;
    config.max_h_events = 8192U;

    const auto compile_begin = Clock::now();
    ExactHadamardPathAmplitudePlan plan(qubits, operations, config);
    const auto compile_end = Clock::now();
    std::array<std::uint8_t, qubits> bits{};
    const auto query_begin = Clock::now();
    auto result = plan.scaled_amplitude_bits(bits);
    const auto query_end = Clock::now();
    return CarrierResult{
        std::move(result),
        std::chrono::duration<double, std::micro>(compile_end - compile_begin).count(),
        std::chrono::duration<double, std::milli>(query_end - query_begin).count(),
    };
}

bool hard_width_rejects(double& elapsed_ms) {
    using qubit::ExactHadamardPathAmplitudePlan;
    using qubit::ExactHadamardPathConfig;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    constexpr std::size_t qubits = 32U;
    std::vector<Operation> operations;
    operations.reserve(600U);
    for (std::size_t first = 0U; first < qubits; ++first) {
        for (std::size_t second = first + 1U; second < qubits; ++second) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(first),
                static_cast<qubit::QubitId>(second),
            });
        }
    }
    for (std::size_t target = 0U; target < qubits; ++target) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(target)});
    }

    ExactHadamardPathConfig config;
    config.factor.max_factor_entries = 1024U;
    config.factor.max_variables = 64U;
    config.factor.max_factors = 4096U;
    config.max_qubits = 64U;
    config.max_operations = 4096U;
    config.max_h_events = 64U;
    ExactHadamardPathAmplitudePlan plan(qubits, operations, config);
    std::array<std::uint8_t, qubits> bits{};
    const auto begin = Clock::now();
    try {
        (void)plan.scaled_amplitude_bits(bits);
    } catch (const QStateError&) {
        const auto end = Clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return true;
    }
    const auto end = Clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return false;
}

}  // namespace

int main() {
    const CarrierResult spatial = spatial_chain();
    const CarrierResult temporal = temporal_grid();
    double hard_ms = 0.0;
    const bool hard_rejected = hard_width_rejects(hard_ms);

    const double spatial_dense_to_peak_log2 = 1536.0 -
        std::log2(static_cast<double>(spatial.result.factor_stats.peak_factor_entries));
    const double temporal_flat_to_peak_log2 =
        static_cast<double>(temporal.result.h_events) -
        std::log2(static_cast<double>(temporal.result.factor_stats.peak_factor_entries));

    std::cout << std::setprecision(17)
              << "spatial_qubits=1536\n"
              << "spatial_h_events=" << spatial.result.h_events << '\n'
              << "spatial_raw_branch_count_log2=" << spatial.result.h_events << '\n'
              << "spatial_factor_variables=" << spatial.result.factor_variables << '\n'
              << "spatial_factor_count=" << spatial.result.factor_count << '\n'
              << "spatial_peak_union_variables=" << spatial.result.factor_stats.peak_union_variables << '\n'
              << "spatial_peak_factor_entries=" << spatial.result.factor_stats.peak_factor_entries << '\n'
              << "spatial_workspace_slots=" << spatial.result.factor_stats.workspace_slots << '\n'
              << "spatial_dense_to_peak_factor_ratio_log2=" << spatial_dense_to_peak_log2 << '\n'
              << "spatial_log2_probability_zero=" << spatial.result.log2_probability() << '\n'
              << "spatial_compile_us=" << spatial.compile_us << '\n'
              << "spatial_query_ms=" << spatial.query_ms << '\n'
              << "temporal_qubits=6\n"
              << "temporal_rounds=682\n"
              << "temporal_h_events=" << temporal.result.h_events << '\n'
              << "temporal_raw_branch_count_log2=" << temporal.result.h_events << '\n'
              << "temporal_h_active_qubits=" << temporal.result.h_active_qubits << '\n'
              << "temporal_factor_variables=" << temporal.result.factor_variables << '\n'
              << "temporal_factor_count=" << temporal.result.factor_count << '\n'
              << "temporal_peak_union_variables=" << temporal.result.factor_stats.peak_union_variables << '\n'
              << "temporal_peak_factor_entries=" << temporal.result.factor_stats.peak_factor_entries << '\n'
              << "temporal_workspace_slots=" << temporal.result.factor_stats.workspace_slots << '\n'
              << "temporal_flat_to_peak_factor_ratio_log2=" << temporal_flat_to_peak_log2 << '\n'
              << "temporal_log2_probability_zero=" << temporal.result.log2_probability() << '\n'
              << "temporal_metadata_estimated_bytes=" << temporal.result.metadata_estimated_bytes << '\n'
              << "temporal_graph_estimated_bytes=" << temporal.result.graph_estimated_bytes << '\n'
              << "temporal_plan_estimated_bytes=" << temporal.result.plan_estimated_bytes << '\n'
              << "temporal_compile_us=" << temporal.compile_us << '\n'
              << "temporal_query_ms=" << temporal.query_ms << '\n'
              << "temporal_physical_dense_amplitudes=64\n"
              << "hard_qubits=32\n"
              << "hard_cz_edges=496\n"
              << "hard_factor_entry_cap=1024\n"
              << "hard_dense_width_rejected=" << (hard_rejected ? 1 : 0) << '\n'
              << "hard_rejection_ms=" << hard_ms << '\n'
              << "dense_state_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n";
    return 0;
}
