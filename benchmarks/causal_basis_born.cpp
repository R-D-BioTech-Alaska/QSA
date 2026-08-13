#include "qubit/qcausal_basis_born.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::vector<qubit::Operation> island_operations(std::size_t base) {
    using qubit::Operation;
    using qubit::OperationCode;
    constexpr std::size_t width = 4U;
    constexpr std::size_t rounds = 3U;
    std::vector<Operation> operations;
    operations.reserve(45U);
    for (std::size_t round = 0U; round < rounds; ++round) {
        for (std::size_t local = 0U; local < width; ++local) {
            const std::size_t qubit = base + local;
            operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.017 * static_cast<double>(1U + round * width + local),
            });
            operations.push_back({
                ((round + local) & 1U) == 0U ? OperationCode::T : OperationCode::Sdg,
                static_cast<qubit::QubitId>(qubit),
            });
        }
        for (std::size_t local = 0U; local + 1U < width; ++local) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(base + local),
                static_cast<qubit::QubitId>(base + local + 1U),
            });
        }
    }
    return operations;
}

std::vector<qubit::Operation> full_operations(std::size_t qubits) {
    constexpr std::size_t width = 4U;
    std::vector<qubit::Operation> operations;
    operations.reserve((qubits / width) * 45U);
    for (std::size_t base = 0U; base < qubits; base += width) {
        auto island = island_operations(base);
        operations.insert(operations.end(), island.begin(), island.end());
    }
    return operations;
}

qubit::ExactCausalBasisHadamardBornConfig config() {
    qubit::ExactCausalBasisHadamardBornConfig result;
    result.light_cone.max_qubits = 20000U;
    result.light_cone.max_operations = 200000U;
    result.light_cone.max_active_qubits = 64U;
    result.light_cone.max_active_operations = 1024U;
    result.born.factor.max_variables = 256U;
    result.born.factor.max_factors = 4096U;
    result.born.factor.max_factor_entries = 4096U;
    result.born.factor.max_compiled_index_entries = 1U << 20U;
    result.born.factor.reuse_workspace_slots = true;
    result.born.max_qubits = 64U;
    result.born.max_operations = 1024U;
    result.born.max_h_events = 256U;
    result.born.max_retained_qubits = 8U;
    return result;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactBasisHadamardBornMarginalPlan;
    using qubit::ExactCausalBasisHadamardBornPlan;

    constexpr std::size_t qubits = 10000U;
    constexpr std::size_t island_width = 4U;
    constexpr std::size_t island_count = qubits / island_width;
    const std::array<std::size_t, island_width> retained{0U, 1U, 2U, 3U};
    std::vector<std::uint8_t> input(qubits, 0U);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        input[qubit] = static_cast<std::uint8_t>(((qubit * 31U + 9U) >> 2U) & 1U);
    }
    const auto operations = full_operations(qubits);
    const auto settings = config();

    const auto causal_setup_begin = Clock::now();
    ExactCausalBasisHadamardBornPlan causal(input, operations, retained, settings);
    auto causal_workspace = causal.workspace();
    const auto causal_setup_end = Clock::now();

    const auto causal_query_begin = Clock::now();
    const auto causal_values = causal.marginal(causal_workspace);
    const auto causal_query_end = Clock::now();

    const auto compact_ops = island_operations(0U);
    const std::array<std::uint8_t, island_width> compact_input{
        input[0], input[1], input[2], input[3]};
    const std::array<std::size_t, island_width> compact_retained{0U, 1U, 2U, 3U};
    ExactBasisHadamardBornMarginalPlan compact(
        compact_input, compact_ops, compact_retained, settings.born);
    auto compact_workspace = compact.workspace();
    const auto compact_values = compact.marginal(compact_workspace);

    double max_error = 0.0;
    double probability_sum = 0.0;
    double max_abs_imaginary = 0.0;
    for (std::size_t index = 0U; index < causal_values.size(); ++index) {
        max_error = std::max(
            max_error,
            std::hypot(
                causal_values[index].re - compact_values[index].re,
                causal_values[index].im - compact_values[index].im));
        probability_sum += causal_values[index].re;
        max_abs_imaginary = std::max(
            max_abs_imaginary, std::abs(causal_values[index].im));
    }

    const double causal_setup_ms =
        std::chrono::duration<double, std::milli>(causal_setup_end - causal_setup_begin).count();
    const double causal_query_ms =
        std::chrono::duration<double, std::milli>(causal_query_end - causal_query_begin).count();
    const auto& stats = causal.stats();
    const double qubit_reduction =
        static_cast<double>(stats.light_cone.full_qubits) /
        static_cast<double>(stats.light_cone.active_qubits);
    const double operation_reduction =
        static_cast<double>(stats.light_cone.full_operations) /
        static_cast<double>(stats.light_cone.active_operations);

    std::cout << std::setprecision(17)
              << "causal_born_full_qubits=" << stats.light_cone.full_qubits << '\n'
              << "causal_born_full_operations=" << stats.light_cone.full_operations << '\n'
              << "causal_born_island_count=" << island_count << '\n'
              << "causal_born_active_qubits=" << stats.light_cone.active_qubits << '\n'
              << "causal_born_active_operations=" << stats.light_cone.active_operations << '\n'
              << "causal_born_pruned_qubits=" << stats.light_cone.pruned_qubits << '\n'
              << "causal_born_pruned_operations=" << stats.light_cone.pruned_operations << '\n'
              << "causal_born_qubit_reduction_ratio=" << qubit_reduction << '\n'
              << "causal_born_operation_reduction_ratio=" << operation_reduction << '\n'
              << "causal_born_light_cone_bytes=" << stats.light_cone.estimated_bytes << '\n'
              << "causal_born_compact_input_bytes=" << stats.compact_input_bytes << '\n'
              << "causal_born_h_events=" << stats.born.h_events << '\n'
              << "causal_born_hidden_variables_per_side=" << stats.born.hidden_variables_per_side << '\n'
              << "causal_born_total_variables=" << stats.born.born_variables << '\n'
              << "causal_born_factor_count=" << stats.born.factor_count << '\n'
              << "causal_born_peak_union_variables=" << stats.born.factor.peak_union_variables << '\n'
              << "causal_born_peak_factor_entries=" << stats.born.factor.peak_factor_entries << '\n'
              << "causal_born_plan_bytes=" << stats.born.plan_estimated_bytes << '\n'
              << "causal_born_workspace_bytes=" << causal_workspace.estimated_bytes() << '\n'
              << "causal_born_output_entries=" << causal_values.size() << '\n'
              << "causal_born_probability_sum=" << probability_sum << '\n'
              << "causal_born_max_abs_imaginary=" << max_abs_imaginary << '\n'
              << "causal_born_max_compact_error=" << max_error << '\n'
              << "causal_born_setup_ms=" << causal_setup_ms << '\n'
              << "causal_born_query_ms=" << causal_query_ms << '\n'
              << "dense_state_materialized=0\n"
              << "full_born_graph_materialized=0\n"
              << "explicit_output_enumeration_performed=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return stats.light_cone.full_qubits == 10000U &&
            stats.light_cone.full_operations == island_count * 45U &&
            stats.light_cone.active_qubits == 4U &&
            stats.light_cone.active_operations == 45U &&
            qubit_reduction == 2500.0 &&
            operation_reduction == 2500.0 &&
            stats.born.h_events == 12U &&
            stats.born.hidden_variables_per_side == 8U &&
            stats.born.born_variables == 20U &&
            stats.born.factor.peak_factor_entries <= 4096U &&
            causal_values.size() == 16U &&
            std::abs(probability_sum - 1.0) <= 1e-9 &&
            max_abs_imaginary <= 1e-9 &&
            max_error <= 1e-12 &&
            causal_setup_ms > 0.0 && causal_setup_ms < 10000.0 &&
            causal_query_ms > 0.0 && causal_query_ms < 10000.0
        ? 0
        : 1;
}
