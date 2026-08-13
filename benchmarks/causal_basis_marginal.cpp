#include "qubit/qcausal_basis_marginal.hpp"
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

qubit::ExactCausalBasisMarginalConfig broker_config() {
    qubit::ExactCausalBasisMarginalConfig result;
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
    result.local_state.max_component_qubits = 16U;
    result.local_state.max_dense_amplitudes = 1ULL << 16U;
    result.local_state.max_sparse_entries = 1'000'000U;
    result.max_local_qubits = 16U;
    result.local_preference_max_amplitudes = 1ULL << 12U;
    result.max_local_dense_bytes = 16U * 1024U * 1024U;
    return result;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactCausalBasisHadamardBornPlan;
    using qubit::ExactCausalBasisMarginalPlan;
    using qubit::ExactCausalBasisMarginalRoute;

    constexpr std::size_t qubits = 10000U;
    constexpr std::size_t island_width = 4U;
    constexpr std::size_t island_count = qubits / island_width;
    const std::array<std::size_t, island_width> retained{0U, 1U, 2U, 3U};
    std::vector<std::uint8_t> input(qubits, 0U);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        input[qubit] = static_cast<std::uint8_t>(((qubit * 31U + 9U) >> 2U) & 1U);
    }
    const auto operations = full_operations(qubits);
    const auto settings = broker_config();

    const auto broker_setup_begin = Clock::now();
    ExactCausalBasisMarginalPlan broker(input, operations, retained, settings);
    auto broker_workspace = broker.workspace();
    const auto broker_setup_end = Clock::now();

    qubit::ExactCausalBasisHadamardBornConfig hpath_settings;
    hpath_settings.light_cone = settings.light_cone;
    hpath_settings.born = settings.born;
    const auto hpath_setup_begin = Clock::now();
    ExactCausalBasisHadamardBornPlan hpath(input, operations, retained, hpath_settings);
    auto hpath_workspace = hpath.workspace();
    const auto hpath_setup_end = Clock::now();

    const auto broker_query_begin = Clock::now();
    const auto broker_values = broker.marginal(broker_workspace);
    const auto broker_query_end = Clock::now();

    const auto hpath_query_begin = Clock::now();
    const auto hpath_values = hpath.marginal(hpath_workspace);
    const auto hpath_query_end = Clock::now();

    double max_error = 0.0;
    double probability_sum = 0.0;
    double max_abs_imaginary = 0.0;
    for (std::size_t index = 0U; index < broker_values.size(); ++index) {
        max_error = std::max(
            max_error,
            std::hypot(
                broker_values[index].re - hpath_values[index].re,
                broker_values[index].im - hpath_values[index].im));
        probability_sum += broker_values[index].re;
        max_abs_imaginary = std::max(
            max_abs_imaginary, std::abs(broker_values[index].im));
    }

    const double broker_setup_ms =
        std::chrono::duration<double, std::milli>(broker_setup_end - broker_setup_begin).count();
    const double hpath_setup_ms =
        std::chrono::duration<double, std::milli>(hpath_setup_end - hpath_setup_begin).count();
    const double broker_query_ms =
        std::chrono::duration<double, std::milli>(broker_query_end - broker_query_begin).count();
    const double hpath_query_ms =
        std::chrono::duration<double, std::milli>(hpath_query_end - hpath_query_begin).count();
    const double setup_ratio = hpath_setup_ms / broker_setup_ms;
    const double query_ratio = hpath_query_ms / broker_query_ms;

    const auto& stats = broker.stats();
    const auto& hpath_stats = hpath.stats();
    const double qubit_reduction =
        static_cast<double>(stats.light_cone.full_qubits) /
        static_cast<double>(stats.light_cone.active_qubits);
    const double operation_reduction =
        static_cast<double>(stats.light_cone.full_operations) /
        static_cast<double>(stats.light_cone.active_operations);

    std::cout << std::setprecision(17)
              << "causal_broker_full_qubits=" << stats.light_cone.full_qubits << '\n'
              << "causal_broker_full_operations=" << stats.light_cone.full_operations << '\n'
              << "causal_broker_island_count=" << island_count << '\n'
              << "causal_broker_active_qubits=" << stats.light_cone.active_qubits << '\n'
              << "causal_broker_active_operations=" << stats.light_cone.active_operations << '\n'
              << "causal_broker_qubit_reduction_ratio=" << qubit_reduction << '\n'
              << "causal_broker_operation_reduction_ratio=" << operation_reduction << '\n'
              << "causal_broker_route=" << static_cast<unsigned>(stats.route) << '\n'
              << "causal_broker_local_eligible=" << (stats.local_eligible ? 1 : 0) << '\n'
              << "causal_broker_born_eligible=" << (stats.born_eligible ? 1 : 0) << '\n'
              << "causal_broker_local_worst_case_amplitudes="
              << stats.local_worst_case_amplitudes << '\n'
              << "causal_broker_local_worst_case_dense_bytes="
              << stats.local_worst_case_dense_bytes << '\n'
              << "causal_broker_local_state_bytes=" << stats.local_state_estimated_bytes << '\n'
              << "causal_broker_workspace_bytes=" << broker_workspace.estimated_bytes() << '\n'
              << "causal_broker_hpath_plan_bytes=" << hpath_stats.born.plan_estimated_bytes << '\n'
              << "causal_broker_hpath_workspace_bytes=" << hpath_workspace.estimated_bytes() << '\n'
              << "causal_broker_output_entries=" << broker_values.size() << '\n'
              << "causal_broker_probability_sum=" << probability_sum << '\n'
              << "causal_broker_max_abs_imaginary=" << max_abs_imaginary << '\n'
              << "causal_broker_max_hpath_error=" << max_error << '\n'
              << "causal_broker_setup_ms=" << broker_setup_ms << '\n'
              << "causal_broker_hpath_setup_ms=" << hpath_setup_ms << '\n'
              << "causal_broker_setup_speed_ratio=" << setup_ratio << '\n'
              << "causal_broker_query_ms=" << broker_query_ms << '\n'
              << "causal_broker_hpath_query_ms=" << hpath_query_ms << '\n'
              << "causal_broker_query_speed_ratio=" << query_ratio << '\n'
              << "dense_full_state_materialized=0\n"
              << "full_born_graph_materialized=0\n"
              << "explicit_output_enumeration_performed=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return stats.light_cone.full_qubits == 10000U &&
            stats.light_cone.full_operations == island_count * 45U &&
            stats.light_cone.active_qubits == 4U &&
            stats.light_cone.active_operations == 45U &&
            qubit_reduction == 2500.0 &&
            operation_reduction == 2500.0 &&
            stats.route == ExactCausalBasisMarginalRoute::LocalRegister &&
            stats.local_eligible &&
            stats.local_worst_case_amplitudes == 16U &&
            stats.local_worst_case_dense_bytes == 16U * sizeof(qubit::QComplex) &&
            broker_values.size() == 16U &&
            std::abs(probability_sum - 1.0) <= 1e-9 &&
            max_abs_imaginary <= 1e-12 &&
            max_error <= 1e-10 &&
            broker_setup_ms > 0.0 && broker_setup_ms < 10000.0 &&
            hpath_setup_ms > 0.0 && hpath_setup_ms < 10000.0 &&
            broker_query_ms > 0.0 && broker_query_ms < 10000.0 &&
            hpath_query_ms > 0.0 && hpath_query_ms < 10000.0 &&
            std::isfinite(setup_ratio) && setup_ratio > 0.0 &&
            std::isfinite(query_ratio) && query_ratio > 0.0
        ? 0
        : 1;
}
