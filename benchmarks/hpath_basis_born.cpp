#include "qubit/qhpath_basis_born.hpp"

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

std::vector<qubit::Operation> basis_carrier(std::size_t qubits) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    operations.reserve(12U * qubits);

    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(qubit),
            0U,
            0.000137 * static_cast<double>(qubit + 1U),
        });
        operations.push_back({
            (qubit & 1U) == 0U ? OperationCode::T : OperationCode::Sdg,
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
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(qubit),
            0U,
            -0.000211 * static_cast<double>(qubit + 3U),
        });
        operations.push_back({
            (qubit & 1U) == 0U ? OperationCode::S : OperationCode::Tdg,
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
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(qubit),
            0U,
            0.000091 * static_cast<double>(qubit + 5U),
        });
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(qubit),
            static_cast<qubit::QubitId>(qubit + 1U),
        });
    }
    return operations;
}

std::vector<qubit::Operation> plus_emulation(
    std::span<const std::uint8_t> input_bits,
    std::span<const qubit::Operation> operations) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> result;
    result.reserve(input_bits.size() * 2U + operations.size());
    for (std::size_t qubit = 0U; qubit < input_bits.size(); ++qubit) {
        if (input_bits[qubit] != 0U) {
            result.push_back({OperationCode::Z, static_cast<qubit::QubitId>(qubit)});
        }
        result.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    result.insert(result.end(), operations.begin(), operations.end());
    return result;
}

qubit::ExactHadamardBornConfig config() {
    qubit::ExactHadamardBornConfig result;
    result.factor.max_variables = 20000U;
    result.factor.max_factors = 100000U;
    result.factor.max_factor_entries = 1U << 20U;
    result.factor.max_compiled_index_entries = 8U * 1024U * 1024U;
    result.factor.reuse_workspace_slots = true;
    result.max_qubits = 4096U;
    result.max_operations = 32768U;
    result.max_h_events = 8192U;
    result.max_retained_qubits = 8U;
    return result;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactBasisHadamardBornMarginalPlan;
    using qubit::ExactHadamardBornMarginalPlan;

    constexpr std::size_t qubits = 2048U;
    const std::array<std::size_t, 4> retained{17U, 683U, 1365U, 2031U};
    std::vector<std::uint8_t> input(qubits, 0U);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        input[qubit] = static_cast<std::uint8_t>(((qubit * 17U + 5U) >> 2U) & 1U);
    }
    const auto operations = basis_carrier(qubits);
    const auto emulated_operations = plus_emulation(input, operations);
    const auto settings = config();

    const auto native_setup_begin = Clock::now();
    ExactBasisHadamardBornMarginalPlan native(input, operations, retained, settings);
    auto native_workspace = native.workspace();
    const auto native_setup_end = Clock::now();

    const auto emulated_setup_begin = Clock::now();
    ExactHadamardBornMarginalPlan emulated(
        qubits, emulated_operations, retained, settings);
    auto emulated_workspace = emulated.workspace();
    const auto emulated_setup_end = Clock::now();

    const auto native_query_begin = Clock::now();
    const auto native_values = native.marginal(native_workspace);
    const auto native_query_end = Clock::now();

    const auto emulated_query_begin = Clock::now();
    const auto emulated_values = emulated.marginal(emulated_workspace);
    const auto emulated_query_end = Clock::now();

    double max_error = 0.0;
    double probability_sum = 0.0;
    double max_abs_imaginary = 0.0;
    for (std::size_t index = 0U; index < native_values.size(); ++index) {
        max_error = std::max(
            max_error,
            std::hypot(
                native_values[index].re - emulated_values[index].re,
                native_values[index].im - emulated_values[index].im));
        probability_sum += native_values[index].re;
        max_abs_imaginary = std::max(max_abs_imaginary, std::abs(native_values[index].im));
    }

    const double native_setup_ms =
        std::chrono::duration<double, std::milli>(native_setup_end - native_setup_begin).count();
    const double emulated_setup_ms =
        std::chrono::duration<double, std::milli>(emulated_setup_end - emulated_setup_begin).count();
    const double native_query_ms =
        std::chrono::duration<double, std::milli>(native_query_end - native_query_begin).count();
    const double emulated_query_ms =
        std::chrono::duration<double, std::milli>(emulated_query_end - emulated_query_begin).count();

    const auto& native_stats = native.stats();
    const auto& emulated_stats = emulated.stats();
    const double variable_ratio =
        static_cast<double>(emulated_stats.born_variables) /
        static_cast<double>(native_stats.born_variables);
    const double plan_memory_ratio =
        static_cast<double>(emulated_stats.plan_estimated_bytes) /
        static_cast<double>(native_stats.plan_estimated_bytes);
    const double setup_ratio = emulated_setup_ms / native_setup_ms;
    const double query_ratio = emulated_query_ms / native_query_ms;

    std::cout << std::setprecision(17)
              << "basis_born_qubits=" << qubits << '\n'
              << "basis_born_operations=" << operations.size() << '\n'
              << "basis_born_emulated_operations=" << emulated_operations.size() << '\n'
              << "basis_born_h_events=" << native_stats.h_events << '\n'
              << "basis_born_h_active_qubits=" << native_stats.h_active_qubits << '\n'
              << "basis_born_hidden_variables_per_side=" << native_stats.hidden_variables_per_side << '\n'
              << "basis_born_removed_first_h_per_side=" << native_stats.removed_first_h_variables_per_side << '\n'
              << "basis_born_physical_output_variables=" << native_stats.physical_output_variables << '\n'
              << "basis_born_native_variables=" << native_stats.born_variables << '\n'
              << "basis_born_emulated_variables=" << emulated_stats.born_variables << '\n'
              << "basis_born_variable_reduction_ratio=" << variable_ratio << '\n'
              << "basis_born_native_factor_count=" << native_stats.factor_count << '\n'
              << "basis_born_emulated_factor_count=" << emulated_stats.factor_count << '\n'
              << "basis_born_native_peak_union_variables=" << native_stats.factor.peak_union_variables << '\n'
              << "basis_born_emulated_peak_union_variables=" << emulated_stats.factor.peak_union_variables << '\n'
              << "basis_born_native_peak_factor_entries=" << native_stats.factor.peak_factor_entries << '\n'
              << "basis_born_emulated_peak_factor_entries=" << emulated_stats.factor.peak_factor_entries << '\n'
              << "basis_born_native_graph_bytes=" << native_stats.graph_estimated_bytes << '\n'
              << "basis_born_emulated_graph_bytes=" << emulated_stats.graph_estimated_bytes << '\n'
              << "basis_born_native_plan_bytes=" << native_stats.plan_estimated_bytes << '\n'
              << "basis_born_emulated_plan_bytes=" << emulated_stats.plan_estimated_bytes << '\n'
              << "basis_born_plan_memory_reduction_ratio=" << plan_memory_ratio << '\n'
              << "basis_born_native_workspace_bytes=" << native_workspace.estimated_bytes() << '\n'
              << "basis_born_emulated_workspace_bytes=" << emulated_workspace.estimated_bytes() << '\n'
              << "basis_born_output_entries=" << native_values.size() << '\n'
              << "basis_born_probability_sum=" << probability_sum << '\n'
              << "basis_born_max_abs_imaginary=" << max_abs_imaginary << '\n'
              << "basis_born_max_error=" << max_error << '\n'
              << "basis_born_native_setup_ms=" << native_setup_ms << '\n'
              << "basis_born_emulated_setup_ms=" << emulated_setup_ms << '\n'
              << "basis_born_setup_speed_ratio=" << setup_ratio << '\n'
              << "basis_born_native_query_ms=" << native_query_ms << '\n'
              << "basis_born_emulated_query_ms=" << emulated_query_ms << '\n'
              << "basis_born_query_speed_ratio=" << query_ratio << '\n'
              << "dense_state_materialized=0\n"
              << "explicit_output_enumeration_performed=0\n"
              << "explicit_h_branch_list_materialized=0\n";

    return native_stats.hidden_variables_per_side == qubits &&
            native_stats.born_variables == 3U * qubits &&
            emulated_stats.born_variables == 7U * qubits &&
            variable_ratio > 2.3 &&
            native_stats.factor_count < emulated_stats.factor_count &&
            plan_memory_ratio > 1.5 &&
            native_values.size() == 16U &&
            std::abs(probability_sum - 1.0) <= 1e-8 &&
            max_abs_imaginary <= 1e-9 &&
            max_error <= 1e-8 &&
            std::isfinite(native_setup_ms) && native_setup_ms > 0.0 &&
            std::isfinite(emulated_setup_ms) && emulated_setup_ms > 0.0 &&
            std::isfinite(native_query_ms) && native_query_ms > 0.0 &&
            std::isfinite(emulated_query_ms) && emulated_query_ms > 0.0
        ? 0
        : 1;
}
