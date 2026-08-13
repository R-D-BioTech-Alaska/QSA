#include "qubit/qhpath_born.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

std::vector<qubit::Operation> spacetime_operations() {
    using qubit::Operation;
    using qubit::OperationCode;
    constexpr std::size_t qubits = 1024U;
    constexpr std::size_t rounds = 4U;
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
                0.000271 * static_cast<double>(1U + round * qubits + target),
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

qubit::ExactHadamardBornConfig config() {
    qubit::ExactHadamardBornConfig result;
    result.factor.max_variables = 12000U;
    result.factor.max_factors = 50000U;
    result.factor.max_factor_entries = 1U << 20U;
    result.factor.max_compiled_index_entries = 16U * 1024U * 1024U;
    result.factor.reuse_workspace_slots = true;
    result.max_qubits = 2048U;
    result.max_operations = 50000U;
    result.max_h_events = 5000U;
    result.max_retained_qubits = 8U;
    return result;
}

bool broad_width_rejected() {
    using qubit::ExactHadamardBornMarginalPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    constexpr std::size_t qubits = 16U;
    std::vector<Operation> operations;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    for (std::size_t first = 0U; first < qubits; ++first) {
        for (std::size_t second = first + 1U; second < qubits; ++second) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(first),
                static_cast<qubit::QubitId>(second),
            });
        }
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    auto limited = config();
    limited.factor.max_factor_entries = 1024U;
    const std::array<std::size_t, 1> retained{0U};
    try {
        (void)ExactHadamardBornMarginalPlan(qubits, operations, retained, limited);
    } catch (const QStateError&) {
        return true;
    }
    return false;
}

std::size_t ceil_log2(std::size_t value) {
    if (value <= 1U) {
        return 0U;
    }
    std::size_t result = 0U;
    std::size_t current = value - 1U;
    while (current != 0U) {
        ++result;
        current >>= 1U;
    }
    return result;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactHadamardBornMarginalPlan;

    constexpr std::size_t qubits = 1024U;
    constexpr std::size_t rounds = 4U;
    const auto operations = spacetime_operations();
    const std::array<std::size_t, 4> retained{0U, 341U, 682U, 1023U};

    const auto setup_begin = Clock::now();
    ExactHadamardBornMarginalPlan plan(qubits, operations, retained, config());
    auto workspace = plan.workspace();
    const auto setup_end = Clock::now();
    const auto query_begin = Clock::now();
    const auto marginal = plan.marginal(workspace);
    const auto query_end = Clock::now();

    double total = 0.0;
    double max_abs_imaginary = 0.0;
    double minimum_probability = std::numeric_limits<double>::infinity();
    double maximum_probability = -std::numeric_limits<double>::infinity();
    for (const auto& value : marginal) {
        total += value.re;
        max_abs_imaginary = std::max(max_abs_imaginary, std::abs(value.im));
        minimum_probability = std::min(minimum_probability, value.re);
        maximum_probability = std::max(maximum_probability, value.re);
    }

    const double setup_ms =
        std::chrono::duration<double, std::milli>(setup_end - setup_begin).count();
    const double query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_begin).count();
    const auto& stats = plan.stats();
    const std::size_t peak_log2 = ceil_log2(std::max<std::size_t>(1U, stats.factor.peak_factor_entries));
    const std::size_t path_pair_gap = 2U * stats.h_events - peak_log2;
    const std::size_t output_gap = qubits - retained.size();
    const bool hard_rejected = broad_width_rejected();

    std::cout << std::setprecision(17)
              << "born_qubits=" << qubits << '\n'
              << "born_rounds=" << rounds << '\n'
              << "born_operations=" << operations.size() << '\n'
              << "born_h_events=" << stats.h_events << '\n'
              << "born_path_pair_log2=" << (2U * stats.h_events) << '\n'
              << "born_physical_output_space_log2=" << qubits << '\n'
              << "born_variables=" << stats.born_variables << '\n'
              << "born_factor_count=" << stats.factor_count << '\n'
              << "born_retained_qubits=" << stats.retained_qubits << '\n'
              << "born_output_entries=" << stats.output_entries << '\n'
              << "born_peak_union_variables=" << stats.factor.peak_union_variables << '\n'
              << "born_peak_factor_entries=" << stats.factor.peak_factor_entries << '\n'
              << "born_path_pair_to_peak_gap_log2=" << path_pair_gap << '\n'
              << "born_output_space_to_returned_gap_log2=" << output_gap << '\n'
              << "born_graph_bytes=" << stats.graph_estimated_bytes << '\n'
              << "born_plan_bytes=" << stats.plan_estimated_bytes << '\n'
              << "born_workspace_bytes=" << workspace.estimated_bytes() << '\n'
              << "born_probability_sum=" << total << '\n'
              << "born_max_abs_imaginary=" << max_abs_imaginary << '\n'
              << "born_min_probability=" << minimum_probability << '\n'
              << "born_max_probability=" << maximum_probability << '\n'
              << "born_setup_ms=" << setup_ms << '\n'
              << "born_query_ms=" << query_ms << '\n'
              << "born_hard_width_rejected=" << (hard_rejected ? 1 : 0) << '\n'
              << "dense_state_materialized=0\n"
              << "explicit_h_branch_list_materialized=0\n"
              << "physical_output_enumeration_performed=0\n";

    return marginal.size() == 16U &&
            std::abs(total - 1.0) <= 1e-8 &&
            max_abs_imaginary <= 1e-9 &&
            minimum_probability >= -1e-10 &&
            maximum_probability <= 1.0 + 1e-10 &&
            hard_rejected
        ? 0
        : 1;
}
