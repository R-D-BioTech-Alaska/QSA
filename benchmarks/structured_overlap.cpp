#include "qubit/qoverlap.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

namespace {

void prepare_block(qubit::QRegister& state, std::size_t base, double phase) {
    const auto q0 = static_cast<qubit::QubitId>(base);
    const auto q1 = static_cast<qubit::QubitId>(base + 1U);
    const auto q2 = static_cast<qubit::QubitId>(base + 2U);
    const auto q3 = static_cast<qubit::QubitId>(base + 3U);
    state.apply_h(q0);
    state.apply_cnot(q0, q1);
    state.apply_cnot(q1, q2);
    state.apply_cnot(q2, q3);
    state.apply_rz(q0, phase);
}

qubit::QComplex dense_control(const qubit::QRegister& left, const qubit::QRegister& right) {
    const auto lhs = left.materialize(4U);
    const auto rhs = right.materialize(4U);
    qubit::QComplex value{};
    for (std::size_t index = 0U; index < lhs.size(); ++index) {
        value += lhs[index].conjugate() * rhs[index];
    }
    return value;
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t blocks = 2500U;
    constexpr std::size_t qubits = blocks * 4U;
    constexpr double phase_delta = 1.2;

    qubit::QRegister local_left(4U);
    qubit::QRegister local_right(4U);
    prepare_block(local_left, 0U, 0.0);
    prepare_block(local_right, 0U, phase_delta);
    const qubit::QComplex local_overlap = dense_control(local_left, local_right);
    const double local_fidelity = local_overlap.norm2();
    const double expected_log2_fidelity = static_cast<double>(blocks) * std::log2(local_fidelity);

    qubit::QRegister left(qubits);
    qubit::QRegister right(qubits);
    const auto prepare_begin = Clock::now();
    for (std::size_t block = 0U; block < blocks; ++block) {
        const std::size_t base = block * 4U;
        prepare_block(left, base, 0.0);
        prepare_block(right, base, phase_delta);
    }
    const auto prepare_end = Clock::now();

    qubit::ExactStructuredOverlapConfig config;
    config.max_qubits = qubits;
    config.max_union_block_qubits = 4U;
    config.max_basis_evaluations = blocks * 16U;
    const auto overlap_begin = Clock::now();
    const auto result = qubit::exact_structured_overlap(left, right, config);
    const auto overlap_end = Clock::now();

    const double prepare_ms =
        std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
    const double overlap_ms =
        std::chrono::duration<double, std::milli>(overlap_end - overlap_begin).count();
    const double structural_log2_gap =
        static_cast<double>(qubits) - std::log2(static_cast<double>(result.stats.basis_evaluations));
    const double log2_error = std::abs(result.log2_fidelity - expected_log2_fidelity);

    std::cout << std::setprecision(17)
              << "overlap_qubits=" << qubits << '\n'
              << "overlap_blocks=" << blocks << '\n'
              << "overlap_left_components=" << result.stats.left_components << '\n'
              << "overlap_right_components=" << result.stats.right_components << '\n'
              << "overlap_union_blocks=" << result.stats.union_blocks << '\n'
              << "overlap_max_union_block_qubits=" << result.stats.max_union_block_qubits << '\n'
              << "overlap_basis_evaluations=" << result.stats.basis_evaluations << '\n'
              << "overlap_dense_log2_amplitudes=" << qubits << '\n'
              << "overlap_structural_log2_gap=" << structural_log2_gap << '\n'
              << "overlap_local_fidelity=" << local_fidelity << '\n'
              << "overlap_expected_log2_fidelity=" << expected_log2_fidelity << '\n'
              << "overlap_observed_log2_fidelity=" << result.log2_fidelity << '\n'
              << "overlap_log2_fidelity_error=" << log2_error << '\n'
              << "overlap_direct_fidelity=" << result.fidelity << '\n'
              << "overlap_prepare_ms=" << prepare_ms << '\n'
              << "overlap_query_ms=" << overlap_ms << '\n'
              << "dense_global_overlap_materialized=0\n";

    return result.stats.union_blocks == blocks &&
            result.stats.max_union_block_qubits == 4U &&
            result.stats.basis_evaluations == blocks * 16U &&
            structural_log2_gap > 9980.0 &&
            std::isfinite(result.log2_fidelity) &&
            log2_error <= 1e-9 &&
            prepare_ms > 0.0 && overlap_ms > 0.0
        ? 0
        : 1;
}
