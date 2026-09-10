#include "qubit/qoverlap.hpp"

#include <cmath>
#include <stdexcept>

int main() {
    qubit::QRegister left(4U);
    qubit::QRegister right(4U);
    left.apply_h(0U);
    left.apply_cnot(0U, 1U);
    right.apply_h(0U);
    right.apply_cnot(0U, 1U);
    const auto same = qubit::exact_structured_overlap(left, right);
    if (std::abs(same.fidelity - 1.0) > 1e-12) {
        throw std::runtime_error("structured self overlap mismatch");
    }

    qubit::QRegister cross_left(4U);
    qubit::QRegister cross_right(4U);
    cross_left.apply_h(0U);
    cross_left.apply_cnot(0U, 1U);
    cross_left.apply_h(2U);
    cross_left.apply_cnot(2U, 3U);
    cross_right.apply_h(1U);
    cross_right.apply_cnot(1U, 2U);
    cross_right.apply_ry(0U, 0.31);
    const auto lhs = cross_left.materialize(4U);
    const auto rhs = cross_right.materialize(4U);
    qubit::QComplex control{};
    for (std::size_t i = 0U; i < lhs.size(); ++i) {
        control += lhs[i].conjugate() * rhs[i];
    }
    const auto observed = qubit::exact_structured_overlap(cross_left, cross_right);
    if (!qubit::almost_equal(observed.inner_product, control, 1e-12)) {
        throw std::runtime_error("structured overlap differs from dense control");
    }
    if (observed.stats.max_union_block_qubits != 4U || observed.stats.basis_evaluations != 16U) {
        throw std::runtime_error("structured overlap union certificate mismatch");
    }
    return 0;
}
