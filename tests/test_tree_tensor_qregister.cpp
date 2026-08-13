#include "qubit/qttn.hpp"
#include "qubit/qstate.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {

using qubit::BasisIndex;
using qubit::QComplex;
using qubit::QMatrix2;
using qubit::QRegister;
using qubit::QubitId;
using qubit::TreeTensorConfig;
using qubit::TreeTensorState;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& message) {
    const double scale = 1.0 + std::max(std::abs(actual), std::abs(expected));
    require(std::abs(actual - expected) <= 5e-10 * scale, message);
}

void apply_single(
    TreeTensorState& tree,
    QRegister& dense,
    QubitId qubit,
    const QMatrix2& matrix) {
    tree.apply_unitary(qubit, matrix);
    dense.apply_single(qubit, matrix);
}

void apply_cnot(TreeTensorState& tree, QRegister& dense, QubitId first, QubitId second) {
    tree.apply_cnot(first, second);
    dense.apply_two(first, second, qubit::gates::cnot());
}

void apply_cz(TreeTensorState& tree, QRegister& dense, QubitId first, QubitId second) {
    tree.apply_cz(first, second);
    dense.apply_two(first, second, qubit::gates::cz());
}

void compare_all(const TreeTensorState& tree, const QRegister& dense) {
    require(tree.qubit_count() < std::numeric_limits<std::size_t>::digits,
            "QRegister differential exceeds size_t width");
    const std::size_t entries = std::size_t{1U} << tree.qubit_count();
    constexpr double tolerance = 5e-10;
    QComplex phase{1.0, 0.0};
    bool have_phase = false;
    for (std::size_t index = 0U; index < entries; ++index) {
        const QComplex actual = tree.amplitude(static_cast<BasisIndex>(index));
        const QComplex expected = dense.amplitude(static_cast<BasisIndex>(index));
        if (!have_phase && actual.norm2() > tolerance * tolerance &&
            expected.norm2() > tolerance * tolerance) {
            phase = QComplex::from_polar(
                1.0,
                std::atan2(actual.im, actual.re) - std::atan2(expected.im, expected.re));
            have_phase = true;
        }
        if (actual.norm2() <= tolerance * tolerance &&
            expected.norm2() <= tolerance * tolerance) {
            continue;
        }
        require(have_phase && qubit::almost_equal(actual, phase * expected, tolerance),
                "TTN state differs from QRegister beyond global phase");
    }
    require(have_phase, "TTN/QRegister state contains no comparable amplitude");
    require_close(tree.norm2(), 1.0, "TTN norm differs from unity");
}

void fixed_branching() {
    TreeTensorConfig config;
    config.max_bond_dimension = 64U;
    config.max_scalars = 2'000'000U;
    TreeTensorState tree(6U, config);
    QRegister dense(6U);

    apply_single(tree, dense, 0U, qubit::gates::h());
    apply_single(tree, dense, 1U, qubit::gates::ry(0.42));
    apply_single(tree, dense, 5U, qubit::gates::rz(-0.27));
    apply_cnot(tree, dense, 0U, 4U);
    apply_cz(tree, dense, 1U, 5U);
    apply_cnot(tree, dense, 4U, 2U);
    apply_cz(tree, dense, 0U, 3U);
    apply_cnot(tree, dense, 5U, 1U);
    compare_all(tree, dense);

    const std::array<QubitId, 3> selected{{0U, 2U, 5U}};
    for (std::uint8_t mask = 0U; mask < 8U; ++mask) {
        const std::array<std::uint8_t, 3> bits{{
            static_cast<std::uint8_t>(mask & 1U),
            static_cast<std::uint8_t>((mask >> 1U) & 1U),
            static_cast<std::uint8_t>((mask >> 2U) & 1U),
        }};
        require_close(
            tree.marginal_probability(selected, bits),
            dense.marginal_probability(selected, bits),
            "TTN marginal differs from QRegister");
    }
}

void randomized_branching() {
    std::mt19937_64 generator(0x7a11d1ffULL);
    for (std::size_t trial = 0U; trial < 20U; ++trial) {
        const std::size_t qubits = 3U + static_cast<std::size_t>(generator() % 5U);
        TreeTensorConfig config;
        config.max_bond_dimension = 64U;
        config.max_scalars = 4'000'000U;
        TreeTensorState tree(qubits, config);
        QRegister dense(qubits);
        std::size_t controlled = 0U;

        for (std::size_t step = 0U; step < 10U; ++step) {
            const bool two_qubit = controlled < 4U && (generator() % 3U == 0U);
            if (!two_qubit) {
                const QubitId qubit = static_cast<QubitId>(generator() % qubits);
                QMatrix2 matrix;
                switch (generator() % 4U) {
                    case 0U:
                        matrix = qubit::gates::h();
                        break;
                    case 1U:
                        matrix = qubit::gates::x();
                        break;
                    case 2U:
                        matrix = qubit::gates::ry(
                            0.1 + 0.07 * static_cast<double>((trial + step) % 9U));
                        break;
                    default:
                        matrix = qubit::gates::rz(
                            -0.2 + 0.05 * static_cast<double>((trial + 2U * step) % 11U));
                        break;
                }
                apply_single(tree, dense, qubit, matrix);
                continue;
            }

            QubitId first = static_cast<QubitId>(generator() % qubits);
            QubitId second = static_cast<QubitId>(generator() % qubits);
            while (second == first) {
                second = static_cast<QubitId>(generator() % qubits);
            }
            if ((generator() & 1U) == 0U) {
                apply_cnot(tree, dense, first, second);
            } else {
                apply_cz(tree, dense, first, second);
            }
            ++controlled;
        }

        compare_all(tree, dense);
        QubitId first = static_cast<QubitId>(generator() % qubits);
        QubitId second = static_cast<QubitId>(generator() % qubits);
        while (second == first) {
            second = static_cast<QubitId>(generator() % qubits);
        }
        const std::array<QubitId, 2> selected{{first, second}};
        const std::array<std::uint8_t, 2> bits{{
            static_cast<std::uint8_t>(generator() & 1U),
            static_cast<std::uint8_t>(generator() & 1U),
        }};
        require_close(
            tree.marginal_probability(selected, bits),
            dense.marginal_probability(selected, bits),
            "random TTN marginal differs from QRegister");
    }
}

}  // namespace

int main() {
    fixed_branching();
    randomized_branching();
    return 0;
}
