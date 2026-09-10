#include "qubit/qmps.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::MPSPauliPlan;
using qubit::MatrixProductState;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& message) {
    require(std::abs(actual - expected) <= 2e-11, message);
}

double reference_marginal(
    const QRegister& state,
    std::span<const QubitId> qubits,
    std::span<const std::uint8_t> bits) {
    const std::vector<qubit::QComplex> amplitudes = state.materialize(20U);
    double probability = 0.0;
    for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
        bool match = true;
        for (std::size_t index = 0U; index < qubits.size(); ++index) {
            const std::uint8_t bit = static_cast<std::uint8_t>(
                (basis >> static_cast<std::size_t>(qubits[index])) & 1U);
            if (bit != bits[index]) {
                match = false;
                break;
            }
        }
        if (match) {
            probability += amplitudes[basis].norm2();
        }
    }
    return probability;
}

}  // namespace

int main() {
    {
        const MatrixProductState ghz = MatrixProductState::ghz(8U);
        const MPSPauliPlan plan(MatrixProductState::ghz(8U));
        const std::array<QubitId, 1> q0{{0U}};
        const std::array<std::uint8_t, 1> zero{{0U}};
        const std::array<std::uint8_t, 1> one{{1U}};
        require_close(ghz.marginal_probability(q0, zero), 0.5,
                      "GHZ direct marginal P(q0=0) is wrong");
        require_close(plan.marginal_probability(q0, zero), 0.5,
                      "GHZ cached marginal P(q0=0) is wrong");
        require_close(plan.marginal_probability(q0, one), 0.5,
                      "GHZ cached marginal P(q0=1) is wrong");

        const std::array<QubitId, 2> ends{{0U, 7U}};
        const std::array<std::uint8_t, 2> consistent{{1U, 1U}};
        const std::array<std::uint8_t, 2> inconsistent{{0U, 1U}};
        require_close(ghz.marginal_probability(ends, consistent), 0.5,
                      "GHZ direct endpoint marginal is wrong");
        require_close(plan.marginal_probability(ends, consistent), 0.5,
                      "GHZ cached endpoint marginal is wrong");
        require(plan.marginal_probability(ends, inconsistent) == 0.0,
                "GHZ inconsistent endpoint marginal was nonzero");

        const std::array<QubitId, 0> empty_qubits{};
        const std::array<std::uint8_t, 0> empty_bits{};
        require_close(ghz.marginal_probability(empty_qubits, empty_bits), 1.0,
                      "GHZ direct empty marginal is not one");
        require_close(plan.marginal_probability(empty_qubits, empty_bits), 1.0,
                      "GHZ cached empty marginal is not one");
    }

    {
        constexpr std::size_t qubits = 6U;
        MatrixProductState mps = MatrixProductState::zero(qubits);
        QRegister reference(qubits);

        mps.apply_unitary(0U, qubit::gates::h());
        reference.apply_h(0U);
        mps.apply_cnot(0U, 1U);
        reference.apply_cnot(0U, 1U);
        mps.apply_unitary(1U, qubit::gates::ry(0.19));
        reference.apply_ry(1U, 0.19);
        mps.apply_unitary(2U, qubit::gates::rx(-0.31));
        reference.apply_rx(2U, -0.31);
        mps.apply_cz(1U, 2U);
        reference.apply_cz(1U, 2U);
        mps.apply_unitary(3U, qubit::gates::rz(0.23));
        reference.apply_rz(3U, 0.23);
        mps.apply_unitary(4U, qubit::gates::h());
        reference.apply_h(4U);
        mps.apply_cz(4U, 5U);
        reference.apply_cz(4U, 5U);

        const MPSPauliPlan plan(mps);
        const std::array<QubitId, 3> selected{{0U, 1U, 2U}};
        const std::array<std::uint8_t, 3> bits{{0U, 0U, 1U}};
        const double expected = reference_marginal(reference, selected, bits);
        require_close(mps.marginal_probability(selected, bits), expected,
                      "non-Clifford direct MPS marginal differs from QRegister");
        require_close(plan.marginal_probability(selected, bits), expected,
                      "non-Clifford cached MPS marginal differs from QRegister");

        const std::array<QubitId, 2> local{{4U, 5U}};
        const std::array<std::uint8_t, 2> local_bits{{1U, 0U}};
        const double local_expected = reference_marginal(reference, local, local_bits);
        require_close(plan.marginal_probability(local, local_bits), local_expected,
                      "local cached MPS marginal differs from QRegister");
    }

    {
        const MPSPauliPlan plan(MatrixProductState::cluster(8U));
        bool rejected = false;
        try {
            const std::array<QubitId, 2> duplicate{{1U, 1U}};
            const std::array<std::uint8_t, 2> bits{{0U, 0U}};
            static_cast<void>(plan.marginal_probability(duplicate, bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS cached marginal accepted duplicate qubits");

        rejected = false;
        try {
            const std::array<QubitId, 1> qubits{{8U}};
            const std::array<std::uint8_t, 1> bits{{0U}};
            static_cast<void>(plan.marginal_probability(qubits, bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS cached marginal accepted an out-of-range qubit");

        rejected = false;
        try {
            const std::array<QubitId, 1> qubits{{1U}};
            const std::array<std::uint8_t, 1> bits{{2U}};
            static_cast<void>(plan.marginal_probability(qubits, bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS cached marginal accepted a non-binary bit");

        rejected = false;
        try {
            const std::array<QubitId, 1> qubits{{1U}};
            const std::array<std::uint8_t, 2> bits{{0U, 1U}};
            static_cast<void>(plan.marginal_probability(qubits, bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS cached marginal accepted mismatched query widths");
    }

    return 0;
}
