#include "qubit/qphase_exact.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

bool close(QComplex left, QComplex right, double tolerance = 4e-11) {
    return almost_equal(left, right, tolerance);
}

bool close(double left, double right, double tolerance = 4e-11) {
    return std::abs(left - right) <= tolerance;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw QStateError(message);
    }
}

std::vector<QComplex> dense_plus(std::size_t qubits) {
    const std::size_t dimension = std::size_t{1U} << qubits;
    return std::vector<QComplex>(
        dimension, QComplex{std::exp2(-0.5 * static_cast<double>(qubits))});
}

void dense_single(
    std::vector<QComplex>& state,
    std::size_t qubit,
    const QMatrix2& matrix) {
    const std::size_t stride = std::size_t{1U} << qubit;
    const std::size_t period = stride << 1U;
    for (std::size_t base = 0U; base < state.size(); base += period) {
        for (std::size_t offset = 0U; offset < stride; ++offset) {
            const std::size_t zero = base + offset;
            const std::size_t one = zero + stride;
            const QComplex left = state[zero];
            const QComplex right = state[one];
            state[zero] = matrix(0U, 0U) * left + matrix(0U, 1U) * right;
            state[one] = matrix(1U, 0U) * left + matrix(1U, 1U) * right;
        }
    }
}

void dense_controlled_phase_ticks(
    std::vector<QComplex>& state,
    std::size_t first,
    std::size_t second,
    std::uint8_t ticks) {
    const std::size_t first_bit = std::size_t{1U} << first;
    const std::size_t second_bit = std::size_t{1U} << second;
    const QComplex phase = QComplex::from_polar(
        1.0, 0.25 * kPi * static_cast<double>(ticks & 7U));
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & first_bit) != 0U && (basis & second_bit) != 0U) {
            state[basis] *= phase;
        }
    }
}

void dense_swap(std::vector<QComplex>& state, std::size_t first, std::size_t second) {
    const std::size_t first_bit = std::size_t{1U} << first;
    const std::size_t second_bit = std::size_t{1U} << second;
    const std::size_t mask = first_bit | second_bit;
    std::vector<QComplex> output(state.size(), QComplex{});
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        const bool first_value = (basis & first_bit) != 0U;
        const bool second_value = (basis & second_bit) != 0U;
        output[first_value == second_value ? basis : basis ^ mask] = state[basis];
    }
    state.swap(output);
}

void dense_equivalence() {
    for (std::size_t qubits = 2U; qubits <= 6U; ++qubits) {
        ExactCompressedPhaseGraphSum exact(qubits);
        std::vector<QComplex> dense = dense_plus(qubits);
        for (std::size_t round = 0U; round < 8U; ++round) {
            const QubitId q = static_cast<QubitId>((round * 3U + 1U) % qubits);
            const QubitId r = static_cast<QubitId>((static_cast<std::size_t>(q) + 1U) % qubits);
            exact.apply_t(q);
            dense_single(dense, q, gates::t());
            exact.apply_controlled_phase_ticks(q, r, static_cast<std::uint8_t>((round % 7U) + 1U));
            dense_controlled_phase_ticks(
                dense, q, r, static_cast<std::uint8_t>((round % 7U) + 1U));
            exact.apply_h(q);
            dense_single(dense, q, gates::h());
            if ((round & 1U) == 0U) {
                exact.apply_y(r);
                dense_single(dense, r, gates::y());
            } else {
                exact.apply_sdg(r);
                dense_single(dense, r, gates::sdg());
            }
            if (qubits > 2U && round % 3U == 0U) {
                const QubitId third = static_cast<QubitId>((static_cast<std::size_t>(r) + 1U) % qubits);
                exact.apply_swap(r, third);
                dense_swap(dense, r, third);
            }
        }

        const std::vector<QComplex> materialized = exact.materialize(8U);
        require(materialized.size() == dense.size(),
                "exact compressed phase dense size mismatch");
        for (std::size_t basis = 0U; basis < dense.size(); ++basis) {
            require(close(materialized[basis], dense[basis]),
                    "exact compressed phase dense amplitude mismatch");
        }
    }
}

void exact_zero_and_cyclotomic_query() {
    ExactCompressedPhaseGraphSum state(1U);
    state.apply_h(0U);
    const std::vector<std::uint8_t> zero{0U};
    const std::vector<std::uint8_t> one{1U};
    const ExactCyclotomicScaledAmplitude a0 = state.exact_amplitude_bits(zero);
    const ExactCyclotomicScaledAmplitude a1 = state.exact_amplitude_bits(one);
    require(!a0.coefficient.zero(), "H|+> zero amplitude vanished");
    require(a1.coefficient.zero(), "H|+> one amplitude did not cancel exactly");
    require(std::isinf(state.log2_probability_bits(one)) &&
                state.log2_probability_bits(one) < 0.0,
            "exact zero amplitude did not produce -infinity log2 probability");
    require(close(state.amplitude_bits(zero), QComplex{1.0}),
            "H|+> zero amplitude mismatch");
    require(close(state.amplitude_bits(one), QComplex{}),
            "H|+> one amplitude mismatch");
}

void repeated_h_reconverges_exactly() {
    ExactCompressedPhaseGraphSum state(4U);
    for (std::size_t repeat = 0U; repeat < 512U; ++repeat) {
        state.apply_h(0U);
    }
    require(state.branch_count() == 1U,
            "512 repeated H gates did not reconverge to one exact branch");
    require(state.stats().hadamard_defects == 512U,
            "repeated-H defect count mismatch");
    require(state.stats().max_live_branches <= 2U,
            "repeated-H exact reconvergence exceeded two live branches");
    require(state.stats().extracted_power_of_two_bits == 256U,
            "repeated-H common coefficient extraction mismatch");
    require(state.stats().scale_half_power == -4,
            "repeated-H common scale did not return to the original |+> normalization");

    const std::vector<QComplex> materialized = state.materialize(8U);
    const std::vector<QComplex> expected = dense_plus(4U);
    for (std::size_t basis = 0U; basis < expected.size(); ++basis) {
        require(close(materialized[basis], expected[basis]),
                "512 repeated H gates did not return the exact initial state");
    }
}

void irreducible_growth_fails_closed() {
    ExactCompressedPhaseConfig config;
    config.max_live_branches = 8U;
    config.max_intermediate_branches = 16U;
    ExactCompressedPhaseGraphSum state(8U, config);
    state.apply_h(0U);
    state.apply_h(1U);
    state.apply_h(2U);
    require(state.branch_count() == 8U,
            "distinct-H branch growth did not reach eight exact branches");

    bool rejected = false;
    try {
        state.apply_h(3U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "exact compressed phase ignored irreducible branch growth cap");
    require(state.branch_count() == 8U && state.stats().hadamard_defects == 3U,
            "irreducible branch-cap rejection mutated accepted state");
}

void rejection_cases() {
    bool rejected = false;
    try {
        ExactCompressedPhaseGraphSum state(2U);
        state.apply(Operation{OperationCode::Rz, 0U, 0U, 0.25 * kPi});
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected,
            "exact compressed phase accepted floating Rz outside the symbolic pi/4 contract");

    rejected = false;
    try {
        ExactCompressedPhaseGraphSum state(2U);
        state.apply_x(2U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "exact compressed phase accepted an out-of-range qubit");

    rejected = false;
    try {
        ExactCompressedPhaseGraphSum state(2U);
        state.apply_controlled_phase_ticks(0U, 0U, 1U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "exact compressed phase accepted a self controlled phase");

    rejected = false;
    try {
        ExactCompressedPhaseConfig config;
        config.max_live_branches = 2U;
        config.max_intermediate_branches = 1U;
        ExactCompressedPhaseGraphSum invalid(2U, config);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "exact compressed phase accepted invalid branch cap ordering");
}

}  // namespace

int main() {
    dense_equivalence();
    exact_zero_and_cyclotomic_query();
    repeated_h_reconverges_exactly();
    irreducible_growth_fails_closed();
    rejection_cases();
    std::cout << "exact compressed phase tests passed\n";
    return 0;
}
