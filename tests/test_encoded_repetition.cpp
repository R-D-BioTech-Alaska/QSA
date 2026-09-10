#include "qubit/qencoded.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

bool close(double left, double right, double tolerance = 2e-11) {
    return std::abs(left - right) <= tolerance;
}

bool close(QComplex left, QComplex right, double tolerance = 2e-11) {
    return almost_equal(left, right, tolerance);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw QStateError(message);
    }
}

std::size_t codeword_mask(std::size_t logical_qubits, std::size_t block_size, std::size_t basis) {
    std::size_t mask = 0U;
    const std::size_t block_ones = (std::size_t{1U} << block_size) - 1U;
    for (std::size_t logical = 0U; logical < logical_qubits; ++logical) {
        if (((basis >> logical) & 1U) != 0U) {
            mask |= block_ones << (logical * block_size);
        }
    }
    return mask;
}

std::vector<QComplex> dense_plus(std::size_t logical_qubits, std::size_t block_size) {
    const std::size_t physical = logical_qubits * block_size;
    std::vector<QComplex> state(std::size_t{1U} << physical, QComplex{});
    const std::size_t logical_dimension = std::size_t{1U} << logical_qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(logical_dimension));
    for (std::size_t basis = 0U; basis < logical_dimension; ++basis) {
        state[codeword_mask(logical_qubits, block_size, basis)] = QComplex{scale};
    }
    return state;
}

void dense_logical_ry(
    std::vector<QComplex>& state,
    std::size_t logical_qubits,
    std::size_t block_size,
    std::size_t logical,
    double angle) {
    const std::size_t bit = std::size_t{1U} << logical;
    const double cosine = std::cos(0.5 * angle);
    const double sine = std::sin(0.5 * angle);
    const std::size_t logical_dimension = std::size_t{1U} << logical_qubits;
    for (std::size_t basis = 0U; basis < logical_dimension; ++basis) {
        if ((basis & bit) != 0U) {
            continue;
        }
        const std::size_t partner = basis | bit;
        const std::size_t zero_index = codeword_mask(logical_qubits, block_size, basis);
        const std::size_t one_index = codeword_mask(logical_qubits, block_size, partner);
        const QComplex zero = state[zero_index];
        const QComplex one = state[one_index];
        state[zero_index] = zero * cosine - one * sine;
        state[one_index] = zero * sine + one * cosine;
    }
}

void dense_physical_rz(std::vector<QComplex>& state, std::size_t physical, double angle) {
    const std::size_t bit = std::size_t{1U} << physical;
    const QComplex zero_phase = QComplex::from_polar(1.0, -0.5 * angle);
    const QComplex one_phase = QComplex::from_polar(1.0, 0.5 * angle);
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        state[basis] *= (basis & bit) == 0U ? zero_phase : one_phase;
    }
}

void dense_physical_z(std::vector<QComplex>& state, std::size_t physical) {
    const std::size_t bit = std::size_t{1U} << physical;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & bit) != 0U) {
            state[basis] = -state[basis];
        }
    }
}

void dense_transversal_x(
    std::vector<QComplex>& state, std::size_t block_size, std::size_t logical) {
    const std::size_t block_mask =
        ((std::size_t{1U} << block_size) - 1U) << (logical * block_size);
    std::vector<QComplex> output(state.size(), QComplex{});
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        output[basis ^ block_mask] = state[basis];
    }
    state.swap(output);
}

void dense_transversal_cnot(
    std::vector<QComplex>& state,
    std::size_t block_size,
    std::size_t control,
    std::size_t target) {
    std::vector<QComplex> output(state.size(), QComplex{});
    const std::size_t control_start = control * block_size;
    const std::size_t target_start = target * block_size;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        std::size_t destination = basis;
        for (std::size_t offset = 0U; offset < block_size; ++offset) {
            const std::size_t control_bit = std::size_t{1U} << (control_start + offset);
            const std::size_t target_bit = std::size_t{1U} << (target_start + offset);
            if ((basis & control_bit) != 0U) {
                destination ^= target_bit;
            }
        }
        output[destination] += state[basis];
    }
    state.swap(output);
}

double dense_z_expectation(const std::vector<QComplex>& state, std::size_t physical) {
    const std::size_t bit = std::size_t{1U} << physical;
    double total = 0.0;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        total += ((basis & bit) == 0U ? 1.0 : -1.0) * state[basis].norm2();
    }
    return total;
}

void dense_equivalence() {
    constexpr std::size_t logical_qubits = 3U;
    for (std::size_t block_size = 1U; block_size <= 6U; ++block_size) {
        EncodedRepetitionState encoded = EncodedRepetitionState::plus(
            logical_qubits, block_size);
        std::vector<QComplex> dense = dense_plus(logical_qubits, block_size);

        encoded.logical_ry(0U, 0.37);
        dense_logical_ry(dense, logical_qubits, block_size, 0U, 0.37);

        encoded.physical_rz(block_size + block_size / 2U, -0.22);
        dense_physical_rz(dense, block_size + block_size / 2U, -0.22);

        encoded.logical_cnot(0U, 2U);
        dense_transversal_cnot(dense, block_size, 0U, 2U);

        encoded.logical_x(1U);
        dense_transversal_x(dense, block_size, 1U);

        encoded.physical_z(0U);
        dense_physical_z(dense, 0U);

        const std::size_t logical_dimension = std::size_t{1U} << logical_qubits;
        for (std::size_t basis = 0U; basis < logical_dimension; ++basis) {
            require(close(
                        encoded.amplitude(basis),
                        dense[codeword_mask(logical_qubits, block_size, basis)]),
                    "encoded repetition codeword amplitude mismatch");
        }
        for (std::size_t physical_basis = 0U; physical_basis < dense.size(); ++physical_basis) {
            bool codeword = false;
            for (std::size_t logical_basis = 0U; logical_basis < logical_dimension; ++logical_basis) {
                if (physical_basis == codeword_mask(logical_qubits, block_size, logical_basis)) {
                    codeword = true;
                    break;
                }
            }
            if (!codeword) {
                require(dense[physical_basis].norm2() <= 1e-24,
                        "dense encoded control leaked outside repetition code space");
            }
        }
        require(close(encoded.norm_squared(), 1.0),
                "encoded repetition norm drifted");
        require(close(
                    encoded.physical_z_expectation(2U * block_size),
                    dense_z_expectation(dense, 2U * block_size)),
                "encoded repetition Z expectation mismatch");
    }
}

void physical_marginals_and_leakage() {
    EncodedRepetitionState state = EncodedRepetitionState::plus(2U, 3U);
    const std::vector<QubitId> same_block{0U, 1U};
    const std::vector<std::uint8_t> same_zero{0U, 0U};
    const std::vector<std::uint8_t> contradiction{0U, 1U};
    require(close(state.physical_marginal_probability(same_block, same_zero), 0.5),
            "encoded repetition same-block marginal mismatch");
    require(close(state.physical_marginal_probability(same_block, contradiction), 0.0),
            "encoded repetition allowed contradictory same-block bits");

    const std::vector<QubitId> two_blocks{0U, 3U};
    const std::vector<std::uint8_t> pattern{1U, 0U};
    require(close(state.physical_marginal_probability(two_blocks, pattern), 0.25),
            "encoded repetition cross-block marginal mismatch");

    bool rejected = false;
    try {
        state.physical_x(0U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "encoded repetition accepted a single physical X leakage operation");

    EncodedRepetitionState unencoded = EncodedRepetitionState::zero(2U, 1U);
    unencoded.physical_x(0U);
    require(close(unencoded.logical_basis_probability(1U), 1.0),
            "block-size-one physical X did not act as logical X");
}

void large_encoded_carrier() {
    EncodedRepetitionState state = EncodedRepetitionState::plus(12U, 10000U);
    for (std::size_t logical = 0U; logical < 12U; ++logical) {
        state.logical_ry(logical, 0.01 * static_cast<double>(logical + 1U));
        state.logical_rz(logical, -0.007 * static_cast<double>(logical + 1U));
    }
    for (std::size_t logical = 0U; logical + 1U < 12U; ++logical) {
        state.logical_cnot(logical, logical + 1U);
    }
    require(state.physical_qubit_count() == 120000U,
            "large encoded repetition physical qubit count mismatch");
    require(state.logical_dimension() == 4096U,
            "large encoded repetition logical dimension mismatch");
    require(close(state.norm_squared(), 1.0, 2e-10),
            "large encoded repetition norm drifted");
    require(std::isfinite(state.logical_x_expectation(0U)) &&
                std::isfinite(state.logical_z_expectation(11U)),
            "large encoded repetition observables became non-finite");
}

void rejection_cases() {
    bool rejected = false;
    try {
        (void)EncodedRepetitionState::zero(25U, 1U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "encoded repetition ignored its logical-qubit cap");

    rejected = false;
    try {
        EncodedRepetitionConfig config;
        config.max_physical_qubits = 5U;
        (void)EncodedRepetitionState::zero(2U, 3U, config);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "encoded repetition ignored its physical-qubit cap");

    rejected = false;
    try {
        EncodedRepetitionState state = EncodedRepetitionState::zero(2U, 2U);
        state.logical_ry(0U, std::numeric_limits<double>::infinity());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "encoded repetition accepted a non-finite rotation");

    rejected = false;
    try {
        EncodedRepetitionState state = EncodedRepetitionState::zero(2U, 2U);
        state.logical_cnot(0U, 0U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "encoded repetition accepted a self logical CNOT");

    rejected = false;
    try {
        EncodedRepetitionState state = EncodedRepetitionState::zero(2U, 2U);
        const std::vector<QubitId> duplicate{0U, 0U};
        const std::vector<std::uint8_t> bits{0U, 0U};
        (void)state.physical_marginal_probability(duplicate, bits);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "encoded repetition accepted a duplicate physical marginal qubit");
}

}  // namespace

int main() {
    dense_equivalence();
    physical_marginals_and_leakage();
    large_encoded_carrier();
    rejection_cases();
    std::cout << "encoded repetition tests passed\n";
    return 0;
}
