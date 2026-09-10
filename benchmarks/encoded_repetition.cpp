#include "qubit/qencoded.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

namespace {

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
    const std::size_t physical_qubits = logical_qubits * block_size;
    std::vector<QComplex> state(std::size_t{1U} << physical_qubits, QComplex{});
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
    const std::size_t logical_dimension = std::size_t{1U} << logical_qubits;
    const double cosine = std::cos(0.5 * angle);
    const double sine = std::sin(0.5 * angle);
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

struct DenseControlResult {
    double encoded_seconds{0.0};
    double dense_seconds{0.0};
    double error{0.0};
    double encoded_z{0.0};
    double dense_z{0.0};
};

DenseControlResult dense_control(std::size_t block_size) {
    constexpr std::size_t logical_qubits = 3U;

    const auto encoded_start = std::chrono::steady_clock::now();
    EncodedRepetitionState encoded = EncodedRepetitionState::plus(logical_qubits, block_size);
    encoded.logical_ry(0U, 0.37);
    encoded.physical_rz(block_size, 0.23);
    encoded.logical_cnot(0U, 2U);
    const double encoded_z = encoded.logical_z_expectation(2U);
    const auto encoded_end = std::chrono::steady_clock::now();

    const auto dense_start = std::chrono::steady_clock::now();
    std::vector<QComplex> dense = dense_plus(logical_qubits, block_size);
    dense_logical_ry(dense, logical_qubits, block_size, 0U, 0.37);
    dense_physical_rz(dense, block_size, 0.23);
    dense_transversal_cnot(dense, block_size, 0U, 2U);
    const double dense_z = dense_z_expectation(dense, 2U * block_size);
    const auto dense_end = std::chrono::steady_clock::now();

    return DenseControlResult{
        std::chrono::duration<double>(encoded_end - encoded_start).count(),
        std::chrono::duration<double>(dense_end - dense_start).count(),
        std::abs(encoded_z - dense_z),
        encoded_z,
        dense_z,
    };
}

}  // namespace

int main() {
    double maximum_error = 0.0;
    double block6_speedup = 0.0;
    for (std::size_t block_size = 1U; block_size <= 6U; ++block_size) {
        const DenseControlResult result = dense_control(block_size);
        maximum_error = std::max(maximum_error, result.error);
        const double speedup = result.dense_seconds / result.encoded_seconds;
        if (block_size == 6U) {
            block6_speedup = speedup;
        }
        std::cout << std::setprecision(17);
        std::cout << "dense_block_" << block_size << "_physical_qubits="
                  << 3U * block_size << '\n';
        std::cout << "dense_block_" << block_size << "_dimension_log2="
                  << 3U * block_size << '\n';
        std::cout << "dense_block_" << block_size << "_encoded_seconds="
                  << result.encoded_seconds << '\n';
        std::cout << "dense_block_" << block_size << "_physical_seconds="
                  << result.dense_seconds << '\n';
        std::cout << "dense_block_" << block_size << "_speedup=" << speedup << '\n';
        std::cout << "dense_block_" << block_size << "_absolute_error=" << result.error << '\n';
    }

    const auto large_start = std::chrono::steady_clock::now();
    EncodedRepetitionState large = EncodedRepetitionState::plus(12U, 10000U);
    large.logical_ry(0U, 0.37);
    large.logical_rz(5U, 0.23);
    large.logical_cnot(0U, 11U);
    large.physical_z(5U * 10000U + 123U);
    const double large_x = large.logical_x_expectation(0U);
    const double large_z = large.logical_z_expectation(11U);
    const auto large_end = std::chrono::steady_clock::now();

    bool leakage_rejected = false;
    try {
        large.physical_x(0U);
    } catch (const QStateError&) {
        leakage_rejected = true;
    }

    const double large_seconds =
        std::chrono::duration<double>(large_end - large_start).count();
    const std::size_t physical_qubits = large.physical_qubit_count();
    const std::size_t logical_qubits = large.logical_qubit_count();

    std::cout << "maximum_dense_error=" << maximum_error << '\n';
    std::cout << "block6_speedup=" << block6_speedup << '\n';
    std::cout << "large_logical_qubits=" << logical_qubits << '\n';
    std::cout << "large_block_size=" << large.block_size() << '\n';
    std::cout << "large_physical_qubits=" << physical_qubits << '\n';
    std::cout << "large_logical_amplitudes=" << large.logical_dimension() << '\n';
    std::cout << "large_physical_to_logical_dimension_ratio_log2="
              << physical_qubits - logical_qubits << '\n';
    std::cout << "large_estimated_bytes=" << large.stats().estimated_bytes << '\n';
    std::cout << "large_scalar_operations=" << large.stats().scalar_operations << '\n';
    std::cout << "large_norm=" << large.norm_squared() << '\n';
    std::cout << "large_x0=" << large_x << '\n';
    std::cout << "large_z11=" << large_z << '\n';
    std::cout << "large_seconds=" << large_seconds << '\n';
    std::cout << "large_physical_state_materialized=0\n";
    std::cout << "single_physical_x_leakage_rejected=" << (leakage_rejected ? 1 : 0) << '\n';
    return 0;
}
