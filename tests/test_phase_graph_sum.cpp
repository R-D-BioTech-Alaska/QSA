#include "qubit/qphase_sum.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

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
    const double scale = std::exp2(-0.5 * static_cast<double>(qubits));
    return std::vector<QComplex>(dimension, QComplex{scale});
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

void dense_controlled_phase(
    std::vector<QComplex>& state,
    std::size_t first,
    std::size_t second,
    double angle) {
    const std::size_t first_bit = std::size_t{1U} << first;
    const std::size_t second_bit = std::size_t{1U} << second;
    const QComplex phase = QComplex::from_polar(1.0, angle);
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
    for (std::size_t qubits = 2U; qubits <= 7U; ++qubits) {
        for (std::size_t h_count = 1U; h_count <= 5U; ++h_count) {
            ExactPhaseGraphBranchSum compact(qubits);
            std::vector<QComplex> dense = dense_plus(qubits);
            for (std::size_t defect = 0U; defect < h_count; ++defect) {
                const QubitId q = static_cast<QubitId>((defect * 3U + 1U) % qubits);
                const QubitId r = static_cast<QubitId>((static_cast<std::size_t>(q) + 1U) % qubits);
                compact.apply_t(q);
                dense_single(dense, q, gates::t());
                compact.apply_rz(r, 0.071 * static_cast<double>(defect + 1U));
                dense_single(dense, r, gates::rz(0.071 * static_cast<double>(defect + 1U)));
                compact.apply_controlled_phase(q, r, 0.13 * static_cast<double>(defect + 1U));
                dense_controlled_phase(dense, q, r, 0.13 * static_cast<double>(defect + 1U));
                compact.apply_h(q);
                dense_single(dense, q, gates::h());
                compact.apply_x(r);
                dense_single(dense, r, gates::x());
                compact.apply_s(q);
                dense_single(dense, q, gates::s());
                if (qubits > 2U && (defect & 1U) != 0U) {
                    const QubitId third = static_cast<QubitId>((static_cast<std::size_t>(r) + 1U) % qubits);
                    compact.apply_swap(r, third);
                    dense_swap(dense, r, third);
                }
            }

            const std::vector<QComplex> actual = compact.materialize(8U);
            require(actual.size() == dense.size(), "phase-graph branch-sum dense size mismatch");
            for (std::size_t basis = 0U; basis < dense.size(); ++basis) {
                require(close(actual[basis], dense[basis]),
                        "phase-graph branch-sum dense amplitude mismatch");
            }
            require(compact.branch_count() == (std::size_t{1U} << h_count),
                    "phase-graph branch-sum did not grow exactly as 2^H");
            require(compact.stats().hadamard_defects == h_count,
                    "phase-graph branch-sum Hadamard count mismatch");
        }
    }
}

void underflow_safe_large_query() {
    ExactPhaseGraphBranchSum state(4096U);
    std::vector<std::uint8_t> zeros(4096U, 0U);
    const ScaledPhaseGraphAmplitude scaled = state.scaled_amplitude_bits(zeros);
    require(close(scaled.mantissa, QComplex{1.0}),
            "phase-graph branch-sum large scaled mantissa mismatch");
    require(close(scaled.log2_scale, -2048.0),
            "phase-graph branch-sum large log2 amplitude scale mismatch");
    require(close(state.log2_probability_bits(zeros), -4096.0),
            "phase-graph branch-sum large log2 probability mismatch");

    bool rejected = false;
    try {
        (void)state.amplitude_bits(zeros);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected,
            "phase-graph branch-sum silently underflowed a large direct amplitude instead of requiring scaled query");

    PhaseGraphState graph(4096U);
    require(close(graph.unit_phase_bits(zeros), QComplex{1.0}),
            "phase-graph unit-phase query failed beyond direct-amplitude range");
}

void branch_and_resource_caps() {
    PhaseGraphBranchSumConfig config;
    config.max_branches = 8U;
    ExactPhaseGraphBranchSum state(8U, config);
    state.apply_h(0U);
    state.apply_h(1U);
    state.apply_h(2U);
    require(state.branch_count() == 8U, "phase-graph branch-sum branch cap setup mismatch");

    bool rejected = false;
    try {
        state.apply_h(3U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "phase-graph branch-sum ignored its branch cap");
    require(state.branch_count() == 8U && state.stats().hadamard_defects == 3U,
            "phase-graph branch-sum branch-cap failure mutated accepted state");

    rejected = false;
    try {
        Operation unsupported{OperationCode::Cnot, 0U, 1U};
        state.apply(unsupported);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "phase-graph branch-sum accepted CNOT outside its certified family");

    PhaseGraphBranchSumConfig tight;
    tight.max_retained_estimated_bytes = 4096U;
    tight.phase_graph.max_edges = 1000U;
    ExactPhaseGraphBranchSum bounded(16U, tight);
    bool byte_rejected = false;
    std::size_t accepted_edges = 0U;
    for (QubitId first = 0U; first < 16U && !byte_rejected; ++first) {
        for (QubitId second = static_cast<QubitId>(first + 1U);
             second < 16U;
             ++second) {
            try {
                bounded.apply_controlled_phase(
                    first, second, 0.01 * static_cast<double>(accepted_edges + 1U));
                accepted_edges = bounded.branches()[0].state.edge_count();
            } catch (const QStateError&) {
                byte_rejected = true;
                require(bounded.branches()[0].state.edge_count() == accepted_edges,
                        "phase-graph branch-sum byte-cap rejection mutated accepted state");
                break;
            }
        }
    }
    require(byte_rejected, "phase-graph branch-sum retained-byte cap was not exercised");
}

void operation_dispatch() {
    ExactPhaseGraphBranchSum state(3U);
    state.apply(Operation{OperationCode::T, 0U});
    state.apply(Operation{OperationCode::H, 1U});
    state.apply(Operation{OperationCode::Cz, 1U, 2U});
    state.apply(Operation{OperationCode::Rz, 2U, 0U, 0.31});
    require(state.branch_count() == 2U,
            "phase-graph branch-sum operation dispatch lost Hadamard branch count");
    const std::vector<QComplex> materialized = state.materialize(4U);
    double norm = 0.0;
    for (const QComplex amplitude : materialized) {
        norm += amplitude.norm2();
    }
    require(close(norm, 1.0), "phase-graph branch-sum operation dispatch lost normalization");
}

}  // namespace

int main() {
    dense_equivalence();
    underflow_safe_large_query();
    branch_and_resource_caps();
    operation_dispatch();
    std::cout << "phase-graph branch-sum tests passed\n";
    return 0;
}
