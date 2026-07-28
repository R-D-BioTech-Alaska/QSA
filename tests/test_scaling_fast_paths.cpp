#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::AmplitudeStore;
using qubit::BasisIndex;
using qubit::ComponentKind;
using qubit::QComplex;
using qubit::QDiagonalPhase;
using qubit::QMatrix2;
using qubit::QRegister;
using qubit::QStateConfig;
using qubit::QubitId;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] double fidelity(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    require(first.size() == second.size(), "state dimensions differ");
    QComplex overlap{};
    for (std::size_t index = 0; index < first.size(); ++index) {
        overlap += second[index].conjugate() * first[index];
    }
    return overlap.norm2();
}

[[nodiscard]] QMatrix2 diagonal_matrix(const QDiagonalPhase& phase) {
    QMatrix2 matrix{};
    matrix.values[0] = phase.zero;
    matrix.values[3] = phase.one;
    return matrix;
}

[[nodiscard]] QRegister random_sparse_state(
    std::mt19937_64& random,
    std::size_t qubits,
    std::size_t support) {
    const std::size_t dimension = std::size_t{1} << qubits;
    std::vector<QComplex> amplitudes(dimension);
    std::vector<std::size_t> indices(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        indices[index] = index;
    }
    std::shuffle(indices.begin(), indices.end(), random);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    for (std::size_t index = 0; index < support; ++index) {
        double re = value(random);
        double im = value(random);
        if (re == 0.0 && im == 0.0) {
            re = 1.0;
        }
        amplitudes[indices[index]] = {re, im};
    }
    QRegister result = QRegister::from_amplitudes(std::move(amplitudes));
    require(result.component_kind(0) == ComponentKind::Sparse,
            "random sparse state was not stored sparsely");
    return result;
}

void require_sorted_sparse(const QRegister& state) {
    const auto view = state.component_read_view(0);
    require(view.kind == ComponentKind::Sparse, "state unexpectedly left sparse storage");
    require(std::is_sorted(
                view.sparse.begin(),
                view.sparse.end(),
                [](const auto& left, const auto& right) {
                    return left.first < right.first;
                }),
            "sparse permutation left entries unsorted");
    require(std::adjacent_find(
                view.sparse.begin(),
                view.sparse.end(),
                [](const auto& left, const auto& right) {
                    return left.first == right.first;
                }) == view.sparse.end(),
            "sparse permutation produced duplicate indices");
}

void test_sparse_pauli_permutations() {
    std::mt19937_64 random(0x94B71D3A5210ULL);
    std::uniform_int_distribution<int> qubit_distribution(4, 16);
    for (int trial = 0; trial < 120; ++trial) {
        const std::size_t qubits = static_cast<std::size_t>(qubit_distribution(random));
        const std::size_t dimension = std::size_t{1} << qubits;
        const std::size_t support = std::max<std::size_t>(
            1U,
            std::min<std::size_t>(dimension / 8U, 4096U));
        QRegister specialized = random_sparse_state(random, qubits, support);
        QRegister generic = specialized;
        std::uniform_int_distribution<std::size_t> bit_distribution(0U, qubits - 1U);
        std::uniform_int_distribution<int> gate_distribution(0, 1);
        for (int operation = 0; operation < 80; ++operation) {
            const QubitId qubit = static_cast<QubitId>(bit_distribution(random));
            if (gate_distribution(random) == 0) {
                specialized.apply_x_structured(qubit);
                generic.apply_single(qubit, qubit::gates::x());
            } else {
                specialized.apply_y_structured(qubit);
                generic.apply_single(qubit, qubit::gates::y());
            }
            require_sorted_sparse(specialized);
            require(specialized.validate(), "specialized sparse state failed validation");
        }
        const auto specialized_state = specialized.materialize(20U);
        const auto generic_state = generic.materialize(20U);
        require(fidelity(specialized_state, generic_state) > 1.0 - 2e-11,
                "linear sparse Pauli permutation changed the quantum state");
    }
}

void test_sparse_pauli_maximum_bit() {
    QStateConfig config;
    config.max_component_qubits = 62U;
    config.max_sparse_entries = 128U;
    const BasisIndex dimension = BasisIndex{1} << 62U;
    const std::vector<AmplitudeStore::SparseEntry> source{
        {0U, {0.25, -0.5}},
        {1U, {-0.125, 0.75}},
        {(BasisIndex{1} << 60U) | 3U, {0.5, 0.125}},
        {(BasisIndex{1} << 61U) | 7U, {-0.375, -0.25}},
        {(BasisIndex{1} << 61U) | (BasisIndex{1} << 60U) | 11U,
         {0.875, -0.125}},
    };
    for (std::size_t bit : {0U, 60U, 61U}) {
        AmplitudeStore x = AmplitudeStore::from_entries(
            dimension, source, config, false);
        AmplitudeStore y = x;
        x.apply_x_structured(bit);
        y.apply_y_structured(bit);

        auto expected_x = source;
        auto expected_y = source;
        const BasisIndex mask = BasisIndex{1} << bit;
        for (auto& [basis, amplitude] : expected_x) {
            basis ^= mask;
        }
        for (auto& [basis, amplitude] : expected_y) {
            const bool original_one = (basis & mask) != 0U;
            amplitude = original_one
                ? -qubit::QI * amplitude
                : qubit::QI * amplitude;
            basis ^= mask;
        }
        const auto by_basis = [](const auto& left, const auto& right) {
            return left.first < right.first;
        };
        std::sort(expected_x.begin(), expected_x.end(), by_basis);
        std::sort(expected_y.begin(), expected_y.end(), by_basis);
        const auto actual_x = x.entries();
        const auto actual_y = y.entries();
        require(actual_x.size() == expected_x.size(),
                "maximum-bit X support size changed");
        require(actual_y.size() == expected_y.size(),
                "maximum-bit Y support size changed");
        for (std::size_t index = 0; index < source.size(); ++index) {
            require(actual_x[index].first == expected_x[index].first &&
                        qubit::almost_equal(
                            actual_x[index].second, expected_x[index].second, 2e-12),
                    "maximum-bit sparse X permutation differs");
            require(actual_y[index].first == expected_y[index].first &&
                        qubit::almost_equal(
                            actual_y[index].second, expected_y[index].second, 2e-12),
                    "maximum-bit sparse Y permutation differs");
        }
    }
}

void compare_component_views(
    const QRegister& first,
    const QRegister& second,
    QubitId qubit) {
    const auto first_view = first.component_read_view(qubit);
    const auto second_view = second.component_read_view(qubit);
    require(first_view.kind == second_view.kind,
            "component storage differs after diagonal update");
    require(first_view.qubits.size() == second_view.qubits.size(),
            "component membership size differs after diagonal update");
    require(std::equal(
                first_view.qubits.begin(), first_view.qubits.end(),
                second_view.qubits.begin()),
            "component membership differs after diagonal update");
    require(first_view.sparse.size() == second_view.sparse.size(),
            "component sparse support differs after diagonal update");
    for (std::size_t index = 0; index < first_view.sparse.size(); ++index) {
        require(first_view.sparse[index].first == second_view.sparse[index].first &&
                    qubit::almost_equal(
                        first_view.sparse[index].second,
                        second_view.sparse[index].second,
                        2e-12),
                "active-component diagonal result differs");
    }
}

void test_diagonal_active_component_grouping() {
    std::mt19937_64 random(0xD1A60A1F44ULL);
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    std::uniform_int_distribution<int> qubit_distribution(0, 11);
    for (int trial = 0; trial < 160; ++trial) {
        QRegister optimized(12);
        optimized.apply_h(0);
        optimized.apply_cnot_structured(0, 1);
        optimized.apply_h(3);
        optimized.apply_cz_structured(3, 4);
        optimized.apply_h(7);
        optimized.apply_cnot_structured(7, 8);
        QRegister literal = optimized;

        std::vector<QDiagonalPhase> phases;
        phases.reserve(40U);
        for (int index = 0; index < 40; ++index) {
            const QubitId qubit = static_cast<QubitId>(qubit_distribution(random));
            const double zero_angle = angle(random);
            const double one_angle = angle(random);
            phases.push_back(QDiagonalPhase{
                qubit,
                QComplex::from_polar(1.0, zero_angle),
                QComplex::from_polar(1.0, one_angle),
            });
        }

        optimized.apply_diagonal_structured(phases);
        for (const QDiagonalPhase& phase : phases) {
            literal.apply_single(phase.qubit, diagonal_matrix(phase));
        }
        require(optimized.validate(), "optimized diagonal state failed validation");
        require(literal.validate(), "literal diagonal state failed validation");
        require(optimized.component_count() == literal.component_count(),
                "diagonal grouping changed component structure");
        require(fidelity(optimized.materialize(16U), literal.materialize(16U)) >
                    1.0 - 3e-10,
                "active-component diagonal grouping changed the quantum state");
    }

    QRegister large(250'000U);
    const QDiagonalPhase phase{
        249'999U,
        QComplex::from_polar(1.0, -0.25),
        QComplex::from_polar(1.0, 0.25),
    };
    large.apply_diagonal_structured(std::span<const QDiagonalPhase>(&phase, 1U));
    require(large.component_count() == 250'000U,
            "single-cell diagonal update changed large-register structure");
    require(large.validate(), "large diagonal scaling state failed validation");

    constexpr std::size_t patch_qubits = 100'000U;
    QRegister active_patch(patch_qubits);
    const QubitId control = static_cast<QubitId>(patch_qubits - 2U);
    const QubitId target = static_cast<QubitId>(patch_qubits - 1U);
    active_patch.apply_h(control);
    active_patch.apply_cnot(control, target);
    QRegister literal_patch = active_patch;
    const QDiagonalPhase patch_phase{
        target,
        QComplex::from_polar(1.0, -0.37),
        QComplex::from_polar(1.0, 0.21),
    };
    active_patch.apply_diagonal_structured(
        std::span<const QDiagonalPhase>(&patch_phase, 1U));
    literal_patch.apply_single(target, diagonal_matrix(patch_phase));
    require(active_patch.component_count() == patch_qubits - 1U,
            "active-patch diagonal changed large-register structure");
    require(active_patch.validate() && literal_patch.validate(),
            "active-patch diagonal produced an invalid state");
    compare_component_views(active_patch, literal_patch, target);
}

}  // namespace

int main() {
    test_sparse_pauli_permutations();
    test_sparse_pauli_maximum_bit();
    test_diagonal_active_component_grouping();
    std::cout << "scaling fast-path differential tests passed\n";
    return 0;
}
