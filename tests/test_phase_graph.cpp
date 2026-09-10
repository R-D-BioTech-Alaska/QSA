#include "qubit/qphase_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PhaseGraphConfig;
using qubit::PhaseGraphPauliConfig;
using qubit::PhaseGraphState;
using qubit::QComplex;
using qubit::QMatrix4;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;
using qubit::almost_equal;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] double fidelity(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    require(first.size() == second.size(), "phase-graph state dimensions differ");
    QComplex overlap{};
    for (std::size_t index = 0; index < first.size(); ++index) {
        overlap += second[index].conjugate() * first[index];
    }
    return overlap.norm2();
}

void apply_controlled_phase(
    QRegister& state,
    QubitId first,
    QubitId second,
    double angle) {
    QMatrix4 matrix{};
    matrix.values[0] = {1.0, 0.0};
    matrix.values[5] = {1.0, 0.0};
    matrix.values[10] = {1.0, 0.0};
    matrix.values[15] = QComplex::from_polar(1.0, angle);
    state.apply_two(first, second, matrix);
}

void compare(
    const PhaseGraphState& graph,
    const QRegister& reference,
    double tolerance = 3e-10) {
    require(
        fidelity(graph.materialize(14), reference.materialize(14)) >= 1.0 - tolerance,
        "phase-graph state differs from QRegister");
}

void compare_pauli(
    const PhaseGraphState& graph,
    const QRegister& reference,
    const std::vector<PauliFactor>& factors,
    double tolerance = 5e-10) {
    PauliObservable observable(graph.qubit_count());
    observable.add_term({1.0, 0.0}, factors);
    const QComplex expected = observable.expectation(reference);
    const auto actual = graph.pauli_expectation(factors);
    require(
        almost_equal(actual.value, expected, tolerance),
        "factorized phase-graph Pauli expectation differs from dense control");
    require(actual.receipt.factorized && !actual.receipt.dense_materialization,
            "phase-graph Pauli expectation lost structural receipt");
}

void compare_equatorial(
    const PhaseGraphState& graph,
    const QRegister& reference,
    double tolerance = 5e-10) {
    const auto result = graph.equatorial_coherence();
    require(result.values.size() == graph.qubit_count(),
            "phase-graph coherence width is wrong");
    require(result.receipt.qubits == graph.qubit_count() &&
            result.receipt.phase_edges == graph.edge_count() &&
            result.receipt.factorized && !result.receipt.dense_materialization,
            "phase-graph coherence receipt is wrong");
    for (QubitId qubit = 0U; qubit < graph.qubit_count(); ++qubit) {
        const std::vector<PauliFactor> x{{qubit, PauliAxis::X}};
        const std::vector<PauliFactor> y{{qubit, PauliAxis::Y}};
        PauliObservable x_observable(graph.qubit_count());
        PauliObservable y_observable(graph.qubit_count());
        x_observable.add_term({1.0, 0.0}, x);
        y_observable.add_term({1.0, 0.0}, y);
        const QComplex expected_x = x_observable.expectation(reference);
        const QComplex expected_y = y_observable.expectation(reference);
        require(std::abs(result.values[qubit].re - expected_x.re) <= tolerance &&
                std::abs(result.values[qubit].im - expected_y.re) <= tolerance,
                "phase-graph equatorial coherence differs from dense X/Y control");
    }
}

}  // namespace

int main() {
    {
        PhaseGraphState graph(3);
        QRegister reference(3);
        for (QubitId qubit = 0; qubit < 3; ++qubit) {
            reference.apply_h(qubit);
        }
        graph.apply_rz(0, 0.37);
        reference.apply_rz(0, 0.37);
        graph.apply_cz(0, 1);
        reference.apply_cz(0, 1);
        graph.apply_controlled_phase(1, 2, -0.29);
        apply_controlled_phase(reference, 1, 2, -0.29);
        graph.apply_x(1);
        reference.apply_x(1);
        graph.apply_y(2);
        reference.apply_y(2);
        graph.apply_swap(0, 2);
        reference.apply_swap(0, 2);
        compare(graph, reference);
        compare_equatorial(graph, reference);
        const std::vector<PauliFactor> tripair{
            {0U, PauliAxis::X},
            {1U, PauliAxis::X},
            {2U, PauliAxis::X},
        };
        compare_pauli(graph, reference, tripair);
        const auto tripair_result = graph.pauli_expectation(tripair);
        require(tripair_result.receipt.flipped_qubits == 3U &&
                tripair_result.receipt.enumerated_assignments == 8U,
                "Tripair XXX correlation did not use bounded factorized readout");
        require(graph.validate(), "basic phase graph failed validation");
    }

    std::mt19937_64 generator(0x5048415345475241ULL);
    std::uniform_real_distribution<double> angle(-2.0, 2.0);
    for (std::size_t test_case = 0; test_case < 320U; ++test_case) {
        const std::size_t qubits = 2U + test_case % 8U;
        PhaseGraphState graph(qubits);
        QRegister reference(qubits);
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            reference.apply_h(static_cast<QubitId>(qubit));
        }
        const std::size_t gates = 30U + test_case % 100U;
        for (std::size_t gate = 0; gate < gates; ++gate) {
            const QubitId first = static_cast<QubitId>(generator() % qubits);
            QubitId second = static_cast<QubitId>(generator() % qubits);
            if (second == first) {
                second = static_cast<QubitId>((second + 1U) % qubits);
            }
            switch (generator() % 10U) {
                case 0U:
                    graph.apply_x(first);
                    reference.apply_x(first);
                    break;
                case 1U:
                    graph.apply_y(first);
                    reference.apply_y(first);
                    break;
                case 2U:
                    graph.apply_z(first);
                    reference.apply_z(first);
                    break;
                case 3U:
                    graph.apply_s(first);
                    reference.apply_s(first);
                    break;
                case 4U:
                    graph.apply_sdg(first);
                    reference.apply_sdg(first);
                    break;
                case 5U:
                    graph.apply_t(first);
                    reference.apply_t(first);
                    break;
                case 6U:
                    graph.apply_tdg(first);
                    reference.apply_tdg(first);
                    break;
                case 7U: {
                    const double value = angle(generator);
                    graph.apply_rz(first, value);
                    reference.apply_rz(first, value);
                    break;
                }
                case 8U: {
                    const double value = angle(generator);
                    graph.apply_controlled_phase(first, second, value);
                    apply_controlled_phase(reference, first, second, value);
                    break;
                }
                default:
                    graph.apply_swap(first, second);
                    reference.apply_swap(first, second);
                    break;
            }
            if ((gate % 17U) == 0U) {
                compare(graph, reference);
                require(graph.validate(), "random phase graph failed validation");
            }
        }
        compare(graph, reference);
        compare_equatorial(graph, reference);

        std::vector<QubitId> selected(qubits);
        for (std::size_t index = 0U; index < qubits; ++index) {
            selected[index] = static_cast<QubitId>(index);
        }
        std::shuffle(selected.begin(), selected.end(), generator);
        const std::size_t support = 1U + static_cast<std::size_t>(generator() % std::min<std::size_t>(4U, qubits));
        std::vector<PauliFactor> factors;
        factors.reserve(support);
        for (std::size_t index = 0U; index < support; ++index) {
            const auto axis = static_cast<PauliAxis>(1U + generator() % 3U);
            factors.push_back({selected[index], axis});
        }
        compare_pauli(graph, reference, factors);
    }

    {
        PhaseGraphConfig config;
        config.max_edges = 300'000U;
        PhaseGraphState large(100'000U, config);
        for (QubitId qubit = 0; qubit < 100'000U; ++qubit) {
            large.apply_rz(qubit, 0.0001 * static_cast<double>(qubit % 17U));
        }
        for (QubitId edge = 1; edge < 100'000U; ++edge) {
            large.apply_controlled_phase(edge - 1U, edge, 0.01);
        }
        require(large.validate(), "large phase graph failed validation");
        require(large.edge_count() == 99'999U, "large phase graph edge count is wrong");
        require(large.estimated_bytes() < 20ULL * 1024ULL * 1024ULL,
                "large phase graph exceeded memory gate");
        const auto coherence = large.equatorial_coherence();
        require(coherence.values.size() == 100'000U &&
                coherence.receipt.structural_factors == 299'998U &&
                !coherence.receipt.dense_materialization,
                "large phase graph coherence did not remain structural");
        const std::vector<PauliFactor> local_tripair{
            {49'999U, PauliAxis::X},
            {50'000U, PauliAxis::X},
            {50'001U, PauliAxis::X},
        };
        const auto correlation = large.pauli_expectation(local_tripair);
        require(correlation.receipt.qubits == 100'000U &&
                correlation.receipt.flipped_qubits == 3U &&
                correlation.receipt.enumerated_assignments == 8U &&
                correlation.receipt.internal_phase_edges == 2U &&
                correlation.receipt.boundary_phase_edges == 2U &&
                !correlation.receipt.dense_materialization &&
                std::isfinite(correlation.value.re) && std::isfinite(correlation.value.im),
                "large phase graph Tripair correlation lost factorized execution");
        const auto sample = large.sample_bits(0x123456789ABCDEF0ULL);
        require(sample.size() == 100'000U, "large phase graph sample has the wrong size");
    }

    {
        PhaseGraphState graph(4);
        bool duplicate_rejected = false;
        try {
            const std::vector<PauliFactor> duplicate{
                {1U, PauliAxis::X},
                {1U, PauliAxis::Z},
            };
            (void)graph.pauli_expectation(duplicate);
        } catch (const QStateError&) {
            duplicate_rejected = true;
        }
        require(duplicate_rejected, "phase graph accepted duplicate Pauli support");

        bool support_rejected = false;
        try {
            const std::vector<PauliFactor> tripair{
                {0U, PauliAxis::X},
                {1U, PauliAxis::X},
                {2U, PauliAxis::X},
            };
            PhaseGraphPauliConfig pauli_config;
            pauli_config.max_flip_qubits = 2U;
            (void)graph.pauli_expectation(tripair, pauli_config);
        } catch (const QStateError&) {
            support_rejected = true;
        }
        require(support_rejected, "phase graph ignored the Pauli flip-support cap");

        bool enumeration_rejected = false;
        try {
            const std::vector<PauliFactor> tripair{
                {0U, PauliAxis::X},
                {1U, PauliAxis::X},
                {2U, PauliAxis::X},
            };
            PhaseGraphPauliConfig pauli_config;
            pauli_config.max_flip_qubits = 3U;
            pauli_config.max_enumerated_assignments = 4U;
            (void)graph.pauli_expectation(tripair, pauli_config);
        } catch (const QStateError&) {
            enumeration_rejected = true;
        }
        require(enumeration_rejected, "phase graph ignored the Pauli enumeration cap");
    }

    {
        PhaseGraphConfig config;
        config.max_edges = 1U;
        PhaseGraphState graph(3, config);
        graph.apply_cz(0, 1);
        bool rejected = false;
        try {
            graph.apply_cz(1, 2);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "phase-graph edge limit was not enforced");
        require(graph.validate(), "edge-limit rejection damaged the phase graph");
    }

    {
        PhaseGraphState graph(2);
        bool rejected = false;
        try {
            graph.apply_rz(0, std::numeric_limits<double>::infinity());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "phase graph accepted an infinite angle");
        require(graph.validate(), "invalid-angle rejection damaged the phase graph");
    }

    std::cout << "phase graph tests passed\n";
    return 0;
}
