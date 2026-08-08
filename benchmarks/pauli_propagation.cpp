#include "qubit/qpauli.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationPlan;
using qubit::PauliPropagationStats;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QubitId;

template <class Function>
[[nodiscard]] double median_ms(Function&& function, int repeats = 7) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        function();
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

[[nodiscard]] PauliObservable endpoint_z(std::size_t qubits) {
    PauliObservable observable(qubits);
    const std::vector<PauliFactor> factors{{static_cast<QubitId>(qubits - 1U), PauliAxis::Z}};
    observable.add_term({1.0, 0.0}, factors);
    return observable;
}

[[nodiscard]] std::vector<Operation> dense_small_circuit(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(qubits * 3U);
    for (QubitId offset = 0; offset < 5U; ++offset) {
        const QubitId qubit = static_cast<QubitId>(qubits - 5U) + offset;
        operations.push_back(
            Operation{OperationCode::Rz, qubit, 0U, 0.13 + 0.07 * static_cast<double>(offset), 0.0});
    }
    for (QubitId qubit = 0; qubit < qubits; ++qubit) {
        operations.push_back(Operation{OperationCode::H, qubit, 0U, 0.0, 0.0});
    }
    for (QubitId qubit = 0; qubit + 1U < qubits; ++qubit) {
        operations.push_back(Operation{OperationCode::Cz, qubit, qubit + 1U, 0.0, 0.0});
    }
    return operations;
}

[[nodiscard]] QComplex direct_expectation(
    std::size_t qubits,
    const std::vector<Operation>& operations,
    const PauliObservable& observable) {
    QRegister state(qubits);
    OperationPlan plan(operations, false);
    plan.execute(state);
    return observable.expectation(state);
}

[[nodiscard]] QComplex propagated_expectation(
    std::size_t qubits,
    const std::vector<Operation>& operations,
    const PauliObservable& observable) {
    QRegister state(qubits);
    return observable.propagated_backward(operations).expectation(state);
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    constexpr std::size_t small_qubits = 18U;
    const std::vector<Operation> small_operations = dense_small_circuit(small_qubits);
    const PauliObservable small_observable = endpoint_z(small_qubits);
    const QComplex direct = direct_expectation(small_qubits, small_operations, small_observable);
    const PauliObservable small_backward = small_observable.propagated_backward(small_operations);
    const QComplex propagated = propagated_expectation(
        small_qubits, small_operations, small_observable);
    const double error = (direct - propagated).magnitude();
    if (error > 2e-11) {
        throw std::runtime_error("Pauli benchmark exactness check failed");
    }

    const double direct_ms = median_ms([&] {
        volatile QComplex value = direct_expectation(
            small_qubits, small_operations, small_observable);
        (void)value;
    }, 5);
    const double pauli_ms = median_ms([&] {
        volatile QComplex value = propagated_expectation(
            small_qubits, small_operations, small_observable);
        (void)value;
    }, 9);

    constexpr std::size_t large_qubits = 100'000U;
    std::vector<Operation> large_operations;
    large_operations.reserve(large_qubits + 1U);
    large_operations.push_back(Operation{OperationCode::H, 0U, 0U, 0.0, 0.0});
    for (QubitId qubit = 0; qubit < large_qubits; ++qubit) {
        large_operations.push_back(Operation{
            OperationCode::Rz,
            qubit,
            0U,
            0.0001 * static_cast<double>((qubit % 17U) + 1U),
            0.0,
        });
    }

    PauliObservable large_observable(large_qubits);
    const std::vector<PauliFactor> large_factors{{0U, PauliAxis::X}};
    large_observable.add_term({1.0, 0.0}, large_factors);

    PauliObservable large_backward = large_observable;
    const double large_propagation_ms = median_ms([&] {
        large_backward = large_observable.propagated_backward(large_operations);
    }, 5);

    const double causal_plan_build_ms = median_ms([&] {
        const PauliPropagationPlan candidate(large_qubits, large_operations);
        volatile std::size_t bytes = candidate.estimated_bytes();
        (void)bytes;
    }, 5);
    const PauliPropagationPlan causal_plan(large_qubits, large_operations);
    PauliPropagationStats causal_stats;
    PauliObservable causal_backward = causal_plan.propagate_backward(
        large_observable, &causal_stats);
    const double causal_query_ms = median_ms([&] {
        PauliPropagationStats stats;
        const PauliObservable candidate = causal_plan.propagate_backward(
            large_observable, &stats);
        volatile std::size_t terms = candidate.term_count();
        (void)terms;
    }, 9);

    QRegister large_state(large_qubits);
    const double large_readout_ms = median_ms([&] {
        volatile QComplex value = large_backward.expectation(large_state);
        (void)value;
    }, 9);
    const QComplex large_value = large_backward.expectation(large_state);
    const QComplex causal_value = causal_backward.expectation(large_state);
    const double causal_error = (large_value - causal_value).magnitude();
    if (causal_error > 2e-12) {
        throw std::runtime_error("Pauli causal benchmark exactness check failed");
    }

    std::cout << "small qubits=" << small_qubits
              << " gates=" << small_operations.size()
              << " terms=" << small_backward.term_count()
              << " support=" << small_backward.support_size()
              << " error=" << error
              << " direct_ms=" << direct_ms
              << " pauli_ms=" << pauli_ms
              << " ratio=" << direct_ms / pauli_ms << '\n';
    std::cout << "large qubits=" << large_qubits
              << " gates=" << large_operations.size()
              << " terms=" << large_backward.term_count()
              << " support=" << large_backward.support_size()
              << " propagation_ms=" << large_propagation_ms
              << " readout_ms=" << large_readout_ms
              << " causal_plan_build_ms=" << causal_plan_build_ms
              << " causal_query_ms=" << causal_query_ms
              << " causal_visited=" << causal_stats.visited_operations
              << " causal_peak_terms=" << causal_stats.peak_terms
              << " causal_peak_support=" << causal_stats.peak_support
              << " causal_plan_bytes=" << causal_plan.estimated_bytes()
              << " causal_error=" << causal_error
              << " causal_query_ratio=" << large_propagation_ms / causal_query_ms
              << " value_re=" << large_value.re
              << " value_im=" << large_value.im << '\n';
    return 0;
}
