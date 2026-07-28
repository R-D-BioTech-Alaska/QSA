#include "qubit/qdiagonal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::CompiledDiagonalPlan;
using qubit::QComplex;
using qubit::QDiagonalPhase;
using qubit::QRegister;
using qubit::QubitId;

[[nodiscard]] QRegister dense_state(std::size_t qubits) {
    const std::size_t dimension = std::size_t{1} << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dimension));
    std::vector<QComplex> amplitudes(dimension);
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        amplitudes[basis] = QComplex::from_polar(
            scale,
            0.00007 * static_cast<double>((basis * 73U) % 997U));
    }
    return QRegister::from_amplitudes(std::move(amplitudes));
}

[[nodiscard]] std::vector<QDiagonalPhase> phases(std::size_t qubits) {
    std::vector<QDiagonalPhase> result;
    result.reserve(qubits);
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        const double angle = 0.003 * static_cast<double>(qubit + 1U);
        result.push_back(QDiagonalPhase{
            static_cast<QubitId>(qubit),
            QComplex::from_polar(1.0, -0.5 * angle),
            QComplex::from_polar(1.0, 0.5 * angle),
        });
    }
    return result;
}

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

}  // namespace

int main() {
    constexpr std::size_t qubits = 18U;
    constexpr int iterations = 80;
    const QRegister prototype = dense_state(qubits);
    const auto diagonal = phases(qubits);
    const CompiledDiagonalPlan plan(prototype, diagonal);

    const double ordinary = median_ms([&] {
        QRegister state = prototype;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            state.apply_diagonal(diagonal);
        }
        volatile double sink = state.probability_one(0);
        (void)sink;
    });
    const double compiled = median_ms([&] {
        QRegister state = prototype;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            plan.execute(state);
        }
        volatile double sink = state.probability_one(0);
        (void)sink;
    });

    std::cout << std::setprecision(12)
              << "compiled_diagonal qubits=" << qubits
              << " phases=" << diagonal.size()
              << " iterations=" << iterations
              << " ordinary_ms=" << ordinary
              << " compiled_ms=" << compiled
              << " speedup=" << ordinary / compiled
              << " coefficient_bytes=" << plan.coefficient_bytes() << '\n';
    return 0;
}
