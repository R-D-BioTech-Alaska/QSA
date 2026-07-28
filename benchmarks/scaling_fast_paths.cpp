#include "qubit/qstate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::QComplex;
using qubit::QDiagonalPhase;
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

[[nodiscard]] QRegister sparse_patch(std::size_t support_bits, std::size_t total_bits) {
    QRegister state(total_bits);
    for (std::size_t qubit = 0; qubit < support_bits; ++qubit) {
        state.apply_h(static_cast<QubitId>(qubit));
    }
    for (std::size_t qubit = 1; qubit < support_bits; ++qubit) {
        state.apply_cz_structured(0U, static_cast<QubitId>(qubit));
    }
    for (std::size_t qubit = support_bits; qubit < total_bits; ++qubit) {
        state.apply_cnot_structured(0U, static_cast<QubitId>(qubit));
    }
    return state;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    for (std::size_t qubits : {10'000U, 100'000U, 1'000'000U}) {
        QRegister state(qubits);
        const QDiagonalPhase phase{
            static_cast<QubitId>(qubits - 1U),
            QComplex::from_polar(1.0, -0.123),
            QComplex::from_polar(1.0, 0.123),
        };
        const double diagonal = median_ms([&] {
            state.apply_diagonal_structured(
                std::span<const QDiagonalPhase>(&phase, 1U));
        }, 9);
        const double direct = median_ms([&] {
            state.apply_z(static_cast<QubitId>(qubits - 1U));
        }, 9);
        std::cout << "diagonal_scaling qubits=" << qubits
                  << " one_phase_ms=" << diagonal
                  << " direct_z_ms=" << direct
                  << " overhead_ratio=" << diagonal / direct << '\n';

        const QubitId control = static_cast<QubitId>(qubits - 2U);
        const QubitId target = static_cast<QubitId>(qubits - 1U);
        state.apply_h(control);
        state.apply_cnot(control, target);
        const QDiagonalPhase patch_phase{
            target,
            QComplex::from_polar(1.0, -0.271),
            QComplex::from_polar(1.0, 0.193),
        };
        const double active_patch = median_ms([&] {
            state.apply_diagonal_structured(
                std::span<const QDiagonalPhase>(&patch_phase, 1U));
        }, 9);
        std::cout << "diagonal_patch_scaling qubits=" << qubits
                  << " components=" << state.component_count()
                  << " one_phase_ms=" << active_patch << '\n';
    }

    QRegister state = sparse_patch(18U, 36U);
    const auto support = state.component_read_view(0U).sparse.size();
    for (std::size_t bit : {0U, 9U, 17U, 30U}) {
        const double x = median_ms([&] {
            state.apply_x_structured(static_cast<QubitId>(bit));
        });
        const double y = median_ms([&] {
            state.apply_y_structured(static_cast<QubitId>(bit));
        });
        std::cout << "sparse_pauli width=36 support=" << support
                  << " bit=" << bit
                  << " x_ms=" << x
                  << " y_ms=" << y << '\n';
    }
    return 0;
}
