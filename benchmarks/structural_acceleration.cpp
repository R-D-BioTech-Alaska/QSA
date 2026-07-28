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
using qubit::QRegister;
using qubit::QStateConfig;
using qubit::QubitId;

[[nodiscard]] QRegister make_dense_groups(std::size_t width) {
    QStateConfig config;
    config.max_component_qubits = 60;
    config.max_dense_amplitudes = 1ULL << 26;
    config.max_sparse_entries = 8'000'000;
    QRegister state(width * 2U, config);
    for (std::size_t qubit = 0; qubit < width * 2U; ++qubit) {
        state.apply_ry(
            static_cast<QubitId>(qubit),
            0.27 + 0.013 * static_cast<double>(qubit));
        state.apply_rz(
            static_cast<QubitId>(qubit),
            -0.19 + 0.009 * static_cast<double>(qubit));
    }
    for (std::size_t qubit = 0; qubit + 1U < width; ++qubit) {
        state.apply_cz(
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U));
        state.apply_cz(
            static_cast<QubitId>(width + qubit),
            static_cast<QubitId>(width + qubit + 1U));
    }
    return state;
}

[[nodiscard]] QRegister make_dense_state(std::size_t qubits) {
    QStateConfig config;
    config.max_component_qubits = 30;
    config.max_dense_amplitudes = 1ULL << 22;
    const std::size_t dimension = std::size_t{1} << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dimension));
    std::vector<QComplex> amplitudes(dimension);
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        const double phase =
            0.000013 * static_cast<double>((basis * 2654435761ULL) & 0xFFFFFFU);
        amplitudes[basis] = QComplex::from_polar(scale, phase);
    }
    return QRegister::from_amplitudes(std::move(amplitudes), config);
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
    std::cout << std::setprecision(12);
    for (std::size_t width : {4U, 6U, 8U, 10U, 12U}) {
        const QRegister seed = make_dense_groups(width);
        const double ordinary = median_ms([&] {
            QRegister state = seed;
            state.apply_swap(0, static_cast<QubitId>(width));
        });
        const double structured = median_ms([&] {
            QRegister state = seed;
            state.apply_swap_structured(0, static_cast<QubitId>(width));
        });
        QRegister result = seed;
        result.apply_swap_structured(0, static_cast<QubitId>(width));
        std::cout << "swap width=" << width
                  << " ordinary_ms=" << ordinary
                  << " structured_ms=" << structured
                  << " speedup=" << ordinary / structured
                  << " bytes=" << result.estimated_bytes()
                  << " components=" << result.component_count() << '\n';
    }

    for (std::size_t qubits : {12U, 16U, 20U}) {
        const QRegister seed = make_dense_state(qubits);
        const double sequential = median_ms([&] {
            QRegister state = seed;
            (void)state.measure_all(0x123456789ABCDEF0ULL);
        });
        const double joint = median_ms([&] {
            QRegister state = seed;
            (void)state.measure_all_joint(0x123456789ABCDEF0ULL);
        });
        std::cout << "measure qubits=" << qubits
                  << " sequential_ms=" << sequential
                  << " joint_ms=" << joint
                  << " speedup=" << sequential / joint << '\n';
    }
    return 0;
}
