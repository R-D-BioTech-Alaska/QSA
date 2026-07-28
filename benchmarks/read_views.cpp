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

[[nodiscard]] QRegister dense_state(std::size_t qubits) {
    const std::size_t dimension = std::size_t{1} << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dimension));
    std::vector<QComplex> amplitudes(dimension);
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        amplitudes[basis] = QComplex::from_polar(
            scale,
            0.000003 * static_cast<double>((basis * 2654435761ULL) & 0xFFFFFFU));
    }
    return QRegister::from_amplitudes(std::move(amplitudes));
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

    QRegister cells(100'000U);
    std::vector<double> buffer(cells.qubit_count());
    constexpr int probability_rounds = 100;
    const double allocating = median_ms([&] {
        volatile double checksum = 0.0;
        for (int round = 0; round < probability_rounds; ++round) {
            const auto values = cells.probabilities_one();
            checksum += values[static_cast<std::size_t>(round) % values.size()];
        }
        (void)checksum;
    });
    const double reused = median_ms([&] {
        volatile double checksum = 0.0;
        for (int round = 0; round < probability_rounds; ++round) {
            cells.probabilities_one_into(buffer);
            checksum += buffer[static_cast<std::size_t>(round) % buffer.size()];
        }
        (void)checksum;
    });
    std::cout << "probabilities qubits=100000 rounds=" << probability_rounds
              << " allocating_ms=" << allocating
              << " reused_ms=" << reused
              << " speedup=" << allocating / reused << '\n';

    QRegister dense = dense_state(20U);
    constexpr int view_rounds = 20;
    const double copying = median_ms([&] {
        volatile double checksum = 0.0;
        for (int round = 0; round < view_rounds; ++round) {
            const auto values = dense.materialize(20U);
            checksum += values[static_cast<std::size_t>(round)].norm2();
        }
        (void)checksum;
    });
    const double viewing = median_ms([&] {
        volatile double checksum = 0.0;
        for (int round = 0; round < view_rounds; ++round) {
            const auto view = dense.component_read_view(0);
            checksum += view.dense[static_cast<std::size_t>(round)].norm2();
        }
        (void)checksum;
    });
    std::cout << "dense_read amplitudes=" << (std::size_t{1} << 20U)
              << " rounds=" << view_rounds
              << " copying_ms=" << copying
              << " viewing_ms=" << viewing
              << " speedup=" << copying / viewing << '\n';
    return 0;
}
