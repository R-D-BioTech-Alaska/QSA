#include "qubit/qsymmetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <typename Function>
double time_ms(Function&& function) {
    const auto start = Clock::now();
    function();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<qubit::QComplex> class_rotation(std::size_t classes, double angle) {
    std::vector<qubit::QComplex> matrix(classes * classes);
    for (std::size_t index = 0; index < classes; ++index) {
        matrix[index * classes + index] = {1.0, 0.0};
    }
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    matrix[0] = {c, 0.0};
    matrix[1] = {-s, 0.0};
    matrix[classes] = {s, 0.0};
    matrix[classes + 1U] = {c, 0.0};
    return matrix;
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 20;
    constexpr std::size_t classes = 8;
    constexpr std::uint64_t iterations = 1000;
    constexpr std::size_t fast_forward_repetitions = 10000;
    const std::size_t dimension = std::size_t{1} << qubits;
    const qubit::BasisIndex per_class = dimension / classes;
    const std::vector<qubit::BasisIndex> counts(classes, per_class);
    const auto matrix = class_rotation(classes, 0.0037);

    qubit::SymmetryState repeated(qubits, counts);
    const double symmetry_repeated_ms = time_ms([&] {
        for (std::uint64_t step = 0; step < iterations; ++step) {
            repeated.apply_class_unitary(matrix);
        }
    });

    qubit::SymmetryState fast_forward(qubits, counts);
    const double fast_forward_total_ms = time_ms([&] {
        for (std::size_t repetition = 0; repetition < fast_forward_repetitions; ++repetition) {
            fast_forward.iterate_class_unitary(matrix, iterations);
        }
    });
    const double symmetry_fast_forward_ms =
        fast_forward_total_ms / static_cast<double>(fast_forward_repetitions);

    std::vector<qubit::QComplex> dense(
        dimension, {1.0 / std::sqrt(static_cast<double>(dimension)), 0.0});
    const double dense_ms = time_ms([&] {
        for (std::uint64_t step = 0; step < iterations; ++step) {
            qubit::QComplex c0{};
            qubit::QComplex c1{};
            for (std::size_t index = 0; index < per_class; ++index) {
                c0 += dense[index];
                c1 += dense[per_class + index];
            }
            c0 /= std::sqrt(static_cast<double>(per_class));
            c1 /= std::sqrt(static_cast<double>(per_class));
            const qubit::QComplex next0 = matrix[0] * c0 + matrix[1] * c1;
            const qubit::QComplex next1 = matrix[classes] * c0 + matrix[classes + 1U] * c1;
            const qubit::QComplex a0 = next0 / std::sqrt(static_cast<double>(per_class));
            const qubit::QComplex a1 = next1 / std::sqrt(static_cast<double>(per_class));
            std::fill(dense.begin(), dense.begin() + static_cast<std::ptrdiff_t>(per_class), a0);
            std::fill(
                dense.begin() + static_cast<std::ptrdiff_t>(per_class),
                dense.begin() + static_cast<std::ptrdiff_t>(2U * per_class),
                a1);
        }
    });

    qubit::QRegister uniform(qubits);
    for (qubit::QubitId qubit = 0; qubit < qubits; ++qubit) {
        uniform.apply_h(qubit);
    }
    std::size_t discovered_classes = 0U;
    std::size_t discovered_bytes = 0U;
    double discovery_error = 0.0;
    const double discovery_ms = time_ms([&] {
        qubit::SymmetryState discovered =
            qubit::SymmetryState::discover(uniform, qubits, 1e-12, 64U);
        discovered_classes = discovered.class_count();
        discovered_bytes = discovered.estimated_bytes();
        discovery_error = discovered.discovery_error();
    });

    qubit::SymmetryState hamming = qubit::SymmetryState::hamming_weight(60);
    constexpr std::size_t hamming_steps = 100000;
    const double hamming_ms = time_ms([&] {
        for (std::size_t step = 0; step < hamming_steps; ++step) {
            hamming.apply_class_phase(step % hamming.class_count(), 1e-5);
            hamming.apply_weighted_reflection();
        }
    });

    const std::size_t dense_bytes = dense.capacity() * sizeof(qubit::QComplex);
    std::cout << std::fixed << std::setprecision(9)
              << "workload,qubits,classes,iterations,time_ms,engine_bytes,extra\n"
              << "symmetry_repeated_unitary," << qubits << ',' << classes << ',' << iterations
              << ',' << symmetry_repeated_ms << ',' << repeated.estimated_bytes() << ",\n"
              << "symmetry_fast_forward," << qubits << ',' << classes << ',' << iterations
              << ',' << symmetry_fast_forward_ms << ',' << fast_forward.estimated_bytes() << ",\n"
              << "dense_class_reference," << qubits << ',' << classes << ',' << iterations
              << ',' << dense_ms << ',' << dense_bytes << ",\n"
              << "automatic_discovery," << qubits << ',' << discovered_classes << ",1,"
              << discovery_ms << ',' << discovered_bytes << ",error=" << discovery_error << '\n'
              << "hamming_weight_60q,60," << hamming.class_count() << ',' << hamming_steps << ','
              << hamming_ms << ',' << hamming.estimated_bytes() << ",logical_states="
              << hamming.space_size() << '\n'
              << "repeated_speedup," << (dense_ms / symmetry_repeated_ms) << "\n"
              << "fast_forward_speedup," << (dense_ms / symmetry_fast_forward_ms) << "\n"
              << "memory_reduction," << (static_cast<double>(dense_bytes) /
                                           static_cast<double>(fast_forward.estimated_bytes()))
              << "\n";
    return 0;
}
