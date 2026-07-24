#include "qubit/qgrover.hpp"
#include "qubit/qstate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
volatile double benchmark_sink = 0.0;

template <typename Function>
double median_microseconds(Function&& function, std::size_t repetitions) {
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = Clock::now();
        benchmark_sink = benchmark_sink + function();
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

void dense_iteration(
    std::vector<qubit::QComplex>& amplitudes,
    std::span<const qubit::BasisIndex> marked) {
    for (qubit::BasisIndex index : marked) {
        amplitudes[static_cast<std::size_t>(index)] =
            -amplitudes[static_cast<std::size_t>(index)];
    }
    long double sum_re = 0.0L;
    long double sum_im = 0.0L;
    for (const auto& value : amplitudes) {
        sum_re += value.re;
        sum_im += value.im;
    }
    const long double inverse = 1.0L / static_cast<long double>(amplitudes.size());
    const qubit::QComplex twice_mean{
        static_cast<double>(2.0L * sum_re * inverse),
        static_cast<double>(2.0L * sum_im * inverse),
    };
    for (auto& value : amplitudes) {
        value = twice_mean - value;
    }
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 16;
    const qubit::BasisIndex marked = 48'731;
    const std::span<const qubit::BasisIndex> marked_span(&marked, 1);
    const qubit::GroverSearch template_search(qubits, marked_span);
    const std::uint64_t iterations = template_search.optimal_iterations();
    const std::size_t dimension = std::size_t{1} << qubits;
    const double uniform = 1.0 / std::sqrt(static_cast<double>(dimension));

    const double dense_us = median_microseconds([&] {
        std::vector<qubit::QComplex> amplitudes(dimension, qubit::QComplex{uniform, 0.0});
        for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
            dense_iteration(amplitudes, marked_span);
        }
        return amplitudes[static_cast<std::size_t>(marked)].norm2();
    }, 7);

    const double exact_us = median_microseconds([&] {
        qubit::QRegister state(qubits);
        for (qubit::QubitId qubit = 0; qubit < qubits; ++qubit) {
            state.apply_h(qubit);
        }
        state.apply_grover_iterations(marked_span, iterations);
        return state.amplitude(marked).norm2();
    }, 7);

    constexpr std::size_t compressed_repetitions = 100'000;
    const auto compressed_start = Clock::now();
    for (std::size_t repetition = 0; repetition < compressed_repetitions; ++repetition) {
        const qubit::BasisIndex varying_marked =
            (marked + static_cast<qubit::BasisIndex>(repetition * 2'654'435'761ULL)) &
            static_cast<qubit::BasisIndex>(dimension - 1U);
        qubit::GroverSearch search(
            qubits, std::span<const qubit::BasisIndex>(&varying_marked, 1));
        search.iterate(iterations + static_cast<std::uint64_t>(repetition % 3U));
        benchmark_sink = benchmark_sink + search.success_probability() +
                         search.amplitude(varying_marked).norm2();
    }
    const auto compressed_end = Clock::now();
    const double compressed_us =
        std::chrono::duration<double, std::micro>(compressed_end - compressed_start).count() /
        static_cast<double>(compressed_repetitions);

    constexpr std::size_t huge_repetitions = 100'000;
    const auto huge_start = Clock::now();
    double huge_probability = 0.0;
    std::uint64_t huge_optimal = 0;
    std::size_t huge_bytes = 0;
    for (std::size_t repetition = 0; repetition < huge_repetitions; ++repetition) {
        qubit::GroverSearch huge = qubit::GroverSearch::from_marked_count(60, 1);
        huge_optimal = huge.optimal_iterations();
        huge.iterate(huge_optimal + static_cast<std::uint64_t>(repetition & 1U));
        huge_probability = huge.success_probability();
        huge_bytes = huge.estimated_bytes();
        benchmark_sink = benchmark_sink + huge_probability;
    }
    const auto huge_end = Clock::now();
    const double huge_us =
        std::chrono::duration<double, std::micro>(huge_end - huge_start).count() /
        static_cast<double>(huge_repetitions);

    const long double dense_60_bytes = std::ldexp(16.0L, 60);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "metric,value,unit\n";
    std::cout << "benchmark_qubits," << qubits << ",qubits\n";
    std::cout << "benchmark_iterations," << iterations << ",iterations\n";
    std::cout << "dense_exact_full," << dense_us << ",microseconds\n";
    std::cout << "qregister_exact_full," << exact_us << ",microseconds\n";
    std::cout << "compressed_exact_full," << compressed_us << ",microseconds\n";
    std::cout << "compressed_vs_dense_speedup," << dense_us / compressed_us << ",x\n";
    std::cout << "compressed_vs_qregister_speedup," << exact_us / compressed_us << ",x\n";
    std::cout << "qregister_vs_dense_speedup," << dense_us / exact_us << ",x\n";
    std::cout << "dense_16_bytes," << dimension * sizeof(qubit::QComplex) << ",bytes\n";
    std::cout << "compressed_16_bytes," << template_search.estimated_bytes() << ",bytes\n";
    std::cout << "compressed_16_memory_reduction,"
              << static_cast<double>(dimension * sizeof(qubit::QComplex)) /
                     static_cast<double>(template_search.estimated_bytes())
              << ",x\n";
    std::cout << "logical_60_states," << (qubit::BasisIndex{1} << 60) << ",states\n";
    std::cout << "logical_60_optimal_iterations," << huge_optimal << ",iterations\n";
    std::cout << "logical_60_fast_forward," << huge_us << ",microseconds\n";
    std::cout << "logical_60_success_probability," << huge_probability << ",probability\n";
    std::cout << "logical_60_compressed_bytes," << huge_bytes << ",bytes\n";
    std::cout << "logical_60_dense_bytes," << static_cast<double>(dense_60_bytes) << ",bytes\n";
    return 0;
}
