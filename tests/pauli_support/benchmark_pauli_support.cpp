#include "pauli_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double milliseconds(
    Clock::time_point start,
    Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

[[nodiscard]] double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main() {
    constexpr std::size_t qubit_count = 10'000U;
    constexpr int repeats = 31;
    constexpr qubit::QubitId first = 17U;
    constexpr qubit::QubitId second = 4'321U;
    constexpr qubit::QubitId third = 9'999U;

    qubit::QRegister state = qsa_pauli_support_test::make_tripair(
        qubit_count,
        first,
        second,
        third);
    const auto supports =
        qsa_pauli_support_test::tripair_support(first, second, third);
    const auto words = qsa_pauli_support_test::tripair_words(
        qubit_count,
        first,
        second,
        third);
    const qubit::PauliSupportPlan compact(qubit_count, supports);
    const qubit::PauliObservablePlan full(qubit_count, words);

    const auto compact_reference = compact.execute(state);
    const auto full_reference = full.execute(state);
    const double maximum_error = qsa_pauli_support_test::maximum_error(
        compact_reference,
        full_reference);

    std::vector<double> full_samples;
    std::vector<double> compact_samples;
    full_samples.reserve(repeats);
    compact_samples.reserve(repeats);
    volatile double sink = 0.0;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        {
            const auto start = Clock::now();
            const auto values = full.execute(state);
            sink += values[static_cast<std::size_t>(repeat) % values.size()];
            const auto stop = Clock::now();
            full_samples.push_back(milliseconds(start, stop));
        }
        {
            const auto start = Clock::now();
            const auto values = compact.execute(state);
            sink += values[static_cast<std::size_t>(repeat) % values.size()];
            const auto stop = Clock::now();
            compact_samples.push_back(milliseconds(start, stop));
        }
    }

    const double full_ms = median(full_samples);
    const double compact_ms = median(compact_samples);
    std::size_t full_word_bytes = 0U;
    for (const std::string& word : words) {
        full_word_bytes += word.size();
    }
    if (sink < -1.0e9) {
        throw std::runtime_error("unreachable benchmark sink");
    }

    std::cout << std::setprecision(12)
              << "pauli_support"
              << " qubits=" << qubit_count
              << " components=" << state.component_count()
              << " observables=" << compact.observable_count()
              << " support_terms=" << compact.term_count()
              << " full_word_bytes=" << full_word_bytes
              << " full_ms=" << full_ms
              << " support_ms=" << compact_ms
              << " speedup=" << full_ms / compact_ms
              << " max_error=" << maximum_error
              << '\n';
    return maximum_error <= 2.0e-12 ? 0 : 1;
}
