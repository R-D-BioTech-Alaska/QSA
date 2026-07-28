#include "qubit/qrouter.hpp"
#include "qubit/qsymmetry.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::QRegister;
using qubit::QubitId;
using qubit::RepresentationAdvisor;
using qubit::SymmetryState;

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

[[nodiscard]] QRegister uniform_register(std::size_t qubits) {
    QRegister state(qubits);
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        state.apply_h(static_cast<QubitId>(qubit));
    }
    return state;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    const QRegister dense_source = uniform_register(20U);
    const double dense_ms = median_ms([&] {
        volatile std::size_t classes = SymmetryState::discover(dense_source, 20U).class_count();
        (void)classes;
    });
    const double component_ms = median_ms([&] {
        volatile std::size_t classes =
            SymmetryState::discover_components(dense_source).class_count();
        (void)classes;
    });
    const SymmetryState large = SymmetryState::discover_components(uniform_register(50U));
    std::cout << "discover qubits=20 dense_ms=" << dense_ms
              << " component_ms=" << component_ms
              << " speedup=" << dense_ms / component_ms << '\n';
    std::cout << "discover qubits=50 classes=" << large.class_count()
              << " bytes=" << large.estimated_bytes() << '\n';

    RepresentationAdvisor advisor;
    const auto features = RepresentationAdvisor::inspect(dense_source, 100'000U, 1U, false);
    const auto recommendation = advisor.recommend(features);
    std::cout << "advisor backend=" << qubit::representation_name(recommendation.kind)
              << " estimated_work=" << recommendation.estimated_work
              << " posterior=" << recommendation.posterior_success
              << " score=" << recommendation.adjusted_score << '\n';
    return 0;
}
