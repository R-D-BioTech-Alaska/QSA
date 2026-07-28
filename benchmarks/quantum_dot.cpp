#include "qubit/qdot.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using qubit::QRegister;
using qubit::qdot::DotInput;
using qubit::qdot::PocketConfig;
using qubit::qdot::QuantumDotPocket;
using qubit::qdot::Topology;

struct Result {
    std::string engine;
    std::size_t dots{};
    std::size_t steps{};
    double milliseconds{};
    std::size_t bytes{};
    std::size_t components{};
    std::size_t max_component{};
    double checksum{};
    bool valid{};
};

std::vector<std::vector<DotInput>> make_sequence(
    std::size_t dots,
    std::size_t steps) {
    std::vector<std::vector<DotInput>> sequence(steps, std::vector<DotInput>(dots));
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t dot = 0; dot < dots; ++dot) {
            const double theta =
                0.21 + 0.071 * static_cast<double>(step) +
                0.037 * static_cast<double>(dot) +
                0.09 * std::sin(
                           0.13 * static_cast<double>((step + 1U) * (dot + 1U)));
            const double phi =
                -0.38 + 0.053 * static_cast<double>(step) -
                0.041 * static_cast<double>(dot) +
                0.07 * std::cos(
                           0.17 * static_cast<double>((step + 2U) * (dot + 1U)));
            const double strength =
                0.82 + 0.14 * std::sin(
                                  0.11 * static_cast<double>(step + 2U * dot));
            sequence[step][dot] = DotInput{theta, phi, strength};
        }
    }
    return sequence;
}

template <class Function>
double median_ms(Function&& function, int repeats = 5) {
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

Result run_specialized(std::size_t dots, std::size_t steps, int repeats) {
    PocketConfig config;
    config.dot_count = dots;
    config.topology =
        (dots % 2U == 0U) ? Topology::PairBlocks : Topology::PairPlusContext;
    const auto sequence = make_sequence(dots, steps);
    Result result{"qdot_specialized_pair_context", dots, steps};
    result.milliseconds = median_ms(
        [&] {
            QuantumDotPocket pocket(config);
            for (const auto& inputs : sequence) {
                pocket.step(inputs);
            }
            const auto probabilities = pocket.probabilities_one();
            volatile double sink =
                std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
            (void)sink;
        },
        repeats);
    QuantumDotPocket pocket(config);
    for (const auto& inputs : sequence) {
        pocket.step(inputs);
    }
    const auto probabilities = pocket.probabilities_one();
    result.checksum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    result.bytes = pocket.estimated_bytes();
    result.components = pocket.component_count();
    result.max_component = pocket.max_component_qubits();
    result.valid = pocket.validate();
    return result;
}

Result run_generic(
    std::size_t dots,
    std::size_t steps,
    Topology topology,
    int repeats) {
    PocketConfig config;
    config.dot_count = dots;
    config.topology = topology;
    const auto sequence = make_sequence(dots, steps);
    const std::string name = topology == Topology::Chain
                                 ? "qsa_0_1_6_generic_chain"
                                 : (topology == Topology::PairBlocks
                                        ? "qsa_0_1_6_generic_pair_blocks"
                                        : "qsa_0_1_6_generic_pair_plus_context");
    Result result{name, dots, steps};
    result.milliseconds = median_ms(
        [&] {
            QRegister state(dots * 2U);
            for (const auto& inputs : sequence) {
                qubit::qdot::apply_reference_step(state, config, inputs);
            }
            const auto probabilities = state.probabilities_one();
            volatile double sink =
                std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
            (void)sink;
        },
        repeats);
    QRegister state(dots * 2U);
    for (const auto& inputs : sequence) {
        qubit::qdot::apply_reference_step(state, config, inputs);
    }
    const auto probabilities = state.probabilities_one();
    result.checksum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    result.bytes = state.estimated_bytes();
    result.components = state.component_count();
    result.max_component = 0U;
    for (std::size_t qubit = 0; qubit < dots * 2U; ++qubit) {
        result.max_component = std::max(
            result.max_component,
            state.component_size(static_cast<qubit::QubitId>(qubit)));
    }
    result.valid = state.validate();
    return result;
}

void print_result(const Result& result) {
    std::cout << "{\"engine\":\"" << result.engine << "\","
              << "\"dots\":" << result.dots << ','
              << "\"steps\":" << result.steps << ','
              << "\"milliseconds\":" << std::setprecision(10)
              << result.milliseconds << ','
              << "\"bytes\":" << result.bytes << ','
              << "\"components\":" << result.components << ','
              << "\"max_component_qubits\":" << result.max_component << ','
              << "\"checksum\":" << std::setprecision(17) << result.checksum << ','
              << "\"valid\":" << (result.valid ? "true" : "false") << "}\n";
}

int main(int argc, char** argv) {
    std::size_t steps = 12U;
    int repeats = 5;
    if (argc > 1) {
        steps = static_cast<std::size_t>(std::stoul(argv[1]));
    }
    if (argc > 2) {
        repeats = std::stoi(argv[2]);
    }

    for (std::size_t dots : {3U, 6U, 12U, 24U, 48U, 96U, 192U, 384U, 768U}) {
        print_result(run_specialized(dots, steps, repeats));
        print_result(run_generic(
            dots,
            steps,
            (dots % 2U == 0U) ? Topology::PairBlocks : Topology::PairPlusContext,
            repeats));
    }
    for (std::size_t dots : {3U, 4U, 5U, 6U}) {
        print_result(run_generic(dots, steps, Topology::Chain, repeats));
    }
    return 0;
}
