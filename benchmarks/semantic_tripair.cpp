#include "qubit/qsemantic_tripair.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t rows = 8192U;
    const std::array<double, 12> trainable{
        0.041, -0.072, 0.113, 0.031, -0.059, 0.087,
        -0.024, 0.068, -0.091, 0.127, 0.052, -0.036,
    };
    const qubit::SemanticTripairProgram program(
        qubit::SemanticTripairMode::FullPhase,
        2U,
        trainable);

    std::vector<qubit::SemanticTripairInput> inputs;
    inputs.reserve(rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        qubit::SemanticTripairInput input;
        for (std::size_t q = 0U; q < 3U; ++q) {
            const double probability =
                0.08 + 0.84 * (0.5 + 0.5 * std::sin(0.0071 * static_cast<double>((row + 1U) * (q + 2U))));
            input.theta[q] = 2.0 * std::asin(std::sqrt(probability));
            input.phase[q] = 1.3 * std::sin(0.0053 * static_cast<double>((row + 3U) * (q + 1U)));
        }
        inputs.push_back(input);
    }

    std::vector<qubit::SemanticTripairState> serial(rows);
    const auto serial_begin = Clock::now();
    for (std::size_t row = 0U; row < rows; ++row) {
        serial[row] = program.evaluate(inputs[row]);
    }
    const auto serial_end = Clock::now();

    std::vector<qubit::SemanticTripairState> batch(rows);
    const auto batch_begin = Clock::now();
    program.evaluate_many(inputs, batch, 0U);
    const auto batch_end = Clock::now();

    double max_observable_error = 0.0;
    double guard = 0.0;
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t observable = 0U; observable < batch[row].observables.size(); ++observable) {
            max_observable_error = std::max(
                max_observable_error,
                std::abs(batch[row].observables[observable] - serial[row].observables[observable]));
            guard += batch[row].observables[observable] * batch[row].observables[observable];
        }
    }

    const qubit::SemanticTripairProgram fingerprint(qubit::SemanticTripairMode::FullPhase, 0U);
    qubit::SemanticTripairInput semantic_base{
        {0.91, 1.07, 0.84},
        {0.31, -0.52, 0.77},
    };
    qubit::SemanticTripairInput phase_distinct = semantic_base;
    phase_distinct.phase = {-0.63, 0.19, -0.28};
    const auto base = fingerprint.evaluate(semantic_base);
    const auto correct = qubit::SemanticTripairProgram::compare(base, base);
    const auto wrong = qubit::SemanticTripairProgram::compare(base, fingerprint.evaluate(phase_distinct));

    const double serial_ms =
        std::chrono::duration<double, std::milli>(serial_end - serial_begin).count();
    const double batch_ms =
        std::chrono::duration<double, std::milli>(batch_end - batch_begin).count();

    std::cout << std::setprecision(17)
              << "tripair_rows=" << rows << '\n'
              << "tripair_depth=" << program.stats().depth << '\n'
              << "tripair_parameters=" << program.stats().parameter_count << '\n'
              << "tripair_operations=" << program.stats().operation_count << '\n'
              << "tripair_observables=" << program.stats().observables << '\n'
              << "tripair_state_amplitudes=" << program.stats().state_amplitudes << '\n'
              << "tripair_scalar_ms=" << serial_ms << '\n'
              << "tripair_native_batch_ms=" << batch_ms << '\n'
              << "tripair_batch_ratio=" << serial_ms / batch_ms << '\n'
              << "tripair_max_scalar_batch_error=" << max_observable_error << '\n'
              << "tripair_guard=" << guard << '\n'
              << "fingerprint_correct_coherent=" << correct.coherent_fidelity << '\n'
              << "fingerprint_phase_distinct_coherent=" << wrong.coherent_fidelity << '\n'
              << "fingerprint_correct_dephased=" << correct.dephased_probability_overlap << '\n'
              << "fingerprint_phase_distinct_dephased=" << wrong.dephased_probability_overlap << '\n'
              << "fingerprint_dephased_difference="
              << std::abs(correct.dephased_probability_overlap - wrong.dephased_probability_overlap) << '\n'
              << "dense_global_state_materialized=0\n";

    return program.stats().parameter_count == 18U &&
            program.stats().operation_count == 25U &&
            program.stats().observables == 14U &&
            max_observable_error <= 1e-12 &&
            guard > 0.0 && std::isfinite(guard) &&
            std::abs(correct.coherent_fidelity - 1.0) <= 1e-12 &&
            wrong.coherent_fidelity < correct.coherent_fidelity - 1e-6 &&
            std::abs(correct.dephased_probability_overlap - wrong.dephased_probability_overlap) <= 1e-12 &&
            serial_ms > 0.0 && batch_ms > 0.0
        ? 0
        : 1;
}
