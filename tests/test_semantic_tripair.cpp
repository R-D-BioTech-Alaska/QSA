#include "qubit/qsemantic_tripair.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

using DenseState = std::array<qubit::QComplex, 8>;

void apply_single(DenseState& state, std::size_t qubit, const qubit::QMatrix2& matrix) {
    const std::size_t mask = std::size_t{1U} << qubit;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & mask) != 0U) {
            continue;
        }
        const std::size_t one = basis | mask;
        const qubit::QComplex zero_value = state[basis];
        const qubit::QComplex one_value = state[one];
        state[basis] = matrix(0U, 0U) * zero_value + matrix(0U, 1U) * one_value;
        state[one] = matrix(1U, 0U) * zero_value + matrix(1U, 1U) * one_value;
    }
}

void apply_cnot(DenseState& state, std::size_t control, std::size_t target) {
    const std::size_t control_mask = std::size_t{1U} << control;
    const std::size_t target_mask = std::size_t{1U} << target;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & control_mask) == 0U || (basis & target_mask) != 0U) {
            continue;
        }
        const std::size_t one = basis | target_mask;
        std::swap(state[basis], state[one]);
    }
}

DenseState reference(
    qubit::SemanticTripairMode mode,
    const qubit::SemanticTripairInput& input,
    std::span<const double> trainable) {
    DenseState state{};
    state[0U] = {1.0, 0.0};
    const bool ablation = mode == qubit::SemanticTripairMode::PhaseAblation;
    const bool entangled = mode != qubit::SemanticTripairMode::ClassicalSeparable;
    for (std::size_t qubit = 0U; qubit < 3U; ++qubit) {
        apply_single(state, qubit, qubit::gates::ry(input.theta[qubit]));
        apply_single(
            state,
            qubit,
            ablation ? qubit::gates::ry(input.phase[qubit]) : qubit::gates::rz(input.phase[qubit]));
    }
    if (entangled) {
        apply_cnot(state, 0U, 1U);
        apply_cnot(state, 1U, 2U);
        apply_cnot(state, 2U, 0U);
    }
    const std::size_t depth = trainable.size() / 6U;
    for (std::size_t layer = 0U; layer < depth; ++layer) {
        for (std::size_t qubit = 0U; qubit < 3U; ++qubit) {
            const std::size_t offset = layer * 6U + qubit * 2U;
            apply_single(state, qubit, qubit::gates::ry(trainable[offset]));
            apply_single(
                state,
                qubit,
                ablation ? qubit::gates::ry(trainable[offset + 1U])
                         : qubit::gates::rz(trainable[offset + 1U]));
        }
        if (entangled) {
            apply_cnot(state, 0U, 1U);
            apply_cnot(state, 1U, 2U);
        }
    }
    return state;
}

void exact_differential() {
    const std::array<double, 12> trainable{
        0.041, -0.072, 0.113, 0.031, -0.059, 0.087,
        -0.024, 0.068, -0.091, 0.127, 0.052, -0.036,
    };
    const std::array modes{
        qubit::SemanticTripairMode::FullPhase,
        qubit::SemanticTripairMode::PhaseAblation,
        qubit::SemanticTripairMode::ClassicalSeparable,
    };
    for (const auto mode : modes) {
        qubit::SemanticTripairProgram program(mode, 2U, trainable);
        require(program.stats().parameter_count == 18U, "Tripair parameter count mismatch");
        require(
            program.stats().operation_count ==
                (mode == qubit::SemanticTripairMode::ClassicalSeparable ? 18U : 25U),
            "Tripair operation count mismatch");
        std::vector<qubit::SemanticTripairInput> inputs;
        inputs.reserve(64U);
        for (std::size_t row = 0U; row < 64U; ++row) {
            qubit::SemanticTripairInput input;
            for (std::size_t qubit = 0U; qubit < 3U; ++qubit) {
                input.theta[qubit] = 0.2 + 0.9 * std::sin(0.071 * static_cast<double>((row + 1U) * (qubit + 2U)));
                input.phase[qubit] = 1.3 * std::sin(0.053 * static_cast<double>((row + 3U) * (qubit + 1U)));
            }
            inputs.push_back(input);
            const auto observed = program.evaluate(input);
            const auto expected = reference(mode, input, trainable);
            for (std::size_t basis = 0U; basis < expected.size(); ++basis) {
                require(
                    qubit::almost_equal(observed.amplitudes[basis], expected[basis], 2e-12),
                    "Tripair amplitude differs from dense reference");
            }
        }
        std::vector<qubit::SemanticTripairState> batch(inputs.size());
        program.evaluate_many(inputs, batch, 4U);
        for (std::size_t row = 0U; row < inputs.size(); ++row) {
            const auto scalar = program.evaluate(inputs[row]);
            for (std::size_t observable = 0U; observable < scalar.observables.size(); ++observable) {
                require(
                    std::abs(batch[row].observables[observable] - scalar.observables[observable]) <= 2e-12,
                    "Tripair batch observable differs from scalar execution");
            }
        }
        const auto same = qubit::SemanticTripairProgram::compare(batch.front(), batch.front());
        require(std::abs(same.coherent_fidelity - 1.0) <= 3e-12, "Tripair self fidelity mismatch");
        require(same.dephased_probability_overlap > 0.0, "Tripair dephased overlap is invalid");
    }
}

void probability_adapter_and_limits() {
    const std::array<double, 3> probabilities{0.0, 0.25, 1.0};
    const std::array<double, 3> phases{0.1, -0.2, 0.3};
    const auto input = qubit::semantic_tripair_from_probabilities(probabilities, phases);
    require(std::abs(input.theta[0U]) <= 1e-15, "Tripair p=0 angle mismatch");
    require(std::abs(input.theta[1U] - std::acos(0.5)) <= 1e-15, "Tripair p=0.25 angle mismatch");
    require(std::abs(input.theta[2U] - std::acos(-1.0)) <= 1e-15, "Tripair p=1 angle mismatch");

    bool rejected = false;
    try {
        const std::array<double, 3> bad{0.0, 1.1, 0.5};
        (void)qubit::semantic_tripair_from_probabilities(bad, phases);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "Tripair probability limit did not reject");

    rejected = false;
    try {
        const std::array<double, 1> bad_angles{std::numeric_limits<double>::quiet_NaN()};
        (void)qubit::SemanticTripairProgram(qubit::SemanticTripairMode::FullPhase, 0U, bad_angles);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "Tripair trainable shape/finite check did not reject");
}

}  // namespace

int main() {
    exact_differential();
    probability_adapter_and_limits();
    return 0;
}
