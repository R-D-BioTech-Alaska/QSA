#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace qubit {

enum class SemanticTripairMode : std::uint8_t {
    FullPhase = 0,
    PhaseAblation = 1,
    ClassicalSeparable = 2,
};

struct SemanticTripairInput {
    std::array<double, 3> theta{};
    std::array<double, 3> phase{};
};

struct SemanticTripairState {
    std::array<QComplex, 8> amplitudes{};
    std::array<double, 14> observables{};
};

struct SemanticTripairComparison {
    double coherent_fidelity{0.0};
    double dephased_probability_overlap{0.0};
};

struct SemanticTripairStats {
    std::size_t depth{0U};
    std::size_t parameter_count{0U};
    std::size_t operation_count{0U};
    std::size_t state_amplitudes{8U};
    std::size_t observables{14U};
};

class SemanticTripairProgram {
public:
    explicit SemanticTripairProgram(
        SemanticTripairMode mode = SemanticTripairMode::FullPhase,
        std::size_t depth = 0U,
        std::span<const double> trainable_angles = {})
        : mode_(mode), depth_(depth), trainable_(trainable_angles.begin(), trainable_angles.end()) {
        if (depth_ > 1024U || trainable_.size() != depth_ * 6U) {
            throw QStateError("Semantic Tripair depth or trainable angle count is invalid");
        }
        for (const double value : trainable_) {
            if (!std::isfinite(value)) {
                throw QStateError("Semantic Tripair trainable angle must be finite");
            }
        }
        const bool entangled = mode_ != SemanticTripairMode::ClassicalSeparable;
        stats_ = SemanticTripairStats{
            depth_,
            6U + trainable_.size(),
            6U + (entangled ? 3U : 0U) + depth_ * (6U + (entangled ? 2U : 0U)),
            8U,
            14U,
        };
    }

    [[nodiscard]] SemanticTripairMode mode() const noexcept { return mode_; }
    [[nodiscard]] const SemanticTripairStats& stats() const noexcept { return stats_; }

    [[nodiscard]] SemanticTripairState evaluate(const SemanticTripairInput& input) const {
        validate_input(input);
        QRegister state(3U);
        for (std::size_t qubit = 0U; qubit < 3U; ++qubit) {
            state.apply_ry(static_cast<QubitId>(qubit), input.theta[qubit]);
            apply_phase(state, static_cast<QubitId>(qubit), input.phase[qubit]);
        }
        if (mode_ != SemanticTripairMode::ClassicalSeparable) {
            state.apply_cnot(0U, 1U);
            state.apply_cnot(1U, 2U);
            state.apply_cnot(2U, 0U);
        }
        for (std::size_t layer = 0U; layer < depth_; ++layer) {
            for (std::size_t qubit = 0U; qubit < 3U; ++qubit) {
                const std::size_t offset = layer * 6U + qubit * 2U;
                state.apply_ry(static_cast<QubitId>(qubit), trainable_[offset]);
                apply_phase(state, static_cast<QubitId>(qubit), trainable_[offset + 1U]);
            }
            if (mode_ != SemanticTripairMode::ClassicalSeparable) {
                state.apply_cnot(0U, 1U);
                state.apply_cnot(1U, 2U);
            }
        }
        const auto dense = state.materialize(3U);
        if (dense.size() != 8U) {
            throw QStateError("Semantic Tripair produced an invalid state dimension");
        }
        SemanticTripairState result;
        std::copy(dense.begin(), dense.end(), result.amplitudes.begin());
        for (std::size_t index = 0U; index < observable_words().size(); ++index) {
            result.observables[index] = expectation(result.amplitudes, observable_words()[index]);
        }
        return result;
    }

    void evaluate_many(
        std::span<const SemanticTripairInput> inputs,
        std::span<SemanticTripairState> output,
        std::size_t worker_count = 0U) const {
        if (output.size() != inputs.size()) {
            throw QStateError("Semantic Tripair batch output size does not match input size");
        }
        if (inputs.empty()) {
            return;
        }
        const std::size_t hardware = std::max<std::size_t>(1U, std::thread::hardware_concurrency());
        const std::size_t requested = worker_count == 0U ? hardware : worker_count;
        const std::size_t workers = std::min<std::size_t>(requested, inputs.size());
        if (workers <= 1U) {
            for (std::size_t index = 0U; index < inputs.size(); ++index) {
                output[index] = evaluate(inputs[index]);
            }
            return;
        }
        std::atomic<std::size_t> next{0U};
        std::atomic<bool> failed{false};
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t worker = 0U; worker < workers; ++worker) {
            threads.emplace_back([&, this]() {
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t index = next.fetch_add(1U, std::memory_order_relaxed);
                    if (index >= inputs.size()) {
                        return;
                    }
                    try {
                        output[index] = evaluate(inputs[index]);
                    } catch (...) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        if (failed.load(std::memory_order_relaxed)) {
            throw QStateError("Semantic Tripair batch evaluation failed");
        }
    }

    [[nodiscard]] static SemanticTripairComparison compare(
        const SemanticTripairState& left,
        const SemanticTripairState& right) {
        QComplex inner{};
        double dephased = 0.0;
        for (std::size_t index = 0U; index < left.amplitudes.size(); ++index) {
            const QComplex lhs = left.amplitudes[index];
            const QComplex rhs = right.amplitudes[index];
            if (!finite(lhs) || !finite(rhs)) {
                throw QStateError("Semantic Tripair comparison received a non-finite amplitude");
            }
            inner += lhs.conjugate() * rhs;
            dephased += lhs.norm2() * rhs.norm2();
        }
        return SemanticTripairComparison{inner.norm2(), dephased};
    }

    [[nodiscard]] static constexpr std::array<const char*, 14> observable_names() noexcept {
        return {
            "X0", "Y0", "Z0", "X1", "Y1", "Z1", "X2", "Y2", "Z2",
            "X0 X1", "Z0 Z1", "X1 X2", "Z1 Z2", "X0 X1 X2",
        };
    }

private:
    SemanticTripairMode mode_{SemanticTripairMode::FullPhase};
    std::size_t depth_{0U};
    std::vector<double> trainable_{};
    SemanticTripairStats stats_{};

    void apply_phase(QRegister& state, QubitId qubit, double angle) const {
        if (mode_ == SemanticTripairMode::PhaseAblation) {
            state.apply_ry(qubit, angle);
        } else {
            state.apply_rz(qubit, angle);
        }
    }

    static void validate_input(const SemanticTripairInput& input) {
        for (const double value : input.theta) {
            if (!std::isfinite(value)) {
                throw QStateError("Semantic Tripair theta must be finite");
            }
        }
        for (const double value : input.phase) {
            if (!std::isfinite(value)) {
                throw QStateError("Semantic Tripair phase must be finite");
            }
        }
    }

    [[nodiscard]] static constexpr std::array<std::array<char, 3>, 14> observable_words() noexcept {
        return {{
            {{'X', 'I', 'I'}}, {{'Y', 'I', 'I'}}, {{'Z', 'I', 'I'}},
            {{'I', 'X', 'I'}}, {{'I', 'Y', 'I'}}, {{'I', 'Z', 'I'}},
            {{'I', 'I', 'X'}}, {{'I', 'I', 'Y'}}, {{'I', 'I', 'Z'}},
            {{'X', 'X', 'I'}}, {{'Z', 'Z', 'I'}},
            {{'I', 'X', 'X'}}, {{'I', 'Z', 'Z'}}, {{'X', 'X', 'X'}},
        }};
    }

    [[nodiscard]] static double expectation(
        const std::array<QComplex, 8>& state,
        const std::array<char, 3>& word) {
        QComplex value{};
        for (std::size_t source = 0U; source < state.size(); ++source) {
            std::size_t target = source;
            QComplex phase{1.0, 0.0};
            for (std::size_t qubit = 0U; qubit < 3U; ++qubit) {
                const bool bit = ((source >> qubit) & 1U) != 0U;
                switch (word[qubit]) {
                    case 'I': break;
                    case 'X': target ^= std::size_t{1U} << qubit; break;
                    case 'Y':
                        target ^= std::size_t{1U} << qubit;
                        phase *= bit ? -QI : QI;
                        break;
                    case 'Z': phase *= bit ? -1.0 : 1.0; break;
                    default: throw QStateError("Semantic Tripair observable is invalid");
                }
            }
            value += state[target].conjugate() * phase * state[source];
        }
        if (!finite(value) || std::abs(value.im) > 1e-10) {
            throw QStateError("Semantic Tripair observable became non-real or non-finite");
        }
        return value.re;
    }

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }
};

[[nodiscard]] inline SemanticTripairInput semantic_tripair_from_probabilities(
    std::span<const double, 3> probabilities,
    std::span<const double, 3> phases) {
    SemanticTripairInput input;
    for (std::size_t index = 0U; index < 3U; ++index) {
        const double probability = probabilities[index];
        const double phase = phases[index];
        if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0 ||
            !std::isfinite(phase)) {
            throw QStateError("Semantic Tripair probability/phase input is invalid");
        }
        input.theta[index] = 2.0 * std::asin(std::sqrt(probability));
        input.phase[index] = phase;
    }
    return input;
}

}  // namespace qubit
