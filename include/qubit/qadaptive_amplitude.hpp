#pragma once

#include "qubit/qamplitude_advisor.hpp"
#include "qubit/qhpath_factor.hpp"
#include "qubit/qphase_sum.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>

namespace qubit {

struct ExactAdaptiveAmplitudeConfig {
    ExactAmplitudeAdvisorConfig advisor{};
    PhaseGraphBranchSumConfig phase_graph{};
    ExactHadamardPathConfig hpath{};
};

class ExactAdaptiveAmplitudePlan {
public:
    ExactAdaptiveAmplitudePlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactAdaptiveAmplitudeConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        ExactAmplitudeAdvisorConfig advisor = config_.advisor;
        advisor.factor = config_.hpath.factor;
        advisor.max_qubits = std::min(advisor.max_qubits, config_.hpath.max_qubits);
        advisor.max_operations = std::min(advisor.max_operations, config_.hpath.max_operations);
        advisor.max_hpath_events = std::min(advisor.max_hpath_events, config_.hpath.max_h_events);
        advisor.max_phase_branches = std::min(
            advisor.max_phase_branches,
            config_.phase_graph.max_branches);
        decision_ = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
            qubit_count_, operations, advisor);

        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum: {
                phase_.emplace(qubit_count_, config_.phase_graph);
                for (const Operation& operation : operations) {
                    phase_->apply(operation);
                }
                break;
            }
            case ExactAmplitudeRoute::HadamardPathFactor:
                hpath_.emplace(qubit_count_, operations, config_.hpath);
                break;
            case ExactAmplitudeRoute::None:
                throw QStateError("Adaptive amplitude plan has no eligible exact route");
            case ExactAmplitudeRoute::Indeterminate:
                throw QStateError(
                    "Adaptive amplitude plan refuses to force an unresolved exact route");
            default:
                throw QStateError("Adaptive amplitude plan received an unknown route");
        }
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] ExactAmplitudeRoute route() const noexcept { return decision_.route; }
    [[nodiscard]] const ExactAmplitudeDecision& decision() const noexcept { return decision_; }
    [[nodiscard]] const ExactAdaptiveAmplitudeConfig& config() const noexcept { return config_; }

    [[nodiscard]] QComplex amplitude_bits(std::span<const std::uint8_t> bits) const {
        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                return phase_->amplitude_bits(bits);
            case ExactAmplitudeRoute::HadamardPathFactor:
                return hpath_->amplitude_bits(bits);
            default:
                throw QStateError("Adaptive amplitude plan has no executable route");
        }
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> bits) const {
        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                return phase_->log2_probability_bits(bits);
            case ExactAmplitudeRoute::HadamardPathFactor:
                return hpath_->scaled_amplitude_bits(bits).log2_probability();
            default:
                throw QStateError("Adaptive amplitude plan has no executable route");
        }
    }

private:
    std::size_t qubit_count_{0U};
    ExactAdaptiveAmplitudeConfig config_{};
    ExactAmplitudeDecision decision_{};
    std::optional<ExactPhaseGraphBranchSum> phase_{};
    std::optional<ExactHadamardPathAmplitudePlan> hpath_{};
};

}  // namespace qubit
