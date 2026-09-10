#pragma once

#include "qubit/qadaptive_amplitude.hpp"
#include "qubit/qhpath_basis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace qubit {

enum class ExactBasisAmplitudeDecisionCode : std::uint8_t {
    NoEligibleRoute = 0U,
    PhaseOnly = 1U,
    HPathOnly = 2U,
    HPathNoHiddenVariables = 3U,
    HPathLowerPeakEnvelope = 4U,
    PhaseLowerBranchEnvelope = 5U,
    CommonEnvelopeUnresolved = 6U,
};

struct ExactBasisAmplitudeDecision {
    ExactAmplitudeRoute route{ExactAmplitudeRoute::None};
    ExactBasisAmplitudeDecisionCode code{ExactBasisAmplitudeDecisionCode::NoEligibleRoute};
    bool phase_graph_eligible{false};
    bool hpath_eligible{false};
    bool hpath_resource_rejected{false};
    std::size_t target_h_events{0U};
    std::size_t basis_preparation_h_events{0U};
    std::size_t phase_h_defects{0U};
    bool phase_branch_envelope_fits_size_t{true};
    std::size_t phase_branch_envelope{1U};
    std::size_t hpath_peak_log2_ceiling{0U};
    std::size_t structural_log2_margin{0U};
    ExactPreparedBasisHadamardPathStats hpath{};
};

class ExactAdaptiveBasisAmplitudePlan;

class ExactAdaptiveBasisAmplitudeWorkspace {
public:
    ExactAdaptiveBasisAmplitudeWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return hpath_.has_value() ? hpath_->estimated_bytes() : sizeof(*this);
    }

    [[nodiscard]] std::size_t hpath_rebind_count() const noexcept {
        return hpath_.has_value() ? hpath_->rebind_count() : 0U;
    }

private:
    const ExactAdaptiveBasisAmplitudePlan* owner_{nullptr};
    std::optional<ExactPreparedBasisHadamardPathWorkspace> hpath_{};

    explicit ExactAdaptiveBasisAmplitudeWorkspace(
        const ExactAdaptiveBasisAmplitudePlan& owner);

    friend class ExactAdaptiveBasisAmplitudePlan;
};

class ExactAdaptiveBasisAmplitudePlan {
public:
    ExactAdaptiveBasisAmplitudePlan(
        std::span<const std::uint8_t> input_bits,
        std::span<const Operation> operations,
        ExactAdaptiveAmplitudeConfig config = {})
        : qubit_count_(input_bits.size()),
          input_bits_(input_bits.begin(), input_bits.end()),
          config_(config) {
        validate_input_bits();
        const ExactAmplitudeAdvisorConfig advisor = aligned_advisor(config_);
        scan_eligibility(operations, advisor);

        if (hpath_candidate_) {
            try {
                prepared_hpath_.emplace(input_bits_, operations, config_.hpath);
                decision_.hpath_eligible = true;
                decision_.hpath = prepared_hpath_->stats();
                decision_.hpath_peak_log2_ceiling = ceil_log2(
                    std::max<std::size_t>(1U, decision_.hpath.factor.peak_factor_entries));
            } catch (const QStateError&) {
                decision_.hpath_resource_rejected = true;
                prepared_hpath_.reset();
            }
        }

        finalize(advisor);
        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                prepared_hpath_.reset();
                prepare_phase(operations);
                break;
            case ExactAmplitudeRoute::HadamardPathFactor:
                if (!prepared_hpath_.has_value()) {
                    throw QStateError(
                        "Adaptive basis amplitude selected Hpath without a prepared basis plan");
                }
                break;
            case ExactAmplitudeRoute::None:
                throw QStateError("Adaptive basis amplitude has no eligible exact route");
            case ExactAmplitudeRoute::Indeterminate:
                throw QStateError(
                    "Adaptive basis amplitude refuses to force an unresolved exact route");
            default:
                throw QStateError("Adaptive basis amplitude received an unknown route");
        }
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::span<const std::uint8_t> input_bits() const noexcept {
        return input_bits_;
    }
    [[nodiscard]] ExactAmplitudeRoute route() const noexcept { return decision_.route; }
    [[nodiscard]] const ExactBasisAmplitudeDecision& decision() const noexcept {
        return decision_;
    }
    [[nodiscard]] const ExactAdaptiveAmplitudeConfig& config() const noexcept { return config_; }
    [[nodiscard]] bool has_prepared_hpath() const noexcept {
        return prepared_hpath_.has_value();
    }

    [[nodiscard]] ExactAdaptiveBasisAmplitudeWorkspace workspace() const {
        return ExactAdaptiveBasisAmplitudeWorkspace(*this);
    }

    [[nodiscard]] QComplex amplitude_bits(
        std::span<const std::uint8_t> bits,
        ExactAdaptiveBasisAmplitudeWorkspace& workspace) const {
        validate_workspace(workspace);
        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                return phase_->amplitude_bits(bits);
            case ExactAmplitudeRoute::HadamardPathFactor:
                return prepared_hpath_->amplitude_bits(bits, *workspace.hpath_);
            default:
                throw QStateError("Adaptive basis amplitude has no executable route");
        }
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> bits,
        ExactAdaptiveBasisAmplitudeWorkspace& workspace) const {
        validate_workspace(workspace);
        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                return phase_->log2_probability_bits(bits);
            case ExactAmplitudeRoute::HadamardPathFactor:
                return prepared_hpath_->log2_probability_bits(bits, *workspace.hpath_);
            default:
                throw QStateError("Adaptive basis amplitude has no executable route");
        }
    }

private:
    std::size_t qubit_count_{0U};
    std::vector<std::uint8_t> input_bits_{};
    ExactAdaptiveAmplitudeConfig config_{};
    ExactBasisAmplitudeDecision decision_{};
    bool hpath_candidate_{false};
    std::optional<ExactPhaseGraphBranchSum> phase_{};
    std::optional<ExactPreparedBasisHadamardPathPlan> prepared_hpath_{};

    [[nodiscard]] static ExactAmplitudeAdvisorConfig aligned_advisor(
        const ExactAdaptiveAmplitudeConfig& config) {
        ExactAmplitudeAdvisorConfig advisor = config.advisor;
        advisor.factor = config.hpath.factor;
        advisor.max_qubits = std::min(advisor.max_qubits, config.hpath.max_qubits);
        advisor.max_operations = std::min(advisor.max_operations, config.hpath.max_operations);
        advisor.max_hpath_events = std::min(advisor.max_hpath_events, config.hpath.max_h_events);
        advisor.max_phase_branches = std::min(
            advisor.max_phase_branches, config.phase_graph.max_branches);
        return advisor;
    }

    void scan_eligibility(
        std::span<const Operation> operations,
        const ExactAmplitudeAdvisorConfig& advisor) {
        validate_configuration(operations.size(), advisor);
        std::vector<std::size_t> h_counts(qubit_count_, 0U);
        bool phase_compatible = true;
        bool hpath_compatible = true;
        for (const Operation& operation : operations) {
            validate_operation(operation);
            if (operation.code == OperationCode::H) {
                ++decision_.target_h_events;
                ++h_counts[static_cast<std::size_t>(operation.first)];
            }
            phase_compatible = phase_compatible && phase_operation(operation.code);
            hpath_compatible = hpath_compatible && hpath_operation(operation.code);
        }

        decision_.basis_preparation_h_events = qubit_count_;
        if (decision_.target_h_events >
            std::numeric_limits<std::size_t>::max() - qubit_count_) {
            decision_.phase_branch_envelope_fits_size_t = false;
            decision_.phase_h_defects = std::numeric_limits<std::size_t>::max();
        } else {
            decision_.phase_h_defects = decision_.target_h_events + qubit_count_;
            set_phase_envelope();
        }
        decision_.phase_graph_eligible =
            phase_compatible &&
            decision_.phase_branch_envelope_fits_size_t &&
            decision_.phase_h_defects <= advisor.max_phase_h_defects &&
            decision_.phase_branch_envelope <= advisor.max_phase_branches;

        std::size_t hidden_variables = 0U;
        for (const std::size_t count : h_counts) {
            if (count == 0U) {
                continue;
            }
            if (hidden_variables >
                std::numeric_limits<std::size_t>::max() - (count - 1U)) {
                hpath_compatible = false;
                break;
            }
            hidden_variables += count - 1U;
        }
        hpath_candidate_ =
            hpath_compatible &&
            decision_.target_h_events <= advisor.max_hpath_events &&
            hidden_variables <= advisor.factor.max_variables;
    }

    void finalize(const ExactAmplitudeAdvisorConfig& advisor) {
        if (!decision_.phase_graph_eligible && !decision_.hpath_eligible) {
            decision_.route = ExactAmplitudeRoute::None;
            decision_.code = ExactBasisAmplitudeDecisionCode::NoEligibleRoute;
            return;
        }
        if (decision_.phase_graph_eligible && !decision_.hpath_eligible) {
            decision_.route = ExactAmplitudeRoute::PhaseGraphBranchSum;
            decision_.code = ExactBasisAmplitudeDecisionCode::PhaseOnly;
            return;
        }
        if (!decision_.phase_graph_eligible && decision_.hpath_eligible) {
            decision_.route = ExactAmplitudeRoute::HadamardPathFactor;
            decision_.code = ExactBasisAmplitudeDecisionCode::HPathOnly;
            return;
        }
        if (decision_.hpath.factor_variables == 0U) {
            decision_.route = ExactAmplitudeRoute::HadamardPathFactor;
            decision_.code = ExactBasisAmplitudeDecisionCode::HPathNoHiddenVariables;
            decision_.structural_log2_margin = decision_.phase_h_defects;
            return;
        }

        const std::size_t phase_log2 = decision_.phase_h_defects;
        const std::size_t hpath_log2 = decision_.hpath_peak_log2_ceiling;
        if (phase_log2 > hpath_log2 &&
            phase_log2 - hpath_log2 >= advisor.minimum_hpath_log2_margin) {
            decision_.route = ExactAmplitudeRoute::HadamardPathFactor;
            decision_.code = ExactBasisAmplitudeDecisionCode::HPathLowerPeakEnvelope;
            decision_.structural_log2_margin = phase_log2 - hpath_log2;
            return;
        }
        if (hpath_log2 > phase_log2 &&
            hpath_log2 - phase_log2 >= advisor.minimum_hpath_log2_margin) {
            decision_.route = ExactAmplitudeRoute::PhaseGraphBranchSum;
            decision_.code = ExactBasisAmplitudeDecisionCode::PhaseLowerBranchEnvelope;
            decision_.structural_log2_margin = hpath_log2 - phase_log2;
            return;
        }

        decision_.route = ExactAmplitudeRoute::Indeterminate;
        decision_.code = ExactBasisAmplitudeDecisionCode::CommonEnvelopeUnresolved;
    }

    void prepare_phase(std::span<const Operation> operations) {
        phase_.emplace(qubit_count_, config_.phase_graph);
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            const QubitId id = static_cast<QubitId>(qubit);
            if (input_bits_[qubit] != 0U) {
                phase_->apply_z(id);
            }
            phase_->apply_h(id);
        }
        for (const Operation& operation : operations) {
            phase_->apply(operation);
        }
    }

    void validate_input_bits() const {
        if (input_bits_.empty()) {
            throw QStateError("Adaptive basis amplitude requires at least one input bit");
        }
        for (const std::uint8_t bit : input_bits_) {
            if (bit > 1U) {
                throw QStateError("Adaptive basis amplitude input bits must be zero or one");
            }
        }
    }

    void validate_configuration(
        std::size_t operation_count,
        const ExactAmplitudeAdvisorConfig& advisor) const {
        if (qubit_count_ > advisor.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operation_count > advisor.max_operations ||
            advisor.max_phase_h_defects == 0U ||
            advisor.max_phase_branches == 0U ||
            advisor.max_hpath_events == 0U ||
            advisor.factor.max_variables == 0U) {
            throw QStateError("Adaptive basis amplitude dimensions or configuration are invalid");
        }
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Adaptive basis amplitude operation target is out of range");
        }
        if (operation.code == OperationCode::Cz ||
            operation.code == OperationCode::Swap ||
            operation.code == OperationCode::Cnot) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Adaptive basis amplitude two-qubit support is invalid");
            }
        }
        if ((operation.code == OperationCode::Rx ||
             operation.code == OperationCode::Ry ||
             operation.code == OperationCode::Rz) &&
            !std::isfinite(operation.parameter)) {
            throw QStateError("Adaptive basis amplitude rotation parameter must be finite");
        }
    }

    void set_phase_envelope() noexcept {
        constexpr std::size_t bits = std::numeric_limits<std::size_t>::digits;
        if (decision_.phase_h_defects >= bits) {
            decision_.phase_branch_envelope_fits_size_t = false;
            decision_.phase_branch_envelope = std::numeric_limits<std::size_t>::max();
            return;
        }
        decision_.phase_branch_envelope =
            std::size_t{1U} << decision_.phase_h_defects;
    }

    [[nodiscard]] static bool phase_operation(OperationCode code) noexcept {
        switch (code) {
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::H:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
            case OperationCode::Rz:
            case OperationCode::Cz:
            case OperationCode::Swap:
                return true;
            default:
                return false;
        }
    }

    [[nodiscard]] static bool hpath_operation(OperationCode code) noexcept {
        switch (code) {
            case OperationCode::H:
            case OperationCode::Z:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
            case OperationCode::Rz:
            case OperationCode::Cz:
                return true;
            default:
                return false;
        }
    }

    [[nodiscard]] static std::size_t ceil_log2(std::size_t value) noexcept {
        if (value <= 1U) {
            return 0U;
        }
        std::size_t result = 0U;
        std::size_t current = value - 1U;
        while (current != 0U) {
            ++result;
            current >>= 1U;
        }
        return result;
    }

    void validate_workspace(const ExactAdaptiveBasisAmplitudeWorkspace& workspace) const {
        if (workspace.owner_ != this) {
            throw QStateError(
                "Adaptive basis amplitude workspace belongs to a different plan");
        }
        const bool should_have_hpath =
            decision_.route == ExactAmplitudeRoute::HadamardPathFactor;
        if (should_have_hpath != workspace.hpath_.has_value()) {
            throw QStateError("Adaptive basis amplitude workspace shape is inconsistent");
        }
    }

    friend class ExactAdaptiveBasisAmplitudeWorkspace;
};

inline ExactAdaptiveBasisAmplitudeWorkspace::ExactAdaptiveBasisAmplitudeWorkspace(
    const ExactAdaptiveBasisAmplitudePlan& owner)
    : owner_(&owner) {
    if (owner.decision_.route == ExactAmplitudeRoute::HadamardPathFactor) {
        hpath_.emplace(owner.prepared_hpath_->workspace());
    }
}

}  // namespace qubit
