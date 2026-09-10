#pragma once

#include "qubit/qadaptive_amplitude.hpp"
#include "qubit/qhpath_prepared.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>

namespace qubit {

class ExactAdaptivePreparedAmplitudePlan;

class ExactAdaptivePreparedAmplitudeWorkspace {
public:
    ExactAdaptivePreparedAmplitudeWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return hpath_.has_value() ? hpath_->estimated_bytes() : sizeof(*this);
    }

    [[nodiscard]] std::size_t hpath_rebind_count() const noexcept {
        return hpath_.has_value() ? hpath_->rebind_count() : 0U;
    }

private:
    const ExactAdaptivePreparedAmplitudePlan* owner_{nullptr};
    std::optional<ExactPreparedHadamardPathWorkspace> hpath_{};

    explicit ExactAdaptivePreparedAmplitudeWorkspace(
        const ExactAdaptivePreparedAmplitudePlan& owner);

    friend class ExactAdaptivePreparedAmplitudePlan;
};

class ExactAdaptivePreparedAmplitudePlan {
public:
    ExactAdaptivePreparedAmplitudePlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactAdaptiveAmplitudeConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        const ExactAmplitudeAdvisorConfig advisor = aligned_advisor(config_);
        if (can_prepare_hpath(operations, advisor)) {
            prepared_hpath_.emplace(qubit_count_, operations, config_.hpath);
            const ExactHadamardPathWidthCertificate certificate =
                certificate_from(prepared_hpath_->stats());
            decision_ =
                ExactAmplitudeRepresentationAdvisor::analyze_plus_state_with_hpath_certificate(
                    qubit_count_, operations, certificate, advisor);
        } else {
            decision_ = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
                qubit_count_, operations, advisor);
        }

        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                prepared_hpath_.reset();
                phase_.emplace(qubit_count_, config_.phase_graph);
                for (const Operation& operation : operations) {
                    phase_->apply(operation);
                }
                break;
            case ExactAmplitudeRoute::HadamardPathFactor:
                if (!prepared_hpath_.has_value()) {
                    throw QStateError(
                        "Prepared adaptive amplitude selected Hpath without a prepared Hpath plan");
                }
                break;
            case ExactAmplitudeRoute::None:
                throw QStateError("Prepared adaptive amplitude has no eligible exact route");
            case ExactAmplitudeRoute::Indeterminate:
                throw QStateError(
                    "Prepared adaptive amplitude refuses to force an unresolved exact route");
            default:
                throw QStateError("Prepared adaptive amplitude received an unknown route");
        }
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] ExactAmplitudeRoute route() const noexcept { return decision_.route; }
    [[nodiscard]] const ExactAmplitudeDecision& decision() const noexcept { return decision_; }
    [[nodiscard]] const ExactAdaptiveAmplitudeConfig& config() const noexcept { return config_; }
    [[nodiscard]] bool has_prepared_hpath() const noexcept {
        return prepared_hpath_.has_value();
    }
    [[nodiscard]] const ExactPreparedHadamardPathStats* prepared_hpath_stats() const noexcept {
        return prepared_hpath_.has_value() ? &prepared_hpath_->stats() : nullptr;
    }

    [[nodiscard]] ExactAdaptivePreparedAmplitudeWorkspace workspace() const {
        return ExactAdaptivePreparedAmplitudeWorkspace(*this);
    }

    [[nodiscard]] QComplex amplitude_bits(
        std::span<const std::uint8_t> bits,
        ExactAdaptivePreparedAmplitudeWorkspace& workspace) const {
        validate_workspace(workspace);
        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                return phase_->amplitude_bits(bits);
            case ExactAmplitudeRoute::HadamardPathFactor:
                return prepared_hpath_->amplitude_bits(bits, *workspace.hpath_);
            default:
                throw QStateError("Prepared adaptive amplitude has no executable route");
        }
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> bits,
        ExactAdaptivePreparedAmplitudeWorkspace& workspace) const {
        validate_workspace(workspace);
        switch (decision_.route) {
            case ExactAmplitudeRoute::PhaseGraphBranchSum:
                return phase_->log2_probability_bits(bits);
            case ExactAmplitudeRoute::HadamardPathFactor:
                return prepared_hpath_->log2_probability_bits(bits, *workspace.hpath_);
            default:
                throw QStateError("Prepared adaptive amplitude has no executable route");
        }
    }

private:
    std::size_t qubit_count_{0U};
    ExactAdaptiveAmplitudeConfig config_{};
    ExactAmplitudeDecision decision_{};
    std::optional<ExactPhaseGraphBranchSum> phase_{};
    std::optional<ExactPreparedHadamardPathPlan> prepared_hpath_{};

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

    [[nodiscard]] static bool can_prepare_hpath(
        std::span<const Operation> operations,
        const ExactAmplitudeAdvisorConfig& advisor) noexcept {
        std::size_t h_events = 0U;
        for (const Operation& operation : operations) {
            switch (operation.code) {
                case OperationCode::H:
                    ++h_events;
                    break;
                case OperationCode::Z:
                case OperationCode::S:
                case OperationCode::Sdg:
                case OperationCode::T:
                case OperationCode::Tdg:
                case OperationCode::Rz:
                case OperationCode::Cz:
                    break;
                default:
                    return false;
            }
        }
        const std::size_t cap =
            std::min(advisor.max_hpath_events, advisor.factor.max_variables);
        return h_events <= cap;
    }

    [[nodiscard]] static ExactHadamardPathWidthCertificate certificate_from(
        const ExactPreparedHadamardPathStats& stats) noexcept {
        return ExactHadamardPathWidthCertificate{
            stats.qubits,
            stats.operations,
            stats.h_events,
            stats.h_active_qubits,
            stats.factor_variables,
            stats.factor_count,
            stats.factor.peak_union_variables,
            stats.factor.peak_factor_entries,
            ceil_log2(std::max<std::size_t>(1U, stats.factor.peak_factor_entries)),
            stats.factor.compiled_index_entries,
            stats.factor.workspace_slots,
            stats.graph_estimated_bytes,
            stats.plan_estimated_bytes,
        };
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

    void validate_workspace(const ExactAdaptivePreparedAmplitudeWorkspace& workspace) const {
        if (workspace.owner_ != this) {
            throw QStateError(
                "Prepared adaptive amplitude workspace belongs to a different plan");
        }
        if (prepared_hpath_.has_value() != workspace.hpath_.has_value()) {
            throw QStateError("Prepared adaptive amplitude workspace shape is inconsistent");
        }
    }

    friend class ExactAdaptivePreparedAmplitudeWorkspace;
};

inline ExactAdaptivePreparedAmplitudeWorkspace::ExactAdaptivePreparedAmplitudeWorkspace(
    const ExactAdaptivePreparedAmplitudePlan& owner)
    : owner_(&owner) {
    if (owner.prepared_hpath_.has_value()) {
        hpath_.emplace(owner.prepared_hpath_->workspace());
    }
}

}  // namespace qubit
