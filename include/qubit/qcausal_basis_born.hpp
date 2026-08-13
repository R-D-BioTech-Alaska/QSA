#pragma once

#include "qubit/qhpath_basis_born.hpp"
#include "qubit/qlightcone.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace qubit {

struct ExactCausalBasisHadamardBornConfig {
    ExactCircuitLightConeConfig light_cone{};
    ExactHadamardBornConfig born{};
};

struct ExactCausalBasisHadamardBornStats {
    ExactCircuitLightConeStats light_cone{};
    ExactBasisHadamardBornStats born{};
    std::size_t compact_input_bytes{0U};
};

class ExactCausalBasisHadamardBornPlan;

class ExactCausalBasisHadamardBornWorkspace {
public:
    ExactCausalBasisHadamardBornWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return born_.has_value() ? born_->estimated_bytes() : sizeof(*this);
    }

private:
    const ExactCausalBasisHadamardBornPlan* owner_{nullptr};
    std::optional<ExactBasisHadamardBornWorkspace> born_{};

    explicit ExactCausalBasisHadamardBornWorkspace(
        const ExactCausalBasisHadamardBornPlan& owner);

    friend class ExactCausalBasisHadamardBornPlan;
};

class ExactCausalBasisHadamardBornPlan {
public:
    ExactCausalBasisHadamardBornPlan(
        std::span<const std::uint8_t> input_bits,
        std::span<const Operation> operations,
        std::span<const std::size_t> retained_qubits,
        ExactCausalBasisHadamardBornConfig config = {})
        : input_bits_(input_bits.begin(), input_bits.end()),
          config_(config),
          cone_(input_bits.size(), operations, retained_qubits, config_.light_cone) {
        validate_input_bits();
        build();
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return input_bits_.size(); }
    [[nodiscard]] const ExactCircuitLightCone& light_cone() const noexcept { return cone_; }
    [[nodiscard]] const ExactCausalBasisHadamardBornStats& stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] const ExactCausalBasisHadamardBornConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] ExactCausalBasisHadamardBornWorkspace workspace() const {
        return ExactCausalBasisHadamardBornWorkspace(*this);
    }

    [[nodiscard]] std::vector<QComplex> marginal(
        ExactCausalBasisHadamardBornWorkspace& workspace) const {
        validate_workspace(workspace);
        return born_->marginal(*workspace.born_);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactCausalBasisHadamardBornWorkspace& workspace) const {
        validate_workspace(workspace);
        return born_->normalized_marginal(*workspace.born_);
    }

private:
    std::vector<std::uint8_t> input_bits_{};
    ExactCausalBasisHadamardBornConfig config_{};
    ExactCircuitLightCone cone_;
    std::vector<std::uint8_t> compact_input_{};
    std::optional<ExactBasisHadamardBornMarginalPlan> born_{};
    ExactCausalBasisHadamardBornStats stats_{};

    void build() {
        compact_input_.reserve(cone_.active_qubits().size());
        for (const std::size_t qubit : cone_.active_qubits()) {
            compact_input_.push_back(input_bits_[qubit]);
        }
        if (compact_input_.empty()) {
            throw QStateError("Causal basis Born light cone has no active input support");
        }
        born_.emplace(
            compact_input_,
            cone_.operations(),
            cone_.retained_local_qubits(),
            config_.born);
        stats_ = ExactCausalBasisHadamardBornStats{
            cone_.stats(),
            born_->stats(),
            compact_input_.capacity() * sizeof(std::uint8_t),
        };
    }

    void validate_input_bits() const {
        if (input_bits_.empty()) {
            throw QStateError("Causal basis Born requires a non-empty input basis state");
        }
        for (const std::uint8_t bit : input_bits_) {
            if (bit > 1U) {
                throw QStateError("Causal basis Born input bits must be zero or one");
            }
        }
    }

    void validate_workspace(const ExactCausalBasisHadamardBornWorkspace& workspace) const {
        if (workspace.owner_ != this || !workspace.born_.has_value()) {
            throw QStateError("Causal basis Born workspace belongs to a different plan");
        }
    }

    friend class ExactCausalBasisHadamardBornWorkspace;
};

inline ExactCausalBasisHadamardBornWorkspace::ExactCausalBasisHadamardBornWorkspace(
    const ExactCausalBasisHadamardBornPlan& owner)
    : owner_(&owner), born_(owner.born_->workspace()) {}

}  // namespace qubit
