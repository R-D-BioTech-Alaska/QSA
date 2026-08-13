#pragma once

#include "qubit/qhpath_basis_born.hpp"
#include "qubit/qlightcone.hpp"
#include "qubit/qplan.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace qubit {

enum class ExactCausalBasisMarginalRoute : std::uint8_t {
    LocalRegister = 0U,
    HadamardBorn = 1U,
};

struct ExactCausalBasisMarginalConfig {
    ExactCircuitLightConeConfig light_cone{};
    ExactHadamardBornConfig born{};
    QStateConfig local_state{};
    std::size_t max_local_qubits{20U};
    std::uint64_t local_preference_max_amplitudes{1ULL << 12U};
    std::size_t max_local_dense_bytes{256U * 1024U * 1024U};
};

struct ExactCausalBasisMarginalStats {
    ExactCircuitLightConeStats light_cone{};
    ExactCausalBasisMarginalRoute route{ExactCausalBasisMarginalRoute::LocalRegister};
    bool local_eligible{false};
    bool born_eligible{false};
    bool born_resource_rejected{false};
    std::uint64_t local_worst_case_amplitudes{0U};
    std::size_t local_worst_case_dense_bytes{0U};
    std::size_t local_state_estimated_bytes{0U};
    std::size_t output_entries{0U};
    std::optional<ExactBasisHadamardBornStats> born{};
};

class ExactCausalBasisMarginalPlan;

class ExactCausalBasisMarginalWorkspace {
public:
    ExactCausalBasisMarginalWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return born_.has_value() ? born_->estimated_bytes() : sizeof(*this);
    }

private:
    const ExactCausalBasisMarginalPlan* owner_{nullptr};
    std::optional<ExactBasisHadamardBornWorkspace> born_{};
    std::vector<std::uint8_t> bits_{};

    explicit ExactCausalBasisMarginalWorkspace(
        const ExactCausalBasisMarginalPlan& owner);

    friend class ExactCausalBasisMarginalPlan;
};

class ExactCausalBasisMarginalPlan {
public:
    ExactCausalBasisMarginalPlan(
        std::span<const std::uint8_t> input_bits,
        std::span<const Operation> operations,
        std::span<const std::size_t> retained_qubits,
        ExactCausalBasisMarginalConfig config = {})
        : input_bits_(input_bits.begin(), input_bits.end()),
          config_(config),
          cone_(input_bits.size(), operations, retained_qubits, config_.light_cone) {
        validate_input_bits();
        validate_configuration();
        build();
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return input_bits_.size(); }
    [[nodiscard]] ExactCausalBasisMarginalRoute route() const noexcept { return stats_.route; }
    [[nodiscard]] const ExactCircuitLightCone& light_cone() const noexcept { return cone_; }
    [[nodiscard]] const ExactCausalBasisMarginalStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ExactCausalBasisMarginalConfig& config() const noexcept { return config_; }

    [[nodiscard]] ExactCausalBasisMarginalWorkspace workspace() const {
        return ExactCausalBasisMarginalWorkspace(*this);
    }

    [[nodiscard]] std::vector<QComplex> marginal(
        ExactCausalBasisMarginalWorkspace& workspace) const {
        validate_workspace(workspace);
        if (stats_.route == ExactCausalBasisMarginalRoute::HadamardBorn) {
            return born_->marginal(*workspace.born_);
        }
        return local_marginal(workspace);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactCausalBasisMarginalWorkspace& workspace) const {
        validate_workspace(workspace);
        if (stats_.route == ExactCausalBasisMarginalRoute::HadamardBorn) {
            return born_->normalized_marginal(*workspace.born_);
        }
        std::vector<QComplex> values = local_marginal(workspace);
        double total = 0.0;
        for (const QComplex value : values) {
            total += value.re;
        }
        if (!std::isfinite(total) || total <= std::numeric_limits<double>::min()) {
            throw QStateError("Causal local-register marginal cannot be normalized");
        }
        for (QComplex& value : values) {
            value.re /= total;
            value.im /= total;
        }
        return values;
    }

private:
    std::vector<std::uint8_t> input_bits_{};
    ExactCausalBasisMarginalConfig config_{};
    ExactCircuitLightCone cone_;
    std::vector<std::uint8_t> compact_input_{};
    std::vector<QubitId> retained_local_ids_{};
    std::optional<QRegister> local_state_{};
    std::optional<ExactBasisHadamardBornMarginalPlan> born_{};
    ExactCausalBasisMarginalStats stats_{};

    void build() {
        compact_input_.reserve(cone_.active_qubits().size());
        for (const std::size_t qubit : cone_.active_qubits()) {
            compact_input_.push_back(input_bits_[qubit]);
        }
        if (compact_input_.empty()) {
            throw QStateError("Causal basis marginal light cone has no active support");
        }

        retained_local_ids_.reserve(cone_.retained_local_qubits().size());
        for (const std::size_t qubit : cone_.retained_local_qubits()) {
            if (qubit > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
                throw QStateError("Causal basis marginal local retained id overflowed");
            }
            retained_local_ids_.push_back(static_cast<QubitId>(qubit));
        }

        const std::size_t output_entries = retained_output_entries();
        const LocalPreflight local = local_preflight();
        if (local.eligible &&
            local.amplitudes <= config_.local_preference_max_amplitudes) {
            prepare_local();
            stats_ = make_stats(
                ExactCausalBasisMarginalRoute::LocalRegister,
                local,
                false,
                false,
                output_entries);
            return;
        }

        bool born_resource_rejected = false;
        if (born_operation_family()) {
            try {
                born_.emplace(
                    compact_input_,
                    cone_.operations(),
                    cone_.retained_local_qubits(),
                    config_.born);
            } catch (const QStateError&) {
                born_resource_rejected = true;
                born_.reset();
            }
        }

        if (born_.has_value()) {
            stats_ = make_stats(
                ExactCausalBasisMarginalRoute::HadamardBorn,
                local,
                true,
                born_resource_rejected,
                output_entries);
            stats_.born = born_->stats();
            return;
        }
        if (local.eligible) {
            prepare_local();
            stats_ = make_stats(
                ExactCausalBasisMarginalRoute::LocalRegister,
                local,
                false,
                born_resource_rejected,
                output_entries);
            return;
        }
        throw QStateError(
            "Causal basis marginal has neither a bounded local register nor a bounded Hpath Born route");
    }

    struct LocalPreflight {
        bool eligible{false};
        std::uint64_t amplitudes{0U};
        std::size_t bytes{0U};
    };

    [[nodiscard]] LocalPreflight local_preflight() const {
        const std::size_t active = cone_.active_qubits().size();
        if (active > config_.max_local_qubits ||
            active > config_.local_state.max_component_qubits ||
            active >= std::numeric_limits<std::uint64_t>::digits) {
            return {};
        }
        const std::uint64_t amplitudes = std::uint64_t{1U} << active;
        if (amplitudes > config_.local_state.max_dense_amplitudes ||
            amplitudes > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max() / sizeof(QComplex))) {
            return {};
        }
        const std::size_t bytes =
            static_cast<std::size_t>(amplitudes) * sizeof(QComplex);
        if (bytes > config_.max_local_dense_bytes) {
            return {};
        }
        return {true, amplitudes, bytes};
    }

    void prepare_local() {
        local_state_.emplace(compact_input_.size(), config_.local_state);
        for (std::size_t qubit = 0U; qubit < compact_input_.size(); ++qubit) {
            if (compact_input_[qubit] != 0U) {
                local_state_->apply_x(static_cast<QubitId>(qubit));
            }
        }
        OperationPlan plan(cone_.operations(), true);
        plan.execute(*local_state_);
        std::string reason;
        if (!local_state_->validate(&reason)) {
            throw QStateError("Causal local-register state is invalid: " + reason);
        }
    }

    [[nodiscard]] std::vector<QComplex> local_marginal(
        ExactCausalBasisMarginalWorkspace& workspace) const {
        if (!local_state_.has_value()) {
            throw QStateError("Causal local-register route has no prepared state");
        }
        const std::size_t entries = stats_.output_entries;
        if (workspace.bits_.size() != retained_local_ids_.size()) {
            workspace.bits_.resize(retained_local_ids_.size());
        }
        std::vector<QComplex> result(entries, QComplex{});
        for (std::size_t index = 0U; index < entries; ++index) {
            for (std::size_t position = 0U; position < workspace.bits_.size(); ++position) {
                workspace.bits_[position] =
                    static_cast<std::uint8_t>((index >> position) & 1U);
            }
            const double probability =
                local_state_->marginal_probability(retained_local_ids_, workspace.bits_);
            if (!std::isfinite(probability)) {
                throw QStateError("Causal local-register marginal became non-finite");
            }
            result[index] = {probability, 0.0};
        }
        return result;
    }

    [[nodiscard]] bool born_operation_family() const noexcept {
        for (const Operation& operation : cone_.operations()) {
            switch (operation.code) {
                case OperationCode::H:
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
        return true;
    }

    [[nodiscard]] ExactCausalBasisMarginalStats make_stats(
        ExactCausalBasisMarginalRoute route_value,
        LocalPreflight local,
        bool born_eligible,
        bool born_resource_rejected,
        std::size_t output_entries) const {
        return ExactCausalBasisMarginalStats{
            cone_.stats(),
            route_value,
            local.eligible,
            born_eligible,
            born_resource_rejected,
            local.amplitudes,
            local.bytes,
            local_state_.has_value() ? local_state_->estimated_bytes() : 0U,
            output_entries,
            std::nullopt,
        };
    }

    [[nodiscard]] std::size_t retained_output_entries() const {
        const std::size_t retained = cone_.retained_local_qubits().size();
        if (retained >= std::numeric_limits<std::size_t>::digits) {
            throw QStateError("Causal basis marginal output size overflowed");
        }
        const std::size_t entries = std::size_t{1U} << retained;
        if (entries > config_.born.factor.max_factor_entries) {
            throw QStateError("Causal basis marginal output exceeds configured entry cap");
        }
        return entries;
    }

    void validate_configuration() const {
        if (config_.max_local_qubits == 0U ||
            config_.local_preference_max_amplitudes == 0U ||
            config_.max_local_dense_bytes == 0U ||
            config_.local_state.max_dense_amplitudes == 0U ||
            config_.local_state.max_component_qubits == 0U) {
            throw QStateError("Causal basis marginal local-route limits must be nonzero");
        }
    }

    void validate_input_bits() const {
        if (input_bits_.empty()) {
            throw QStateError("Causal basis marginal requires a non-empty input basis state");
        }
        for (const std::uint8_t bit : input_bits_) {
            if (bit > 1U) {
                throw QStateError("Causal basis marginal input bits must be zero or one");
            }
        }
    }

    void validate_workspace(const ExactCausalBasisMarginalWorkspace& workspace) const {
        if (workspace.owner_ != this) {
            throw QStateError("Causal basis marginal workspace belongs to a different plan");
        }
        const bool should_have_born = stats_.route == ExactCausalBasisMarginalRoute::HadamardBorn;
        if (workspace.born_.has_value() != should_have_born) {
            throw QStateError("Causal basis marginal workspace shape is inconsistent");
        }
    }

    friend class ExactCausalBasisMarginalWorkspace;
};

inline ExactCausalBasisMarginalWorkspace::ExactCausalBasisMarginalWorkspace(
    const ExactCausalBasisMarginalPlan& owner)
    : owner_(&owner), bits_(owner.retained_local_ids_.size()) {
    if (owner.stats_.route == ExactCausalBasisMarginalRoute::HadamardBorn) {
        born_.emplace(owner.born_->workspace());
    }
}

}  // namespace qubit
