#pragma once

#include "qubit/qhpath_factor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct ExactPreparedHadamardPathStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t h_events{0U};
    std::size_t h_active_qubits{0U};
    std::size_t factor_variables{0U};
    std::size_t factor_count{0U};
    std::size_t output_bindings{0U};
    std::size_t fixed_cz_pairs{0U};
    std::size_t metadata_estimated_bytes{0U};
    std::size_t graph_estimated_bytes{0U};
    std::size_t plan_estimated_bytes{0U};
    ExactFactorStats factor{};
};

class ExactPreparedHadamardPathPlan;

class ExactPreparedHadamardPathWorkspace {
public:
    ExactPreparedHadamardPathWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        const std::size_t workspace_bytes = factor_workspace_.estimated_bytes();
        if (workspace_bytes > std::numeric_limits<std::size_t>::max() - sizeof(*this)) {
            return std::numeric_limits<std::size_t>::max();
        }
        return sizeof(*this) + workspace_bytes;
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept { return rebind_count_; }

private:
    const ExactPreparedHadamardPathPlan* owner_{nullptr};
    ExactFactorWorkspace factor_workspace_{};
    std::size_t rebind_count_{0U};
    bool factor_workspace_active_{false};

    explicit ExactPreparedHadamardPathWorkspace(
        const ExactPreparedHadamardPathPlan& owner);

    friend class ExactPreparedHadamardPathPlan;
};

class ExactPreparedHadamardPathPlan {
public:
    ExactPreparedHadamardPathPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactHadamardPathConfig config = {})
        : qubit_count_(qubit_count),
          operation_count_(operations.size()),
          config_(config),
          fixed_phase_(qubit_count, std::array<QComplex, 2>{QComplex{1.0}, QComplex{1.0}}) {
        validate_configuration();
        build(operations);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] const ExactHadamardPathConfig& config() const noexcept { return config_; }
    [[nodiscard]] const ExactPreparedHadamardPathStats& stats() const noexcept { return stats_; }

    [[nodiscard]] ExactPreparedHadamardPathWorkspace workspace() const {
        return ExactPreparedHadamardPathWorkspace(*this);
    }

    [[nodiscard]] ExactHadamardPathAmplitude scaled_amplitude_bits(
        std::span<const std::uint8_t> bits,
        ExactPreparedHadamardPathWorkspace& workspace) const {
        validate_workspace(workspace);
        validate_bits(bits);

        QComplex global{1.0};
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            global *= fixed_phase_[qubit][bits[qubit]];
        }
        for (const auto& pair : fixed_cz_) {
            if (bits[pair.first] != 0U && bits[pair.second] != 0U) {
                global *= -1.0;
            }
        }

        QComplex partition{1.0};
        ExactFactorStats factor_stats{};
        if (plan_.has_value()) {
            std::lock_guard<std::mutex> lock(*query_mutex_);
            for (const OutputBinding& binding : bindings_) {
                const auto& values = bits[binding.bit] == 0U
                    ? binding.zero_values
                    : binding.one_values;
                plan_->rebind_dense_factor(binding.factor, values);
            }
            workspace.rebind_count_ = checked_sum(
                workspace.rebind_count_, bindings_.size(),
                "Prepared Hadamard-path workspace rebind count overflowed");
            partition = plan_->partition(workspace.factor_workspace_);
            factor_stats = plan_->stats();
        }

        const QComplex mantissa = global * partition;
        if (!finite(mantissa)) {
            throw QStateError("Prepared Hadamard-path amplitude became non-finite");
        }
        return ExactHadamardPathAmplitude{
            mantissa,
            -0.5 * static_cast<double>(qubit_count_ - h_active_qubits_),
            h_events_,
            h_active_qubits_,
            stats_.factor_variables,
            stats_.factor_count,
            stats_.metadata_estimated_bytes,
            stats_.graph_estimated_bytes,
            stats_.plan_estimated_bytes,
            factor_stats,
        };
    }

    [[nodiscard]] QComplex amplitude_bits(
        std::span<const std::uint8_t> bits,
        ExactPreparedHadamardPathWorkspace& workspace) const {
        return scaled_amplitude_bits(bits, workspace).amplitude();
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> bits,
        ExactPreparedHadamardPathWorkspace& workspace) const {
        return scaled_amplitude_bits(bits, workspace).log2_probability();
    }

private:
    struct OutputBinding {
        FactorId factor{0U};
        std::size_t bit{0U};
        std::array<QComplex, 2> zero_values{};
        std::array<QComplex, 2> one_values{};
    };

    std::size_t qubit_count_{0U};
    std::size_t operation_count_{0U};
    ExactHadamardPathConfig config_{};
    std::size_t h_events_{0U};
    std::size_t h_active_qubits_{0U};
    std::vector<OutputBinding> bindings_{};
    std::vector<std::pair<std::size_t, std::size_t>> fixed_cz_{};
    std::vector<std::array<QComplex, 2>> fixed_phase_{};
    mutable std::optional<ExactFactorPlan> plan_{};
    mutable std::shared_ptr<std::mutex> query_mutex_{std::make_shared<std::mutex>()};
    ExactPreparedHadamardPathStats stats_{};

    void build(std::span<const Operation> operations) {
        const std::size_t npos = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> first(qubit_count_, npos);
        std::vector<std::size_t> last(qubit_count_, npos);
        std::vector<std::size_t> next;
        std::vector<std::size_t> operation_event(operation_count_, npos);
        next.reserve(std::min(operation_count_, config_.max_h_events));

        for (std::size_t operation_index = 0U;
             operation_index < operation_count_;
             ++operation_index) {
            const Operation& operation = operations[operation_index];
            validate_operation(operation);
            if (operation.code != OperationCode::H) {
                continue;
            }
            if (h_events_ >= config_.max_h_events ||
                h_events_ >= config_.factor.max_variables) {
                throw QStateError("Prepared Hadamard-path compiler exceeds its H-event variable cap");
            }
            const std::size_t qubit = static_cast<std::size_t>(operation.first);
            const std::size_t event = h_events_++;
            operation_event[operation_index] = event;
            next.push_back(npos);
            if (last[qubit] == npos) {
                first[qubit] = event;
                ++h_active_qubits_;
            } else {
                next[last[qubit]] = event;
            }
            last[qubit] = event;
        }

        ExactFactorGraph graph(config_.factor);
        std::vector<FactorVariableId> variables(h_events_);
        for (std::size_t event = 0U; event < h_events_; ++event) {
            variables[event] = graph.add_variable(2U);
        }
        std::vector<std::size_t> current = first;

        const auto active_event = [&](QubitId qubit) {
            return current[static_cast<std::size_t>(qubit)];
        };
        const auto add_active_unary = [&](QubitId qubit, QComplex zero, QComplex one) {
            const std::size_t event = active_event(qubit);
            if (event == npos) {
                multiply_fixed_phase(static_cast<std::size_t>(qubit), zero, one);
                return;
            }
            const std::array<FactorVariableId, 1> scope{variables[event]};
            const std::array<QComplex, 2> values{zero, one};
            (void)graph.add_dense_factor(scope, values);
        };

        constexpr double inverse_sqrt_two =
            0.707106781186547524400844362104849039;
        for (std::size_t operation_index = 0U;
             operation_index < operation_count_;
             ++operation_index) {
            const Operation& operation = operations[operation_index];
            switch (operation.code) {
                case OperationCode::H: {
                    const std::size_t qubit = static_cast<std::size_t>(operation.first);
                    const std::size_t event = operation_event[operation_index];
                    if (event == npos || current[qubit] != event) {
                        throw QStateError("Prepared Hadamard-path event sequence is inconsistent");
                    }
                    const std::size_t following = next[event];
                    const double normalization =
                        event == first[qubit] ? 0.5 : inverse_sqrt_two;
                    if (following != npos) {
                        const std::array<FactorVariableId, 2> scope{
                            variables[event], variables[following]};
                        const std::array<QComplex, 4> values{
                            QComplex{normalization}, QComplex{normalization},
                            QComplex{normalization}, QComplex{-normalization},
                        };
                        (void)graph.add_dense_factor(scope, values);
                    } else {
                        const std::array<FactorVariableId, 1> scope{variables[event]};
                        const std::array<QComplex, 2> placeholder{
                            QComplex{normalization}, QComplex{normalization}};
                        const FactorId factor = graph.add_dense_factor(scope, placeholder);
                        bindings_.push_back(OutputBinding{
                            factor,
                            qubit,
                            {QComplex{normalization}, QComplex{normalization}},
                            {QComplex{normalization}, QComplex{-normalization}},
                        });
                    }
                    current[qubit] = following;
                    break;
                }
                case OperationCode::Z:
                    add_active_unary(operation.first, {1.0, 0.0}, {-1.0, 0.0});
                    break;
                case OperationCode::S:
                    add_active_unary(operation.first, {1.0, 0.0}, QI);
                    break;
                case OperationCode::Sdg:
                    add_active_unary(operation.first, {1.0, 0.0}, -QI);
                    break;
                case OperationCode::T:
                    add_active_unary(
                        operation.first,
                        {1.0, 0.0},
                        QComplex::from_polar(1.0, 0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Tdg:
                    add_active_unary(
                        operation.first,
                        {1.0, 0.0},
                        QComplex::from_polar(1.0, -0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Rz:
                    add_active_unary(
                        operation.first,
                        QComplex::from_polar(1.0, -0.5 * operation.parameter),
                        QComplex::from_polar(1.0, 0.5 * operation.parameter));
                    break;
                case OperationCode::Cz: {
                    const std::size_t first_qubit = static_cast<std::size_t>(operation.first);
                    const std::size_t second_qubit = static_cast<std::size_t>(operation.second);
                    const std::size_t first_event = active_event(operation.first);
                    const std::size_t second_event = active_event(operation.second);
                    if (first_event != npos && second_event != npos) {
                        const std::array<FactorVariableId, 2> scope{
                            variables[first_event], variables[second_event]};
                        const std::array<QComplex, 4> values{
                            QComplex{1.0}, QComplex{1.0},
                            QComplex{1.0}, QComplex{-1.0}};
                        (void)graph.add_dense_factor(scope, values);
                    } else if (first_event != npos || second_event != npos) {
                        const std::size_t active = first_event != npos ? first_event : second_event;
                        const std::size_t fixed = first_event != npos ? second_qubit : first_qubit;
                        const std::array<FactorVariableId, 1> scope{variables[active]};
                        const std::array<QComplex, 2> placeholder{QComplex{1.0}, QComplex{1.0}};
                        const FactorId factor = graph.add_dense_factor(scope, placeholder);
                        bindings_.push_back(OutputBinding{
                            factor,
                            fixed,
                            {QComplex{1.0}, QComplex{1.0}},
                            {QComplex{1.0}, QComplex{-1.0}},
                        });
                    } else {
                        fixed_cz_.push_back({first_qubit, second_qubit});
                    }
                    break;
                }
                default:
                    throw QStateError(
                        "Operation left the validated prepared Hadamard-path diagonal contract");
            }
        }

        ExactFactorStats factor_stats{};
        std::size_t plan_bytes = 0U;
        if (h_events_ != 0U) {
            plan_.emplace(graph);
            factor_stats = plan_->stats();
            plan_bytes = plan_->estimated_bytes();
        }
        const std::size_t metadata_bytes = estimate_metadata_bytes();
        if (metadata_bytes > config_.max_metadata_bytes) {
            throw QStateError("Prepared Hadamard-path metadata exceeds configured byte cap");
        }
        stats_ = ExactPreparedHadamardPathStats{
            qubit_count_,
            operation_count_,
            h_events_,
            h_active_qubits_,
            graph.variable_count(),
            graph.factor_count(),
            bindings_.size(),
            fixed_cz_.size(),
            metadata_bytes,
            graph.estimated_bytes(),
            plan_bytes,
            factor_stats,
        };
    }

    void validate_configuration() const {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operation_count_ > config_.max_operations ||
            config_.max_h_events == 0U || config_.max_metadata_bytes == 0U) {
            throw QStateError("Prepared Hadamard-path dimensions or configuration are invalid");
        }
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Prepared Hadamard-path operation target is out of range");
        }
        if (operation.code == OperationCode::Cz) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Prepared Hadamard-path CZ support is invalid");
            }
        }
        if (operation.code == OperationCode::Rz && !std::isfinite(operation.parameter)) {
            throw QStateError("Prepared Hadamard-path Rz angle must be finite");
        }
        switch (operation.code) {
            case OperationCode::H:
            case OperationCode::Z:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
            case OperationCode::Rz:
            case OperationCode::Cz:
                return;
            default:
                throw QStateError(
                    "Operation is outside the exact prepared Hadamard-path diagonal contract");
        }
    }

    void validate_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Prepared Hadamard-path output bit count does not match qubit count");
        }
        for (const std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Prepared Hadamard-path output bits must be zero or one");
            }
        }
    }

    void validate_workspace(const ExactPreparedHadamardPathWorkspace& workspace) const {
        if (workspace.owner_ != this) {
            throw QStateError("Prepared Hadamard-path workspace belongs to a different plan");
        }
        if (plan_.has_value() != workspace.factor_workspace_active_) {
            throw QStateError("Prepared Hadamard-path workspace shape is inconsistent");
        }
    }

    void multiply_fixed_phase(std::size_t qubit, QComplex zero, QComplex one) {
        fixed_phase_[qubit][0] *= zero;
        fixed_phase_[qubit][1] *= one;
        if (!finite(fixed_phase_[qubit][0]) || !finite(fixed_phase_[qubit][1])) {
            throw QStateError("Prepared Hadamard-path fixed phase became non-finite");
        }
    }

    [[nodiscard]] std::size_t estimate_metadata_bytes() const {
        std::size_t total = sizeof(*this);
        total = checked_sum(total, checked_product(
            bindings_.capacity(), sizeof(OutputBinding),
            "Prepared Hadamard-path binding metadata overflowed"));
        total = checked_sum(total, checked_product(
            fixed_cz_.capacity(), sizeof(std::pair<std::size_t, std::size_t>),
            "Prepared Hadamard-path fixed-CZ metadata overflowed"));
        total = checked_sum(total, checked_product(
            fixed_phase_.capacity(), sizeof(std::array<QComplex, 2>),
            "Prepared Hadamard-path phase metadata overflowed"));
        return total;
    }

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left,
        std::size_t right,
        const char* message = "Prepared Hadamard-path metadata overflowed") {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }

    friend class ExactPreparedHadamardPathWorkspace;
};

inline ExactPreparedHadamardPathWorkspace::ExactPreparedHadamardPathWorkspace(
    const ExactPreparedHadamardPathPlan& owner)
    : owner_(&owner) {
    if (owner.plan_.has_value()) {
        factor_workspace_ = owner.plan_->workspace();
        factor_workspace_active_ = true;
    }
}

}  // namespace qubit
