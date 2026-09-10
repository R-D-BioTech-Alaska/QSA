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

struct ExactPreparedBasisHadamardPathStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t h_events{0U};
    std::size_t h_active_qubits{0U};
    std::size_t removed_first_h_variables{0U};
    std::size_t factor_variables{0U};
    std::size_t factor_count{0U};
    std::size_t output_bindings{0U};
    std::size_t output_cz_pairs{0U};
    std::size_t fixed_support_qubits{0U};
    std::size_t metadata_estimated_bytes{0U};
    std::size_t graph_estimated_bytes{0U};
    std::size_t plan_estimated_bytes{0U};
    ExactFactorStats factor{};
};

class ExactPreparedBasisHadamardPathPlan;

class ExactPreparedBasisHadamardPathWorkspace {
public:
    ExactPreparedBasisHadamardPathWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        const std::size_t workspace_bytes = factor_workspace_.estimated_bytes();
        if (workspace_bytes > std::numeric_limits<std::size_t>::max() - sizeof(*this)) {
            return std::numeric_limits<std::size_t>::max();
        }
        return sizeof(*this) + workspace_bytes;
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept { return rebind_count_; }

private:
    const ExactPreparedBasisHadamardPathPlan* owner_{nullptr};
    ExactFactorWorkspace factor_workspace_{};
    std::size_t rebind_count_{0U};
    bool factor_workspace_active_{false};

    explicit ExactPreparedBasisHadamardPathWorkspace(
        const ExactPreparedBasisHadamardPathPlan& owner);

    friend class ExactPreparedBasisHadamardPathPlan;
};

class ExactPreparedBasisHadamardPathPlan {
public:
    ExactPreparedBasisHadamardPathPlan(
        std::span<const std::uint8_t> input_bits,
        std::span<const Operation> operations,
        ExactHadamardPathConfig config = {})
        : qubit_count_(input_bits.size()),
          operation_count_(operations.size()),
          config_(config),
          input_bits_(input_bits.begin(), input_bits.end()),
          output_phase_(qubit_count_,
              std::array<QComplex, 2>{QComplex{1.0, 0.0}, QComplex{1.0, 0.0}}) {
        validate_configuration();
        validate_input_bits();
        build(operations);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] const ExactHadamardPathConfig& config() const noexcept { return config_; }
    [[nodiscard]] const ExactPreparedBasisHadamardPathStats& stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] std::span<const std::uint8_t> input_bits() const noexcept {
        return input_bits_;
    }

    [[nodiscard]] ExactPreparedBasisHadamardPathWorkspace workspace() const {
        return ExactPreparedBasisHadamardPathWorkspace(*this);
    }

    [[nodiscard]] ExactHadamardPathAmplitude scaled_amplitude_bits(
        std::span<const std::uint8_t> output_bits,
        ExactPreparedBasisHadamardPathWorkspace& workspace) const {
        validate_workspace(workspace);
        validate_output_bits(output_bits);

        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            if (h_counts_[qubit] == 0U && output_bits[qubit] != input_bits_[qubit]) {
                return zero_result();
            }
        }

        QComplex global = compile_global_;
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            if (h_counts_[qubit] != 0U) {
                global *= output_phase_[qubit][output_bits[qubit]];
            }
        }
        for (const auto& pair : output_cz_) {
            if (output_bits[pair.first] != 0U && output_bits[pair.second] != 0U) {
                global *= -1.0;
            }
        }

        QComplex partition{1.0, 0.0};
        ExactFactorStats factor_stats{};
        if (plan_.has_value()) {
            std::lock_guard<std::mutex> lock(*query_mutex_);
            for (const OutputBinding& binding : bindings_) {
                const auto& values = output_bits[binding.bit] == 0U
                    ? binding.zero_values
                    : binding.one_values;
                plan_->rebind_dense_factor(binding.factor, values);
            }
            workspace.rebind_count_ = checked_sum(
                workspace.rebind_count_, bindings_.size(),
                "Basis Hpath workspace rebind count overflowed");
            partition = plan_->partition(workspace.factor_workspace_);
            factor_stats = plan_->stats();
        }

        const QComplex mantissa = global * partition;
        if (!finite(mantissa)) {
            throw QStateError("Basis Hpath amplitude became non-finite");
        }
        return ExactHadamardPathAmplitude{
            mantissa,
            0.0,
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
        std::span<const std::uint8_t> output_bits,
        ExactPreparedBasisHadamardPathWorkspace& workspace) const {
        return scaled_amplitude_bits(output_bits, workspace).amplitude();
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> output_bits,
        ExactPreparedBasisHadamardPathWorkspace& workspace) const {
        return scaled_amplitude_bits(output_bits, workspace).log2_probability();
    }

private:
    enum class SegmentState : std::uint8_t {
        InputFixed = 0U,
        Hidden = 1U,
        OutputFixed = 2U,
    };

    struct OutputBinding {
        FactorId factor{0U};
        std::size_t bit{0U};
        std::array<QComplex, 2> zero_values{};
        std::array<QComplex, 2> one_values{};
    };

    std::size_t qubit_count_{0U};
    std::size_t operation_count_{0U};
    ExactHadamardPathConfig config_{};
    std::vector<std::uint8_t> input_bits_{};
    std::vector<std::size_t> h_counts_{};
    std::size_t h_events_{0U};
    std::size_t h_active_qubits_{0U};
    QComplex compile_global_{1.0, 0.0};
    std::vector<std::array<QComplex, 2>> output_phase_{};
    std::vector<OutputBinding> bindings_{};
    std::vector<std::pair<std::size_t, std::size_t>> output_cz_{};
    mutable std::optional<ExactFactorPlan> plan_{};
    mutable std::shared_ptr<std::mutex> query_mutex_{std::make_shared<std::mutex>()};
    ExactPreparedBasisHadamardPathStats stats_{};

    void build(std::span<const Operation> operations) {
        const std::size_t npos = std::numeric_limits<std::size_t>::max();
        h_counts_.assign(qubit_count_, 0U);
        for (const Operation& operation : operations) {
            validate_operation(operation);
            if (operation.code != OperationCode::H) {
                continue;
            }
            if (h_events_ >= config_.max_h_events) {
                throw QStateError("Basis Hpath compiler exceeds its H-event cap");
            }
            ++h_events_;
            const std::size_t qubit = static_cast<std::size_t>(operation.first);
            if (h_counts_[qubit] == 0U) {
                ++h_active_qubits_;
            }
            ++h_counts_[qubit];
        }

        ExactFactorGraph graph(config_.factor);
        std::vector<std::size_t> seen_h(qubit_count_, 0U);
        std::vector<std::size_t> current_variable(qubit_count_, npos);
        std::vector<std::uint8_t> before_first(qubit_count_, 1U);

        const auto state = [&](std::size_t qubit) {
            if (before_first[qubit] != 0U) {
                return SegmentState::InputFixed;
            }
            return current_variable[qubit] == npos
                ? SegmentState::OutputFixed
                : SegmentState::Hidden;
        };

        const auto add_unary_factor = [&](std::size_t variable, QComplex zero, QComplex one) {
            const std::array<FactorVariableId, 1> scope{
                static_cast<FactorVariableId>(variable)};
            const std::array<QComplex, 2> values{zero, one};
            (void)graph.add_dense_factor(scope, values);
        };

        const auto add_diagonal = [&](std::size_t qubit, QComplex zero, QComplex one) {
            switch (state(qubit)) {
                case SegmentState::InputFixed:
                    compile_global_ *= input_bits_[qubit] == 0U ? zero : one;
                    break;
                case SegmentState::Hidden:
                    add_unary_factor(current_variable[qubit], zero, one);
                    break;
                case SegmentState::OutputFixed:
                    multiply_output_phase(qubit, zero, one);
                    break;
            }
            if (!finite(compile_global_)) {
                throw QStateError("Basis Hpath fixed input phase became non-finite");
            }
        };

        const auto add_output_binding = [&](std::size_t variable, std::size_t output_bit) {
            const std::array<FactorVariableId, 1> scope{
                static_cast<FactorVariableId>(variable)};
            const std::array<QComplex, 2> placeholder{
                QComplex{1.0, 0.0}, QComplex{1.0, 0.0}};
            const FactorId factor = graph.add_dense_factor(scope, placeholder);
            bindings_.push_back(OutputBinding{
                factor,
                output_bit,
                {QComplex{1.0, 0.0}, QComplex{1.0, 0.0}},
                {QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}},
            });
        };

        const auto add_cz = [&](std::size_t first, std::size_t second) {
            const SegmentState first_state = state(first);
            const SegmentState second_state = state(second);
            if (first_state == SegmentState::Hidden &&
                second_state == SegmentState::Hidden) {
                const std::array<FactorVariableId, 2> scope{
                    static_cast<FactorVariableId>(current_variable[first]),
                    static_cast<FactorVariableId>(current_variable[second])};
                const std::array<QComplex, 4> values{
                    QComplex{1.0, 0.0}, QComplex{1.0, 0.0},
                    QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}};
                (void)graph.add_dense_factor(scope, values);
                return;
            }

            if (first_state == SegmentState::Hidden ||
                second_state == SegmentState::Hidden) {
                const bool first_hidden = first_state == SegmentState::Hidden;
                const std::size_t hidden = first_hidden ? first : second;
                const std::size_t other = first_hidden ? second : first;
                const SegmentState other_state = first_hidden ? second_state : first_state;
                if (other_state == SegmentState::InputFixed) {
                    if (input_bits_[other] != 0U) {
                        add_unary_factor(
                            current_variable[hidden],
                            {1.0, 0.0}, {-1.0, 0.0});
                    }
                } else {
                    add_output_binding(current_variable[hidden], other);
                }
                return;
            }

            if (first_state == SegmentState::InputFixed &&
                second_state == SegmentState::InputFixed) {
                if (input_bits_[first] != 0U && input_bits_[second] != 0U) {
                    compile_global_ *= -1.0;
                }
                return;
            }

            if (first_state == SegmentState::OutputFixed &&
                second_state == SegmentState::OutputFixed) {
                output_cz_.push_back({first, second});
                return;
            }

            const std::size_t input =
                first_state == SegmentState::InputFixed ? first : second;
            const std::size_t output = input == first ? second : first;
            if (input_bits_[input] != 0U) {
                output_phase_[output][1] *= -1.0;
            }
        };

        constexpr double inverse_sqrt_two =
            0.707106781186547524400844362104849039;

        for (const Operation& operation : operations) {
            const std::size_t first = static_cast<std::size_t>(operation.first);
            switch (operation.code) {
                case OperationCode::H: {
                    const bool is_first = seen_h[first] == 0U;
                    const bool is_last = seen_h[first] + 1U == h_counts_[first];
                    const std::size_t previous = current_variable[first];
                    if (is_first && (previous != npos || before_first[first] == 0U)) {
                        throw QStateError("Basis Hpath first-H state is inconsistent");
                    }
                    if (!is_first && previous == npos) {
                        throw QStateError("Basis Hpath hidden segment was lost before H");
                    }

                    if (!is_last) {
                        const FactorVariableId next = graph.add_variable(2U);
                        if (is_first) {
                            const double one = input_bits_[first] == 0U
                                ? inverse_sqrt_two
                                : -inverse_sqrt_two;
                            const std::array<FactorVariableId, 1> scope{next};
                            const std::array<QComplex, 2> values{
                                QComplex{inverse_sqrt_two, 0.0}, QComplex{one, 0.0}};
                            (void)graph.add_dense_factor(scope, values);
                        } else {
                            const std::array<FactorVariableId, 2> scope{
                                static_cast<FactorVariableId>(previous), next};
                            const std::array<QComplex, 4> values{
                                QComplex{inverse_sqrt_two, 0.0},
                                QComplex{inverse_sqrt_two, 0.0},
                                QComplex{inverse_sqrt_two, 0.0},
                                QComplex{-inverse_sqrt_two, 0.0}};
                            (void)graph.add_dense_factor(scope, values);
                        }
                        current_variable[first] = static_cast<std::size_t>(next);
                    } else if (is_first) {
                        const double output_one = input_bits_[first] == 0U
                            ? inverse_sqrt_two
                            : -inverse_sqrt_two;
                        multiply_output_phase(
                            first,
                            {inverse_sqrt_two, 0.0},
                            {output_one, 0.0});
                        current_variable[first] = npos;
                    } else {
                        const std::array<FactorVariableId, 1> scope{
                            static_cast<FactorVariableId>(previous)};
                        const std::array<QComplex, 2> placeholder{
                            QComplex{inverse_sqrt_two, 0.0},
                            QComplex{inverse_sqrt_two, 0.0}};
                        const FactorId factor = graph.add_dense_factor(scope, placeholder);
                        bindings_.push_back(OutputBinding{
                            factor,
                            first,
                            {QComplex{inverse_sqrt_two, 0.0},
                             QComplex{inverse_sqrt_two, 0.0}},
                            {QComplex{inverse_sqrt_two, 0.0},
                             QComplex{-inverse_sqrt_two, 0.0}},
                        });
                        current_variable[first] = npos;
                    }
                    before_first[first] = 0U;
                    ++seen_h[first];
                    break;
                }
                case OperationCode::Z:
                    add_diagonal(first, {1.0, 0.0}, {-1.0, 0.0});
                    break;
                case OperationCode::S:
                    add_diagonal(first, {1.0, 0.0}, QI);
                    break;
                case OperationCode::Sdg:
                    add_diagonal(first, {1.0, 0.0}, -QI);
                    break;
                case OperationCode::T:
                    add_diagonal(
                        first,
                        {1.0, 0.0},
                        QComplex::from_polar(
                            1.0, 0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Tdg:
                    add_diagonal(
                        first,
                        {1.0, 0.0},
                        QComplex::from_polar(
                            1.0, -0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Rz:
                    add_diagonal(
                        first,
                        QComplex::from_polar(1.0, -0.5 * operation.parameter),
                        QComplex::from_polar(1.0, 0.5 * operation.parameter));
                    break;
                case OperationCode::Cz:
                    add_cz(first, static_cast<std::size_t>(operation.second));
                    break;
                default:
                    throw QStateError(
                        "Operation left the validated basis Hpath diagonal contract");
            }
        }

        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            if (seen_h[qubit] != h_counts_[qubit]) {
                throw QStateError("Basis Hpath H-event accounting is inconsistent");
            }
            if (h_counts_[qubit] != 0U && current_variable[qubit] != npos) {
                throw QStateError("Basis Hpath final hidden segment was not closed");
            }
        }

        ExactFactorStats factor_stats{};
        std::size_t plan_bytes = 0U;
        if (graph.variable_count() != 0U) {
            plan_.emplace(graph);
            factor_stats = plan_->stats();
            plan_bytes = plan_->estimated_bytes();
        }
        const std::size_t metadata_bytes = estimate_metadata_bytes();
        if (metadata_bytes > config_.max_metadata_bytes) {
            throw QStateError("Basis Hpath metadata exceeds configured byte cap");
        }
        stats_ = ExactPreparedBasisHadamardPathStats{
            qubit_count_,
            operation_count_,
            h_events_,
            h_active_qubits_,
            h_active_qubits_,
            graph.variable_count(),
            graph.factor_count(),
            bindings_.size(),
            output_cz_.size(),
            qubit_count_ - h_active_qubits_,
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
            throw QStateError("Basis Hpath dimensions or configuration are invalid");
        }
    }

    void validate_input_bits() const {
        for (const std::uint8_t bit : input_bits_) {
            if (bit > 1U) {
                throw QStateError("Basis Hpath input bits must be zero or one");
            }
        }
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Basis Hpath operation target is out of range");
        }
        if (operation.code == OperationCode::Cz) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Basis Hpath CZ support is invalid");
            }
        }
        if (operation.code == OperationCode::Rz && !std::isfinite(operation.parameter)) {
            throw QStateError("Basis Hpath Rz angle must be finite");
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
                    "Operation is outside the exact basis Hpath diagonal contract");
        }
    }

    void validate_output_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Basis Hpath output bit count does not match qubit count");
        }
        for (const std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Basis Hpath output bits must be zero or one");
            }
        }
    }

    void validate_workspace(const ExactPreparedBasisHadamardPathWorkspace& workspace) const {
        if (workspace.owner_ != this) {
            throw QStateError("Basis Hpath workspace belongs to a different plan");
        }
        if (plan_.has_value() != workspace.factor_workspace_active_) {
            throw QStateError("Basis Hpath workspace shape is inconsistent");
        }
    }

    void multiply_output_phase(std::size_t qubit, QComplex zero, QComplex one) {
        output_phase_[qubit][0] *= zero;
        output_phase_[qubit][1] *= one;
        if (!finite(output_phase_[qubit][0]) || !finite(output_phase_[qubit][1])) {
            throw QStateError("Basis Hpath output phase became non-finite");
        }
    }

    [[nodiscard]] ExactHadamardPathAmplitude zero_result() const {
        ExactFactorStats factor_stats{};
        if (plan_.has_value()) {
            factor_stats = plan_->stats();
        }
        return ExactHadamardPathAmplitude{
            QComplex{},
            0.0,
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

    [[nodiscard]] std::size_t estimate_metadata_bytes() const {
        std::size_t total = sizeof(*this);
        total = checked_sum(total, checked_product(
            input_bits_.capacity(), sizeof(std::uint8_t),
            "Basis Hpath input metadata overflowed"));
        total = checked_sum(total, checked_product(
            h_counts_.capacity(), sizeof(std::size_t),
            "Basis Hpath H-count metadata overflowed"));
        total = checked_sum(total, checked_product(
            output_phase_.capacity(), sizeof(std::array<QComplex, 2>),
            "Basis Hpath output-phase metadata overflowed"));
        total = checked_sum(total, checked_product(
            bindings_.capacity(), sizeof(OutputBinding),
            "Basis Hpath binding metadata overflowed"));
        total = checked_sum(total, checked_product(
            output_cz_.capacity(), sizeof(std::pair<std::size_t, std::size_t>),
            "Basis Hpath output-CZ metadata overflowed"));
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
        const char* message = "Basis Hpath metadata overflowed") {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }

    friend class ExactPreparedBasisHadamardPathWorkspace;
};

inline ExactPreparedBasisHadamardPathWorkspace::ExactPreparedBasisHadamardPathWorkspace(
    const ExactPreparedBasisHadamardPathPlan& owner)
    : owner_(&owner) {
    if (owner.plan_.has_value()) {
        factor_workspace_ = owner.plan_->workspace();
        factor_workspace_active_ = true;
    }
}

}  // namespace qubit
