#pragma once

#include "qubit/qhpath_born.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace qubit {

struct ExactBasisHadamardBornStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t h_events{0U};
    std::size_t h_active_qubits{0U};
    std::size_t hidden_variables_per_side{0U};
    std::size_t removed_first_h_variables_per_side{0U};
    std::size_t physical_output_variables{0U};
    std::size_t retained_fixed_support_qubits{0U};
    std::size_t born_variables{0U};
    std::size_t factor_count{0U};
    std::size_t retained_qubits{0U};
    std::size_t output_entries{0U};
    std::size_t graph_estimated_bytes{0U};
    std::size_t plan_estimated_bytes{0U};
    ExactFactorStats factor{};
};

class ExactBasisHadamardBornMarginalPlan;

class ExactBasisHadamardBornWorkspace {
public:
    ExactBasisHadamardBornWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return factor_.estimated_bytes();
    }

private:
    const ExactBasisHadamardBornMarginalPlan* owner_{nullptr};
    ExactFactorWorkspace factor_{};

    explicit ExactBasisHadamardBornWorkspace(
        const ExactBasisHadamardBornMarginalPlan& owner);

    friend class ExactBasisHadamardBornMarginalPlan;
};

class ExactBasisHadamardBornMarginalPlan {
public:
    ExactBasisHadamardBornMarginalPlan(
        std::span<const std::uint8_t> input_bits,
        std::span<const Operation> operations,
        std::span<const std::size_t> retained_qubits,
        ExactHadamardBornConfig config = {})
        : qubit_count_(input_bits.size()),
          operation_count_(operations.size()),
          input_bits_(input_bits.begin(), input_bits.end()),
          retained_qubits_(retained_qubits.begin(), retained_qubits.end()),
          config_(config) {
        validate_configuration();
        validate_input_bits();
        validate_retained_qubits();
        build(operations);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::span<const std::uint8_t> input_bits() const noexcept {
        return input_bits_;
    }
    [[nodiscard]] std::span<const std::size_t> retained_qubits() const noexcept {
        return retained_qubits_;
    }
    [[nodiscard]] const ExactHadamardBornConfig& config() const noexcept { return config_; }
    [[nodiscard]] const ExactBasisHadamardBornStats& stats() const noexcept { return stats_; }

    [[nodiscard]] ExactBasisHadamardBornWorkspace workspace() const {
        return ExactBasisHadamardBornWorkspace(*this);
    }

    [[nodiscard]] std::vector<QComplex> marginal(
        ExactBasisHadamardBornWorkspace& workspace) const {
        validate_workspace(workspace);
        std::vector<QComplex> result = plan_->evaluate(workspace.factor_);
        validate_result(result, "Basis Hadamard Born marginal");
        return result;
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactBasisHadamardBornWorkspace& workspace) const {
        validate_workspace(workspace);
        std::vector<QComplex> result = plan_->normalized_marginal(workspace.factor_);
        validate_result(result, "Basis Hadamard Born normalized marginal");
        return result;
    }

private:
    enum class SegmentState : std::uint8_t {
        InputFixed = 0U,
        Hidden = 1U,
        OutputFixed = 2U,
    };

    std::size_t qubit_count_{0U};
    std::size_t operation_count_{0U};
    std::vector<std::uint8_t> input_bits_{};
    std::vector<std::size_t> retained_qubits_{};
    ExactHadamardBornConfig config_{};
    std::optional<ExactFactorPlan> plan_{};
    ExactBasisHadamardBornStats stats_{};

    void build(std::span<const Operation> operations) {
        const std::size_t npos = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> h_counts(qubit_count_, 0U);
        std::size_t h_events = 0U;
        std::size_t h_active_qubits = 0U;
        for (const Operation& operation : operations) {
            validate_operation(operation);
            if (operation.code != OperationCode::H) {
                continue;
            }
            if (h_events >= config_.max_h_events) {
                throw QStateError("Basis Hadamard Born compiler exceeds its H-event cap");
            }
            ++h_events;
            const std::size_t qubit = static_cast<std::size_t>(operation.first);
            if (h_counts[qubit] == 0U) {
                ++h_active_qubits;
            }
            ++h_counts[qubit];
        }

        std::vector<std::uint8_t> retained_mask(qubit_count_, 0U);
        std::size_t retained_fixed_support = 0U;
        for (const std::size_t qubit : retained_qubits_) {
            retained_mask[qubit] = 1U;
            if (h_counts[qubit] == 0U) {
                ++retained_fixed_support;
            }
        }

        std::size_t hidden_per_side = 0U;
        for (const std::size_t count : h_counts) {
            if (count != 0U) {
                hidden_per_side = checked_sum(
                    hidden_per_side,
                    count - 1U,
                    "Basis Hadamard Born hidden-variable count overflowed");
            }
        }
        const std::size_t doubled_hidden = checked_product(
            hidden_per_side, 2U,
            "Basis Hadamard Born doubled hidden-variable count overflowed");
        const std::size_t physical_outputs = checked_sum(
            h_active_qubits,
            retained_fixed_support,
            "Basis Hadamard Born physical-output count overflowed");
        const std::size_t expected_variables = checked_sum(
            doubled_hidden,
            physical_outputs,
            "Basis Hadamard Born total variable count overflowed");
        if (expected_variables > config_.factor.max_variables) {
            throw QStateError("Basis Hadamard Born graph exceeds configured variable cap");
        }

        ExactFactorGraph graph(config_.factor);
        std::vector<std::size_t> output_variable(qubit_count_, npos);
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            if (h_counts[qubit] != 0U || retained_mask[qubit] != 0U) {
                output_variable[qubit] = static_cast<std::size_t>(graph.add_variable(2U));
            }
        }

        for (const std::size_t qubit : retained_qubits_) {
            if (h_counts[qubit] != 0U) {
                continue;
            }
            const std::array<FactorVariableId, 1> scope{
                static_cast<FactorVariableId>(output_variable[qubit])};
            const std::array<QComplex, 2> delta = input_bits_[qubit] == 0U
                ? std::array<QComplex, 2>{QComplex{1.0, 0.0}, QComplex{0.0, 0.0}}
                : std::array<QComplex, 2>{QComplex{0.0, 0.0}, QComplex{1.0, 0.0}};
            (void)graph.add_dense_factor(scope, delta);
        }

        std::vector<std::size_t> seen_h(qubit_count_, 0U);
        std::vector<std::size_t> current_ket(qubit_count_, npos);
        std::vector<std::size_t> current_bra(qubit_count_, npos);
        std::vector<std::uint8_t> before_first(qubit_count_, 1U);

        const auto state = [&](std::size_t qubit) {
            if (before_first[qubit] != 0U) {
                return SegmentState::InputFixed;
            }
            return current_ket[qubit] == npos
                ? SegmentState::OutputFixed
                : SegmentState::Hidden;
        };

        const auto add_unary_pair = [&](std::size_t qubit, QComplex zero, QComplex one) {
            if (state(qubit) != SegmentState::Hidden) {
                return;
            }
            const std::array<FactorVariableId, 1> ket_scope{
                static_cast<FactorVariableId>(current_ket[qubit])};
            const std::array<FactorVariableId, 1> bra_scope{
                static_cast<FactorVariableId>(current_bra[qubit])};
            const std::array<QComplex, 2> ket_values{zero, one};
            const std::array<QComplex, 2> bra_values{zero.conjugate(), one.conjugate()};
            (void)graph.add_dense_factor(ket_scope, ket_values);
            (void)graph.add_dense_factor(bra_scope, bra_values);
        };

        const std::array<QComplex, 4> cz_values{
            QComplex{1.0, 0.0}, QComplex{1.0, 0.0},
            QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}};

        const auto add_hidden_output_cz = [&](std::size_t hidden, std::size_t output) {
            if (output_variable[output] == npos) {
                throw QStateError("Basis Hadamard Born output variable is missing");
            }
            const std::array<FactorVariableId, 2> ket_scope{
                static_cast<FactorVariableId>(current_ket[hidden]),
                static_cast<FactorVariableId>(output_variable[output])};
            const std::array<FactorVariableId, 2> bra_scope{
                static_cast<FactorVariableId>(current_bra[hidden]),
                static_cast<FactorVariableId>(output_variable[output])};
            (void)graph.add_dense_factor(ket_scope, cz_values);
            (void)graph.add_dense_factor(bra_scope, cz_values);
        };

        const auto add_cz_pair = [&](std::size_t first, std::size_t second) {
            const SegmentState first_state = state(first);
            const SegmentState second_state = state(second);
            if (first_state == SegmentState::Hidden &&
                second_state == SegmentState::Hidden) {
                const std::array<FactorVariableId, 2> ket_scope{
                    static_cast<FactorVariableId>(current_ket[first]),
                    static_cast<FactorVariableId>(current_ket[second])};
                const std::array<FactorVariableId, 2> bra_scope{
                    static_cast<FactorVariableId>(current_bra[first]),
                    static_cast<FactorVariableId>(current_bra[second])};
                (void)graph.add_dense_factor(ket_scope, cz_values);
                (void)graph.add_dense_factor(bra_scope, cz_values);
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
                        add_unary_pair(hidden, {1.0, 0.0}, {-1.0, 0.0});
                    }
                } else {
                    add_hidden_output_cz(hidden, other);
                }
            }
        };

        constexpr double inverse_sqrt_two =
            0.707106781186547524400844362104849039;
        const std::array<QComplex, 4> h_values{
            QComplex{inverse_sqrt_two, 0.0},
            QComplex{inverse_sqrt_two, 0.0},
            QComplex{inverse_sqrt_two, 0.0},
            QComplex{-inverse_sqrt_two, 0.0}};

        for (const Operation& operation : operations) {
            const std::size_t first = static_cast<std::size_t>(operation.first);
            switch (operation.code) {
                case OperationCode::H: {
                    const bool is_first = seen_h[first] == 0U;
                    const bool is_last = seen_h[first] + 1U == h_counts[first];
                    const std::size_t previous_ket = current_ket[first];
                    const std::size_t previous_bra = current_bra[first];
                    if (is_first &&
                        (previous_ket != npos || previous_bra != npos || before_first[first] == 0U)) {
                        throw QStateError("Basis Hadamard Born first-H state is inconsistent");
                    }
                    if (!is_first && (previous_ket == npos || previous_bra == npos)) {
                        throw QStateError("Basis Hadamard Born hidden segment was lost before H");
                    }

                    if (!is_last) {
                        const FactorVariableId next_ket = graph.add_variable(2U);
                        const FactorVariableId next_bra = graph.add_variable(2U);
                        if (is_first) {
                            const double one = input_bits_[first] == 0U
                                ? inverse_sqrt_two
                                : -inverse_sqrt_two;
                            const std::array<FactorVariableId, 1> ket_scope{next_ket};
                            const std::array<FactorVariableId, 1> bra_scope{next_bra};
                            const std::array<QComplex, 2> values{
                                QComplex{inverse_sqrt_two, 0.0}, QComplex{one, 0.0}};
                            (void)graph.add_dense_factor(ket_scope, values);
                            (void)graph.add_dense_factor(bra_scope, values);
                        } else {
                            const std::array<FactorVariableId, 2> ket_scope{
                                static_cast<FactorVariableId>(previous_ket), next_ket};
                            const std::array<FactorVariableId, 2> bra_scope{
                                static_cast<FactorVariableId>(previous_bra), next_bra};
                            (void)graph.add_dense_factor(ket_scope, h_values);
                            (void)graph.add_dense_factor(bra_scope, h_values);
                        }
                        current_ket[first] = static_cast<std::size_t>(next_ket);
                        current_bra[first] = static_cast<std::size_t>(next_bra);
                    } else if (is_first) {
                        const std::array<FactorVariableId, 1> scope{
                            static_cast<FactorVariableId>(output_variable[first])};
                        const std::array<QComplex, 2> probability{
                            QComplex{0.5, 0.0}, QComplex{0.5, 0.0}};
                        (void)graph.add_dense_factor(scope, probability);
                        current_ket[first] = npos;
                        current_bra[first] = npos;
                    } else {
                        const FactorVariableId output =
                            static_cast<FactorVariableId>(output_variable[first]);
                        const std::array<FactorVariableId, 2> ket_scope{
                            static_cast<FactorVariableId>(previous_ket), output};
                        const std::array<FactorVariableId, 2> bra_scope{
                            static_cast<FactorVariableId>(previous_bra), output};
                        (void)graph.add_dense_factor(ket_scope, h_values);
                        (void)graph.add_dense_factor(bra_scope, h_values);
                        current_ket[first] = npos;
                        current_bra[first] = npos;
                    }
                    before_first[first] = 0U;
                    ++seen_h[first];
                    break;
                }
                case OperationCode::Z:
                    add_unary_pair(first, {1.0, 0.0}, {-1.0, 0.0});
                    break;
                case OperationCode::S:
                    add_unary_pair(first, {1.0, 0.0}, QI);
                    break;
                case OperationCode::Sdg:
                    add_unary_pair(first, {1.0, 0.0}, -QI);
                    break;
                case OperationCode::T:
                    add_unary_pair(
                        first,
                        {1.0, 0.0},
                        QComplex::from_polar(
                            1.0, 0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Tdg:
                    add_unary_pair(
                        first,
                        {1.0, 0.0},
                        QComplex::from_polar(
                            1.0, -0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Rz:
                    add_unary_pair(
                        first,
                        QComplex::from_polar(1.0, -0.5 * operation.parameter),
                        QComplex::from_polar(1.0, 0.5 * operation.parameter));
                    break;
                case OperationCode::Cz:
                    add_cz_pair(first, static_cast<std::size_t>(operation.second));
                    break;
                default:
                    throw QStateError(
                        "Operation left the validated basis Hadamard Born contract");
            }
        }

        if (graph.variable_count() != expected_variables) {
            throw QStateError("Basis Hadamard Born variable accounting changed during build");
        }

        std::vector<FactorVariableId> retained_variables;
        retained_variables.reserve(retained_qubits_.size());
        for (const std::size_t qubit : retained_qubits_) {
            if (output_variable[qubit] == npos) {
                throw QStateError("Basis Hadamard Born retained output variable is missing");
            }
            retained_variables.push_back(
                static_cast<FactorVariableId>(output_variable[qubit]));
        }
        plan_.emplace(graph, retained_variables);
        const std::size_t expected_output_entries = retained_output_entries();
        if (plan_->stats().output_entries != expected_output_entries) {
            throw QStateError(
                "Basis Hadamard Born compiled output shape differs from retained qubits");
        }

        stats_ = ExactBasisHadamardBornStats{
            qubit_count_,
            operation_count_,
            h_events,
            h_active_qubits,
            hidden_per_side,
            h_active_qubits,
            physical_outputs,
            retained_fixed_support,
            graph.variable_count(),
            graph.factor_count(),
            retained_qubits_.size(),
            expected_output_entries,
            graph.estimated_bytes(),
            plan_->estimated_bytes(),
            plan_->stats(),
        };
    }

    void validate_configuration() const {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operation_count_ > config_.max_operations ||
            config_.max_h_events == 0U ||
            config_.max_retained_qubits > config_.factor.max_variables) {
            throw QStateError("Basis Hadamard Born dimensions or configuration are invalid");
        }
    }

    void validate_input_bits() const {
        for (const std::uint8_t bit : input_bits_) {
            if (bit > 1U) {
                throw QStateError("Basis Hadamard Born input bits must be zero or one");
            }
        }
    }

    void validate_retained_qubits() const {
        if (retained_qubits_.size() > config_.max_retained_qubits) {
            throw QStateError("Basis Hadamard Born retained-qubit count exceeds configured cap");
        }
        std::vector<std::uint8_t> seen(qubit_count_, 0U);
        for (const std::size_t qubit : retained_qubits_) {
            if (qubit >= qubit_count_) {
                throw QStateError("Basis Hadamard Born retained qubit is out of range");
            }
            if (seen[qubit] != 0U) {
                throw QStateError("Basis Hadamard Born retained qubits contain duplicates");
            }
            seen[qubit] = 1U;
        }
        (void)retained_output_entries();
    }

    [[nodiscard]] std::size_t retained_output_entries() const {
        constexpr std::size_t bits = std::numeric_limits<std::size_t>::digits;
        if (retained_qubits_.size() >= bits) {
            throw QStateError("Basis Hadamard Born retained output size overflowed");
        }
        const std::size_t entries = std::size_t{1U} << retained_qubits_.size();
        if (entries > config_.factor.max_factor_entries) {
            throw QStateError("Basis Hadamard Born retained output exceeds factor-entry cap");
        }
        return entries;
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Basis Hadamard Born operation target is out of range");
        }
        if (operation.code == OperationCode::Cz) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Basis Hadamard Born CZ support is invalid");
            }
        }
        if (operation.code == OperationCode::Rz && !std::isfinite(operation.parameter)) {
            throw QStateError("Basis Hadamard Born Rz angle must be finite");
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
                    "Operation is outside the exact basis Hadamard Born contract");
        }
    }

    void validate_workspace(const ExactBasisHadamardBornWorkspace& workspace) const {
        if (workspace.owner_ != this) {
            throw QStateError("Basis Hadamard Born workspace belongs to a different plan");
        }
    }

    void validate_result(
        std::span<const QComplex> result,
        const char* label) const {
        if (result.size() != stats_.output_entries) {
            throw QStateError(std::string(label) + " output size changed");
        }
        for (const QComplex& value : result) {
            if (!finite(value)) {
                throw QStateError(std::string(label) + " became non-finite");
            }
        }
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
        const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }

    friend class ExactBasisHadamardBornWorkspace;
};

inline ExactBasisHadamardBornWorkspace::ExactBasisHadamardBornWorkspace(
    const ExactBasisHadamardBornMarginalPlan& owner)
    : owner_(&owner), factor_(owner.plan_->workspace()) {}

}  // namespace qubit
