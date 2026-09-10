#pragma once

#include "qubit/qfactor.hpp"
#include "qubit/qplan.hpp"

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

struct ExactHadamardBornConfig {
    ExactFactorConfig factor{};
    std::size_t max_qubits{1U << 20U};
    std::size_t max_operations{1U << 24U};
    std::size_t max_h_events{1U << 20U};
    std::size_t max_retained_qubits{20U};
};

struct ExactHadamardBornStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t h_events{0U};
    std::size_t h_active_qubits{0U};
    std::size_t born_variables{0U};
    std::size_t factor_count{0U};
    std::size_t retained_qubits{0U};
    std::size_t output_entries{0U};
    std::size_t graph_estimated_bytes{0U};
    std::size_t plan_estimated_bytes{0U};
    ExactFactorStats factor{};
};

class ExactHadamardBornMarginalPlan;

class ExactHadamardBornWorkspace {
public:
    ExactHadamardBornWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return factor_.estimated_bytes();
    }

private:
    const ExactHadamardBornMarginalPlan* owner_{nullptr};
    ExactFactorWorkspace factor_{};

    explicit ExactHadamardBornWorkspace(const ExactHadamardBornMarginalPlan& owner);

    friend class ExactHadamardBornMarginalPlan;
};

class ExactHadamardBornMarginalPlan {
public:
    ExactHadamardBornMarginalPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const std::size_t> retained_qubits,
        ExactHadamardBornConfig config = {})
        : qubit_count_(qubit_count),
          operation_count_(operations.size()),
          retained_qubits_(retained_qubits.begin(), retained_qubits.end()),
          config_(config) {
        validate_configuration();
        validate_retained_qubits();
        build(operations);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::span<const std::size_t> retained_qubits() const noexcept {
        return retained_qubits_;
    }
    [[nodiscard]] const ExactHadamardBornConfig& config() const noexcept { return config_; }
    [[nodiscard]] const ExactHadamardBornStats& stats() const noexcept { return stats_; }

    [[nodiscard]] ExactHadamardBornWorkspace workspace() const {
        return ExactHadamardBornWorkspace(*this);
    }

    [[nodiscard]] std::vector<QComplex> marginal(
        ExactHadamardBornWorkspace& workspace) const {
        validate_workspace(workspace);
        std::vector<QComplex> result = plan_->evaluate(workspace.factor_);
        if (result.size() != stats_.output_entries) {
            throw QStateError("Hadamard Born marginal output size changed");
        }
        for (const QComplex& value : result) {
            if (!finite(value)) {
                throw QStateError("Hadamard Born marginal became non-finite");
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactHadamardBornWorkspace& workspace) const {
        validate_workspace(workspace);
        std::vector<QComplex> result = plan_->normalized_marginal(workspace.factor_);
        if (result.size() != stats_.output_entries) {
            throw QStateError("Hadamard Born normalized marginal output size changed");
        }
        return result;
    }

private:
    std::size_t qubit_count_{0U};
    std::size_t operation_count_{0U};
    std::vector<std::size_t> retained_qubits_{};
    ExactHadamardBornConfig config_{};
    std::optional<ExactFactorPlan> plan_{};
    ExactHadamardBornStats stats_{};

    void build(std::span<const Operation> operations) {
        const std::size_t npos = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> first(qubit_count_, npos);
        std::vector<std::size_t> last(qubit_count_, npos);
        std::vector<std::size_t> next;
        std::vector<std::size_t> operation_event(operation_count_, npos);
        next.reserve(std::min(operation_count_, config_.max_h_events));

        std::size_t h_events = 0U;
        std::size_t h_active_qubits = 0U;
        for (std::size_t operation_index = 0U;
             operation_index < operation_count_;
             ++operation_index) {
            const Operation& operation = operations[operation_index];
            validate_operation(operation);
            if (operation.code != OperationCode::H) {
                continue;
            }
            if (h_events >= config_.max_h_events) {
                throw QStateError("Hadamard Born compiler exceeds its H-event cap");
            }
            const std::size_t qubit = static_cast<std::size_t>(operation.first);
            const std::size_t event = h_events++;
            operation_event[operation_index] = event;
            next.push_back(npos);
            if (last[qubit] == npos) {
                first[qubit] = event;
                ++h_active_qubits;
            } else {
                next[last[qubit]] = event;
            }
            last[qubit] = event;
        }

        const std::size_t hidden_variables = checked_product(
            h_events, 2U, "Hadamard Born hidden-variable count overflowed");
        const std::size_t born_variables = checked_sum(
            hidden_variables, qubit_count_, "Hadamard Born variable count overflowed");
        if (born_variables > config_.factor.max_variables) {
            throw QStateError("Hadamard Born graph exceeds configured variable cap");
        }

        ExactFactorGraph graph(config_.factor);
        std::vector<FactorVariableId> ket(h_events);
        std::vector<FactorVariableId> bra(h_events);
        for (std::size_t event = 0U; event < h_events; ++event) {
            ket[event] = graph.add_variable(2U);
            bra[event] = graph.add_variable(2U);
        }
        std::vector<FactorVariableId> output(qubit_count_);
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            output[qubit] = graph.add_variable(2U);
        }

        const std::array<QComplex, 2> uniform_probability{
            QComplex{0.5, 0.0}, QComplex{0.5, 0.0}};
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            if (first[qubit] == npos) {
                const std::array<FactorVariableId, 1> scope{output[qubit]};
                (void)graph.add_dense_factor(scope, uniform_probability);
            }
        }

        std::vector<std::size_t> current = first;
        const std::array<QComplex, 4> cz_values{
            QComplex{1.0, 0.0}, QComplex{1.0, 0.0},
            QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}};

        const auto add_unary_pair = [&](QubitId qubit, QComplex zero, QComplex one) {
            const std::size_t event = current[static_cast<std::size_t>(qubit)];
            if (event == npos) {
                return;
            }
            const std::array<FactorVariableId, 1> ket_scope{ket[event]};
            const std::array<FactorVariableId, 1> bra_scope{bra[event]};
            const std::array<QComplex, 2> ket_values{zero, one};
            const std::array<QComplex, 2> bra_values{zero.conjugate(), one.conjugate()};
            (void)graph.add_dense_factor(ket_scope, ket_values);
            (void)graph.add_dense_factor(bra_scope, bra_values);
        };
        const auto add_cz_pair = [&](QubitId first_qubit, QubitId second_qubit) {
            const std::size_t first_index = static_cast<std::size_t>(first_qubit);
            const std::size_t second_index = static_cast<std::size_t>(second_qubit);
            const std::size_t first_event = current[first_index];
            const std::size_t second_event = current[second_index];
            if (first_event == npos && second_event == npos) {
                return;
            }
            if (first_event != npos && second_event != npos) {
                const std::array<FactorVariableId, 2> ket_scope{
                    ket[first_event], ket[second_event]};
                const std::array<FactorVariableId, 2> bra_scope{
                    bra[first_event], bra[second_event]};
                (void)graph.add_dense_factor(ket_scope, cz_values);
                (void)graph.add_dense_factor(bra_scope, cz_values);
                return;
            }
            const std::size_t active_event =
                first_event != npos ? first_event : second_event;
            const std::size_t fixed_qubit =
                first_event != npos ? second_index : first_index;
            const std::array<FactorVariableId, 2> ket_scope{
                ket[active_event], output[fixed_qubit]};
            const std::array<FactorVariableId, 2> bra_scope{
                bra[active_event], output[fixed_qubit]};
            (void)graph.add_dense_factor(ket_scope, cz_values);
            (void)graph.add_dense_factor(bra_scope, cz_values);
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
                        throw QStateError("Hadamard Born event sequence is inconsistent");
                    }
                    const std::size_t following = next[event];
                    const double normalization =
                        event == first[qubit] ? 0.5 : inverse_sqrt_two;
                    const std::array<QComplex, 4> values{
                        QComplex{normalization, 0.0},
                        QComplex{normalization, 0.0},
                        QComplex{normalization, 0.0},
                        QComplex{-normalization, 0.0},
                    };
                    if (following != npos) {
                        const std::array<FactorVariableId, 2> ket_scope{
                            ket[event], ket[following]};
                        const std::array<FactorVariableId, 2> bra_scope{
                            bra[event], bra[following]};
                        (void)graph.add_dense_factor(ket_scope, values);
                        (void)graph.add_dense_factor(bra_scope, values);
                    } else {
                        const std::array<FactorVariableId, 2> ket_scope{
                            ket[event], output[qubit]};
                        const std::array<FactorVariableId, 2> bra_scope{
                            bra[event], output[qubit]};
                        (void)graph.add_dense_factor(ket_scope, values);
                        (void)graph.add_dense_factor(bra_scope, values);
                    }
                    current[qubit] = following;
                    break;
                }
                case OperationCode::Z:
                    add_unary_pair(operation.first, {1.0, 0.0}, {-1.0, 0.0});
                    break;
                case OperationCode::S:
                    add_unary_pair(operation.first, {1.0, 0.0}, QI);
                    break;
                case OperationCode::Sdg:
                    add_unary_pair(operation.first, {1.0, 0.0}, -QI);
                    break;
                case OperationCode::T:
                    add_unary_pair(
                        operation.first,
                        {1.0, 0.0},
                        QComplex::from_polar(1.0, 0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Tdg:
                    add_unary_pair(
                        operation.first,
                        {1.0, 0.0},
                        QComplex::from_polar(1.0, -0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Rz:
                    add_unary_pair(
                        operation.first,
                        QComplex::from_polar(1.0, -0.5 * operation.parameter),
                        QComplex::from_polar(1.0, 0.5 * operation.parameter));
                    break;
                case OperationCode::Cz:
                    add_cz_pair(operation.first, operation.second);
                    break;
                default:
                    throw QStateError("Operation left the validated Hadamard Born contract");
            }
        }

        std::vector<FactorVariableId> retained_variables;
        retained_variables.reserve(retained_qubits_.size());
        for (const std::size_t qubit : retained_qubits_) {
            retained_variables.push_back(output[qubit]);
        }
        plan_.emplace(graph, retained_variables);
        const std::size_t expected_output_entries = retained_output_entries();
        if (plan_->stats().output_entries != expected_output_entries) {
            throw QStateError("Hadamard Born compiled output shape differs from retained qubits");
        }
        stats_ = ExactHadamardBornStats{
            qubit_count_,
            operation_count_,
            h_events,
            h_active_qubits,
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
            config_.max_h_events == 0U || config_.max_retained_qubits > config_.factor.max_variables) {
            throw QStateError("Hadamard Born dimensions or configuration are invalid");
        }
    }

    void validate_retained_qubits() const {
        if (retained_qubits_.size() > config_.max_retained_qubits) {
            throw QStateError("Hadamard Born retained-qubit count exceeds configured cap");
        }
        std::vector<std::uint8_t> seen(qubit_count_, 0U);
        for (const std::size_t qubit : retained_qubits_) {
            if (qubit >= qubit_count_) {
                throw QStateError("Hadamard Born retained qubit is out of range");
            }
            if (seen[qubit] != 0U) {
                throw QStateError("Hadamard Born retained qubits contain duplicates");
            }
            seen[qubit] = 1U;
        }
        (void)retained_output_entries();
    }

    [[nodiscard]] std::size_t retained_output_entries() const {
        constexpr std::size_t bits = std::numeric_limits<std::size_t>::digits;
        if (retained_qubits_.size() >= bits) {
            throw QStateError("Hadamard Born retained output size overflowed");
        }
        const std::size_t entries = std::size_t{1U} << retained_qubits_.size();
        if (entries > config_.factor.max_factor_entries) {
            throw QStateError("Hadamard Born retained output exceeds factor-entry cap");
        }
        return entries;
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Hadamard Born operation target is out of range");
        }
        if (operation.code == OperationCode::Cz) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Hadamard Born CZ support is invalid");
            }
        }
        if (operation.code == OperationCode::Rz && !std::isfinite(operation.parameter)) {
            throw QStateError("Hadamard Born Rz angle must be finite");
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
                throw QStateError("Operation is outside the exact Hadamard Born contract");
        }
    }

    void validate_workspace(const ExactHadamardBornWorkspace& workspace) const {
        if (workspace.owner_ != this) {
            throw QStateError("Hadamard Born workspace belongs to a different plan");
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

    friend class ExactHadamardBornWorkspace;
};

inline ExactHadamardBornWorkspace::ExactHadamardBornWorkspace(
    const ExactHadamardBornMarginalPlan& owner)
    : owner_(&owner), factor_(owner.plan_->workspace()) {}

}  // namespace qubit
