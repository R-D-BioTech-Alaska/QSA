#pragma once

#include "qubit/qfactor.hpp"
#include "qubit/qplan.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace qubit {

struct ExactSingleHPathConfig {
    ExactFactorConfig factor{};
    std::size_t max_qubits{1U << 20U};
    std::size_t max_operations{1U << 24U};
    std::size_t max_h_defects{1U << 20U};
};

struct ExactSingleHPathAmplitude {
    QComplex mantissa{};
    double log2_scale{0.0};
    std::size_t h_defects{0U};
    std::size_t factor_variables{0U};
    std::size_t factor_count{0U};
    std::size_t graph_estimated_bytes{0U};
    std::size_t plan_estimated_bytes{0U};
    ExactFactorStats factor_stats{};

    [[nodiscard]] QComplex amplitude() const {
        const double scale = std::exp2(log2_scale);
        if (scale == 0.0 && mantissa.norm2() != 0.0) {
            throw QStateError(
                "Single-H path amplitude underflows; use scaled result or log2_probability");
        }
        return mantissa * scale;
    }

    [[nodiscard]] double log2_probability() const {
        const double value = mantissa.norm2();
        if (!std::isfinite(value)) {
            throw QStateError("Single-H path probability became non-finite");
        }
        if (value == 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        return std::log2(value) + 2.0 * log2_scale;
    }
};

class ExactSingleHPathAmplitudePlan {
public:
    ExactSingleHPathAmplitudePlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactSingleHPathConfig config = {})
        : qubit_count_(qubit_count),
          operations_(operations.begin(), operations.end()),
          config_(config),
          has_h_(qubit_count, 0U) {
        validate_configuration();
        for (const Operation& operation : operations_) {
            validate_operation(operation);
            if (operation.code == OperationCode::H) {
                const std::size_t qubit = static_cast<std::size_t>(operation.first);
                if (has_h_[qubit] != 0U) {
                    throw QStateError(
                        "Single-H path compiler accepts at most one Hadamard per qubit");
                }
                has_h_[qubit] = 1U;
                ++h_defects_;
            }
        }
        if (h_defects_ > config_.max_h_defects ||
            h_defects_ > config_.factor.max_variables) {
            throw QStateError("Single-H path compiler exceeds its Hadamard-variable cap");
        }
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operations_.size(); }
    [[nodiscard]] std::size_t h_defects() const noexcept { return h_defects_; }
    [[nodiscard]] const ExactSingleHPathConfig& config() const noexcept { return config_; }

    [[nodiscard]] ExactSingleHPathAmplitude scaled_amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        validate_bits(bits);

        ExactFactorGraph graph(config_.factor);
        const FactorVariableId invalid = std::numeric_limits<FactorVariableId>::max();
        std::vector<FactorVariableId> variables(qubit_count_, invalid);
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            if (has_h_[qubit] != 0U) {
                variables[qubit] = graph.add_variable(2U);
            }
        }

        std::vector<std::uint8_t> passed_h(qubit_count_, 0U);
        QComplex global{1.0, 0.0};
        const auto variable_active = [&](QubitId qubit) noexcept {
            const std::size_t index = static_cast<std::size_t>(qubit);
            return has_h_[index] != 0U && passed_h[index] == 0U;
        };
        const auto add_unary = [&](QubitId qubit, QComplex zero, QComplex one) {
            const std::size_t index = static_cast<std::size_t>(qubit);
            if (variable_active(qubit)) {
                const std::array<FactorVariableId, 1> scope{variables[index]};
                const std::array<QComplex, 2> values{zero, one};
                (void)graph.add_dense_factor(scope, values);
            } else {
                global *= bits[index] == 0U ? zero : one;
            }
        };
        const auto add_cz = [&](QubitId first, QubitId second) {
            const std::size_t first_index = static_cast<std::size_t>(first);
            const std::size_t second_index = static_cast<std::size_t>(second);
            const bool first_variable = variable_active(first);
            const bool second_variable = variable_active(second);
            if (!first_variable && !second_variable) {
                if (bits[first_index] != 0U && bits[second_index] != 0U) {
                    global *= -1.0;
                }
                return;
            }
            if (first_variable && second_variable) {
                const std::array<FactorVariableId, 2> scope{
                    variables[first_index], variables[second_index]};
                const std::array<QComplex, 4> values{
                    QComplex{1.0, 0.0}, QComplex{1.0, 0.0},
                    QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}};
                (void)graph.add_dense_factor(scope, values);
                return;
            }
            const QubitId variable_qubit = first_variable ? first : second;
            const std::size_t fixed_index = first_variable ? second_index : first_index;
            if (bits[fixed_index] != 0U) {
                const std::size_t variable_index =
                    static_cast<std::size_t>(variable_qubit);
                const std::array<FactorVariableId, 1> scope{variables[variable_index]};
                const std::array<QComplex, 2> values{
                    QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}};
                (void)graph.add_dense_factor(scope, values);
            }
        };

        for (const Operation& operation : operations_) {
            switch (operation.code) {
                case OperationCode::H: {
                    const std::size_t index = static_cast<std::size_t>(operation.first);
                    const double one_sign = bits[index] == 0U ? 0.5 : -0.5;
                    const std::array<FactorVariableId, 1> scope{variables[index]};
                    const std::array<QComplex, 2> values{
                        QComplex{0.5, 0.0}, QComplex{one_sign, 0.0}};
                    (void)graph.add_dense_factor(scope, values);
                    passed_h[index] = 1U;
                    break;
                }
                case OperationCode::Z:
                    add_unary(operation.first, {1.0, 0.0}, {-1.0, 0.0});
                    break;
                case OperationCode::S:
                    add_unary(operation.first, {1.0, 0.0}, QI);
                    break;
                case OperationCode::Sdg:
                    add_unary(operation.first, {1.0, 0.0}, -QI);
                    break;
                case OperationCode::T:
                    add_unary(
                        operation.first,
                        {1.0, 0.0},
                        QComplex::from_polar(1.0, 0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Tdg:
                    add_unary(
                        operation.first,
                        {1.0, 0.0},
                        QComplex::from_polar(1.0, -0.785398163397448309615660845819875721));
                    break;
                case OperationCode::Rz:
                    add_unary(
                        operation.first,
                        QComplex::from_polar(1.0, -0.5 * operation.parameter),
                        QComplex::from_polar(1.0, 0.5 * operation.parameter));
                    break;
                case OperationCode::Cz:
                    add_cz(operation.first, operation.second);
                    break;
                default:
                    throw QStateError(
                        "Operation left the validated single-H diagonal path contract");
            }
        }

        QComplex partition{1.0, 0.0};
        ExactFactorStats factor_stats{};
        std::size_t plan_bytes = 0U;
        if (h_defects_ != 0U) {
            ExactFactorPlan plan = graph.compile();
            partition = plan.partition();
            factor_stats = plan.stats();
            plan_bytes = plan.estimated_bytes();
        }
        const QComplex mantissa = global * partition;
        if (!std::isfinite(mantissa.re) || !std::isfinite(mantissa.im)) {
            throw QStateError("Single-H path amplitude became non-finite");
        }

        return ExactSingleHPathAmplitude{
            mantissa,
            -0.5 * static_cast<double>(qubit_count_ - h_defects_),
            h_defects_,
            graph.variable_count(),
            graph.factor_count(),
            graph.estimated_bytes(),
            plan_bytes,
            factor_stats,
        };
    }

    [[nodiscard]] QComplex amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        return scaled_amplitude_bits(bits).amplitude();
    }

private:
    std::size_t qubit_count_{0U};
    std::vector<Operation> operations_{};
    ExactSingleHPathConfig config_{};
    std::vector<std::uint8_t> has_h_{};
    std::size_t h_defects_{0U};

    void validate_configuration() const {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operations_.size() > config_.max_operations ||
            config_.max_h_defects == 0U) {
            throw QStateError("Single-H path compiler dimensions or configuration are invalid");
        }
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Single-H path operation target is out of range");
        }
        if (operation.code == OperationCode::Cz) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Single-H path CZ support is invalid");
            }
        }
        if (operation.code == OperationCode::Rz && !std::isfinite(operation.parameter)) {
            throw QStateError("Single-H path Rz angle must be finite");
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
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Cnot:
            case OperationCode::Swap:
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError(
                    "Operation is outside the exact single-H diagonal path contract");
            default:
                throw QStateError("Single-H path compiler received an unknown opcode");
        }
    }

    void validate_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Single-H path output bit count does not match qubit count");
        }
        for (const std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Single-H path output bits must be zero or one");
            }
        }
    }
};

}  // namespace qubit
