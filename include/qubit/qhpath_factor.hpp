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

struct ExactHadamardPathConfig {
    ExactFactorConfig factor{};
    std::size_t max_qubits{1U << 20U};
    std::size_t max_operations{1U << 24U};
    std::size_t max_h_events{1U << 20U};
    std::size_t max_metadata_bytes{1U << 30U};
};

struct ExactHadamardPathAmplitude {
    QComplex mantissa{};
    double log2_scale{0.0};
    std::size_t h_events{0U};
    std::size_t h_active_qubits{0U};
    std::size_t factor_variables{0U};
    std::size_t factor_count{0U};
    std::size_t metadata_estimated_bytes{0U};
    std::size_t graph_estimated_bytes{0U};
    std::size_t plan_estimated_bytes{0U};
    ExactFactorStats factor_stats{};

    [[nodiscard]] QComplex amplitude() const {
        const double scale = std::exp2(log2_scale);
        if (scale == 0.0 && (mantissa.re != 0.0 || mantissa.im != 0.0)) {
            throw QStateError(
                "Hadamard-path amplitude underflows; use scaled result or log2_probability");
        }
        return mantissa * scale;
    }

    [[nodiscard]] double log2_probability() const {
        const double real = std::abs(mantissa.re);
        const double imaginary = std::abs(mantissa.im);
        const double maximum = std::max(real, imaginary);
        if (!std::isfinite(maximum)) {
            throw QStateError("Hadamard-path probability became non-finite");
        }
        if (maximum == 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        const double scaled_real = mantissa.re / maximum;
        const double scaled_imaginary = mantissa.im / maximum;
        const double scaled_norm2 =
            scaled_real * scaled_real + scaled_imaginary * scaled_imaginary;
        return 2.0 * std::log2(maximum) + std::log2(scaled_norm2) +
            2.0 * log2_scale;
    }
};

class ExactHadamardPathAmplitudePlan {
public:
    ExactHadamardPathAmplitudePlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactHadamardPathConfig config = {})
        : qubit_count_(qubit_count),
          operations_(operations.begin(), operations.end()),
          config_(config),
          operation_event_(operations.size(), npos()),
          first_h_event_(qubit_count, npos()) {
        validate_configuration();
        std::vector<std::size_t> last_h_event(qubit_count_, npos());
        for (std::size_t operation_index = 0U;
             operation_index < operations_.size();
             ++operation_index) {
            const Operation& operation = operations_[operation_index];
            validate_operation(operation);
            if (operation.code != OperationCode::H) {
                continue;
            }
            if (h_events_ >= config_.max_h_events ||
                h_events_ >= config_.factor.max_variables) {
                throw QStateError("Hadamard-path compiler exceeds its H-event variable cap");
            }
            const std::size_t qubit = static_cast<std::size_t>(operation.first);
            const std::size_t event = h_events_++;
            operation_event_[operation_index] = event;
            next_h_event_.push_back(npos());
            if (last_h_event[qubit] == npos()) {
                first_h_event_[qubit] = event;
                ++h_active_qubits_;
            } else {
                next_h_event_[last_h_event[qubit]] = event;
            }
            last_h_event[qubit] = event;
        }
        metadata_estimated_bytes_ = estimate_metadata_bytes(last_h_event.capacity());
        if (metadata_estimated_bytes_ > config_.max_metadata_bytes) {
            throw QStateError("Hadamard-path compiler metadata exceeds configured byte cap");
        }
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operations_.size(); }
    [[nodiscard]] std::size_t h_events() const noexcept { return h_events_; }
    [[nodiscard]] std::size_t h_active_qubits() const noexcept { return h_active_qubits_; }
    [[nodiscard]] std::size_t metadata_estimated_bytes() const noexcept {
        return metadata_estimated_bytes_;
    }
    [[nodiscard]] const ExactHadamardPathConfig& config() const noexcept { return config_; }

    [[nodiscard]] ExactHadamardPathAmplitude scaled_amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        validate_bits(bits);

        ExactFactorGraph graph(config_.factor);
        std::vector<FactorVariableId> variables(h_events_);
        for (std::size_t event = 0U; event < h_events_; ++event) {
            variables[event] = graph.add_variable(2U);
        }
        std::vector<std::size_t> current_event = first_h_event_;
        QComplex global{1.0, 0.0};

        const auto active_variable = [&](QubitId qubit) -> const FactorVariableId* {
            const std::size_t event = current_event[static_cast<std::size_t>(qubit)];
            return event == npos() ? nullptr : &variables[event];
        };
        const auto add_unary = [&](QubitId qubit, QComplex zero, QComplex one) {
            const std::size_t index = static_cast<std::size_t>(qubit);
            const FactorVariableId* variable = active_variable(qubit);
            if (variable != nullptr) {
                const std::array<FactorVariableId, 1> scope{*variable};
                const std::array<QComplex, 2> values{zero, one};
                (void)graph.add_dense_factor(scope, values);
            } else {
                global *= bits[index] == 0U ? zero : one;
            }
        };
        const auto add_cz = [&](QubitId first, QubitId second) {
            const std::size_t first_index = static_cast<std::size_t>(first);
            const std::size_t second_index = static_cast<std::size_t>(second);
            const FactorVariableId* first_variable = active_variable(first);
            const FactorVariableId* second_variable = active_variable(second);
            if (first_variable == nullptr && second_variable == nullptr) {
                if (bits[first_index] != 0U && bits[second_index] != 0U) {
                    global *= -1.0;
                }
                return;
            }
            if (first_variable != nullptr && second_variable != nullptr) {
                const std::array<FactorVariableId, 2> scope{
                    *first_variable, *second_variable};
                const std::array<QComplex, 4> values{
                    QComplex{1.0, 0.0}, QComplex{1.0, 0.0},
                    QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}};
                (void)graph.add_dense_factor(scope, values);
                return;
            }
            const FactorVariableId variable =
                first_variable != nullptr ? *first_variable : *second_variable;
            const std::size_t fixed_index =
                first_variable != nullptr ? second_index : first_index;
            if (bits[fixed_index] != 0U) {
                const std::array<FactorVariableId, 1> scope{variable};
                const std::array<QComplex, 2> values{
                    QComplex{1.0, 0.0}, QComplex{-1.0, 0.0}};
                (void)graph.add_dense_factor(scope, values);
            }
        };

        constexpr double inverse_sqrt_two =
            0.707106781186547524400844362104849039;
        for (std::size_t operation_index = 0U;
             operation_index < operations_.size();
             ++operation_index) {
            const Operation& operation = operations_[operation_index];
            switch (operation.code) {
                case OperationCode::H: {
                    const std::size_t qubit = static_cast<std::size_t>(operation.first);
                    const std::size_t event = operation_event_[operation_index];
                    if (event == npos() || current_event[qubit] != event) {
                        throw QStateError("Hadamard-path event sequence is inconsistent");
                    }
                    const std::size_t next = next_h_event_[event];
                    const double normalization =
                        event == first_h_event_[qubit] ? 0.5 : inverse_sqrt_two;
                    if (next != npos()) {
                        const std::array<FactorVariableId, 2> scope{
                            variables[event], variables[next]};
                        const std::array<QComplex, 4> values{
                            QComplex{normalization, 0.0},
                            QComplex{normalization, 0.0},
                            QComplex{normalization, 0.0},
                            QComplex{-normalization, 0.0},
                        };
                        (void)graph.add_dense_factor(scope, values);
                    } else {
                        const double one_sign =
                            bits[qubit] == 0U ? normalization : -normalization;
                        const std::array<FactorVariableId, 1> scope{variables[event]};
                        const std::array<QComplex, 2> values{
                            QComplex{normalization, 0.0}, QComplex{one_sign, 0.0}};
                        (void)graph.add_dense_factor(scope, values);
                    }
                    current_event[qubit] = next;
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
                        "Operation left the validated Hadamard-path diagonal contract");
            }
        }

        QComplex partition{1.0, 0.0};
        ExactFactorStats factor_stats{};
        std::size_t plan_bytes = 0U;
        if (h_events_ != 0U) {
            ExactFactorPlan plan = graph.compile();
            partition = plan.partition();
            factor_stats = plan.stats();
            plan_bytes = plan.estimated_bytes();
        }
        const QComplex mantissa = global * partition;
        if (!std::isfinite(mantissa.re) || !std::isfinite(mantissa.im)) {
            throw QStateError("Hadamard-path amplitude became non-finite");
        }

        return ExactHadamardPathAmplitude{
            mantissa,
            -0.5 * static_cast<double>(qubit_count_ - h_active_qubits_),
            h_events_,
            h_active_qubits_,
            graph.variable_count(),
            graph.factor_count(),
            metadata_estimated_bytes_,
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
    ExactHadamardPathConfig config_{};
    std::vector<std::size_t> operation_event_{};
    std::vector<std::size_t> next_h_event_{};
    std::vector<std::size_t> first_h_event_{};
    std::size_t h_events_{0U};
    std::size_t h_active_qubits_{0U};
    std::size_t metadata_estimated_bytes_{0U};

    [[nodiscard]] static constexpr std::size_t npos() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }

    void validate_configuration() const {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operations_.size() > config_.max_operations ||
            config_.max_h_events == 0U || config_.max_metadata_bytes == 0U) {
            throw QStateError("Hadamard-path compiler dimensions or configuration are invalid");
        }
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Hadamard-path operation target is out of range");
        }
        if (operation.code == OperationCode::Cz) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Hadamard-path CZ support is invalid");
            }
        }
        if (operation.code == OperationCode::Rz && !std::isfinite(operation.parameter)) {
            throw QStateError("Hadamard-path Rz angle must be finite");
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
                    "Operation is outside the exact Hadamard-path diagonal contract");
            default:
                throw QStateError("Hadamard-path compiler received an unknown opcode");
        }
    }

    void validate_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Hadamard-path output bit count does not match qubit count");
        }
        for (const std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Hadamard-path output bits must be zero or one");
            }
        }
    }

    [[nodiscard]] std::size_t estimate_metadata_bytes(std::size_t last_capacity) const {
        std::size_t total = sizeof(*this);
        total = checked_sum(total, checked_product(
            operations_.capacity(), sizeof(Operation),
            "Hadamard-path operation metadata overflowed"));
        total = checked_sum(total, checked_product(
            operation_event_.capacity(), sizeof(std::size_t),
            "Hadamard-path event-map metadata overflowed"));
        total = checked_sum(total, checked_product(
            next_h_event_.capacity(), sizeof(std::size_t),
            "Hadamard-path next-event metadata overflowed"));
        total = checked_sum(total, checked_product(
            first_h_event_.capacity(), sizeof(std::size_t),
            "Hadamard-path first-event metadata overflowed"));
        total = checked_sum(total, checked_product(
            last_capacity, sizeof(std::size_t),
            "Hadamard-path construction metadata overflowed"));
        return total;
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
        const char* message = "Hadamard-path metadata size overflowed") {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }
};

}  // namespace qubit
