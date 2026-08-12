#pragma once

#include "qubit/qmps.hpp"
#include "qubit/qpauli.hpp"
#include "qubit/qphase_graph.hpp"
#include "qubit/qplan.hpp"
#include "qubit/qstate.hpp"
#include "qubit/qtensor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace qubit {

enum class ExactExecutionRoute : std::uint8_t {
    Register = 0,
    CausalPauli = 1,
    TensorNetwork = 2,
    PersistentMPS = 3,
    PhaseGraph = 4,
    UniformMagnitude = 5,
    BasisPermutation = 6,
};

struct ExactExecutionBrokerConfig {
    TensorNetworkConfig tensor{};
    MPSConfig mps{};
    PhaseGraphConfig phase_graph{};
    QStateConfig register_state{};
};

struct ExactExpectationResult {
    QComplex value{};
    ExactExecutionRoute route{ExactExecutionRoute::Register};
    PauliPropagationStats pauli_stats{};
    std::string fallback_reason{};
};

struct ExactProbabilityResult {
    double value{0.0};
    ExactExecutionRoute route{ExactExecutionRoute::Register};
    TensorContractionStats tensor_stats{};
    std::string fallback_reason{};
};

class ExactExecutionBroker {
public:
    explicit ExactExecutionBroker(ExactExecutionBrokerConfig config = {});

    [[nodiscard]] ExactExpectationResult expectation(
        const QRegister& input,
        std::span<const Operation> operations,
        const PauliObservable& observable) const;

    [[nodiscard]] ExactExpectationResult expectation_from_zero(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        const PauliObservable& observable) const;

    [[nodiscard]] ExactProbabilityResult basis_probability_from_zero(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const std::uint8_t> basis_bits) const;

    [[nodiscard]] ExactProbabilityResult basis_probability_from_zero(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        BasisIndex basis) const;

    [[nodiscard]] ExactProbabilityResult marginal_probability_from_zero(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const QubitId> qubits,
        std::span<const std::uint8_t> bits) const;

    [[nodiscard]] const ExactExecutionBrokerConfig& config() const noexcept {
        return config_;
    }

private:
    ExactExecutionBrokerConfig config_{};
};

class ExactPreparedExpectationPlan {
public:
    ExactPreparedExpectationPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactExecutionBrokerConfig config = {});

    [[nodiscard]] ExactExpectationResult expectation(
        const PauliObservable& observable) const;
    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] ExactExecutionRoute prepared_fallback_route() const noexcept;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

private:
    std::size_t qubit_count_{0};
    ExactExecutionBrokerConfig config_{};
    QRegister zero_input_;
    std::optional<PauliPropagationPlan> causal_plan_{};
    std::optional<MPSPauliPlan> mps_plan_{};
    std::optional<QRegister> register_state_{};
    std::string causal_preparation_reason_{};
    std::string mps_preparation_reason_{};
};

class ExactPreparedProbabilityPlan {
public:
    ExactPreparedProbabilityPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactExecutionBrokerConfig config = {});

    [[nodiscard]] ExactProbabilityResult probability(
        std::span<const std::uint8_t> basis_bits) const;
    [[nodiscard]] ExactProbabilityResult probability(BasisIndex basis) const;
    [[nodiscard]] ExactProbabilityResult marginal_probability(
        std::span<const QubitId> qubits,
        std::span<const std::uint8_t> bits) const;
    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] ExactExecutionRoute prepared_route() const noexcept { return route_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

private:
    std::size_t qubit_count_{0};
    ExactExecutionBrokerConfig config_{};
    ExactExecutionRoute route_{ExactExecutionRoute::Register};
    std::vector<std::uint8_t> basis_permutation_bits_{};
    double uniform_probability_{0.0};
    std::optional<TensorContractionPlan> tensor_plan_{};
    std::optional<MatrixProductState> mps_state_{};
    std::optional<PhaseGraphState> phase_graph_state_{};
    std::optional<QRegister> register_state_{};
    std::string fallback_reason_{};
};

inline ExactProbabilityResult ExactPreparedProbabilityPlan::marginal_probability(
    std::span<const QubitId> qubits,
    std::span<const std::uint8_t> bits) const {
    if (qubits.size() != bits.size()) {
        throw QStateError("Prepared marginal query qubit and bit counts do not match");
    }
    for (std::size_t index = 0U; index < qubits.size(); ++index) {
        if (static_cast<std::size_t>(qubits[index]) >= qubit_count_) {
            throw QStateError("Prepared marginal query qubit is out of range");
        }
        if (bits[index] > 1U) {
            throw QStateError("Prepared marginal query bits must be zero or one");
        }
        for (std::size_t previous = 0U; previous < index; ++previous) {
            if (qubits[previous] == qubits[index]) {
                throw QStateError("Prepared marginal query contains duplicate qubits");
            }
        }
    }

    ExactProbabilityResult result;
    result.route = route_;
    result.fallback_reason = fallback_reason_;
    if (route_ == ExactExecutionRoute::BasisPermutation) {
        if (basis_permutation_bits_.size() != qubit_count_) {
            throw QStateError("Prepared BasisPermutation plan is missing its terminal state");
        }
        for (std::size_t index = 0U; index < qubits.size(); ++index) {
            if (basis_permutation_bits_[qubits[index]] != bits[index]) {
                result.value = 0.0;
                return result;
            }
        }
        result.value = 1.0;
        return result;
    }
    if (route_ == ExactExecutionRoute::UniformMagnitude) {
        result.value = bits.size() > 1074U
            ? 0.0
            : std::ldexp(1.0, -static_cast<int>(bits.size()));
        return result;
    }
    throw QStateError(
        "Prepared marginal probability currently requires a BasisPermutation or UniformMagnitude route");
}

inline ExactProbabilityResult ExactExecutionBroker::marginal_probability_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    std::span<const QubitId> qubits,
    std::span<const std::uint8_t> bits) const {
    ExactPreparedProbabilityPlan prepared(qubit_count, operations, config_);
    return prepared.marginal_probability(qubits, bits);
}

[[nodiscard]] const char* exact_execution_route_name(ExactExecutionRoute route) noexcept;

}  // namespace qubit
