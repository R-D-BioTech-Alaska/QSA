#pragma once

#include "qubit/qmps.hpp"
#include "qubit/qpauli.hpp"
#include "qubit/qphase_graph.hpp"
#include "qubit/qplan.hpp"
#include "qubit/qstate.hpp"
#include "qubit/qtensor.hpp"

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

[[nodiscard]] const char* exact_execution_route_name(ExactExecutionRoute route) noexcept;

}  // namespace qubit
