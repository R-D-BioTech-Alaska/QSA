#pragma once

#include "qubit/qmps.hpp"
#include "qubit/qpauli.hpp"
#include "qubit/qplan.hpp"
#include "qubit/qstate.hpp"
#include "qubit/qtensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace qubit {

enum class ExactExecutionRoute : std::uint8_t {
    Register = 0,
    CausalPauli = 1,
    TensorNetwork = 2,
    PersistentMPS = 3,
};

struct ExactExecutionBrokerConfig {
    TensorNetworkConfig tensor{};
    MPSConfig mps{};
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

[[nodiscard]] const char* exact_execution_route_name(ExactExecutionRoute route) noexcept;

}  // namespace qubit
