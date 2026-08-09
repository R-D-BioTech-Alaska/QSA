#pragma once

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
};

struct ExactExecutionBrokerConfig {
    TensorNetworkConfig tensor{};
    QStateConfig register_state{};
};

struct ExactExpectationResult {
    QComplex value{};
    ExactExecutionRoute route{ExactExecutionRoute::Register};
    PauliPropagationStats pauli_stats{};
    std::string fallback_reason{};
};

struct ExactAmplitudeResult {
    QComplex value{};
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

    [[nodiscard]] ExactAmplitudeResult amplitude_from_zero(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const std::uint8_t> basis_bits) const;

    [[nodiscard]] ExactAmplitudeResult amplitude_from_zero(
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
