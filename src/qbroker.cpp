#include "qubit/qbroker.hpp"

#include <limits>
#include <vector>

namespace qubit {
namespace {

[[nodiscard]] std::string failure_message(const QStateError& error) {
    return error.what();
}

void validate_basis_width(std::size_t qubit_count, std::span<const std::uint8_t> basis_bits) {
    if (qubit_count == 0U) {
        throw QStateError("Execution broker requires at least one qubit");
    }
    if (basis_bits.size() != qubit_count) {
        throw QStateError("Execution broker basis width does not match qubit count");
    }
    for (const std::uint8_t bit : basis_bits) {
        if (bit > 1U) {
            throw QStateError("Execution broker basis bits must be zero or one");
        }
    }
}

}  // namespace

ExactExecutionBroker::ExactExecutionBroker(ExactExecutionBrokerConfig config)
    : config_(config) {
    if (config_.tensor.max_contraction_entries < 2U ||
        config_.tensor.max_factors == 0U) {
        throw QStateError("Execution broker tensor limits are invalid");
    }
}

const char* exact_execution_route_name(ExactExecutionRoute route) noexcept {
    switch (route) {
        case ExactExecutionRoute::Register:
            return "QRegister";
        case ExactExecutionRoute::CausalPauli:
            return "CausalPauli";
        case ExactExecutionRoute::TensorNetwork:
            return "TensorNetwork";
    }
    return "unknown";
}

ExactExpectationResult ExactExecutionBroker::expectation(
    const QRegister& input,
    std::span<const Operation> operations,
    const PauliObservable& observable) const {
    if (observable.qubit_count() != input.qubit_count()) {
        throw QStateError("Execution broker Pauli observable width does not match input");
    }
    if (!observable.validate()) {
        throw QStateError("Execution broker Pauli observable failed validation");
    }

    ExactExpectationResult result;
    try {
        PauliPropagationPlan plan(input.qubit_count(), operations);
        const PauliObservable propagated = plan.propagate_backward(
            observable,
            &result.pauli_stats);
        result.value = propagated.expectation(input);
        result.route = ExactExecutionRoute::CausalPauli;
        return result;
    } catch (const QStateError& error) {
        result.fallback_reason = failure_message(error);
    }

    QRegister state = input;
    OperationPlan plan(operations);
    plan.execute(state);
    result.value = observable.expectation(state);
    result.route = ExactExecutionRoute::Register;
    return result;
}

ExactAmplitudeResult ExactExecutionBroker::amplitude_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    std::span<const std::uint8_t> basis_bits) const {
    validate_basis_width(qubit_count, basis_bits);

    ExactAmplitudeResult result;
    try {
        TensorNetworkCircuit tensor(qubit_count, operations, config_.tensor);
        result.value = tensor.amplitude(basis_bits, &result.tensor_stats);
        result.route = ExactExecutionRoute::TensorNetwork;
        return result;
    } catch (const QStateError& error) {
        result.fallback_reason = failure_message(error);
    }

    QRegister state(qubit_count, config_.register_state);
    OperationPlan plan(operations);
    plan.execute(state);
    result.value = state.amplitude_bits(basis_bits);
    result.route = ExactExecutionRoute::Register;
    return result;
}

ExactAmplitudeResult ExactExecutionBroker::amplitude_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    BasisIndex basis) const {
    if (qubit_count == 0U) {
        throw QStateError("Execution broker requires at least one qubit");
    }
    if (qubit_count > std::numeric_limits<BasisIndex>::digits) {
        throw QStateError("Execution broker BasisIndex query is limited to 64 qubits");
    }
    if (qubit_count < std::numeric_limits<BasisIndex>::digits &&
        basis >= (BasisIndex{1} << qubit_count)) {
        throw QStateError("Execution broker basis index is out of range");
    }

    std::vector<std::uint8_t> bits(qubit_count, 0U);
    for (std::size_t qubit = 0; qubit < qubit_count; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & BasisIndex{1});
    }
    return amplitude_from_zero(qubit_count, operations, bits);
}

}  // namespace qubit
