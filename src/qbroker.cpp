#include "qubit/qbroker.hpp"

#include <limits>
#include <utility>
#include <vector>

namespace qubit {
namespace {

[[nodiscard]] std::string failure_message(const QStateError& error) {
    return error.what();
}

[[nodiscard]] std::size_t validated_qubit_count(std::size_t qubit_count) {
    if (qubit_count == 0U) {
        throw QStateError("Execution broker requires at least one qubit");
    }
    return qubit_count;
}

void validate_broker_config(const ExactExecutionBrokerConfig& config) {
    if (config.tensor.max_contraction_entries < 2U ||
        config.tensor.max_factors == 0U) {
        throw QStateError("Execution broker tensor limits are invalid");
    }
}

void validate_basis_width(std::size_t qubit_count, std::span<const std::uint8_t> basis_bits) {
    validated_qubit_count(qubit_count);
    if (basis_bits.size() != qubit_count) {
        throw QStateError("Execution broker basis width does not match qubit count");
    }
    for (const std::uint8_t bit : basis_bits) {
        if (bit > 1U) {
            throw QStateError("Execution broker basis bits must be zero or one");
        }
    }
}

void validate_observable(std::size_t qubit_count, const PauliObservable& observable) {
    if (observable.qubit_count() != qubit_count) {
        throw QStateError("Execution broker Pauli observable width does not match input");
    }
    if (!observable.validate()) {
        throw QStateError("Execution broker Pauli observable failed validation");
    }
}

void execute_mps(MatrixProductState& state, std::span<const Operation> operations) {
    for (const Operation& operation : operations) {
        switch (operation.code) {
            case OperationCode::X:
                state.apply_unitary(operation.first, gates::x());
                break;
            case OperationCode::Y:
                state.apply_unitary(operation.first, gates::y());
                break;
            case OperationCode::Z:
                state.apply_unitary(operation.first, gates::z());
                break;
            case OperationCode::H:
                state.apply_unitary(operation.first, gates::h());
                break;
            case OperationCode::S:
                state.apply_unitary(operation.first, gates::s());
                break;
            case OperationCode::Sdg:
                state.apply_unitary(operation.first, gates::sdg());
                break;
            case OperationCode::T:
                state.apply_unitary(operation.first, gates::t());
                break;
            case OperationCode::Tdg:
                state.apply_unitary(operation.first, gates::tdg());
                break;
            case OperationCode::Rx:
                state.apply_unitary(operation.first, gates::rx(operation.parameter));
                break;
            case OperationCode::Ry:
                state.apply_unitary(operation.first, gates::ry(operation.parameter));
                break;
            case OperationCode::Rz:
                state.apply_unitary(operation.first, gates::rz(operation.parameter));
                break;
            case OperationCode::Cnot:
                state.apply_cnot(operation.first, operation.second);
                break;
            case OperationCode::Cz:
                state.apply_cz(operation.first, operation.second);
                break;
            case OperationCode::Swap:
                throw QStateError("Persistent MPS route does not support SWAP");
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError("Persistent MPS route supports unitary operations only");
            default:
                throw QStateError("Persistent MPS route received an unknown opcode");
        }
    }
}

[[nodiscard]] PhaseGraphState execute_phase_graph_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    PhaseGraphConfig config) {
    if (operations.size() < qubit_count) {
        throw QStateError("PhaseGraph route requires a complete leading H layer");
    }

    std::vector<std::uint8_t> seen(qubit_count, 0U);
    for (std::size_t index = 0; index < qubit_count; ++index) {
        const Operation& operation = operations[index];
        if (operation.code != OperationCode::H) {
            throw QStateError("PhaseGraph route requires a complete leading H layer");
        }
        if (operation.first >= qubit_count) {
            throw QStateError("PhaseGraph route leading H qubit is out of range");
        }
        if (seen[operation.first] != 0U) {
            throw QStateError("PhaseGraph route leading H layer contains duplicate qubits");
        }
        seen[operation.first] = 1U;
    }

    PhaseGraphState state(qubit_count, config);
    for (std::size_t index = qubit_count; index < operations.size(); ++index) {
        const Operation& operation = operations[index];
        switch (operation.code) {
            case OperationCode::X:
                state.apply_x(operation.first);
                break;
            case OperationCode::Y:
                state.apply_y(operation.first);
                break;
            case OperationCode::Z:
                state.apply_z(operation.first);
                break;
            case OperationCode::S:
                state.apply_s(operation.first);
                break;
            case OperationCode::Sdg:
                state.apply_sdg(operation.first);
                break;
            case OperationCode::T:
                state.apply_t(operation.first);
                break;
            case OperationCode::Tdg:
                state.apply_tdg(operation.first);
                break;
            case OperationCode::Rz:
                state.apply_rz(operation.first, operation.parameter);
                break;
            case OperationCode::Cz:
                state.apply_cz(operation.first, operation.second);
                break;
            case OperationCode::Swap:
                state.apply_swap(operation.first, operation.second);
                break;
            case OperationCode::H:
                throw QStateError("PhaseGraph route does not support H after the leading layer");
            case OperationCode::Rx:
            case OperationCode::Ry:
                throw QStateError("PhaseGraph route supports phase-preserving rotations only");
            case OperationCode::Cnot:
                throw QStateError("PhaseGraph route does not support CNOT");
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError("PhaseGraph route supports unitary phase-graph operations only");
            default:
                throw QStateError("PhaseGraph route received an unknown opcode");
        }
    }
    return state;
}

[[nodiscard]] std::size_t dynamic_bytes(std::size_t estimated, std::size_t object_size) noexcept {
    return estimated > object_size ? estimated - object_size : 0U;
}

}  // namespace

ExactExecutionBroker::ExactExecutionBroker(ExactExecutionBrokerConfig config)
    : config_(config) {
    validate_broker_config(config_);
}

const char* exact_execution_route_name(ExactExecutionRoute route) noexcept {
    switch (route) {
        case ExactExecutionRoute::Register:
            return "QRegister";
        case ExactExecutionRoute::CausalPauli:
            return "CausalPauli";
        case ExactExecutionRoute::TensorNetwork:
            return "TensorNetwork";
        case ExactExecutionRoute::PersistentMPS:
            return "PersistentMPS";
        case ExactExecutionRoute::PhaseGraph:
            return "PhaseGraph";
    }
    return "unknown";
}

ExactExpectationResult ExactExecutionBroker::expectation(
    const QRegister& input,
    std::span<const Operation> operations,
    const PauliObservable& observable) const {
    validate_observable(input.qubit_count(), observable);

    ExactExpectationResult result;
    try {
        PauliPropagationPlan plan(input.qubit_count(), operations);
        const char* reason = nullptr;
        auto propagated = plan.try_propagate_backward(
            observable,
            &reason,
            &result.pauli_stats);
        if (propagated.has_value()) {
            result.value = propagated->expectation(input);
            result.route = ExactExecutionRoute::CausalPauli;
            return result;
        }
        result.fallback_reason = reason != nullptr ? reason : "Pauli propagation failed exact routing";
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

ExactExpectationResult ExactExecutionBroker::expectation_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    const PauliObservable& observable) const {
    validated_qubit_count(qubit_count);
    validate_observable(qubit_count, observable);

    QRegister input(qubit_count, config_.register_state);
    ExactExpectationResult result;
    try {
        PauliPropagationPlan plan(qubit_count, operations);
        const char* reason = nullptr;
        auto propagated = plan.try_propagate_backward(
            observable,
            &reason,
            &result.pauli_stats);
        if (propagated.has_value()) {
            result.value = propagated->expectation(input);
            result.route = ExactExecutionRoute::CausalPauli;
            return result;
        }
        result.fallback_reason = "causal: ";
        result.fallback_reason += reason != nullptr ? reason : "Pauli propagation failed exact routing";
    } catch (const QStateError& error) {
        result.fallback_reason = "causal: " + failure_message(error);
    }

    try {
        MatrixProductState state = MatrixProductState::zero(qubit_count, config_.mps);
        execute_mps(state, operations);
        result.value = state.expectation(observable);
        result.route = ExactExecutionRoute::PersistentMPS;
        return result;
    } catch (const QStateError& error) {
        result.fallback_reason += "; mps: ";
        result.fallback_reason += failure_message(error);
    }

    OperationPlan plan(operations);
    plan.execute(input);
    result.value = observable.expectation(input);
    result.route = ExactExecutionRoute::Register;
    return result;
}

ExactProbabilityResult ExactExecutionBroker::basis_probability_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    std::span<const std::uint8_t> basis_bits) const {
    validate_basis_width(qubit_count, basis_bits);

    ExactProbabilityResult result;
    try {
        TensorNetworkCircuit tensor(qubit_count, operations, config_.tensor);
        result.value = tensor.amplitude(basis_bits, &result.tensor_stats).norm2();
        result.route = ExactExecutionRoute::TensorNetwork;
        return result;
    } catch (const QStateError& error) {
        result.fallback_reason = "tensor: " + failure_message(error);
    }

    try {
        MatrixProductState state = MatrixProductState::zero(qubit_count, config_.mps);
        execute_mps(state, operations);
        result.value = state.amplitude(basis_bits).norm2();
        result.route = ExactExecutionRoute::PersistentMPS;
        return result;
    } catch (const QStateError& error) {
        result.fallback_reason += "; mps: ";
        result.fallback_reason += failure_message(error);
    }

    try {
        PhaseGraphState state = execute_phase_graph_from_zero(
            qubit_count,
            operations,
            config_.phase_graph);
        result.value = state.amplitude_bits(basis_bits).norm2();
        result.route = ExactExecutionRoute::PhaseGraph;
        return result;
    } catch (const QStateError& error) {
        result.fallback_reason += "; phase_graph: ";
        result.fallback_reason += failure_message(error);
    }

    QRegister state(qubit_count, config_.register_state);
    OperationPlan plan(operations);
    plan.execute(state);
    result.value = state.amplitude_bits(basis_bits).norm2();
    result.route = ExactExecutionRoute::Register;
    return result;
}

ExactProbabilityResult ExactExecutionBroker::basis_probability_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    BasisIndex basis) const {
    validated_qubit_count(qubit_count);
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
    return basis_probability_from_zero(qubit_count, operations, bits);
}

ExactPreparedExpectationPlan::ExactPreparedExpectationPlan(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    ExactExecutionBrokerConfig config)
    : qubit_count_(validated_qubit_count(qubit_count)),
      config_(config),
      zero_input_(qubit_count_, config_.register_state) {
    validate_broker_config(config_);

    try {
        causal_plan_.emplace(qubit_count_, operations);
    } catch (const QStateError& error) {
        causal_preparation_reason_ = failure_message(error);
    }

    try {
        MatrixProductState state = MatrixProductState::zero(qubit_count_, config_.mps);
        execute_mps(state, operations);
        mps_plan_.emplace(std::move(state));
    } catch (const QStateError& error) {
        mps_preparation_reason_ = failure_message(error);
        register_state_.emplace(qubit_count_, config_.register_state);
        OperationPlan plan(operations);
        plan.execute(*register_state_);
    }
}

ExactExpectationResult ExactPreparedExpectationPlan::expectation(
    const PauliObservable& observable) const {
    validate_observable(qubit_count_, observable);

    ExactExpectationResult result;
    if (causal_plan_.has_value()) {
        const char* reason = nullptr;
        auto propagated = causal_plan_->try_propagate_backward(
            observable,
            &reason,
            &result.pauli_stats);
        if (propagated.has_value()) {
            result.value = propagated->expectation(zero_input_);
            result.route = ExactExecutionRoute::CausalPauli;
            return result;
        }
        result.fallback_reason = "causal: ";
        result.fallback_reason += reason != nullptr ? reason : "Pauli propagation failed exact routing";
    } else {
        result.fallback_reason = "causal: " + causal_preparation_reason_;
    }

    if (mps_plan_.has_value()) {
        result.value = mps_plan_->expectation(observable);
        result.route = ExactExecutionRoute::PersistentMPS;
        return result;
    }

    if (!mps_preparation_reason_.empty()) {
        if (!result.fallback_reason.empty()) {
            result.fallback_reason += "; ";
        }
        result.fallback_reason += "mps: " + mps_preparation_reason_;
    }
    if (!register_state_.has_value()) {
        throw QStateError("Prepared expectation plan has no exact fallback state");
    }
    result.value = observable.expectation(*register_state_);
    result.route = ExactExecutionRoute::Register;
    return result;
}

ExactExecutionRoute ExactPreparedExpectationPlan::prepared_fallback_route() const noexcept {
    return mps_plan_.has_value()
        ? ExactExecutionRoute::PersistentMPS
        : ExactExecutionRoute::Register;
}

std::size_t ExactPreparedExpectationPlan::estimated_bytes() const noexcept {
    std::size_t total = sizeof(*this);
    total += dynamic_bytes(zero_input_.estimated_bytes(), sizeof(QRegister));
    if (causal_plan_.has_value()) {
        total += dynamic_bytes(causal_plan_->estimated_bytes(), sizeof(PauliPropagationPlan));
    }
    if (mps_plan_.has_value()) {
        total += dynamic_bytes(mps_plan_->estimated_bytes(), sizeof(MPSPauliPlan));
    }
    if (register_state_.has_value()) {
        total += dynamic_bytes(register_state_->estimated_bytes(), sizeof(QRegister));
    }
    total += causal_preparation_reason_.capacity();
    total += mps_preparation_reason_.capacity();
    return total;
}

}  // namespace qubit
