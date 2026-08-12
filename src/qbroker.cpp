#include "qubit/qbroker.hpp"

#include <cmath>
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

[[nodiscard]] bool mps_structure_eligible(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    const char** reason) noexcept {
    const auto fail = [&](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    for (const Operation& operation : operations) {
        switch (operation.code) {
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::H:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
                if (operation.first >= qubit_count) {
                    return fail("MPS single-qubit target is out of range");
                }
                break;
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Rz:
                if (operation.first >= qubit_count) {
                    return fail("MPS single-qubit target is out of range");
                }
                if (!std::isfinite(operation.parameter)) {
                    return fail("MPS single-qubit operation is not unitary");
                }
                break;
            case OperationCode::Cnot:
            case OperationCode::Cz: {
                const std::size_t first = static_cast<std::size_t>(operation.first);
                const std::size_t second = static_cast<std::size_t>(operation.second);
                if (first >= qubit_count || second >= qubit_count) {
                    return fail("MPS controlled gate qubit is out of range");
                }
                if (first == second ||
                    (first + 1U != second && second + 1U != first)) {
                    return fail("MPS controlled gates require adjacent distinct qubits");
                }
                break;
            }
            case OperationCode::Swap:
                return fail("Persistent MPS route does not support SWAP");
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                return fail("Persistent MPS route supports unitary operations only");
            default:
                return fail("Persistent MPS route received an unknown opcode");
        }
    }
    return true;
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

[[nodiscard]] bool basis_permutation_state(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    std::vector<std::uint8_t>* terminal_bits,
    const char** reason) {
    const auto fail = [&](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    std::vector<std::uint8_t> bits(qubit_count, 0U);
    for (const Operation& operation : operations) {
        switch (operation.code) {
            case OperationCode::X:
            case OperationCode::Y:
                if (operation.first >= qubit_count) {
                    return fail("BasisPermutation route single-qubit target is out of range");
                }
                bits[operation.first] ^= 1U;
                break;
            case OperationCode::Z:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
                if (operation.first >= qubit_count) {
                    return fail("BasisPermutation route single-qubit target is out of range");
                }
                break;
            case OperationCode::Rz:
                if (operation.first >= qubit_count) {
                    return fail("BasisPermutation route single-qubit target is out of range");
                }
                if (!std::isfinite(operation.parameter)) {
                    return fail("BasisPermutation route Rz angle must be finite");
                }
                break;
            case OperationCode::Cnot:
                if (operation.first >= qubit_count || operation.second >= qubit_count) {
                    return fail("BasisPermutation route CNOT qubit is out of range");
                }
                if (operation.first == operation.second) {
                    return fail("BasisPermutation route CNOT requires distinct qubits");
                }
                if (bits[operation.first] != 0U) {
                    bits[operation.second] ^= 1U;
                }
                break;
            case OperationCode::Cz:
                if (operation.first >= qubit_count || operation.second >= qubit_count) {
                    return fail("BasisPermutation route CZ qubit is out of range");
                }
                if (operation.first == operation.second) {
                    return fail("BasisPermutation route CZ requires distinct qubits");
                }
                break;
            case OperationCode::Swap:
                if (operation.first >= qubit_count || operation.second >= qubit_count) {
                    return fail("BasisPermutation route SWAP qubit is out of range");
                }
                if (operation.first == operation.second) {
                    return fail("BasisPermutation route SWAP requires distinct qubits");
                }
                std::swap(bits[operation.first], bits[operation.second]);
                break;
            case OperationCode::H:
                return fail("BasisPermutation route does not support H");
            case OperationCode::Rx:
            case OperationCode::Ry:
                return fail("BasisPermutation route supports monomial rotations only");
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                return fail("BasisPermutation route supports unitary operations only");
            default:
                return fail("BasisPermutation route received an unknown opcode");
        }
    }

    if (terminal_bits != nullptr) {
        *terminal_bits = std::move(bits);
    }
    return true;
}

[[nodiscard]] bool basis_permutation_probability(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    std::span<const std::uint8_t> basis_bits,
    double* probability,
    const char** reason) {
    std::vector<std::uint8_t> bits;
    if (!basis_permutation_state(qubit_count, operations, &bits, reason)) {
        return false;
    }

    bool match = true;
    for (std::size_t qubit = 0U; qubit < qubit_count; ++qubit) {
        if (bits[qubit] != basis_bits[qubit]) {
            match = false;
            break;
        }
    }
    if (probability != nullptr) {
        *probability = match ? 1.0 : 0.0;
    }
    return true;
}

[[nodiscard]] bool monomial_operation(
    std::size_t qubit_count,
    const Operation& operation,
    const char** reason) noexcept {
    const auto fail = [&](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    switch (operation.code) {
        case OperationCode::X:
        case OperationCode::Y:
        case OperationCode::Z:
        case OperationCode::S:
        case OperationCode::Sdg:
        case OperationCode::T:
        case OperationCode::Tdg:
            if (operation.first >= qubit_count) {
                return fail("UniformMagnitude route single-qubit target is out of range");
            }
            return true;
        case OperationCode::Rz:
            if (operation.first >= qubit_count) {
                return fail("UniformMagnitude route single-qubit target is out of range");
            }
            if (!std::isfinite(operation.parameter)) {
                return fail("UniformMagnitude route Rz angle must be finite");
            }
            return true;
        case OperationCode::Cnot:
        case OperationCode::Cz:
        case OperationCode::Swap:
            if (operation.first >= qubit_count || operation.second >= qubit_count) {
                return fail("UniformMagnitude route two-qubit target is out of range");
            }
            if (operation.first == operation.second) {
                return fail("UniformMagnitude route two-qubit operation requires distinct qubits");
            }
            return true;
        case OperationCode::H:
            return fail("UniformMagnitude route allows H only in one complete layer");
        case OperationCode::Rx:
        case OperationCode::Ry:
            return fail("UniformMagnitude route supports monomial rotations only");
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            return fail("UniformMagnitude route supports unitary operations only");
        default:
            return fail("UniformMagnitude route received an unknown opcode");
    }
}

[[nodiscard]] bool uniform_magnitude_eligible(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    const char** reason) {
    const auto fail = [&](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    std::size_t h_begin = 0U;
    while (h_begin < operations.size() && operations[h_begin].code != OperationCode::H) {
        if (!monomial_operation(qubit_count, operations[h_begin], reason)) {
            return false;
        }
        ++h_begin;
    }
    if (h_begin == operations.size() || operations.size() - h_begin < qubit_count) {
        return fail("UniformMagnitude route requires one complete H layer");
    }

    std::vector<std::uint8_t> seen(qubit_count, 0U);
    for (std::size_t offset = 0U; offset < qubit_count; ++offset) {
        const Operation& operation = operations[h_begin + offset];
        if (operation.code != OperationCode::H) {
            return fail("UniformMagnitude route requires one complete contiguous H layer");
        }
        if (operation.first >= qubit_count) {
            return fail("UniformMagnitude route H-layer qubit is out of range");
        }
        if (seen[operation.first] != 0U) {
            return fail("UniformMagnitude route H layer contains duplicate qubits");
        }
        seen[operation.first] = 1U;
    }

    for (std::size_t index = h_begin + qubit_count; index < operations.size(); ++index) {
        if (!monomial_operation(qubit_count, operations[index], reason)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] double uniform_probability(std::size_t qubit_count) noexcept {
    if (qubit_count > 1074U) {
        return 0.0;
    }
    return std::ldexp(1.0, -static_cast<int>(qubit_count));
}

[[nodiscard]] bool phase_graph_structure_eligible(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    const char** reason) {
    const auto fail = [&](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    if (operations.size() < qubit_count) {
        return fail("PhaseGraph route requires a complete leading H layer");
    }
    std::vector<std::uint8_t> seen(qubit_count, 0U);
    for (std::size_t index = 0U; index < qubit_count; ++index) {
        const Operation& operation = operations[index];
        if (operation.code != OperationCode::H) {
            return fail("PhaseGraph route requires a complete leading H layer");
        }
        if (operation.first >= qubit_count) {
            return fail("PhaseGraph route leading H qubit is out of range");
        }
        if (seen[operation.first] != 0U) {
            return fail("PhaseGraph route leading H layer contains duplicate qubits");
        }
        seen[operation.first] = 1U;
    }

    for (std::size_t index = qubit_count; index < operations.size(); ++index) {
        const Operation& operation = operations[index];
        switch (operation.code) {
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
                if (operation.first >= qubit_count) {
                    return fail("Phase-graph qubit index is out of range");
                }
                break;
            case OperationCode::Rz:
                if (operation.first >= qubit_count) {
                    return fail("Phase-graph qubit index is out of range");
                }
                if (!std::isfinite(operation.parameter)) {
                    return fail("Phase-graph Rz angle must be finite");
                }
                break;
            case OperationCode::Cz:
            case OperationCode::Swap:
                if (operation.first >= qubit_count || operation.second >= qubit_count) {
                    return fail("Phase-graph qubit index is out of range");
                }
                if (operation.first == operation.second) {
                    return fail(operation.code == OperationCode::Cz
                        ? "Phase-graph controlled phase requires distinct qubits"
                        : "Phase-graph SWAP requires distinct qubits");
                }
                break;
            case OperationCode::H:
                return fail("PhaseGraph route does not support H after the leading layer");
            case OperationCode::Rx:
            case OperationCode::Ry:
                return fail("PhaseGraph route supports phase-preserving rotations only");
            case OperationCode::Cnot:
                return fail("PhaseGraph route does not support CNOT");
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                return fail("PhaseGraph route supports unitary phase-graph operations only");
            default:
                return fail("PhaseGraph route received an unknown opcode");
        }
    }
    return true;
}

[[nodiscard]] PhaseGraphState execute_phase_graph_from_zero(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    PhaseGraphConfig config) {
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
            default:
                throw QStateError("PhaseGraph route execution violated structural preflight");
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
        case ExactExecutionRoute::UniformMagnitude:
            return "UniformMagnitude";
        case ExactExecutionRoute::BasisPermutation:
            return "BasisPermutation";
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

    const char* mps_reason = nullptr;
    if (mps_structure_eligible(qubit_count, operations, &mps_reason)) {
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
    } else {
        result.fallback_reason += "; mps: ";
        result.fallback_reason += mps_reason != nullptr ? mps_reason : "structural preflight failed";
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
    const char* basis_reason = nullptr;
    double basis_probability = 0.0;
    if (basis_permutation_probability(
            qubit_count,
            operations,
            basis_bits,
            &basis_probability,
            &basis_reason)) {
        result.value = basis_probability;
        result.route = ExactExecutionRoute::BasisPermutation;
        return result;
    }

    const char* uniform_reason = nullptr;
    if (uniform_magnitude_eligible(qubit_count, operations, &uniform_reason)) {
        result.value = uniform_probability(qubit_count);
        result.route = ExactExecutionRoute::UniformMagnitude;
        return result;
    }

    try {
        TensorNetworkCircuit tensor(qubit_count, operations, config_.tensor);
        result.value = tensor.amplitude(basis_bits, &result.tensor_stats).norm2();
        result.route = ExactExecutionRoute::TensorNetwork;
        return result;
    } catch (const QStateError& error) {
        result.fallback_reason = "basis_permutation: ";
        result.fallback_reason += basis_reason != nullptr
            ? basis_reason
            : "basis-permutation certificate failed";
        result.fallback_reason += "; uniform: ";
        result.fallback_reason += uniform_reason != nullptr
            ? uniform_reason
            : "uniform-magnitude certificate failed";
        result.fallback_reason += "; tensor: ";
        result.fallback_reason += failure_message(error);
    }

    const char* mps_reason = nullptr;
    if (mps_structure_eligible(qubit_count, operations, &mps_reason)) {
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
    } else {
        result.fallback_reason += "; mps: ";
        result.fallback_reason += mps_reason != nullptr ? mps_reason : "structural preflight failed";
    }

    const char* phase_reason = nullptr;
    if (phase_graph_structure_eligible(qubit_count, operations, &phase_reason)) {
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
    } else {
        result.fallback_reason += "; phase_graph: ";
        result.fallback_reason += phase_reason != nullptr ? phase_reason : "structural preflight failed";
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

    const char* mps_reason = nullptr;
    if (mps_structure_eligible(qubit_count_, operations, &mps_reason)) {
        try {
            MatrixProductState state = MatrixProductState::zero(qubit_count_, config_.mps);
            execute_mps(state, operations);
            mps_plan_.emplace(std::move(state));
        } catch (const QStateError& error) {
            mps_preparation_reason_ = failure_message(error);
        }
    } else {
        mps_preparation_reason_ = mps_reason != nullptr ? mps_reason : "structural preflight failed";
    }

    if (!mps_plan_.has_value()) {
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

ExactPreparedProbabilityPlan::ExactPreparedProbabilityPlan(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    ExactExecutionBrokerConfig config)
    : qubit_count_(validated_qubit_count(qubit_count)),
      config_(config) {
    validate_broker_config(config_);

    const char* basis_reason = nullptr;
    if (basis_permutation_state(
            qubit_count_,
            operations,
            &basis_permutation_bits_,
            &basis_reason)) {
        route_ = ExactExecutionRoute::BasisPermutation;
        return;
    }
    basis_permutation_bits_.clear();
    basis_permutation_bits_.shrink_to_fit();

    const char* uniform_reason = nullptr;
    if (uniform_magnitude_eligible(qubit_count_, operations, &uniform_reason)) {
        route_ = ExactExecutionRoute::UniformMagnitude;
        uniform_probability_ = uniform_probability(qubit_count_);
        return;
    }

    try {
        TensorNetworkCircuit tensor(qubit_count_, operations, config_.tensor);
        tensor_plan_.emplace(tensor.compile());
        route_ = ExactExecutionRoute::TensorNetwork;
        return;
    } catch (const QStateError& error) {
        fallback_reason_ = "basis_permutation: ";
        fallback_reason_ += basis_reason != nullptr
            ? basis_reason
            : "basis-permutation certificate failed";
        fallback_reason_ += "; uniform: ";
        fallback_reason_ += uniform_reason != nullptr
            ? uniform_reason
            : "uniform-magnitude certificate failed";
        fallback_reason_ += "; tensor: ";
        fallback_reason_ += failure_message(error);
    }

    const char* mps_reason = nullptr;
    if (mps_structure_eligible(qubit_count_, operations, &mps_reason)) {
        try {
            MatrixProductState state = MatrixProductState::zero(qubit_count_, config_.mps);
            execute_mps(state, operations);
            mps_state_.emplace(std::move(state));
            route_ = ExactExecutionRoute::PersistentMPS;
            return;
        } catch (const QStateError& error) {
            fallback_reason_ += "; mps: ";
            fallback_reason_ += failure_message(error);
        }
    } else {
        fallback_reason_ += "; mps: ";
        fallback_reason_ += mps_reason != nullptr ? mps_reason : "structural preflight failed";
    }

    const char* phase_reason = nullptr;
    if (phase_graph_structure_eligible(qubit_count_, operations, &phase_reason)) {
        try {
            phase_graph_state_.emplace(execute_phase_graph_from_zero(
                qubit_count_,
                operations,
                config_.phase_graph));
            route_ = ExactExecutionRoute::PhaseGraph;
            return;
        } catch (const QStateError& error) {
            fallback_reason_ += "; phase_graph: ";
            fallback_reason_ += failure_message(error);
        }
    } else {
        fallback_reason_ += "; phase_graph: ";
        fallback_reason_ += phase_reason != nullptr ? phase_reason : "structural preflight failed";
    }

    register_state_.emplace(qubit_count_, config_.register_state);
    OperationPlan plan(operations);
    plan.execute(*register_state_);
    route_ = ExactExecutionRoute::Register;
}

ExactProbabilityResult ExactPreparedProbabilityPlan::probability(
    std::span<const std::uint8_t> basis_bits) const {
    validate_basis_width(qubit_count_, basis_bits);

    ExactProbabilityResult result;
    result.route = route_;
    result.fallback_reason = fallback_reason_;
    switch (route_) {
        case ExactExecutionRoute::BasisPermutation: {
            if (basis_permutation_bits_.size() != qubit_count_) {
                throw QStateError("Prepared BasisPermutation plan is missing its terminal state");
            }
            bool match = true;
            for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
                if (basis_permutation_bits_[qubit] != basis_bits[qubit]) {
                    match = false;
                    break;
                }
            }
            result.value = match ? 1.0 : 0.0;
            return result;
        }
        case ExactExecutionRoute::UniformMagnitude:
            result.value = uniform_probability_;
            return result;
        case ExactExecutionRoute::TensorNetwork:
            if (!tensor_plan_.has_value()) {
                throw QStateError("Prepared tensor probability plan is missing its contraction plan");
            }
            result.value = tensor_plan_->amplitude(basis_bits, &result.tensor_stats).norm2();
            return result;
        case ExactExecutionRoute::PersistentMPS:
            if (!mps_state_.has_value()) {
                throw QStateError("Prepared MPS probability plan is missing its state");
            }
            result.value = mps_state_->amplitude(basis_bits).norm2();
            return result;
        case ExactExecutionRoute::PhaseGraph:
            if (!phase_graph_state_.has_value()) {
                throw QStateError("Prepared PhaseGraph probability plan is missing its state");
            }
            result.value = phase_graph_state_->amplitude_bits(basis_bits).norm2();
            return result;
        case ExactExecutionRoute::Register:
            if (!register_state_.has_value()) {
                throw QStateError("Prepared register probability plan is missing its state");
            }
            result.value = register_state_->amplitude_bits(basis_bits).norm2();
            return result;
        case ExactExecutionRoute::CausalPauli:
            throw QStateError("Prepared probability plan cannot use CausalPauli");
    }
    throw QStateError("Prepared probability plan has an unknown route");
}

ExactProbabilityResult ExactPreparedProbabilityPlan::probability(BasisIndex basis) const {
    if (qubit_count_ > std::numeric_limits<BasisIndex>::digits) {
        throw QStateError("Prepared probability BasisIndex query is limited to 64 qubits");
    }
    if (qubit_count_ < std::numeric_limits<BasisIndex>::digits &&
        basis >= (BasisIndex{1} << qubit_count_)) {
        throw QStateError("Prepared probability basis index is out of range");
    }

    std::vector<std::uint8_t> bits(qubit_count_, 0U);
    for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & BasisIndex{1});
    }
    return probability(bits);
}

std::size_t ExactPreparedProbabilityPlan::estimated_bytes() const noexcept {
    std::size_t total = sizeof(*this);
    total += basis_permutation_bits_.capacity() * sizeof(std::uint8_t);
    if (tensor_plan_.has_value()) {
        total += dynamic_bytes(tensor_plan_->estimated_bytes(), sizeof(TensorContractionPlan));
    }
    if (mps_state_.has_value()) {
        total += dynamic_bytes(mps_state_->estimated_bytes(), sizeof(MatrixProductState));
    }
    if (phase_graph_state_.has_value()) {
        total += dynamic_bytes(phase_graph_state_->estimated_bytes(), sizeof(PhaseGraphState));
    }
    if (register_state_.has_value()) {
        total += dynamic_bytes(register_state_->estimated_bytes(), sizeof(QRegister));
    }
    total += fallback_reason_.capacity();
    return total;
}

}  // namespace qubit
