#pragma once

#include "qubit/qpauli_support.hpp"
#include "qubit/qplan.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct WeightedAdjointResult {
    std::vector<double> values{};
    std::vector<double> gradient{};
};

namespace adjoint_detail {

using StateVector = std::vector<QComplex>;

[[nodiscard]] inline QMatrix2 single_matrix(const Operation& operation) {
    switch (operation.code) {
        case OperationCode::X: return gates::x();
        case OperationCode::Y: return gates::y();
        case OperationCode::Z: return gates::z();
        case OperationCode::H: return gates::h();
        case OperationCode::S: return gates::s();
        case OperationCode::Sdg: return gates::sdg();
        case OperationCode::T: return gates::t();
        case OperationCode::Tdg: return gates::tdg();
        case OperationCode::Rx: return gates::rx(operation.parameter);
        case OperationCode::Ry: return gates::ry(operation.parameter);
        case OperationCode::Rz: return gates::rz(operation.parameter);
        default:
            throw QStateError(
                "Adjoint operation is not a supported single-qubit gate");
    }
}

[[nodiscard]] inline QMatrix2 dagger(const QMatrix2& matrix) {
    QMatrix2 result{};
    for (std::size_t row = 0; row < 2U; ++row) {
        for (std::size_t column = 0; column < 2U; ++column) {
            result.values[row * 2U + column] =
                matrix(column, row).conjugate();
        }
    }
    return result;
}

[[nodiscard]] inline QMatrix2 derivative_matrix(
    OperationCode code,
    double parameter) {
    const double half = parameter / 2.0;
    const double cosine = std::cos(half);
    const double sine = std::sin(half);
    QMatrix2 result{};
    switch (code) {
        case OperationCode::Rx:
            result.values = {
                QComplex{-0.5 * sine, 0.0},
                QComplex{0.0, -0.5 * cosine},
                QComplex{0.0, -0.5 * cosine},
                QComplex{-0.5 * sine, 0.0},
            };
            return result;
        case OperationCode::Ry:
            result.values = {
                QComplex{-0.5 * sine, 0.0},
                QComplex{-0.5 * cosine, 0.0},
                QComplex{0.5 * cosine, 0.0},
                QComplex{-0.5 * sine, 0.0},
            };
            return result;
        case OperationCode::Rz: {
            const QComplex zero = QComplex::from_polar(1.0, -half);
            const QComplex one = QComplex::from_polar(1.0, half);
            result.values = {
                QComplex{0.0, -0.5} * zero,
                QComplex{},
                QComplex{},
                QComplex{0.0, 0.5} * one,
            };
            return result;
        }
        default:
            throw QStateError(
                "Adjoint parameters are supported only on Rx, Ry, and Rz gates");
    }
}

inline void validate_qubit(std::size_t qubit_count, QubitId qubit) {
    if (static_cast<std::size_t>(qubit) >= qubit_count) {
        throw QStateError(
            "Adjoint operation qubit is outside the local register");
    }
}

inline void apply_single(
    StateVector& state,
    std::size_t qubit_count,
    QubitId qubit,
    const QMatrix2& matrix) {
    validate_qubit(qubit_count, qubit);
    const BasisIndex mask = BasisIndex{1} << qubit;
    const BasisIndex dimension = static_cast<BasisIndex>(state.size());
    for (BasisIndex base = 0; base < dimension; ++base) {
        if ((base & mask) != 0U) {
            continue;
        }
        const BasisIndex one = base | mask;
        const QComplex zero_value = state[static_cast<std::size_t>(base)];
        const QComplex one_value = state[static_cast<std::size_t>(one)];
        state[static_cast<std::size_t>(base)] =
            matrix(0, 0) * zero_value + matrix(0, 1) * one_value;
        state[static_cast<std::size_t>(one)] =
            matrix(1, 0) * zero_value + matrix(1, 1) * one_value;
    }
}

inline void apply_cnot(
    StateVector& state,
    std::size_t qubit_count,
    QubitId control,
    QubitId target) {
    validate_qubit(qubit_count, control);
    validate_qubit(qubit_count, target);
    if (control == target) {
        throw QStateError("Adjoint CNOT qubits must differ");
    }
    const BasisIndex control_mask = BasisIndex{1} << control;
    const BasisIndex target_mask = BasisIndex{1} << target;
    const BasisIndex dimension = static_cast<BasisIndex>(state.size());
    for (BasisIndex basis = 0; basis < dimension; ++basis) {
        if ((basis & control_mask) == 0U || (basis & target_mask) != 0U) {
            continue;
        }
        const BasisIndex paired = basis | target_mask;
        std::swap(
            state[static_cast<std::size_t>(basis)],
            state[static_cast<std::size_t>(paired)]);
    }
}

inline void apply_cz(
    StateVector& state,
    std::size_t qubit_count,
    QubitId first,
    QubitId second) {
    validate_qubit(qubit_count, first);
    validate_qubit(qubit_count, second);
    if (first == second) {
        throw QStateError("Adjoint CZ qubits must differ");
    }
    const BasisIndex first_mask = BasisIndex{1} << first;
    const BasisIndex second_mask = BasisIndex{1} << second;
    const BasisIndex dimension = static_cast<BasisIndex>(state.size());
    for (BasisIndex basis = 0; basis < dimension; ++basis) {
        if ((basis & first_mask) != 0U && (basis & second_mask) != 0U) {
            state[static_cast<std::size_t>(basis)] *= -1.0;
        }
    }
}

inline void apply_swap(
    StateVector& state,
    std::size_t qubit_count,
    QubitId first,
    QubitId second) {
    validate_qubit(qubit_count, first);
    validate_qubit(qubit_count, second);
    if (first == second) {
        return;
    }
    const BasisIndex first_mask = BasisIndex{1} << first;
    const BasisIndex second_mask = BasisIndex{1} << second;
    const BasisIndex dimension = static_cast<BasisIndex>(state.size());
    for (BasisIndex basis = 0; basis < dimension; ++basis) {
        const bool first_one = (basis & first_mask) != 0U;
        const bool second_one = (basis & second_mask) != 0U;
        if (first_one || !second_one) {
            continue;
        }
        const BasisIndex paired = basis ^ first_mask ^ second_mask;
        std::swap(
            state[static_cast<std::size_t>(basis)],
            state[static_cast<std::size_t>(paired)]);
    }
}

inline void apply_operation(
    StateVector& state,
    std::size_t qubit_count,
    const Operation& operation,
    bool reverse) {
    switch (operation.code) {
        case OperationCode::X:
        case OperationCode::Y:
        case OperationCode::Z:
        case OperationCode::H:
        case OperationCode::S:
        case OperationCode::Sdg:
        case OperationCode::T:
        case OperationCode::Tdg:
        case OperationCode::Rx:
        case OperationCode::Ry:
        case OperationCode::Rz: {
            QMatrix2 matrix = single_matrix(operation);
            if (reverse) {
                matrix = dagger(matrix);
            }
            apply_single(state, qubit_count, operation.first, matrix);
            return;
        }
        case OperationCode::Cnot:
            apply_cnot(state, qubit_count, operation.first, operation.second);
            return;
        case OperationCode::Cz:
            apply_cz(state, qubit_count, operation.first, operation.second);
            return;
        case OperationCode::Swap:
            apply_swap(state, qubit_count, operation.first, operation.second);
            return;
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            throw QStateError(
                "Adjoint differentiation does not support trajectory noise");
        default:
            throw QStateError("Adjoint operation contains an unknown opcode");
    }
}

[[nodiscard]] inline QComplex derivative_inner_product(
    const StateVector& state_before,
    const StateVector& adjoint_after,
    std::size_t qubit_count,
    QubitId qubit,
    const QMatrix2& derivative) {
    validate_qubit(qubit_count, qubit);
    const BasisIndex mask = BasisIndex{1} << qubit;
    const BasisIndex dimension = static_cast<BasisIndex>(state_before.size());
    QComplex result{};
    for (BasisIndex base = 0; base < dimension; ++base) {
        if ((base & mask) != 0U) {
            continue;
        }
        const BasisIndex one = base | mask;
        const QComplex zero_value = state_before[static_cast<std::size_t>(base)];
        const QComplex one_value = state_before[static_cast<std::size_t>(one)];
        const QComplex derivative_zero =
            derivative(0, 0) * zero_value + derivative(0, 1) * one_value;
        const QComplex derivative_one =
            derivative(1, 0) * zero_value + derivative(1, 1) * one_value;
        result += adjoint_after[static_cast<std::size_t>(base)].conjugate()
            * derivative_zero;
        result += adjoint_after[static_cast<std::size_t>(one)].conjugate()
            * derivative_one;
    }
    return result;
}

[[nodiscard]] inline QComplex pauli_coefficient(
    BasisIndex source,
    std::span<const PauliSupportTerm> observable,
    BasisIndex& target) {
    QComplex coefficient{1.0, 0.0};
    target = source;
    for (const PauliSupportTerm& term : observable) {
        const bool one = ((source >> term.qubit) & BasisIndex{1}) != 0U;
        switch (term.axis) {
            case 'X':
                target ^= BasisIndex{1} << term.qubit;
                break;
            case 'Y':
                target ^= BasisIndex{1} << term.qubit;
                coefficient *= one
                    ? QComplex{0.0, -1.0}
                    : QComplex{0.0, 1.0};
                break;
            case 'Z':
                if (one) {
                    coefficient *= -1.0;
                }
                break;
            default:
                throw QStateError(
                    "Adjoint Pauli support contains an invalid axis");
        }
    }
    return coefficient;
}

inline void weighted_pauli_action(
    const StateVector& state,
    const PauliSupportPlan& observables,
    std::span<const double> cotangent,
    StateVector& adjoint) {
    if (cotangent.size() != observables.observable_count()) {
        throw QStateError(
            "Adjoint cotangent width differs from observable count");
    }
    adjoint.assign(state.size(), QComplex{});
    const auto& supports = observables.observables();
    for (std::size_t observable = 0;
         observable < supports.size();
         ++observable) {
        const double weight = cotangent[observable];
        if (!std::isfinite(weight)) {
            throw QStateError(
                "Adjoint cotangent contains a non-finite value");
        }
        if (weight == 0.0) {
            continue;
        }
        const auto& support = supports[observable];
        const BasisIndex dimension = static_cast<BasisIndex>(state.size());
        for (BasisIndex source = 0; source < dimension; ++source) {
            BasisIndex target = 0;
            const QComplex coefficient = pauli_coefficient(
                source,
                support,
                target);
            adjoint[static_cast<std::size_t>(target)] +=
                weight * coefficient * state[static_cast<std::size_t>(source)];
        }
    }
}

}  // namespace adjoint_detail

[[nodiscard]] inline WeightedAdjointResult weighted_adjoint(
    const QRegister& initial,
    const ParameterizedOperationPlan& plan,
    std::span<const double> parameters,
    const PauliSupportPlan& observables,
    std::span<const double> cotangent,
    std::size_t max_qubits = 16U) {
    const std::size_t qubit_count = initial.qubit_count();
    if (max_qubits == 0U || max_qubits > 24U) {
        throw QStateError("Adjoint max_qubits must be between 1 and 24");
    }
    if (qubit_count > max_qubits) {
        throw QStateError("Adjoint local register exceeds max_qubits");
    }
    if (observables.qubit_count() != qubit_count) {
        throw QStateError(
            "Adjoint observable width differs from local register");
    }
    if (parameters.size() != plan.parameter_count()) {
        throw QStateError(
            "Adjoint parameter width differs from plan parameter count");
    }
    for (double value : parameters) {
        if (!std::isfinite(value)) {
            throw QStateError(
                "Adjoint parameter vector contains a non-finite value");
        }
    }

    const auto& templates = plan.operations();
    std::vector<Operation> operations;
    operations.reserve(templates.size());
    for (const ParameterizedOperation& templated : templates) {
        if (templated.sample_slot >= 0) {
            throw QStateError(
                "Adjoint differentiation does not support sample slots");
        }
        Operation operation = templated.operation;
        if (templated.parameter_slot >= 0) {
            const std::size_t slot =
                static_cast<std::size_t>(templated.parameter_slot);
            if (slot >= parameters.size()) {
                throw QStateError(
                    "Adjoint parameter slot is outside the parameter vector");
            }
            switch (operation.code) {
                case OperationCode::Rx:
                case OperationCode::Ry:
                case OperationCode::Rz:
                    break;
                default:
                    throw QStateError(
                        "Adjoint parameter slots are supported only on Rx, Ry, and Rz");
            }
            operation.parameter = parameters[slot];
        }
        operations.push_back(operation);
    }

    using adjoint_detail::StateVector;
    StateVector primal = initial.materialize(max_qubits);
    for (const Operation& operation : operations) {
        adjoint_detail::apply_operation(
            primal,
            qubit_count,
            operation,
            false);
    }

    WeightedAdjointResult result;
    {
        QRegister final_register = QRegister::from_amplitudes(
            primal,
            initial.config());
        result.values = observables.execute(final_register);
    }
    result.gradient.assign(plan.parameter_count(), 0.0);

    StateVector adjoint;
    adjoint_detail::weighted_pauli_action(
        primal,
        observables,
        cotangent,
        adjoint);
    for (std::size_t reverse_index = operations.size();
         reverse_index > 0U;
         --reverse_index) {
        const std::size_t index = reverse_index - 1U;
        const Operation& operation = operations[index];
        const ParameterizedOperation& templated = templates[index];

        adjoint_detail::apply_operation(
            primal,
            qubit_count,
            operation,
            true);
        if (templated.parameter_slot >= 0) {
            const QMatrix2 derivative = adjoint_detail::derivative_matrix(
                operation.code,
                operation.parameter);
            const QComplex inner = adjoint_detail::derivative_inner_product(
                primal,
                adjoint,
                qubit_count,
                operation.first,
                derivative);
            result.gradient[
                static_cast<std::size_t>(templated.parameter_slot)] +=
                2.0 * inner.re;
        }
        adjoint_detail::apply_operation(
            adjoint,
            qubit_count,
            operation,
            true);
    }
    return result;
}

}  // namespace qubit
