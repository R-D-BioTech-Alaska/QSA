#pragma once

#include "qubit/qadjoint_batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace qubit {

namespace vectorized_adjoint_detail {

using BatchedState = std::vector<QComplex>;

[[nodiscard]] inline bool is_single_qubit(OperationCode code) noexcept {
    switch (code) {
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
        case OperationCode::Rz:
            return true;
        default:
            return false;
    }
}

inline void fill_bound_matrices(
    const ParameterizedOperation& templated,
    std::span<const double> parameter_rows,
    std::size_t row_count,
    std::size_t parameter_count,
    bool reverse,
    std::vector<QMatrix2>& matrices) {
    matrices.clear();
    if (templated.parameter_slot < 0) {
        QMatrix2 matrix = adjoint_detail::single_matrix(templated.operation);
        if (reverse) {
            matrix = adjoint_detail::dagger(matrix);
        }
        matrices.push_back(matrix);
        return;
    }

    const std::size_t slot =
        static_cast<std::size_t>(templated.parameter_slot);
    if (slot >= parameter_count) {
        throw QStateError(
            "Vectorized adjoint parameter slot is outside the parameter row");
    }
    matrices.resize(row_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        const double value = parameter_rows[row * parameter_count + slot];
        if (!std::isfinite(value)) {
            throw QStateError(
                "Vectorized adjoint parameter row contains a non-finite value");
        }
        Operation operation = templated.operation;
        operation.parameter = value;
        QMatrix2 matrix = adjoint_detail::single_matrix(operation);
        matrices[row] = reverse ? adjoint_detail::dagger(matrix) : matrix;
    }
}

inline void apply_single(
    BatchedState& state,
    std::size_t row_count,
    std::size_t qubit_count,
    QubitId qubit,
    std::span<const QMatrix2> matrices) {
    adjoint_detail::validate_qubit(qubit_count, qubit);
    if (matrices.size() != 1U && matrices.size() != row_count) {
        throw QStateError("Vectorized single-gate matrix count is invalid");
    }
    const BasisIndex mask = BasisIndex{1} << qubit;
    const BasisIndex dimension = BasisIndex{1} << qubit_count;
    for (BasisIndex base = 0U; base < dimension; ++base) {
        if ((base & mask) != 0U) {
            continue;
        }
        const BasisIndex one = base | mask;
        const std::size_t zero_offset =
            static_cast<std::size_t>(base) * row_count;
        const std::size_t one_offset =
            static_cast<std::size_t>(one) * row_count;
        for (std::size_t row = 0U; row < row_count; ++row) {
            const QMatrix2& matrix =
                matrices.size() == 1U ? matrices.front() : matrices[row];
            const QComplex zero_value = state[zero_offset + row];
            const QComplex one_value = state[one_offset + row];
            state[zero_offset + row] =
                matrix(0, 0) * zero_value + matrix(0, 1) * one_value;
            state[one_offset + row] =
                matrix(1, 0) * zero_value + matrix(1, 1) * one_value;
        }
    }
}

inline void apply_cnot(
    BatchedState& state,
    std::size_t row_count,
    std::size_t qubit_count,
    QubitId control,
    QubitId target) {
    adjoint_detail::validate_qubit(qubit_count, control);
    adjoint_detail::validate_qubit(qubit_count, target);
    if (control == target) {
        throw QStateError("Vectorized adjoint CNOT qubits must differ");
    }
    const BasisIndex control_mask = BasisIndex{1} << control;
    const BasisIndex target_mask = BasisIndex{1} << target;
    const BasisIndex dimension = BasisIndex{1} << qubit_count;
    for (BasisIndex basis = 0U; basis < dimension; ++basis) {
        if ((basis & control_mask) == 0U || (basis & target_mask) != 0U) {
            continue;
        }
        const BasisIndex paired = basis | target_mask;
        const std::size_t first_offset =
            static_cast<std::size_t>(basis) * row_count;
        const std::size_t second_offset =
            static_cast<std::size_t>(paired) * row_count;
        for (std::size_t row = 0U; row < row_count; ++row) {
            std::swap(state[first_offset + row], state[second_offset + row]);
        }
    }
}

inline void apply_cz(
    BatchedState& state,
    std::size_t row_count,
    std::size_t qubit_count,
    QubitId first,
    QubitId second) {
    adjoint_detail::validate_qubit(qubit_count, first);
    adjoint_detail::validate_qubit(qubit_count, second);
    if (first == second) {
        throw QStateError("Vectorized adjoint CZ qubits must differ");
    }
    const BasisIndex first_mask = BasisIndex{1} << first;
    const BasisIndex second_mask = BasisIndex{1} << second;
    const BasisIndex dimension = BasisIndex{1} << qubit_count;
    for (BasisIndex basis = 0U; basis < dimension; ++basis) {
        if ((basis & first_mask) == 0U || (basis & second_mask) == 0U) {
            continue;
        }
        const std::size_t offset =
            static_cast<std::size_t>(basis) * row_count;
        for (std::size_t row = 0U; row < row_count; ++row) {
            state[offset + row] *= -1.0;
        }
    }
}

inline void apply_swap(
    BatchedState& state,
    std::size_t row_count,
    std::size_t qubit_count,
    QubitId first,
    QubitId second) {
    adjoint_detail::validate_qubit(qubit_count, first);
    adjoint_detail::validate_qubit(qubit_count, second);
    if (first == second) {
        return;
    }
    const BasisIndex first_mask = BasisIndex{1} << first;
    const BasisIndex second_mask = BasisIndex{1} << second;
    const BasisIndex dimension = BasisIndex{1} << qubit_count;
    for (BasisIndex basis = 0U; basis < dimension; ++basis) {
        const bool first_one = (basis & first_mask) != 0U;
        const bool second_one = (basis & second_mask) != 0U;
        if (first_one || !second_one) {
            continue;
        }
        const BasisIndex paired = basis ^ first_mask ^ second_mask;
        const std::size_t first_offset =
            static_cast<std::size_t>(basis) * row_count;
        const std::size_t second_offset =
            static_cast<std::size_t>(paired) * row_count;
        for (std::size_t row = 0U; row < row_count; ++row) {
            std::swap(state[first_offset + row], state[second_offset + row]);
        }
    }
}

inline void apply_operation(
    BatchedState& state,
    std::size_t row_count,
    std::size_t qubit_count,
    const ParameterizedOperation& templated,
    std::span<const double> parameter_rows,
    std::size_t parameter_count,
    bool reverse,
    std::vector<QMatrix2>& matrices) {
    const OperationCode code = templated.operation.code;
    if (is_single_qubit(code)) {
        fill_bound_matrices(
            templated,
            parameter_rows,
            row_count,
            parameter_count,
            reverse,
            matrices);
        apply_single(
            state,
            row_count,
            qubit_count,
            templated.operation.first,
            matrices);
        return;
    }
    switch (code) {
        case OperationCode::Cnot:
            apply_cnot(
                state,
                row_count,
                qubit_count,
                templated.operation.first,
                templated.operation.second);
            return;
        case OperationCode::Cz:
            apply_cz(
                state,
                row_count,
                qubit_count,
                templated.operation.first,
                templated.operation.second);
            return;
        case OperationCode::Swap:
            apply_swap(
                state,
                row_count,
                qubit_count,
                templated.operation.first,
                templated.operation.second);
            return;
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            throw QStateError(
                "Vectorized adjoint does not support trajectory noise");
        default:
            throw QStateError(
                "Vectorized adjoint operation contains an unknown opcode");
    }
}

[[nodiscard]] inline BatchedState broadcast_initial(
    const adjoint_detail::StateVector& initial,
    std::size_t row_count) {
    BatchedState state(initial.size() * row_count);
    for (std::size_t basis = 0U; basis < initial.size(); ++basis) {
        const std::size_t offset = basis * row_count;
        for (std::size_t row = 0U; row < row_count; ++row) {
            state[offset + row] = initial[basis];
        }
    }
    return state;
}

inline void observable_values(
    const BatchedState& state,
    std::size_t row_count,
    const PauliSupportPlan& observables,
    double imaginary_tolerance,
    std::span<double> output) {
    const auto& supports = observables.observables();
    if (output.size() != row_count * supports.size()) {
        throw QStateError(
            "Vectorized observable output size is incorrect");
    }
    const BasisIndex dimension =
        static_cast<BasisIndex>(state.size() / row_count);
    std::vector<QComplex> expectations(row_count);
    for (std::size_t observable = 0U;
         observable < supports.size();
         ++observable) {
        std::fill(expectations.begin(), expectations.end(), QComplex{});
        for (BasisIndex source = 0U; source < dimension; ++source) {
            BasisIndex target = 0U;
            const QComplex coefficient = adjoint_detail::pauli_coefficient(
                source,
                supports[observable],
                target);
            const std::size_t source_offset =
                static_cast<std::size_t>(source) * row_count;
            const std::size_t target_offset =
                static_cast<std::size_t>(target) * row_count;
            for (std::size_t row = 0U; row < row_count; ++row) {
                expectations[row] += state[target_offset + row].conjugate()
                    * coefficient
                    * state[source_offset + row];
            }
        }
        for (std::size_t row = 0U; row < row_count; ++row) {
            const double scale = 1.0 + std::abs(expectations[row].re);
            if (std::abs(expectations[row].im) > imaginary_tolerance * scale) {
                throw QStateError(
                    "Vectorized Hermitian Pauli support developed a non-real value");
            }
            output[row * supports.size() + observable] = expectations[row].re;
        }
    }
}

inline void weighted_pauli_action(
    const BatchedState& state,
    std::size_t row_count,
    const PauliSupportPlan& observables,
    std::span<const double> cotangent_rows,
    BatchedState& adjoint) {
    const auto& supports = observables.observables();
    if (cotangent_rows.size() != row_count * supports.size()) {
        throw QStateError(
            "Vectorized cotangent buffer size is incorrect");
    }
    adjoint.assign(state.size(), QComplex{});
    const BasisIndex dimension =
        static_cast<BasisIndex>(state.size() / row_count);
    for (std::size_t observable = 0U;
         observable < supports.size();
         ++observable) {
        for (BasisIndex source = 0U; source < dimension; ++source) {
            BasisIndex target = 0U;
            const QComplex coefficient = adjoint_detail::pauli_coefficient(
                source,
                supports[observable],
                target);
            const std::size_t source_offset =
                static_cast<std::size_t>(source) * row_count;
            const std::size_t target_offset =
                static_cast<std::size_t>(target) * row_count;
            for (std::size_t row = 0U; row < row_count; ++row) {
                const double weight =
                    cotangent_rows[row * supports.size() + observable];
                if (!std::isfinite(weight)) {
                    throw QStateError(
                        "Vectorized cotangent row contains a non-finite value");
                }
                if (weight == 0.0) {
                    continue;
                }
                adjoint[target_offset + row] +=
                    weight * coefficient * state[source_offset + row];
            }
        }
    }
}

inline void derivative_inner_products(
    const BatchedState& state_before,
    const BatchedState& adjoint_after,
    std::size_t row_count,
    std::size_t qubit_count,
    const ParameterizedOperation& templated,
    std::span<const double> parameter_rows,
    std::size_t parameter_count,
    std::span<double> gradient_output) {
    const std::size_t slot =
        static_cast<std::size_t>(templated.parameter_slot);
    std::vector<QMatrix2> derivatives(row_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        const double value = parameter_rows[row * parameter_count + slot];
        derivatives[row] = adjoint_detail::derivative_matrix(
            templated.operation.code,
            value);
    }
    std::vector<QComplex> inner(row_count);
    const BasisIndex mask = BasisIndex{1} << templated.operation.first;
    const BasisIndex dimension = BasisIndex{1} << qubit_count;
    for (BasisIndex base = 0U; base < dimension; ++base) {
        if ((base & mask) != 0U) {
            continue;
        }
        const BasisIndex one = base | mask;
        const std::size_t zero_offset =
            static_cast<std::size_t>(base) * row_count;
        const std::size_t one_offset =
            static_cast<std::size_t>(one) * row_count;
        for (std::size_t row = 0U; row < row_count; ++row) {
            const QComplex zero_value = state_before[zero_offset + row];
            const QComplex one_value = state_before[one_offset + row];
            const QMatrix2& derivative = derivatives[row];
            const QComplex derivative_zero =
                derivative(0, 0) * zero_value
                + derivative(0, 1) * one_value;
            const QComplex derivative_one =
                derivative(1, 0) * zero_value
                + derivative(1, 1) * one_value;
            inner[row] += adjoint_after[zero_offset + row].conjugate()
                * derivative_zero;
            inner[row] += adjoint_after[one_offset + row].conjugate()
                * derivative_one;
        }
    }
    for (std::size_t row = 0U; row < row_count; ++row) {
        gradient_output[row * parameter_count + slot] += 2.0 * inner[row].re;
    }
}

}  // namespace vectorized_adjoint_detail

inline void observables_many_vectorized(
    const QRegister& initial,
    const ParameterizedOperationPlan& plan,
    std::span<const double> parameter_rows,
    std::size_t row_count,
    const PauliSupportPlan& observables,
    std::size_t max_qubits,
    double imaginary_tolerance,
    std::span<double> values_output) {
    const std::size_t parameter_count = plan.parameter_count();
    adjoint_batch_detail::validate_common(
        initial,
        plan,
        observables,
        parameter_count,
        max_qubits,
        imaginary_tolerance);
    if (parameter_rows.size() != row_count * parameter_count) {
        throw QStateError(
            "Vectorized primal parameter buffer size is incorrect");
    }
    if (values_output.size() != row_count * observables.observable_count()) {
        throw QStateError("Vectorized primal value output size is incorrect");
    }
    if (row_count == 0U) {
        return;
    }

    const std::size_t qubit_count = initial.qubit_count();
    const adjoint_detail::StateVector local = initial.materialize(max_qubits);
    vectorized_adjoint_detail::BatchedState state =
        vectorized_adjoint_detail::broadcast_initial(local, row_count);
    std::vector<QMatrix2> matrices;
    matrices.reserve(row_count);
    for (const ParameterizedOperation& templated : plan.operations()) {
        vectorized_adjoint_detail::apply_operation(
            state,
            row_count,
            qubit_count,
            templated,
            parameter_rows,
            parameter_count,
            false,
            matrices);
    }
    vectorized_adjoint_detail::observable_values(
        state,
        row_count,
        observables,
        imaginary_tolerance,
        values_output);
}

inline void weighted_adjoint_many_vectorized(
    const QRegister& initial,
    const ParameterizedOperationPlan& plan,
    std::span<const double> parameter_rows,
    std::size_t row_count,
    const PauliSupportPlan& observables,
    std::span<const double> cotangent_rows,
    std::size_t max_qubits,
    double imaginary_tolerance,
    std::span<double> values_output,
    std::span<double> gradient_output) {
    const std::size_t parameter_count = plan.parameter_count();
    const std::size_t observable_count = observables.observable_count();
    adjoint_batch_detail::validate_common(
        initial,
        plan,
        observables,
        parameter_count,
        max_qubits,
        imaginary_tolerance);
    if (parameter_rows.size() != row_count * parameter_count) {
        throw QStateError(
            "Vectorized adjoint parameter buffer size is incorrect");
    }
    if (cotangent_rows.size() != row_count * observable_count) {
        throw QStateError(
            "Vectorized adjoint cotangent buffer size is incorrect");
    }
    if (values_output.size() != row_count * observable_count) {
        throw QStateError("Vectorized adjoint value output size is incorrect");
    }
    if (gradient_output.size() != row_count * parameter_count) {
        throw QStateError(
            "Vectorized adjoint gradient output size is incorrect");
    }
    if (row_count == 0U) {
        return;
    }

    const std::size_t qubit_count = initial.qubit_count();
    const adjoint_detail::StateVector local = initial.materialize(max_qubits);
    vectorized_adjoint_detail::BatchedState primal =
        vectorized_adjoint_detail::broadcast_initial(local, row_count);
    std::vector<QMatrix2> matrices;
    matrices.reserve(row_count);
    for (const ParameterizedOperation& templated : plan.operations()) {
        vectorized_adjoint_detail::apply_operation(
            primal,
            row_count,
            qubit_count,
            templated,
            parameter_rows,
            parameter_count,
            false,
            matrices);
    }
    vectorized_adjoint_detail::observable_values(
        primal,
        row_count,
        observables,
        imaginary_tolerance,
        values_output);
    std::fill(gradient_output.begin(), gradient_output.end(), 0.0);

    vectorized_adjoint_detail::BatchedState adjoint;
    vectorized_adjoint_detail::weighted_pauli_action(
        primal,
        row_count,
        observables,
        cotangent_rows,
        adjoint);
    const auto& templates = plan.operations();
    for (std::size_t reverse_index = templates.size();
         reverse_index > 0U;
         --reverse_index) {
        const ParameterizedOperation& templated = templates[reverse_index - 1U];
        vectorized_adjoint_detail::apply_operation(
            primal,
            row_count,
            qubit_count,
            templated,
            parameter_rows,
            parameter_count,
            true,
            matrices);
        if (templated.parameter_slot >= 0) {
            vectorized_adjoint_detail::derivative_inner_products(
                primal,
                adjoint,
                row_count,
                qubit_count,
                templated,
                parameter_rows,
                parameter_count,
                gradient_output);
        }
        vectorized_adjoint_detail::apply_operation(
            adjoint,
            row_count,
            qubit_count,
            templated,
            parameter_rows,
            parameter_count,
            true,
            matrices);
    }
}

}  // namespace qubit
