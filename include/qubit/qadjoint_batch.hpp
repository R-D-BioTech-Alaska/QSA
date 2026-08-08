#pragma once

#include "qubit/qadjoint.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace qubit {

namespace adjoint_batch_detail {

using adjoint_detail::StateVector;

inline void validate_common(
    const QRegister& initial,
    const ParameterizedOperationPlan& plan,
    const PauliSupportPlan& observables,
    std::size_t parameter_count,
    std::size_t max_qubits,
    double imaginary_tolerance) {
    const std::size_t qubit_count = initial.qubit_count();
    if (max_qubits == 0U || max_qubits > 24U) {
        throw QStateError("Batch adjoint max_qubits must be between 1 and 24");
    }
    if (qubit_count > max_qubits) {
        throw QStateError("Batch adjoint local register exceeds max_qubits");
    }
    if (observables.qubit_count() != qubit_count) {
        throw QStateError(
            "Batch adjoint observable width differs from local register");
    }
    if (parameter_count != plan.parameter_count()) {
        throw QStateError(
            "Batch adjoint parameter width differs from plan parameter count");
    }
    if (!std::isfinite(imaginary_tolerance) || imaginary_tolerance <= 0.0) {
        throw QStateError(
            "Batch adjoint imaginary tolerance must be finite and positive");
    }
    for (const ParameterizedOperation& templated : plan.operations()) {
        if (templated.sample_slot >= 0) {
            throw QStateError(
                "Batch adjoint differentiation does not support sample slots");
        }
        if (templated.parameter_slot < 0) {
            continue;
        }
        switch (templated.operation.code) {
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Rz:
                break;
            default:
                throw QStateError(
                    "Batch adjoint parameter slots are supported only on Rx, Ry, and Rz");
        }
    }
}

[[nodiscard]] inline Operation bind_operation(
    const ParameterizedOperation& templated,
    std::span<const double> parameters) {
    Operation operation = templated.operation;
    if (templated.parameter_slot < 0) {
        return operation;
    }
    const std::size_t slot =
        static_cast<std::size_t>(templated.parameter_slot);
    if (slot >= parameters.size()) {
        throw QStateError(
            "Batch adjoint parameter slot is outside the parameter row");
    }
    const double value = parameters[slot];
    if (!std::isfinite(value)) {
        throw QStateError(
            "Batch adjoint parameter row contains a non-finite value");
    }
    operation.parameter = value;
    return operation;
}

inline void observable_values(
    const StateVector& state,
    const PauliSupportPlan& observables,
    double imaginary_tolerance,
    std::span<double> output) {
    const auto& supports = observables.observables();
    if (output.size() != supports.size()) {
        throw QStateError(
            "Batch adjoint observable output width is incorrect");
    }
    const BasisIndex dimension = static_cast<BasisIndex>(state.size());
    for (std::size_t observable = 0U;
         observable < supports.size();
         ++observable) {
        QComplex expectation{};
        for (BasisIndex source = 0U; source < dimension; ++source) {
            BasisIndex target = 0U;
            const QComplex coefficient = adjoint_detail::pauli_coefficient(
                source,
                supports[observable],
                target);
            expectation += state[static_cast<std::size_t>(target)].conjugate()
                * coefficient
                * state[static_cast<std::size_t>(source)];
        }
        const double scale = 1.0 + std::abs(expectation.re);
        if (std::abs(expectation.im) > imaginary_tolerance * scale) {
            throw QStateError(
                "Batch adjoint Hermitian Pauli support developed a non-real value");
        }
        output[observable] = expectation.re;
    }
}

inline void apply_bound_plan(
    StateVector& primal,
    std::size_t qubit_count,
    std::span<const ParameterizedOperation> templates,
    std::span<const double> parameters) {
    for (const ParameterizedOperation& templated : templates) {
        const Operation operation = bind_operation(templated, parameters);
        adjoint_detail::apply_operation(
            primal,
            qubit_count,
            operation,
            false);
    }
}

}  // namespace adjoint_batch_detail

// Executes all exact primal rows directly on one materialized local root. It
// avoids structural branch creation, QRegister reconstruction, and repeated
// Python/native transitions while preserving row-major deterministic output.
inline void observables_many_dense(
    const QRegister& initial,
    const ParameterizedOperationPlan& plan,
    std::span<const double> parameter_rows,
    std::size_t row_count,
    const PauliSupportPlan& observables,
    std::size_t max_qubits,
    double imaginary_tolerance,
    std::span<double> values_output) {
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
        throw QStateError("Batch primal parameter buffer size is incorrect");
    }
    if (values_output.size() != row_count * observable_count) {
        throw QStateError("Batch primal value output size is incorrect");
    }

    const std::size_t qubit_count = initial.qubit_count();
    const auto& templates = plan.operations();
    const adjoint_batch_detail::StateVector initial_primal =
        initial.materialize(max_qubits);
    adjoint_batch_detail::StateVector primal = initial_primal;
    for (std::size_t row = 0U; row < row_count; ++row) {
        primal = initial_primal;
        const std::span<const double> parameters(
            parameter_rows.data() + row * parameter_count,
            parameter_count);
        adjoint_batch_detail::apply_bound_plan(
            primal,
            qubit_count,
            templates,
            parameters);
        adjoint_batch_detail::observable_values(
            primal,
            observables,
            imaginary_tolerance,
            values_output.subspan(row * observable_count, observable_count));
    }
}

// Executes exact row-wise VJPs directly on one materialized local root. Plan
// validation and root materialization happen once. Each row is reconstructed
// in reverse without storing a depth trajectory, and repeated parameter slots
// accumulate into the same deterministic gradient location.
inline void weighted_adjoint_many_dense(
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
        throw QStateError("Batch adjoint parameter buffer size is incorrect");
    }
    if (cotangent_rows.size() != row_count * observable_count) {
        throw QStateError("Batch adjoint cotangent buffer size is incorrect");
    }
    if (values_output.size() != row_count * observable_count) {
        throw QStateError("Batch adjoint value output size is incorrect");
    }
    if (gradient_output.size() != row_count * parameter_count) {
        throw QStateError("Batch adjoint gradient output size is incorrect");
    }

    const std::size_t qubit_count = initial.qubit_count();
    const auto& templates = plan.operations();
    const adjoint_batch_detail::StateVector initial_primal =
        initial.materialize(max_qubits);
    adjoint_batch_detail::StateVector primal = initial_primal;
    adjoint_batch_detail::StateVector adjoint;

    for (std::size_t row = 0U; row < row_count; ++row) {
        primal = initial_primal;
        const std::span<const double> parameters(
            parameter_rows.data() + row * parameter_count,
            parameter_count);
        const std::span<const double> cotangent(
            cotangent_rows.data() + row * observable_count,
            observable_count);
        for (double value : cotangent) {
            if (!std::isfinite(value)) {
                throw QStateError(
                    "Batch adjoint cotangent row contains a non-finite value");
            }
        }

        adjoint_batch_detail::apply_bound_plan(
            primal,
            qubit_count,
            templates,
            parameters);
        adjoint_batch_detail::observable_values(
            primal,
            observables,
            imaginary_tolerance,
            values_output.subspan(row * observable_count, observable_count));

        std::span<double> gradient = gradient_output.subspan(
            row * parameter_count,
            parameter_count);
        std::fill(gradient.begin(), gradient.end(), 0.0);
        adjoint_detail::weighted_pauli_action(
            primal,
            observables,
            cotangent,
            adjoint);
        for (std::size_t reverse_index = templates.size();
             reverse_index > 0U;
             --reverse_index) {
            const std::size_t index = reverse_index - 1U;
            const ParameterizedOperation& templated = templates[index];
            const Operation operation =
                adjoint_batch_detail::bind_operation(templated, parameters);
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
                gradient[static_cast<std::size_t>(templated.parameter_slot)] +=
                    2.0 * inner.re;
            }
            adjoint_detail::apply_operation(
                adjoint,
                qubit_count,
                operation,
                true);
        }
    }
}

}  // namespace qubit
