#pragma once

#include "qubit/qadjoint.hpp"
#include "qubit/qadjoint_batch.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>

extern "C" {

int qcausal_weighted_adjoint(
    qcausal_handle initial,
    qcausal_parameterized_plan_handle plan,
    qcausal_pauli_support_plan_handle observables,
    const double* parameters,
    size_t parameter_count,
    const double* cotangent,
    size_t cotangent_count,
    size_t max_qubits,
    double* values_output,
    size_t values_output_size,
    double* gradient_output,
    size_t gradient_output_size) {
    return guarded_causal([&] {
        if (parameter_count != 0U && parameters == nullptr) {
            throw qubit::QStateError("Adjoint parameter buffer is null");
        }
        if (cotangent_count != 0U && cotangent == nullptr) {
            throw qubit::QStateError("Adjoint cotangent buffer is null");
        }
        CausalHandle* causal = as_causal(initial);
        CausalParameterizedPlanHandle* parameterized =
            as_causal_parameterized_plan(plan);
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(observables);

        const std::size_t required_values = support->plan.observable_count();
        const std::size_t required_gradient = parameterized->plan.parameter_count();
        if (values_output_size < required_values ||
            (required_values != 0U && values_output == nullptr)) {
            throw qubit::QStateError("Adjoint value output buffer is too small");
        }
        if (gradient_output_size < required_gradient ||
            (required_gradient != 0U && gradient_output == nullptr)) {
            throw qubit::QStateError("Adjoint gradient output buffer is too small");
        }

        const qubit::WeightedAdjointResult result = qubit::weighted_adjoint(
            causal->state.read(),
            parameterized->plan,
            std::span<const double>(parameters, parameter_count),
            support->plan,
            std::span<const double>(cotangent, cotangent_count),
            max_qubits);
        std::copy(result.values.begin(), result.values.end(), values_output);
        std::copy(result.gradient.begin(), result.gradient.end(), gradient_output);
    });
}

int qcausal_observables_many_dense(
    qcausal_handle initial,
    qcausal_parameterized_plan_handle plan,
    qcausal_pauli_support_plan_handle observables,
    const double* parameter_rows,
    size_t row_count,
    size_t parameter_count,
    size_t max_qubits,
    double imaginary_tolerance,
    double* values_output,
    size_t values_output_size,
    size_t* completed_row_count) {
    return guarded_causal([&] {
        if (completed_row_count == nullptr) {
            throw qubit::QStateError(
                "Batch primal completed-row output is null");
        }
        *completed_row_count = 0U;
        CausalHandle* causal = as_causal(initial);
        CausalParameterizedPlanHandle* parameterized =
            as_causal_parameterized_plan(plan);
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(observables);

        const std::size_t required_parameter_count =
            parameterized->plan.parameter_count();
        const std::size_t observable_count = support->plan.observable_count();
        if (parameter_count != required_parameter_count) {
            throw qubit::QStateError(
                "Batch primal parameter width differs from plan parameter count");
        }
        const std::size_t limit = std::numeric_limits<std::size_t>::max();
        if ((parameter_count != 0U && row_count > limit / parameter_count) ||
            (observable_count != 0U && row_count > limit / observable_count)) {
            throw qubit::QStateError("Batch primal size overflow");
        }
        const std::size_t required_parameters = row_count * parameter_count;
        const std::size_t required_values = row_count * observable_count;
        if (required_parameters != 0U && parameter_rows == nullptr) {
            throw qubit::QStateError("Batch primal parameter buffer is null");
        }
        if (values_output_size < required_values ||
            (required_values != 0U && values_output == nullptr)) {
            throw qubit::QStateError(
                "Batch primal value output buffer is too small");
        }

        qubit::observables_many_dense(
            causal->state.read(),
            parameterized->plan,
            std::span<const double>(parameter_rows, required_parameters),
            row_count,
            support->plan,
            max_qubits,
            imaginary_tolerance,
            std::span<double>(values_output, required_values));
        *completed_row_count = row_count;
    });
}

int qcausal_weighted_adjoint_many(
    qcausal_handle initial,
    qcausal_parameterized_plan_handle plan,
    qcausal_pauli_support_plan_handle observables,
    const double* parameter_rows,
    size_t row_count,
    size_t parameter_count,
    const double* cotangent_rows,
    size_t cotangent_count,
    size_t max_qubits,
    double imaginary_tolerance,
    double* values_output,
    size_t values_output_size,
    double* gradient_output,
    size_t gradient_output_size,
    size_t* completed_row_count) {
    return guarded_causal([&] {
        if (completed_row_count == nullptr) {
            throw qubit::QStateError(
                "Batch adjoint completed-row output is null");
        }
        *completed_row_count = 0U;

        CausalHandle* causal = as_causal(initial);
        CausalParameterizedPlanHandle* parameterized =
            as_causal_parameterized_plan(plan);
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(observables);

        const std::size_t required_values_per_row =
            support->plan.observable_count();
        const std::size_t required_gradient_per_row =
            parameterized->plan.parameter_count();
        if (parameter_count != required_gradient_per_row) {
            throw qubit::QStateError(
                "Batch adjoint parameter width differs from plan parameter count");
        }
        if (cotangent_count != required_values_per_row) {
            throw qubit::QStateError(
                "Batch adjoint cotangent width differs from observable count");
        }

        const std::size_t limit = std::numeric_limits<std::size_t>::max();
        if ((parameter_count != 0U && row_count > limit / parameter_count) ||
            (cotangent_count != 0U && row_count > limit / cotangent_count) ||
            (required_values_per_row != 0U &&
             row_count > limit / required_values_per_row) ||
            (required_gradient_per_row != 0U &&
             row_count > limit / required_gradient_per_row)) {
            throw qubit::QStateError("Batch adjoint size overflow");
        }

        const std::size_t required_parameter_values = row_count * parameter_count;
        const std::size_t required_cotangent_values = row_count * cotangent_count;
        const std::size_t required_values =
            row_count * required_values_per_row;
        const std::size_t required_gradients =
            row_count * required_gradient_per_row;

        if (required_parameter_values != 0U && parameter_rows == nullptr) {
            throw qubit::QStateError("Batch adjoint parameter buffer is null");
        }
        if (required_cotangent_values != 0U && cotangent_rows == nullptr) {
            throw qubit::QStateError("Batch adjoint cotangent buffer is null");
        }
        if (values_output_size < required_values ||
            (required_values != 0U && values_output == nullptr)) {
            throw qubit::QStateError(
                "Batch adjoint value output buffer is too small");
        }
        if (gradient_output_size < required_gradients ||
            (required_gradients != 0U && gradient_output == nullptr)) {
            throw qubit::QStateError(
                "Batch adjoint gradient output buffer is too small");
        }

        qubit::weighted_adjoint_many_dense(
            causal->state.read(),
            parameterized->plan,
            std::span<const double>(
                parameter_rows,
                required_parameter_values),
            row_count,
            support->plan,
            std::span<const double>(
                cotangent_rows,
                required_cotangent_values),
            max_qubits,
            imaginary_tolerance,
            std::span<double>(values_output, required_values),
            std::span<double>(gradient_output, required_gradients));
        *completed_row_count = row_count;
    });
}

}  // extern "C"
