#pragma once

#include "qubit/qadjoint_vectorized_batch.hpp"

#include <cstddef>
#include <limits>
#include <span>

namespace {

struct VectorizedBatchSizes {
    std::size_t parameters{0U};
    std::size_t cotangents{0U};
    std::size_t values{0U};
    std::size_t gradients{0U};
};

VectorizedBatchSizes validate_vectorized_batch_sizes(
    std::size_t row_count,
    std::size_t parameter_count,
    std::size_t observable_count,
    bool include_cotangents) {
    const std::size_t limit = std::numeric_limits<std::size_t>::max();
    if ((parameter_count != 0U && row_count > limit / parameter_count) ||
        (observable_count != 0U && row_count > limit / observable_count)) {
        throw qubit::QStateError("Vectorized batch size overflow");
    }
    return VectorizedBatchSizes{
        row_count * parameter_count,
        include_cotangents ? row_count * observable_count : 0U,
        row_count * observable_count,
        include_cotangents ? row_count * parameter_count : 0U,
    };
}

}  // namespace

extern "C" {

int qcausal_observables_many_vectorized(
    qcausal_handle initial,
    qcausal_parameterized_plan_handle plan,
    qcausal_pauli_support_plan_handle observables,
    const double* parameter_rows,
    size_t row_count,
    size_t parameter_count,
    size_t max_qubits,
    double* values_output,
    size_t values_output_size,
    size_t* completed_row_count) {
    return guarded_causal([&] {
        if (completed_row_count == nullptr) {
            throw qubit::QStateError(
                "Vectorized primal completed-row output is null");
        }
        *completed_row_count = 0U;
        CausalHandle* causal = as_causal(initial);
        CausalParameterizedPlanHandle* parameterized =
            as_causal_parameterized_plan(plan);
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(observables);
        if (parameter_count != parameterized->plan.parameter_count()) {
            throw qubit::QStateError(
                "Vectorized primal parameter width differs from plan");
        }
        const VectorizedBatchSizes sizes = validate_vectorized_batch_sizes(
            row_count,
            parameter_count,
            support->plan.observable_count(),
            false);
        if (sizes.parameters != 0U && parameter_rows == nullptr) {
            throw qubit::QStateError(
                "Vectorized primal parameter buffer is null");
        }
        if (values_output_size < sizes.values ||
            (sizes.values != 0U && values_output == nullptr)) {
            throw qubit::QStateError(
                "Vectorized primal value output buffer is too small");
        }
        qubit::observables_many_vectorized(
            causal->state.read(),
            parameterized->plan,
            std::span<const double>(parameter_rows, sizes.parameters),
            row_count,
            support->plan,
            max_qubits,
            support->imaginary_tolerance,
            std::span<double>(values_output, sizes.values));
        *completed_row_count = row_count;
    });
}

int qcausal_weighted_adjoint_many_vectorized(
    qcausal_handle initial,
    qcausal_parameterized_plan_handle plan,
    qcausal_pauli_support_plan_handle observables,
    const double* parameter_rows,
    size_t row_count,
    size_t parameter_count,
    const double* cotangent_rows,
    size_t cotangent_count,
    size_t max_qubits,
    double* values_output,
    size_t values_output_size,
    double* gradient_output,
    size_t gradient_output_size,
    size_t* completed_row_count) {
    return guarded_causal([&] {
        if (completed_row_count == nullptr) {
            throw qubit::QStateError(
                "Vectorized adjoint completed-row output is null");
        }
        *completed_row_count = 0U;
        CausalHandle* causal = as_causal(initial);
        CausalParameterizedPlanHandle* parameterized =
            as_causal_parameterized_plan(plan);
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(observables);
        if (parameter_count != parameterized->plan.parameter_count()) {
            throw qubit::QStateError(
                "Vectorized adjoint parameter width differs from plan");
        }
        if (cotangent_count != support->plan.observable_count()) {
            throw qubit::QStateError(
                "Vectorized adjoint cotangent width differs from observables");
        }
        const VectorizedBatchSizes sizes = validate_vectorized_batch_sizes(
            row_count,
            parameter_count,
            cotangent_count,
            true);
        if (sizes.parameters != 0U && parameter_rows == nullptr) {
            throw qubit::QStateError(
                "Vectorized adjoint parameter buffer is null");
        }
        if (sizes.cotangents != 0U && cotangent_rows == nullptr) {
            throw qubit::QStateError(
                "Vectorized adjoint cotangent buffer is null");
        }
        if (values_output_size < sizes.values ||
            (sizes.values != 0U && values_output == nullptr)) {
            throw qubit::QStateError(
                "Vectorized adjoint value output buffer is too small");
        }
        if (gradient_output_size < sizes.gradients ||
            (sizes.gradients != 0U && gradient_output == nullptr)) {
            throw qubit::QStateError(
                "Vectorized adjoint gradient output buffer is too small");
        }
        qubit::weighted_adjoint_many_vectorized(
            causal->state.read(),
            parameterized->plan,
            std::span<const double>(parameter_rows, sizes.parameters),
            row_count,
            support->plan,
            std::span<const double>(cotangent_rows, sizes.cotangents),
            max_qubits,
            support->imaginary_tolerance,
            std::span<double>(values_output, sizes.values),
            std::span<double>(gradient_output, sizes.gradients));
        *completed_row_count = row_count;
    });
}

}  // extern "C"
