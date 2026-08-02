#pragma once

#include "qubit/qadjoint.hpp"

#include <algorithm>
#include <cstddef>
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

}  // extern "C"
