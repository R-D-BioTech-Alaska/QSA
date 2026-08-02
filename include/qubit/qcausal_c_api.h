#pragma once

#include "qubit/c_api.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(QSTATE_BUILD_SHARED)
    #define QCAUSAL_API __declspec(dllexport)
  #else
    #define QCAUSAL_API __declspec(dllimport)
  #endif
#else
  #define QCAUSAL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* qcausal_handle;
typedef void* qcausal_parameterized_plan_handle;
typedef void* qcausal_pauli_plan_handle;

// Experimental QSA 0.2 causal runtime surface. It is additive and separate
// from the stable QSA 0.1 C ABI while the runtime contract is validated by
// Brain/QELM integration.
QCAUSAL_API uint32_t qcausal_api_version_major(void);
QCAUSAL_API uint32_t qcausal_api_version_minor(void);
QCAUSAL_API uint32_t qcausal_api_version_patch(void);
QCAUSAL_API const char* qcausal_last_error(void);

QCAUSAL_API qcausal_handle qcausal_create(size_t qubit_count);
QCAUSAL_API qcausal_handle qcausal_from_qsc(const uint8_t* data, size_t data_size);
QCAUSAL_API qcausal_handle qcausal_fork(qcausal_handle source);
QCAUSAL_API int qcausal_adopt(qcausal_handle target, qcausal_handle selected);
QCAUSAL_API void qcausal_destroy(qcausal_handle handle);

QCAUSAL_API size_t qcausal_qubit_count(qcausal_handle handle);
QCAUSAL_API size_t qcausal_component_count(qcausal_handle handle);
QCAUSAL_API size_t qcausal_estimated_bytes(qcausal_handle handle);
QCAUSAL_API size_t qcausal_shared_owner_count(qcausal_handle handle);
QCAUSAL_API int qcausal_validate(qcausal_handle handle);
QCAUSAL_API int qcausal_amplitude(
    qcausal_handle handle,
    uint64_t basis_index,
    double* real,
    double* imag);
QCAUSAL_API int qcausal_probabilities_one(
    qcausal_handle handle,
    double* output,
    size_t output_size);

QCAUSAL_API size_t qcausal_qsc_size(qcausal_handle handle);
QCAUSAL_API int qcausal_qsc_write(
    qcausal_handle handle,
    uint8_t* output,
    size_t output_size);

QCAUSAL_API qcausal_parameterized_plan_handle qcausal_parameterized_plan_create(
    const qstate_parameterized_operation* operations,
    size_t operation_count,
    uint32_t flags);
QCAUSAL_API void qcausal_parameterized_plan_destroy(
    qcausal_parameterized_plan_handle plan);
QCAUSAL_API size_t qcausal_parameterized_plan_parameter_count(
    qcausal_parameterized_plan_handle plan);
QCAUSAL_API int qcausal_parameterized_plan_execute(
    qcausal_handle handle,
    qcausal_parameterized_plan_handle plan,
    const double* parameters,
    size_t parameter_count,
    size_t* completed_operation_count);

// Pauli words are supplied as one contiguous word_count * qubit_count byte
// buffer using ASCII I, X, Y, or Z without terminators between words.
QCAUSAL_API qcausal_pauli_plan_handle qcausal_pauli_plan_create(
    size_t qubit_count,
    const char* words,
    size_t word_count,
    double imaginary_tolerance);
QCAUSAL_API void qcausal_pauli_plan_destroy(qcausal_pauli_plan_handle plan);
QCAUSAL_API size_t qcausal_pauli_plan_observable_count(
    qcausal_pauli_plan_handle plan);
QCAUSAL_API int qcausal_pauli_plan_execute(
    qcausal_pauli_plan_handle plan,
    qcausal_handle handle,
    double* output,
    size_t output_size);

#ifdef __cplusplus
}
#endif
