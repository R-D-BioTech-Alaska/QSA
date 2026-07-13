#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(QSTATE_BUILD_SHARED)
    #define QSTATE_API __declspec(dllexport)
  #else
    #define QSTATE_API __declspec(dllimport)
  #endif
#else
  #define QSTATE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* qstate_handle;

QSTATE_API qstate_handle qstate_create(size_t qubit_count);
QSTATE_API void qstate_destroy(qstate_handle handle);
QSTATE_API const char* qstate_last_error(void);

QSTATE_API int qstate_apply_x(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_y(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_z(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_h(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_s(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_t(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_rx(qstate_handle handle, uint32_t qubit, double theta);
QSTATE_API int qstate_apply_ry(qstate_handle handle, uint32_t qubit, double theta);
QSTATE_API int qstate_apply_rz(qstate_handle handle, uint32_t qubit, double theta);
QSTATE_API int qstate_apply_cnot(qstate_handle handle, uint32_t control, uint32_t target);
QSTATE_API int qstate_apply_cz(qstate_handle handle, uint32_t first, uint32_t second);
QSTATE_API int qstate_apply_swap(qstate_handle handle, uint32_t first, uint32_t second);

QSTATE_API int qstate_probability_one(qstate_handle handle, uint32_t qubit, double* result);
QSTATE_API int qstate_measure(qstate_handle handle, uint32_t qubit, double sample, int* result);
QSTATE_API int qstate_amplitude(qstate_handle handle, uint64_t basis_index, double* real, double* imag);

QSTATE_API size_t qstate_qubit_count(qstate_handle handle);
QSTATE_API size_t qstate_component_count(qstate_handle handle);
QSTATE_API size_t qstate_component_size(qstate_handle handle, uint32_t qubit);
QSTATE_API size_t qstate_component_nonzero_count(qstate_handle handle, uint32_t qubit);
QSTATE_API size_t qstate_estimated_bytes(qstate_handle handle);

QSTATE_API size_t qstate_description_size(qstate_handle handle);
QSTATE_API int qstate_description_write(qstate_handle handle, char* output, size_t output_size);

QSTATE_API size_t qstate_qsc_size(qstate_handle handle);
QSTATE_API int qstate_qsc_write(qstate_handle handle, uint8_t* output, size_t output_size);
QSTATE_API qstate_handle qstate_qsc_read(const uint8_t* data, size_t data_size);

#ifdef __cplusplus
}
#endif
