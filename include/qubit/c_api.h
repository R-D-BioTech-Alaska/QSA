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
typedef void* qstate_plan_handle;
typedef void* qstate_parameterized_plan_handle;
typedef void* qstate_grover_handle;
typedef void* qstate_symmetry_handle;

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

typedef enum qstate_opcode {
    QSTATE_OP_X = 1,
    QSTATE_OP_Y = 2,
    QSTATE_OP_Z = 3,
    QSTATE_OP_H = 4,
    QSTATE_OP_S = 5,
    QSTATE_OP_SDG = 6,
    QSTATE_OP_T = 7,
    QSTATE_OP_TDG = 8,
    QSTATE_OP_RX = 9,
    QSTATE_OP_RY = 10,
    QSTATE_OP_RZ = 11,
    QSTATE_OP_CNOT = 12,
    QSTATE_OP_CZ = 13,
    QSTATE_OP_SWAP = 14,
    QSTATE_OP_BIT_FLIP_TRAJECTORY = 15,
    QSTATE_OP_PHASE_FLIP_TRAJECTORY = 16,
    QSTATE_OP_DEPOLARIZING_TRAJECTORY = 17,
    QSTATE_OP_AMPLITUDE_DAMPING_TRAJECTORY = 18
} qstate_opcode;

typedef struct qstate_operation {
    uint32_t opcode;
    uint32_t first;
    uint32_t second;
    uint32_t reserved;
    double parameter;
    double sample;
} qstate_operation;

QSTATE_API uint32_t qstate_abi_version_major(void);
QSTATE_API uint32_t qstate_abi_version_minor(void);
QSTATE_API uint32_t qstate_abi_version_patch(void);
QSTATE_API const char* qstate_version_string(void);

QSTATE_API int qstate_apply_sdg(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_tdg(qstate_handle handle, uint32_t qubit);
QSTATE_API int qstate_apply_bit_flip_trajectory(
    qstate_handle handle, uint32_t qubit, double probability, double sample);
QSTATE_API int qstate_apply_phase_flip_trajectory(
    qstate_handle handle, uint32_t qubit, double probability, double sample);
QSTATE_API int qstate_apply_depolarizing_trajectory(
    qstate_handle handle, uint32_t qubit, double probability, double sample);
QSTATE_API int qstate_apply_amplitude_damping_trajectory(
    qstate_handle handle, uint32_t qubit, double gamma, double sample);

QSTATE_API int qstate_measure_all(
    qstate_handle handle, uint64_t seed, uint8_t* output, size_t output_size);
QSTATE_API int qstate_amplitude_bits(
    qstate_handle handle,
    const uint8_t* bits,
    size_t bit_count,
    double* real,
    double* imag);
QSTATE_API int qstate_component_kind(qstate_handle handle, uint32_t qubit, int* result);
QSTATE_API int qstate_validate(qstate_handle handle);
QSTATE_API int qstate_apply_operations(
    qstate_handle handle,
    const qstate_operation* operations,
    size_t operation_count,
    size_t* completed_count);

typedef enum qstate_plan_flags {
    QSTATE_PLAN_DEFAULT = 0,
    QSTATE_PLAN_OPTIMIZE = 1
} qstate_plan_flags;

QSTATE_API qstate_plan_handle qstate_plan_create(
    const qstate_operation* operations,
    size_t operation_count,
    uint32_t flags);
QSTATE_API void qstate_plan_destroy(qstate_plan_handle plan);
QSTATE_API size_t qstate_plan_source_operation_count(qstate_plan_handle plan);
QSTATE_API size_t qstate_plan_compiled_step_count(qstate_plan_handle plan);
QSTATE_API int qstate_plan_execute(
    qstate_handle handle,
    qstate_plan_handle plan,
    size_t* completed_operation_count);
QSTATE_API int qstate_plan_execute_many(
    qstate_plan_handle plan,
    qstate_handle* handles,
    size_t handle_count,
    size_t worker_count,
    size_t* completed_handle_count);
QSTATE_API int qstate_probabilities_one(
    qstate_handle handle,
    double* output,
    size_t output_size);

typedef struct qstate_parameterized_operation {
    uint32_t opcode;
    uint32_t first;
    uint32_t second;
    uint32_t reserved;
    double parameter;
    double sample;
    int32_t parameter_slot;
    int32_t sample_slot;
} qstate_parameterized_operation;

QSTATE_API qstate_parameterized_plan_handle qstate_parameterized_plan_create(
    const qstate_parameterized_operation* operations,
    size_t operation_count,
    uint32_t flags);
QSTATE_API void qstate_parameterized_plan_destroy(qstate_parameterized_plan_handle plan);
QSTATE_API size_t qstate_parameterized_plan_source_operation_count(
    qstate_parameterized_plan_handle plan);
QSTATE_API size_t qstate_parameterized_plan_parameter_count(
    qstate_parameterized_plan_handle plan);
QSTATE_API int qstate_parameterized_plan_execute(
    qstate_handle handle,
    qstate_parameterized_plan_handle plan,
    const double* parameters,
    size_t parameter_count,
    size_t* completed_operation_count);
QSTATE_API int qstate_parameterized_plan_execute_many(
    qstate_parameterized_plan_handle plan,
    qstate_handle* handles,
    size_t handle_count,
    const double* parameters,
    size_t parameter_count,
    size_t worker_count,
    size_t* completed_handle_count);

QSTATE_API int qstate_apply_grover_oracle(
    qstate_handle handle,
    const uint64_t* marked_indices,
    size_t marked_count);
QSTATE_API int qstate_apply_grover_diffusion(qstate_handle handle);
QSTATE_API int qstate_apply_grover_iterations(
    qstate_handle handle,
    const uint64_t* marked_indices,
    size_t marked_count,
    uint64_t iteration_count);

QSTATE_API qstate_grover_handle qstate_grover_create(
    size_t qubit_count,
    const uint64_t* marked_indices,
    size_t marked_count);
QSTATE_API qstate_grover_handle qstate_grover_create_count(
    size_t qubit_count,
    uint64_t marked_count);
QSTATE_API void qstate_grover_destroy(qstate_grover_handle handle);
QSTATE_API int qstate_grover_reset(qstate_grover_handle handle);
QSTATE_API int qstate_grover_apply_oracle(qstate_grover_handle handle);
QSTATE_API int qstate_grover_apply_diffusion(qstate_grover_handle handle);
QSTATE_API int qstate_grover_iterate(qstate_grover_handle handle, uint64_t iteration_count);
QSTATE_API int qstate_grover_run_optimal(qstate_grover_handle handle);
QSTATE_API size_t qstate_grover_qubit_count(qstate_grover_handle handle);
QSTATE_API uint64_t qstate_grover_space_size(qstate_grover_handle handle);
QSTATE_API uint64_t qstate_grover_marked_count(qstate_grover_handle handle);
QSTATE_API uint64_t qstate_grover_iteration_count(qstate_grover_handle handle);
QSTATE_API uint64_t qstate_grover_optimal_iterations(qstate_grover_handle handle);
QSTATE_API size_t qstate_grover_estimated_bytes(qstate_grover_handle handle);
QSTATE_API int qstate_grover_has_explicit_marked_indices(qstate_grover_handle handle);
QSTATE_API int qstate_grover_success_probability(
    qstate_grover_handle handle,
    double* result);
QSTATE_API int qstate_grover_marked_amplitude(
    qstate_grover_handle handle,
    double* real,
    double* imag);
QSTATE_API int qstate_grover_unmarked_amplitude(
    qstate_grover_handle handle,
    double* real,
    double* imag);
QSTATE_API int qstate_grover_amplitude(
    qstate_grover_handle handle,
    uint64_t basis_index,
    double* real,
    double* imag);
QSTATE_API int qstate_grover_sample_basis(
    qstate_grover_handle handle,
    double branch_sample,
    double index_sample,
    uint64_t* result);
QSTATE_API int qstate_grover_validate(qstate_grover_handle handle);
QSTATE_API size_t qstate_grover_description_size(qstate_grover_handle handle);
QSTATE_API int qstate_grover_description_write(
    qstate_grover_handle handle,
    char* output,
    size_t output_size);

QSTATE_API qstate_symmetry_handle qstate_symmetry_create_ordered(
    size_t qubit_count,
    const uint64_t* class_counts,
    size_t class_count);
QSTATE_API qstate_symmetry_handle qstate_symmetry_create_count_only(
    size_t qubit_count,
    const uint64_t* class_counts,
    size_t class_count);
QSTATE_API qstate_symmetry_handle qstate_symmetry_create_labels(
    size_t qubit_count,
    const uint32_t* labels,
    size_t label_count);
QSTATE_API qstate_symmetry_handle qstate_symmetry_create_hamming_weight(
    size_t qubit_count);
QSTATE_API qstate_symmetry_handle qstate_symmetry_discover(
    qstate_handle source,
    size_t max_qubits,
    double tolerance,
    size_t max_classes);
QSTATE_API void qstate_symmetry_destroy(qstate_symmetry_handle handle);
QSTATE_API int qstate_symmetry_reset_uniform(qstate_symmetry_handle handle);
QSTATE_API int qstate_symmetry_set_amplitudes(
    qstate_symmetry_handle handle,
    const double* real,
    const double* imag,
    size_t amplitude_count,
    int normalize);
QSTATE_API int qstate_symmetry_apply_class_phase(
    qstate_symmetry_handle handle,
    size_t class_index,
    double angle);
QSTATE_API int qstate_symmetry_apply_class_phases(
    qstate_symmetry_handle handle,
    const double* angles,
    size_t angle_count);
QSTATE_API int qstate_symmetry_apply_reflection(qstate_symmetry_handle handle);
QSTATE_API int qstate_symmetry_split_class(
    qstate_symmetry_handle handle,
    size_t class_index,
    uint64_t first_count,
    size_t* new_class_index);
QSTATE_API int qstate_symmetry_merge_equivalent(
    qstate_symmetry_handle handle,
    double tolerance,
    size_t* removed_count);
QSTATE_API int qstate_symmetry_apply_unitary(
    qstate_symmetry_handle handle,
    const double* real,
    const double* imag,
    size_t element_count,
    double tolerance);
QSTATE_API int qstate_symmetry_iterate_unitary(
    qstate_symmetry_handle handle,
    const double* real,
    const double* imag,
    size_t element_count,
    uint64_t iteration_count,
    double tolerance);
QSTATE_API size_t qstate_symmetry_qubit_count(qstate_symmetry_handle handle);
QSTATE_API uint64_t qstate_symmetry_space_size(qstate_symmetry_handle handle);
QSTATE_API size_t qstate_symmetry_class_count(qstate_symmetry_handle handle);
QSTATE_API int qstate_symmetry_membership_mode(qstate_symmetry_handle handle);
QSTATE_API uint64_t qstate_symmetry_class_size(
    qstate_symmetry_handle handle,
    size_t class_index);
QSTATE_API int qstate_symmetry_class_amplitude(
    qstate_symmetry_handle handle,
    size_t class_index,
    double* real,
    double* imag);
QSTATE_API int qstate_symmetry_class_probability(
    qstate_symmetry_handle handle,
    size_t class_index,
    double* result);
QSTATE_API int qstate_symmetry_amplitude(
    qstate_symmetry_handle handle,
    uint64_t basis_index,
    double* real,
    double* imag);
QSTATE_API int qstate_symmetry_sample_class(
    qstate_symmetry_handle handle,
    double sample,
    size_t* result);
QSTATE_API int qstate_symmetry_sample_basis(
    qstate_symmetry_handle handle,
    double class_sample,
    double index_sample,
    uint64_t* result);
QSTATE_API qstate_handle qstate_symmetry_to_register(
    qstate_symmetry_handle handle,
    size_t max_qubits);
QSTATE_API size_t qstate_symmetry_estimated_bytes(qstate_symmetry_handle handle);
QSTATE_API int qstate_symmetry_discovery_error(
    qstate_symmetry_handle handle,
    double* result);
QSTATE_API int qstate_symmetry_validate(qstate_symmetry_handle handle);
QSTATE_API size_t qstate_symmetry_description_size(qstate_symmetry_handle handle);
QSTATE_API int qstate_symmetry_description_write(
    qstate_symmetry_handle handle,
    char* output,
    size_t output_size);

#ifdef __cplusplus
}
#endif
