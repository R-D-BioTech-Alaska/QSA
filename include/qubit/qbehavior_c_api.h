#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(QSTATE_BUILD_SHARED)
    #define QBEHAVIOR_API __declspec(dllexport)
  #else
    #define QBEHAVIOR_API __declspec(dllimport)
  #endif
#else
  #define QBEHAVIOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* qbehavior_fold_handle;

typedef struct qbehavior_fold_config {
    size_t max_states_per_level;
    size_t max_index_bytes;
    uint64_t max_destination_lookups;
} qbehavior_fold_config;

typedef struct qbehavior_fold_stats {
    size_t branch_count;
    size_t probe_bit_count;
    size_t max_leaves;
    size_t behavior_class_count;
    size_t deep_behavior_count;
    size_t estimated_bytes;
    uint64_t eligible_pairs;
    uint64_t destination_lookups;
    double fold_seconds;
    double index_seconds;
    double build_seconds;
} qbehavior_fold_stats;

enum {
    QBEHAVIOR_JOIN_AND = 0,
    QBEHAVIOR_JOIN_OR = 1,
    QBEHAVIOR_JOIN_XOR = 2,
};

QBEHAVIOR_API uint32_t qbehavior_api_version_major(void);
QBEHAVIOR_API uint32_t qbehavior_api_version_minor(void);
QBEHAVIOR_API uint32_t qbehavior_api_version_patch(void);
QBEHAVIOR_API const char* qbehavior_last_error(void);

QBEHAVIOR_API qbehavior_fold_handle qbehavior_fold_create(
    const uint64_t* branch_masks,
    size_t branch_count,
    size_t probe_bit_count,
    size_t max_leaves,
    const uint8_t* join_codes,
    size_t join_count,
    const qbehavior_fold_config* config);
QBEHAVIOR_API void qbehavior_fold_destroy(qbehavior_fold_handle handle);

QBEHAVIOR_API int qbehavior_fold_stats_read(
    qbehavior_fold_handle handle,
    qbehavior_fold_stats* output);
QBEHAVIOR_API int qbehavior_fold_depth_state_count(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    size_t* output);
QBEHAVIOR_API int qbehavior_fold_structural_count(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    uint64_t* output);
QBEHAVIOR_API int qbehavior_fold_contains_at_depth(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    uint64_t behavior,
    int* output);

QBEHAVIOR_API int qbehavior_fold_canonical_plan(
    qbehavior_fold_handle handle,
    uint64_t behavior,
    uint32_t* branch_indices,
    size_t branch_capacity,
    uint8_t* join_codes,
    size_t join_capacity,
    size_t* leaf_count,
    int* found);

QBEHAVIOR_API int qbehavior_fold_conditioned_count(
    qbehavior_fold_handle handle,
    const size_t* positions,
    const uint8_t* bits,
    size_t observation_count,
    size_t* output);

QBEHAVIOR_API int qbehavior_fold_conditioned_unique(
    qbehavior_fold_handle handle,
    const size_t* positions,
    const uint8_t* bits,
    size_t observation_count,
    size_t* count,
    int* unique,
    uint64_t* behavior,
    uint32_t* branch_indices,
    size_t branch_capacity,
    uint8_t* join_codes,
    size_t join_capacity,
    size_t* leaf_count);

QBEHAVIOR_API int qbehavior_fold_state_digest(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    char* output,
    size_t output_size);
QBEHAVIOR_API int qbehavior_fold_canonical_digest(
    qbehavior_fold_handle handle,
    char* output,
    size_t output_size);

#ifdef __cplusplus
}
#endif
