#pragma once

#include "qubit/qcausal_c_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Extracts the complete QSA components containing the requested global qubits.
// The returned causal state uses compact local qubit indices. global_qubits
// maps each local index back to the source register.
QCAUSAL_API qcausal_handle qcausal_extract_component_closure(
    qcausal_handle source,
    const uint32_t* requested_qubits,
    size_t requested_count,
    size_t max_local_qubits,
    uint32_t* global_qubits,
    size_t global_qubit_capacity,
    size_t* global_qubit_count);

#ifdef __cplusplus
}
#endif
