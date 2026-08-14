#pragma once

#define QSTATE_VERSION_MAJOR 0
#define QSTATE_VERSION_MINOR 2
#define QSTATE_VERSION_PATCH 0
#define QSTATE_VERSION_STRING "0.2.0"

// The ABI version changes only when an existing exported C contract becomes
// incompatible. Additive symbols do not change the ABI major version.
#define QSTATE_ABI_VERSION_MAJOR 1
#define QSTATE_ABI_VERSION_MINOR 5
#define QSTATE_ABI_VERSION_PATCH 0

// Experimental runtimes are separate from the stable QSA 0.1 C ABI. Only the
// shared-library translation unit defines QSTATE_BUILD_SHARED, so each bridge
// is linked once while installed headers remain declaration-only.
#if defined(QSTATE_BUILD_SHARED) && defined(__cplusplus)
#include "qubit/detail/qcausal_c_api_impl.hpp"
#include "qubit/detail/qcausal_batch_impl.hpp"
#include "qubit/detail/qcausal_pauli_support_impl.hpp"
#include "qubit/detail/qcausal_component_impl.hpp"
#include "qubit/detail/qcausal_adjoint_impl.hpp"
#include "qubit/detail/qcausal_vectorized_adjoint_impl.hpp"
#include "qubit/detail/qbehavior_c_api_impl.hpp"
#endif
