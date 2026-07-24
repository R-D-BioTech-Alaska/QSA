# QSA Compatibility Contract

This contract freezes the public behavior established by QSA 0.1.0. Later
releases may add capabilities and replace internal algorithms, but they must not
silently invalidate existing Qubit, QELM, Brain, node, C, C++, or Python code.

## C++ contract

The following remain source-compatible within ABI-major 1 after rebuilding
against the current headers. Binary compatibility for compiled consumers is
provided by the C ABI; the concrete C++ class layout is not a frozen binary
interface.

The following remain source-compatible:

- Namespace `qubit`
- `QComplex`, `QStateConfig`, `QMatrix2`, `QMatrix4`, `BlochCell`
- `StorageMode`, `AmplitudeStore`, `StateComponent`, `QRegister`, `QStateCodec`
- Existing gate, measurement, amplitude, component, validation, and QSC methods
- Qubit numbering and basis ordering: qubit 0 is the least-significant basis bit
- Existing default tolerances and capacity limits

New methods and component types may be added. Existing names, parameter meaning,
and numerical conventions are not to be repurposed.

## Stable C ABI

Every function that existed in `include/qubit/c_api.h` at version 0.1.0 remains
exported with the same name, argument types, ownership rules, and return
convention. A zero return value means success for `int` operations; errors are
reported through `qstate_last_error()`.

ABI version queries were added in 0.1.1. Additive symbols increase only the ABI
minor or patch version. An incompatible signature or handle contract requires a
new ABI major and a parallel compatibility library.

QSA 0.1.3 adds two new opaque handle families without changing `qstate_handle`:

- `qstate_plan_handle` for fixed immutable operation plans
- `qstate_parameterized_plan_handle` for reusable numeric templates

QSA 0.1.4 adds `qstate_grover_handle` for the symmetry-compressed Grover
engine. It has independent create/destroy ownership and does not alter
`qstate_handle` layout or behavior. Exact Grover operations on `qstate_handle`
are exposed only through additive functions.

QSA 0.1.5 adds `qstate_symmetry_handle` for general amplitude-class states.
Symmetry handles are isolated from `qstate_handle`, Grover handles, operation
plans, and QSC v1. `qstate_symmetry_to_register` creates a new ordinary register
through the existing ownership rules when exact fallback is explicitly requested.

Each handle is created and destroyed by its matching function family. Plan
handles may execute only against registers loaded from the same native library
instance.

## Python contract

Both imports remain valid:

```python
from qsa import QubitRegister
from qubit_native import QubitRegister
```

The second form is a compatibility alias, not a separate engine.

The following are preserved:

- Chainable gate methods
- Context-manager and explicit `close()` ownership
- `measure`, `probability_one`, `amplitude`, component queries, descriptions,
  and QSC methods
- `QSA_NATIVE_LIB` and historical `QUBIT_NATIVE_LIB`
- Property and method query forms where both appeared historically, such as
  `state.qubit_count` and `state.qubit_count()`

New Python methods must be additive. A new wrapper must continue to load an ABI
1.0 native library for the original method set.

## QSC v1 contract

QSC v1 remains decodable indefinitely. The frozen contract includes:

- Magic `QSC1QBT\0`
- Little-endian integers and IEEE-754 float64 values
- Major/minor fields and decoder version rules
- Header field order
- Component kinds 0, 1, and 2
- Qubit membership and local ordering
- Sparse and dense payload layouts
- FNV-1a trailer checksum

The byte-for-byte fixtures are:

- `tests/fixtures/qsc_v1_sparse_exact.bin`
- `tests/fixtures/qsc_v1_component_order_exact.bin`

The second packet was emitted by QSA 0.1.1 and exercises multiple components,
merges, compaction, measurement splitting, and component ordering. The test
suite decodes and re-encodes both without changes, and current releases reconstruct the
same multi-component packet from the same operation sequence.

A future QSC2 may add new representations. QSC2 must not replace the QSC1
decoder, and QSC1 export should remain available whenever the state can be
represented by QSC1.

## Architectural invariants

- Independent pure qubits remain compact geometric cells.
- Components merge only when an operation connects them.
- Sparse structure is not expanded merely because a register is large.
- Measurement collapses only the affected component.
- Exact separability recovery remains part of the engine behavior.
- Global materialization is explicit and bounded.
- Grover dense promotion is opt-in; compressed Grover state is isolated from QSC v1.
- Symmetry discovery and fallback are opt-in; symmetry handles are isolated from QSC v1.
- Hamming-weight and count-only symbolic states never allocate the logical statevector.
- The core remains independent of NumPy, Qiskit, `std::complex`, and BLAS.

## Acceptance gates

Before merging an internal redesign:

1. Existing C++ tests pass.
2. Stable C ABI tests pass.
3. Canonical and legacy Python-import tests pass.
4. Frozen QSC v1 fixtures pass.
5. Differential validation passes at the established fidelity threshold.
6. Structured-state memory behavior does not regress without an explicit,
   measured reason.
7. Known downstream projects pass their integration smoke tests.

These are bounded release gates, not a substitute for forward implementation.
