# Experimental quantum-dot engine

QSA provides an opt-in native quantum-dot pocket alongside the general register engine. It does not replace `QRegister`, change QSC, or alter the current C/Python ABI.

## Model

Each dot is a four-level local state represented by two logical qubits. The specialized engine stores either:

- fixed two-dot blocks (`PairBlocks`), or
- fixed two-dot blocks plus one final independent context dot (`PairPlusContext`).

The implementation includes exact local rotations, capacitive `ZZ` coupling, spin exchange, charge tunneling, second-order Trotter stepping, validation, materialization, and a gate-level `QRegister` reference path.

## Safety properties

- All configuration and input values must be finite.
- Input strength is restricted to `[0, 1]`.
- All derived rotation and coupling angles are validated before mutation.
- Rejected steps leave the pocket unchanged.
- Pair topology remains fixed; no cross-block coupling is permitted.
- `PairPlusContext` requires an odd dot count; `PairBlocks` requires an even dot count.
- No existing QSA state, QSC, C ABI, Python API, Grover, plan, or symmetry behavior is modified.

## Validation

The test suite compares the specialized engine against the unchanged generic `QRegister` gate decomposition, exercises invalid and overflow inputs, verifies exception safety and deterministic concurrent execution, and runs long-duration scaling and normalization checks.

The benchmark reports the specialized path and the equivalent generic-QSA path separately. Results are workload-specific and must not be described as a universal QSA speedup.
