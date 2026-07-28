# Exact stabilizer tableau

`StabilizerState` is an exact polynomial-memory backend for Clifford circuits. It supports X, Y, Z, H, S, S-dagger, CNOT, CZ, SWAP, computational-basis probabilities, and projective Z measurement.

The tableau stores destabilizer and stabilizer generators in packed binary symplectic form. It does not allocate a `2^n` statevector. This provides exponential workload-class savings for circuits that remain Clifford.

The backend is separate from `QRegister`. It does not change ordinary state execution and does not accept non-Clifford operations. Callers must choose it explicitly or use representation advice that marks a workload as Clifford-only.

## Validation

- Tableau phases retain Hermitian Pauli conventions.
- Paired destabilizer and stabilizer rows anticommute.
- Stabilizer generators commute.
- Random Clifford circuits are compared against `QRegister` populations.
- Sequential projective measurements are compared against `QRegister` using identical samples.
- Tableau allocation is bounded by `max_tableau_bytes`.

`validate()` performs structural and paired-generator checks suitable for large tableaux. `validate_full()` checks every generator commutation relation and is intended for tests and moderate qubit counts.

The reproducible benchmark target is `qstate_stabilizer_benchmark`.
