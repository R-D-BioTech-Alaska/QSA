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

## Transposed batch execution

`apply_batch()` is an additive exact path for Clifford-only gate sequences. It validates the complete batch before mutation, transposes the packed tableau into qubit-column bitsets, updates all tableau rows with word-level Boolean operations, and restores the canonical row-major tableau before returning.

The executor discovers the qubits touched by the batch. Localized batches transpose only those columns. When at least three quarters of the register is active and the configured scratch limit permits it, QSA selects a full-width transpose because contiguous whole-table conversion is faster for broad circuits. This hybrid keeps short and localized batches compact while retaining the highest throughput for large random circuits.

The scalar gate APIs remain unchanged and are still appropriate for isolated interactive gates. Batch break-even depends on qubit count, gate count, active-qubit coverage, and hardware; the benchmark reports both paths rather than claiming a universal multiplier.

Batch scratch memory, including transposed columns and operation-local workspace, is bounded independently through `StabilizerConfig::max_batch_scratch_bytes`. A rejected batch does not modify the state.
