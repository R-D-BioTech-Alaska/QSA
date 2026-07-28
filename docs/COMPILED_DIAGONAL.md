# Compiled exact diagonal plans

`CompiledDiagonalPlan` accelerates repeated unitary diagonal evolution on a fixed QSA component layout.

The constructor combines duplicate logical-qubit phases, verifies finite unit-magnitude coefficients, records the prototype component membership and storage mode, and precomputes one coefficient per dense basis state within a caller-supplied memory limit. Sparse components retain only their local phase list and evaluate coefficients over existing support entries.

Execution applies one cached complex coefficient per dense amplitude. It never changes support, normalization, component membership, or storage mode. A state whose component structure or storage mode differs from the prototype is rejected before the plan mutates the state.

The plan is intended for repeated training circuits, parameter blocks that remain fixed across a batch, and ensembles sharing one exact component layout. `execute_many()` applies the immutable plan concurrently across independent registers.

The existing `apply_diagonal()` and `OperationPlan` paths remain unchanged. The reproducible benchmark target is `qstate_compiled_diagonal_benchmark`.
