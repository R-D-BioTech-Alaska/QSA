# Exact execution acceleration

QSA provides two opt-in execution plans for workloads that retain ordinary `QRegister` state.

## Dependency operation plan

`DependencyOperationPlan` tracks the most recent active operation touching each qubit. It can move an operation across intervening operations on disjoint qubits and then apply exact reductions:

- cancel X, Y, Z, H, CNOT, CZ, and SWAP pairs;
- cancel S/S-dagger and T/T-dagger pairs;
- combine same-axis Rx, Ry, and Rz rotations;
- remove a combined rotation only when its angle is exactly zero.

The reduced operations are passed through the existing immutable `OperationPlan` compiler for ordinary matrix and diagonal fusion.

## Independent component plan

`IndependentComponentPlan` accepts single-qubit unitary operations only. At execution time it groups them by the current QSA component partition. Operations within one component remain sequential; different components can execute concurrently because no operation can merge, split, or resize the component store.

The plan rejects two-qubit gates, trajectory noise, and non-finite rotation parameters. It does not add locks to scalar `QRegister` methods and does not alter existing execution behavior.

The reproducible benchmark target is `qstate_execution_acceleration_benchmark`.
