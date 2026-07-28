# Exact structural acceleration

QSA can avoid some tensor-product merges when the gate semantics prove that no new entanglement can be created. These paths remain exact and fall back to the ordinary component engine whenever the structural condition is not satisfied.

## Structure-aware gates

`QRegister` provides exact structure-aware variants for CNOT, CZ, and SWAP.

- A disconnected SWAP exchanges logical qubit labels and local tensor axes without merging the two components.
- CNOT with an exact `|0>` control is an identity.
- CNOT with an exact `|1>` control applies X to the target only.
- CNOT with an exact `|+>` target is an identity.
- CNOT with an exact `|->` target applies Z to the control only.
- CZ with an exact `|0>` operand is an identity.
- CZ with an exact `|1>` operand applies Z to the other qubit only.

Compiled operation plans use the structure-aware variants. Existing scalar C++ and C/Python calls remain unchanged.

When a merge is still required, the structure-aware path caches remapped support indices only after the Cartesian product is large enough to repay the allocation cost.

## Joint measurement

`measure_all_joint(seed)` samples each independent component once, decodes the selected basis state, and collapses the register directly to basis cells. It preserves the exact joint computational-basis distribution while avoiding repeated component scans and repeated separability recovery.

The existing `measure_all(seed)` method remains unchanged, including its historical seed-to-result mapping.

## Scope

The acceleration is workload-specific. It does not make arbitrary entangling circuits polynomial. It prevents exponential work only when QSA can prove that the work is unnecessary.

The reproducible benchmark target is `qstate_structural_acceleration_benchmark`.
