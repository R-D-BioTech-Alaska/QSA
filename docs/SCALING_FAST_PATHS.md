# Exact scaling fast paths

QSA keeps structured registers compact by operating only on components touched by an operation. Two exact hot paths remove avoidable work that previously scaled with unrelated state.

## Active-component diagonal evolution

`QRegister::apply_diagonal()` applies geometric-cell phases immediately and groups amplitude-patch phases only for components that actually receive a phase. A single phase on one cell or one entangled patch no longer allocates a grouping vector for every independent component in the register.

The dense multi-phase path retains the compact grouped loop used by ordinary training layers. The active-component path is selected only when the register is much wider than the phase set, so small and dense workloads keep their historical performance.

The transformation is exact: every selected basis amplitude is multiplied by the same product of unit-magnitude coefficients as before. Component membership, support, and storage mode are unchanged.

## Linear sparse Pauli permutations

Sparse X and Y gates are index permutations. XOR by one bit exchanges the lower and upper halves of each naturally sorted index block. QSA now rotates those two sorted halves and applies the Pauli-Y phase in one linear pass instead of permuting every index and sorting the complete support again.

For support size `k`, the sparse X/Y index work changes from `O(k log k)` sorting to `O(k)` exact permutation. The support remains sorted and duplicate-free without rebuilding the amplitude store.

## Validation

The dedicated scaling tests compare:

- randomized sparse X/Y sequences against generic matrix application;
- bit positions through the maximum supported 62-qubit component width;
- randomized diagonal layers against sequential literal matrices;
- single active cells and active entangled patches inside registers containing up to hundreds of thousands of independent components.

The reproducible benchmark target is `qstate_scaling_fast_paths_benchmark`.
