# QSA Architecture

QSA keeps exact state mathematics, structural representations, differentiation, and higher-level execution separate even when they share one repository.

The dependency direction is deliberate. Lower layers must not depend on higher ones.

## Core

The core owns the general exact state and persistence contracts:

- `QComplex`
- `QRegister`
- component storage and copy-on-write payloads
- gates and trajectory evolution
- QSC
- operation plans
- read views and compiled diagonal execution

The core is the exact fallback. Specialized paths may avoid work, but they must not weaken the core contract or become required for ordinary QRegister execution.

## Structural representations

Structural engines exploit a proof or a bounded mathematical variable:

- exact symmetry classes
- quantum-dot pockets
- Grover structure
- stabilizer tableaux
- phase graphs
- sparse Pauli observables
- causal Pauli light cones

A structural route is eligible only while its governing structure is valid. If support, term count, component closure, or another controlling quantity grows beyond its configured bound, the route must reject or return to an exact general path. It must not approximate silently.

`RepresentationAdvisor::inspect_operations()` derives Clifford eligibility from the actual operation list. Callers do not need to assert that property manually. Existing explicit feature inspection remains available for compatibility where an external subsystem already owns a stronger certificate.

## Differentiation

Gradient machinery consumes exact state or structural execution and returns mathematical derivatives:

- parameter shift
- weighted adjoints
- component-local adjoints
- VJP/JVP surfaces

Differentiation does not own model policy, candidate admission, optimizer policy, durable memory, or production routing.

## Runtime

The causal runtime owns temporary execution structure:

- forks
- compact support plans
- candidate transactions
- component extraction
- persistent component workspaces
- batch and row-parallel execution

Runtime state is subordinate to the exact state contract. Stale generations, mismatched roots, invalid closures, and unsupported operations fail closed.

## Transition invariants

Representation and storage changes are tested against the same invariants:

1. state equivalence against an independent dense reference for bounded systems;
2. probability equivalence for every qubit;
3. QSC round-trip equivalence;
4. copy-on-write ancestor isolation;
5. branch mutation isolation;
6. deterministic measurement agreement for identical samples;
7. exact trajectory agreement for identical samples;
8. Pauli expectation agreement;
9. native validation after every transition.

`qstate_transition_invariant_tests` runs deterministic randomized sequences that mix gates, structured two-qubit paths, measurement, trajectory noise, QSC replacement, branch mutation, and Pauli reads. This is intended to exercise combinations rather than only individual features.

## Review gate for new specialized paths

A new exact backend or fast path should include all of the following before integration:

- an explicit structural eligibility rule;
- an adversarial collapse case showing that the eligibility rule does not remain true after structure is lost;
- a bounded independent reference comparison;
- unchanged fallback behavior;
- state or root immutability checks where sharing is involved;
- a measured resource receipt that reports setup costs separately from repeated-query costs;
- no production or model-policy authority.

Performance evidence cannot replace an exactness gate.

## Next structural expansion

The largest remaining representation gap is general bounded-entanglement structure. The next major backend should target tensor-network or Schmidt-width scaling rather than another narrow three-qubit optimization. Its acceptance criterion is exact reconstruction or exact observable agreement under a bounded contraction/bond certificate, with fail-closed behavior when that certificate grows beyond the configured limit.
