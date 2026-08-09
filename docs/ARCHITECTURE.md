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
- bounded exact tensor-network contraction

A structural route is eligible only while its governing structure is valid. If support, term count, component closure, contraction width, or another controlling quantity grows beyond its configured bound, the route must reject or return to an exact general path. It must not approximate silently.

`RepresentationAdvisor::inspect_operations()` derives Clifford eligibility from the actual operation list. Callers do not need to assert that property manually. Existing explicit feature inspection remains available for compatibility where an external subsystem already owns a stronger certificate.

### Exact tensor-network contraction

`TensorNetworkCircuit` represents a unitary circuit as binary tensor factors over wire-segment variables. Initial |0> states, one-qubit gates, two-qubit gates, and requested output-basis pins become factors. A basis amplitude is evaluated by deterministic variable elimination.

The controlling structural resource is the largest binary factor union encountered during contraction. `max_contraction_entries` is a hard bound on that intermediate work. If the requested contraction would exceed the bound, the query throws instead of truncating a bond or approximating an amplitude.

The first tensor route supports exact unitary circuit amplitudes. It intentionally does not accept trajectory-noise operations, automatically convert arbitrary QRegister state, or replace QRegister. Arbitrary state migration requires its own exact eligibility and reconstruction certificate.

## Exact execution broker

`ExactExecutionBroker` is an execution dispatcher above the exact primitives. It does not own a new state representation and it does not mutate a representation in place.

For Pauli expectations, the broker first attempts exact causal backward propagation. If operator growth, trajectory semantics, or another Pauli contract rejects that route, the broker executes the original operations on a copied `QRegister` and evaluates the same observable there.

For basis amplitudes from |0>, the broker first attempts exact bounded tensor contraction. If the contraction certificate fails or the circuit is unsupported by the tensor route, the broker executes the original operations on `QRegister` and reads the requested amplitude.

Every result reports the route that actually produced it. Specialized-route rejection is preserved as a fallback reason so an exact fallback is auditable rather than silent.

The broker therefore automates exact query routing without pretending that arbitrary QRegister states can already migrate losslessly among every backend. A future state-migration broker must prove its input-state and reconstruction certificates separately.

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

The tensor backend has a separate bounded-system differential gate against QRegister, a wide structured circuit with an analytic amplitude check, deterministic resource statistics, and an explicit contraction-limit rejection case.

The execution broker has explicit collapse gates requiring both Pauli term-growth and tensor contraction-width failures to return the same exact result through QRegister.

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

The tensor-network route and exact execution broker close two major gaps: general bounded-entanglement query execution and automatic exact fallback. The next structural step is persistent tensor forms such as Schmidt trees or matrix-product states, followed by certified state migration only where an exact reconstruction or state-family proof exists. Those paths must preserve the same fail-closed rule when bond or contraction width ceases to be bounded.
