# Exact representation advisor

QSA contains several exact state engines whose performance depends on workload structure. `RepresentationAdvisor` ranks the existing `QRegister`, `SymmetryState`, `QuantumDotPocket`, and `StabilizerState` paths using explicit workload evidence and Bayesian outcome history.

The advisor does not convert state, execute a backend, or approximate a result. It returns a recommendation with eligibility, estimated work, posterior success, adjusted score, and a short reason. The caller remains responsible for choosing and constructing the representation.

## Component-aware symmetry discovery

`SymmetryState::discover_components()` derives exact count-only amplitude classes directly from the independent QSA component partition. It combines local amplitude multiplicities symbolically and never constructs the full register statevector.

The result intentionally has count-only membership. It supports class probability, phases, reflections, class-unitary evolution, sampling by class, validation, and memory reporting. Operations requiring a basis-to-class mapping must continue to use ordinary discovery or an explicitly declared partition.

Discovery stops before class growth exceeds `max_classes`. This prevents an unhelpful symbolic Cartesian product from becoming another exponential path.

## Bayesian evidence

Each representation has a Beta posterior within five workload contexts: general, fragmented, symmetry-supported, Clifford-only, and declared quantum-dot work. `observe()` records whether a backend was fastest for a comparable workload. The posterior adjusts future rankings but cannot make an ineligible backend eligible.

The reproducible benchmark target is `qstate_representation_advisor_benchmark`.
