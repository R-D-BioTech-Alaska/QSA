# Exact representation advisor

QSA contains several exact state engines whose performance depends on workload structure. `RepresentationAdvisor` ranks the existing `QRegister`, `SymmetryState`, `QuantumDotPocket`, `StabilizerState`, `PhaseGraphState`, and sparse Pauli paths using explicit workload evidence and Bayesian outcome history.

The advisor does not convert state, execute a backend, or approximate a result. It returns a recommendation with eligibility, estimated work, posterior success, adjusted score, and a short reason. The caller remains responsible for choosing and constructing the representation.

## Operation-derived certification

`RepresentationAdvisor::inspect_operations()` accepts the actual QSA `Operation` sequence and derives Clifford compatibility from the operation codes. X, Y, Z, H, S, S-dagger, CNOT, CZ, and SWAP remain compatible. T, arbitrary rotations, and trajectory-noise operations remove the Clifford certificate.

Clifford gates alone are not sufficient to prove that an arbitrary QRegister can be converted to a stabilizer tableau. Automatic stabilizer eligibility therefore also requires an input-state certificate. The current automatic certificate is deliberately conservative: every QRegister component must be one independent Bloch cell lying on a Pauli eigenaxis. This exactly covers computational-basis, X-eigenstate, and Y-eigenstate product inputs. Entangled states and non-Pauli product states remain on QRegister unless a stronger external certificate is supplied through the existing explicit feature path.

This route is preferred when the caller has an operation list. Bayesian timing history can rank a certified route but cannot manufacture eligibility after either the operation or input-state contract rejects it.

Phase-graph eligibility is not inferred from a gate list alone because `PhaseGraphState` also requires the correct uniform-magnitude state family. The existing explicit feature path remains available when an external subsystem already owns that stronger state certificate.

Pauli eligibility is derived by `inspect_pauli()` from an actual validated `PauliObservable` with matching register width.

## Component-aware symmetry discovery

`SymmetryState::discover_components()` derives exact count-only amplitude classes directly from the independent QSA component partition. It combines local amplitude multiplicities symbolically and never constructs the full register statevector.

The result intentionally has count-only membership. It supports class probability, phases, reflections, class-unitary evolution, sampling by class, validation, and memory reporting. Operations requiring a basis-to-class mapping must continue to use ordinary discovery or an explicitly declared partition.

Discovery stops before class growth exceeds `max_classes`. This prevents an unhelpful symbolic Cartesian product from becoming another exponential path.

## Bayesian evidence

Each representation has a Beta posterior within seven workload contexts: general, fragmented, symmetry-supported, uniform phase-graph, certified Clifford/stabilizer, declared quantum-dot work, and Pauli-observable work. `observe()` records whether a backend was fastest for a comparable workload. The posterior adjusts future rankings but cannot make an ineligible backend eligible.

The reproducible benchmark target is `qstate_representation_advisor_benchmark`.
