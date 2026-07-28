# Uniform phase-graph state

`PhaseGraphState` is an exact compact backend for uniform-magnitude states whose phase is a quadratic function of computational-basis bits.

It begins in `|+>^n` and supports X, Y, Z, S, S-dagger, T, T-dagger, Rz, SWAP, CZ, and arbitrary controlled-phase operations. These operations preserve the quadratic phase form. H, CNOT, arbitrary two-qubit matrices, and nonunitary operations are intentionally absent because they do not preserve this representation in general.

The state stores one local phase per qubit, one sparse phase per interacting pair, and one global phase. Computational-basis populations remain uniform. Amplitude lookup, bounded materialization, and uniform computational-basis sampling are exact within QSA's double-precision arithmetic.

This backend targets graph states, commuting diagonal circuits, and IQP-style phase workloads. It is separate from `QRegister` and must be selected explicitly or recommended only when the caller declares the supported phase-graph gate family.

The reproducible benchmark target is `qstate_phase_graph_benchmark`.
