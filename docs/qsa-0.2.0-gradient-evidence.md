# QSA 0.2.0 exact gradient evidence

This evidence lane compares QSA's exact reverse-mode adjoint scheduler against Qiskit Aer exact parameter shift on matched parameterized circuits and Pauli observables. It is a workload-specific hosted-runner comparison, not a universal performance claim.

The 18-qubit case uses Aer double-precision statevector. The 100-qubit case uses Aer double-precision matrix-product state with truncation threshold and chop threshold set to zero. Aer uses a four-thread ceiling, disables experiment and shot parallelism, transpiles the parameterized circuit once at optimization level 0, and evaluates one base circuit plus exact `+/- pi/2` shifts for each parameter. QSA uses the accepted single-plan runtime-width exact-adjoint scheduler with the same four-worker ceiling.

Both implementations use the same five-layer nearest-neighbor brickwork topology, the same first-N parameterized `Ry`/`Rz` operation positions, the same deterministic parameter vector, and the same Pauli-observable generator. QSA emits every value and gradient component as well as aggregate checksums. The Aer comparison fails closed if the maximum absolute value or gradient disagreement exceeds `5e-9`.

Timing is separated into setup and execution components. QSA reports workload construction, scheduler compilation, workspace/output allocation, and first/best gradient execution. Aer reports backend initialization, workload construction, one transpilation, observable attachment, parameter binding, backend execution, and full first/best parameter-shift gradient wall time. Wrapper and binding overheads remain visible rather than being subtracted from headline execution measurements.

No approximation, stochastic sampling, or silent truncation is allowed in this benchmark. The candidate is eligible for the QSA 0.2.0 evidence branch only after Runtime Evidence, Structural Hardening, and the full Build and Test matrix are green on the exact candidate head.