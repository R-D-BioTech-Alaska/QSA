# Structural hardening

QSA 0.1.8 remains the exact baseline while the next representation work is developed separately.

The transition invariant test runs deterministic randomized five-qubit sequences against an independent dense reference. It mixes ordinary and structure-aware gates, measurement, trajectory noise, QSC encode/decode replacement, copy-on-write branches, and sparse Pauli expectation reads.

A separate dense copy-on-write case starts from one dense six-qubit component, copies the register, mutates only the branch, and requires the original amplitude state to remain unchanged while storage ownership detaches correctly.

The architecture-layer test prevents the QRegister core and structural public headers from acquiring causal-runtime internals.

Representation routing also has an operation-derived Clifford inspection path. Unsupported non-Clifford and trajectory operations remove the certificate and retain QRegister as the exact fallback.

Tensor-network or Schmidt-width execution remains the next major structural backend and should land separately so its exactness and collapse behavior can be reviewed independently.
