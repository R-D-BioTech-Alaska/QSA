# Changelog

## Unreleased

- Added deterministic randomized state-transition differential validation against an independent dense reference
- Added explicit copy-on-write branch-isolation, QSC round-trip, measurement, trajectory-noise, and Pauli-read invariants
- Added architecture-layer dependency checks so the QRegister core cannot acquire causal-runtime internals
- Added operation-derived Clifford certification for representation routing instead of relying only on caller-supplied flags
- Documented the core, structural, differentiation, and runtime boundaries and the acceptance gate for new specialized paths

## 0.1.8 — 2026-08-08

- Preserved weak entanglement during component factorization by requiring exact amplitude reconstruction before accepting a split
- Added batch-native exact weighted-adjoint execution for Tripair training without changing the matched optimization contract
- Added deterministic row-parallel exact V10 execution with bitwise-identical serial and parallel results on the validated row batch
- Added exact sparse Pauli observables with backward Heisenberg propagation through supported Clifford and non-Clifford gates
- Added exact causal Pauli propagation plans that execute only the observable's backward light cone
- Added fail-closed Pauli term-growth limits and adversarial collapse gates for both operator growth and causal-cone growth
- Added Pauli-derived representation-advisor eligibility without making unsupported representations eligible
- Preserved the existing QSA register path as the exact general fallback

## 0.1.7 — 2026-07-24

- Added native quantum-dot pocket systems and exact QRegister comparison paths
- Added exact structural gate acceleration for disconnected and structure-preserving operations
- Added Bayesian adaptive compaction with mandatory exact audits
- Added exact representation advice for QRegister, symmetry, quantum-dot, stabilizer, and phase-graph workloads
- Added component-aware exact symmetry discovery without requiring global statevector materialization first
- Added packed exact stabilizer tableaux and word-parallel stabilizer batches
- Added exact phase-graph states for supported commuting phase circuits
- Added dependency-aware execution, parallel independent-component execution, and zero-copy component read views
- Added compiled diagonal plans and structured sparse kernels
- Preserved the established QSA 0.1 interfaces and QSC v1 compatibility surface

## 0.1.6 — 2026-07-24

- Added installable CMake package targets for native C and C++ consumers
- Added independent installed-package consumer tests for `QSA::qstate` and `QSA::qstate_core`
- Made Python wheel builds omit native tests, examples, benchmarks, headers, and static development files
- Split benchmark builds from example builds without changing source-build defaults
- Added synchronized release-version validation across native, Python, package, README, and citation metadata
- Added hostile and malformed QSC decoder regression coverage
- Added Linux ASan/UBSan, package artifact, wheel install, and cross-platform minimum/current Python CI gates
- Added typed-package markers for canonical and legacy Python imports
- Added security, threading, contribution, citation, release-checklist, issue-template, and dependency-update metadata
- Preserved all QSA 0.1 mathematical behavior, C++/C/Python contracts, and QSC v1 bytes

## 0.1.5 — 2026-07-24

- Generalized Grover's two-class compression into a reusable exact amplitude-class symmetry algebra
- Added ordered-range, count-only symbolic, explicit-label, and Hamming-weight partitions
- Added exact Hamming-weight states using only `n + 1` amplitudes for `n` qubits
- Added arbitrary class-space unitary evolution over normalized class coefficients
- Added logarithmic fast-forward for repeated class-space unitaries
- Added weighted reflections and per-class phase operations
- Added class refinement and equivalence merging without logical-state materialization
- Added automatic exact or tolerance-bounded amplitude-class discovery from `QRegister`
- Added explicit `discovery_error` reporting for approximate grouping
- Added exact symmetry-to-`QRegister` fallback for operations that break the partition
- Added additive C ABI 1.5 and canonical/legacy Python `SymmetryState` interfaces
- Added randomized dense-reference symmetry differential validation
- Preserved QSC v1 bytes and every QSA 0.1 C++, C, Python, plan, and Grover contract
- Measured 386.97x repeated-operation and 130,750.18x fast-forward gains over the equivalent dense 20-qubit class workload

## 0.1.4 — 2026-07-24

- Added an exact symmetry-compressed Grover engine with two amplitude classes
- Added logarithmic-time fast-forward across any number of ideal Grover iterations
- Added explicit marked-index and count-only search-space modes up to 62 logical qubits
- Added exact full-register Grover oracle, diffusion, and iteration primitives to `QRegister`
- Added additive C ABI 1.4 and canonical/legacy Python `GroverSearch` interfaces
- Added exact sampling, amplitude queries, optimal-iteration selection, and validation
- Added dense-reference differential tests for single- and multiple-target Grover search
- Added QSC v1 round-trip coverage for exact dense Grover states without changing QSC bytes
- Optimized global materialization from repeated amplitude queries to direct component expansion
- Fixed unspecified C++ evaluation order in QSC decoding of Bloch and complex values
- Preserved every QSA 0.1 C++, C, Python, and QSC v1 compatibility contract
- Added reproducible Grover benchmarks showing over 160,000x acceleration versus dense exact evolution on the measured 16-qubit optimal-search workload

## 0.1.3 — 2026-07-24

- Added immutable native operation-plan handles to avoid rebuilding C buffers on repeated execution
- Added adjacent single-qubit matrix fusion and safe identity elision
- Added one-pass diagonal layers spanning multiple qubits in the same local patch
- Added parallel plan execution across independent register ensembles
- Added reusable parameterized plans with named Python parameters and native value binding
- Added bulk all-qubit probability readout through C++, C, and Python
- Removed temporary amplitude-vector construction from scalar probability queries
- Added randomized optimized-versus-literal plan equivalence validation
- Added native and Python benchmarks for plan fusion, diagonal layers, ensembles, parameter sweeps, and bulk readout
- Preserved the QSA 0.1 C++, C, Python, and QSC v1 compatibility contracts

## 0.1.2 — 2026-07-23

- Replaced register-wide component erasure and reindexing with stable swap-pop storage
- Preserved historical QSC v1 and `describe()` component order through compact logical order keys
- Added direct sparse and dense kernels for permutation and diagonal gates
- Replaced sparse single-qubit hash accumulation with ordered linear pair merging
- Restricted post-unitary separability checks to the two affected qubits
- Added direct dense and ordered sparse singleton-factorization paths
- Avoided dense sparse-conversion allocation unless conversion is actually selected
- Enabled supported release-time interprocedural optimization
- Added an opt-in `QSTATE_NATIVE_ARCH` build for local CPU-specific deployments
- Added a frozen multi-component QSC ordering fixture and large stable-component regression test
- Added reproducible native hot-path benchmarks showing 3.11x to 104.64x gains on measured workloads

## 0.1.1 — 2026-07-23

- Froze the QSA 0.1 C++, C, Python, and QSC compatibility contracts
- Added byte-for-byte QSC v1 fixtures and compatibility tests
- Preserved both `qsa` and historical `qubit_native` Python imports
- Preserved both `QSA_NATIVE_LIB` and `QUBIT_NATIVE_LIB` library overrides
- Made query values compatible with property and historical method call forms
- Fixed Python QSC class methods and shared native binding initialization
- Exposed S†, T†, trajectories, `measure_all`, bit-vector amplitudes, component kind, and validation through the C and Python layers
- Added ordered batched native operations for low-overhead QELM and node workloads
- Added a mutation-aware QSC serialization cache to avoid duplicate size/write encoding
- Replaced register-width component-merge remapping with component-local remapping
- Added C ABI, canonical Python, legacy Python, and frozen-packet test gates
- Added a downstream dependency scanner and compatibility/performance documentation

## 0.1.0 — 2026-07-13

- Introduced Qubit State Algebra (QSA)
- Added custom dependency-free `QComplex` scalar
- Added geometric Bloch cells for independent qubits
- Added adaptive sparse/dense entangled patches
- Added direct one- and two-qubit gate kernels
- Added exact singleton separability recovery
- Added measurement collapse and component splitting
- Added pure-state trajectory noise channels
- Added Qubit State Code (QSC) binary serialization and checksum
- Added C API and NumPy-free Python bridge
- Added C++ unit tests, sanitizer validation, and random differential tests
- Added structured-state benchmarks and integration documentation
