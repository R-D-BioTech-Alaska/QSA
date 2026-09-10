<p align="center">
  <a href="https://discord.gg/sr9QBj3k36">
    <img src="https://img.shields.io/badge/Discord-Join%20the%20Server-blue?style=for-the-badge"
         alt="Join our Discord" />
  </a>
</p>

<div align="center">

# Qubit State Algebra

[![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.19653.20965-blue)](https://doi.org/10.13140/RG.2.2.19653.20965)
[![Build and Test](https://github.com/R-D-BioTech-Alaska/QSA/actions/workflows/qsa.yml/badge.svg)](https://github.com/R-D-BioTech-Alaska/QSA/actions/workflows/qsa.yml)
[![Version](https://img.shields.io/badge/version-0.3.0-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-orange)](LICENSE)

### An exact, structure-aware quantum runtime for ordinary computers

**QSA keeps quantum work in the smallest exact representation the mathematics permits.**

</div>

---

## What QSA is

**Qubit State Algebra (QSA)** is a C++20 quantum-state and numerical runtime built around structure rather than a mandatory global statevector. It is the native quantum execution layer used by the QELM and Qubit projects, but it can also be built and used independently.

The central rule is simple:

> **Do not construct a global `2^n` state unless the requested state or calculation actually requires one.**

Independent qubits can remain Bloch cells. Local entanglement can remain local. Sparse components remain sparse until density makes another representation more appropriate. Clifford circuits can stay in stabilizer form, supported phase circuits can stay phase graphs, symmetry can be represented by amplitude classes, and local Pauli questions can be answered by propagating the observable through only the part of the circuit that can affect it.

QSA is exact on the routes described here. Specialized routes are admitted only when their structural requirements are certified. If a bounded exact route cannot safely handle a workload, it either falls back to the general `QRegister` path or rejects that route rather than silently approximating the calculation.

The numerical core does not use NumPy, Qiskit, `std::complex`, BLAS, or a full-system Kronecker matrix as its state engine. Python is a control surface over the native library through `ctypes`.

---

## QSA 0.3.0

0.3.0 extends QSA from structure-aware state execution into a broader exact representation and mathematical runtime. The 0.2 numerical, tensor, estimator, gradient, state, and compatibility surfaces remain in place; the new work adds exact algebraic, relational, factorized, and bounded physical representations without changing QSC v1 or the established C ABI.

### Exact mathematical language and persistent representations

QSA now includes arbitrary-precision rational arithmetic, signed structural arithmetic, Weyl and cyclotomic algebra, exact qutrit Clifford symplectic operations, typed strict-order and affine relations, Horn-style exact logic, exact rational linear systems, and bounded exact univariate polynomial operations including Euclidean division, GCD, square-free reduction, Sturm sequences, and rational-interval root counts.

The persistent representation compiler and component fabric keep eligible mathematical structure explicit across repeated work. Admission remains typed and bounded: unsupported structure is not silently promoted into an exact route, and resource limits remain part of the representation contract.

### Structured physical representations

The existing bosonic Gaussian, sparse fixed-number Fock, coherent branch-sum, QTT, stabilizer, phase-graph, tensor, and symmetry routes remain available. 0.3.0 adds a bounded fermionic Gaussian covariance representation using real Majorana covariance matrices with explicit mode, scalar, memory, structural, and pure-state checks.

Quadratic phase-graph work also gains exact structured coherence and bounded Pauli readout where the requested support fits the certified route. These mechanisms preserve compact structure only when the mathematics allows it; they do not imply a compact route for arbitrary states or observables.

### Exact factor decisions and persistent messages

The exact factor runtime now supports deterministic max-sum variable elimination with sparse infeasibility semantics, bounded intermediate tables and backpointers, and deterministic tie handling. A persistent exact message cache can reuse unaffected elimination messages across repeated local dense-factor bindings on fixed topology.

The cache keeps the source plan and bound workspace contracts explicit, rejects stale source-plan state, rejects temporary source plans at construction, propagates dirtiness only through dependent messages, and preflights persistent message/dependency storage against the existing compiled-index cap.

On the recorded 4,096-variable chain/local-update workload, the persistent-message carrier measured a **28.38x repeat-update ratio** against the matched full-update path with zero numerical error. This number is specific to that topology and update pattern; it is not a universal factor-graph speedup claim.

### Source-bound translation contracts

0.3.0 also includes a source-bound Qwen3.5 LCT translation surface that keeps source topology, runtime context, reconstruction evidence, and translation receipts explicit and fail-closed. It is an engineering contract for bounded source translation; it does **not** establish full donor-activation equivalence or end-to-end model acceptance.

---

## QSA 0.2 foundations retained

0.2.0 introduced the compiled numerical and tensor runtime that remains part of 0.3.0.

### Native numerical core

The numerical layer provides persistent bounded CPU workers, deterministic reductions, fused real and complex arithmetic, dot and inner-product operations, and specialized small complex matrix batches. x86 builds can select AVX2/FMA kernels at runtime when the host supports them.

The point of this layer is not to reproduce a general array package. It exists to remove temporary allocations and repeated memory passes from the numerical shapes QSA actually uses.

### Reusable tensor execution

`TensorNetworkCircuit` can compile bounded-width exact contractions once and reuse the resulting plan and workspace across repeated queries. The same approach is used for exact Pauli expectations.

Compilation records the contraction structure, resource bounds, peak contraction width, and workspace requirements. `max_factors` and `max_contraction_entries` remain hard exactness/resource gates; no hidden truncation is introduced when those limits are exceeded.

This lets structured circuits reach logical widths that would be impossible to represent as a dense statevector when the contraction width stays bounded.

### Estimation and parameter sweeps

QSA includes exact estimator planning for Pauli observables and a compile-once parameterized estimator for Rx, Ry, and Rz parameter sweeps.

Parameterized tensor execution does not rebuild the complete circuit for each point when the topology is reusable. Parameterized gate sources are rebound directly, fixed tensor sources are retained, and observable terms proven independent of the parameters can be cached once. Shared parameter slots are supported, and the general QRegister route remains the exact fallback.

### Exact gradients

The tensor runtime includes exact reverse-mode adjoint gradients, bounded term scheduling, point batching, and causal pruning.

Before a gradient calculation, QSA can walk a Pauli term backward through the circuit and determine whether its support can reach any parameterized operation. Terms that cannot depend on a parameter are evaluated once. Only the dynamic terms enter the reverse-mode calculation. This is a structural proof, not a numerical guess based on a small derivative.

The result is especially useful for large logical circuits where a requested observable touches only a small causal region.

---

## Core systems

| System | Role |
| --- | --- |
| `QRegister` | General exact register using Bloch cells plus sparse/dense local components |
| `OperationPlan` / parameterized plans | Reusable native circuit execution and binding |
| `SymmetryState` | Exact amplitude-class and Hamming-weight representations |
| `StabilizerState` | Packed exact Clifford tableau with word-parallel batch execution |
| `PhaseGraphState` | Compact exact representation for supported commuting phase structure |
| `QuantumDotPocket` | Native bounded quantum-dot systems and reference comparisons |
| `PauliObservable` | Exact sparse Pauli operators and Heisenberg propagation |
| `PauliPropagationPlan` | Exact observable light-cone indexing and execution |
| `TensorNetworkCircuit` | Bounded-width exact tensor execution and reusable contraction plans |
| Exact estimator plans | Observable evaluation, parameter sweeps, and exact route fallback |
| Exact adjoint plans | Reverse-mode parameter gradients with bounded scheduling and causal pruning |
| Exact representation compiler / fabric | Typed persistent exact mathematical and structural representations |
| QMath / typed relation and logic | Exact rational, signed, Weyl, qutrit, linear, relational, logic, and polynomial operations |
| Exact factor runtime | Bounded exact elimination, decisions, bindings, and persistent local messages |
| Fermionic Gaussian state | Bounded real Majorana covariance representation |
| QSC v1 | Checksummed binary storage for normal QSA registers |

The representations are complementary. QSA does not force every calculation through the newest backend; the representation has to fit the mathematics of the workload.

---

## Measured results

The benchmark suite records both speed and numerical agreement. These numbers are **workload-specific measurements**, not a claim that QSA is universally faster than NumPy, Qiskit, SymPy, or every dense simulator.

### Compiled runtime evidence

| Workload | Measured result |
| --- | ---: |
| Fused native real/complex numerical kernels and 2x2 complex batches | **4.52x to 7.96x faster than the matched NumPy runs** on the recorded hosted-CPU workloads |
| Repeated 18q exact tensor amplitude queries after compilation | **32.6x** faster than rebuilding/direct contraction |
| Repeated 100q bounded-width tensor amplitude queries after compilation | **36.4x** faster than rebuilding/direct contraction |
| 18q exact gradient, 24 observables / 6 parameters | **96.2x** best execution ratio vs matched Aer statevector parameter shift; **5.79x** setup-plus-first ratio |
| 100q exact gradient, 8 observables / 4 parameters | **364x** best execution ratio vs matched exact Aer MPS parameter shift; **4.51x** setup-plus-first ratio |
| Parameterized estimator sweep, 18q | roughly **29x** vs the matched Aer sweep in the recorded workload |
| Parameterized estimator sweep, 100q | roughly **161x to 239x** vs matched exact Aer MPS in the recorded workload |
| Persistent exact factor messages, 4,096-variable chain/local update | **28.38x** repeat-update ratio vs the matched full-update path, zero numerical error |

The NumPy comparison uses the same numerical work and preallocated outputs; the recorded evidence compares QSA's four persistent workers with NumPy pinned to one backend thread. It is evidence for these fused QSA workloads, not a generic BLAS/GEMM comparison.

The Aer comparisons use the same circuit family, observables, parameters, and exact-result checks. The 100-qubit Aer route uses MPS with truncation disabled. The exact-rational evidence lane checks QSA output against a matched SymPy control before reporting timing. These results demonstrate the value of preserving structure and causal locality; they do not imply the same multiplier for arbitrary circuits.

### Earlier structural results

QSA's older structural engines remain part of 0.3.0. Representative measurements include an approximately 1,744x gain on the recorded 18-qubit Clifford workload, approximately 120x to 138x for a 4,096-qubit stabilizer batch, a 100,000-qubit phase graph evolved in roughly 4.6 ms, and exact 50-qubit GHZ storage in a few KiB instead of a dense 16 PiB statevector.

Benchmark sources live in [`benchmarks/`](benchmarks/). The runtime evidence workflow records the matching configuration and error checks alongside the timings.

---

## A few concrete examples

A normal dense representation for 50 qubits contains `2^50` complex amplitudes. A GHZ state only needs two nonzero amplitudes, so QSA keeps it sparse.

A 60-qubit permutation-symmetric state can be represented by 61 Hamming-weight classes instead of enumerating its full basis.

A Clifford circuit can remain a stabilizer tableau without becoming an amplitude vector at all.

A 100,000-qubit circuit can still admit a tiny exact Pauli query when only a small backward light cone can affect that observable. If the light cone expands across the circuit, QSA follows it; it does not pretend the query stayed local.

This distinction is important: QSA removes unnecessary exponential work. It does not remove information that is genuinely exponential.

---

## Installation

Install QSA 0.3.0 from the release tag:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@v0.3.0"
```

Install the current `main` branch:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@main"
```

Import the Python interface:

```python
from qsa import QubitRegister
```

---

## Quick start

Create a Bell state:

```python
from qsa import QubitRegister

with QubitRegister(2) as state:
    state.h(0)
    state.cnot(0, 1)

    print(state.amplitude(0))
    print(state.amplitude(3))
    print(state.describe())
```

The nonzero amplitudes are approximately:

```text
|00> = 0.7071067811865476 + 0j
|11> = 0.7071067811865475 + 0j
```

A large sparse GHZ state uses the same interface:

```python
from qsa import QubitRegister

with QubitRegister(50) as state:
    state.h(0)
    for target in range(1, 50):
        state.cnot(0, target)

    print(state.component_nonzero_count(0))
    print(state.estimated_bytes)
```

This remains an exact 50-qubit state without allocating a dense `2^50` array.

---

## Build from source

Requirements:

* CMake 3.20 or newer
* A C++20 compiler
* Python 3.9 or newer for the Python package

```bash
git clone https://github.com/R-D-BioTech-Alaska/QSA.git
cd QSA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For a local CPU-specific build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQSTATE_NATIVE_ARCH=ON
```

Leave `QSTATE_NATIVE_ARCH` off for portable packages and wheels.

---

## Exactness, fallback, and limits

QSA's fast paths are conditional on exact structural eligibility.

A fully general globally entangled state may still require exponentially many independent amplitudes. A Pauli observable propagated through a sufficiently broad non-Clifford circuit can also grow exponentially in exact term count. A tensor contraction can become too wide to be useful. Those are properties of the calculation, not conditions QSA hides.

The runtime therefore uses explicit boundaries:

* exact reconstruction is required before component factorization is accepted;
* sparse Pauli propagation has a fail-closed term cap;
* causal execution expands when the real backward light cone expands;
* tensor execution enforces factor and contraction-entry limits;
* parameter-independent caching is based on structural dependency proofs;
* exact factor and representation routes enforce their declared size and compiled-index limits;
* no accepted release path silently truncates amplitudes, Pauli terms, MPS bonds, gradients, exact factors, or typed algebraic results;
* `QRegister` remains the general exact fallback where a specialized representation is not eligible.

Performance should be measured on the actual workload being run.

---

## Compatibility

QSA 0.3.0 keeps the established 0.1 compatibility surface while adding the newer exact C++ systems.

* The C ABI remains **1.5.0**.
* QSC remains **version 1**.
* Existing C++ names and C symbols remain available.
* Existing `qsa` and historical `qubit_native` Python imports remain available.
* Existing gate ordering and qubit ordering are unchanged.
* `QSA_NATIVE_LIB` remains the preferred native-library override; `QUBIT_NATIVE_LIB` remains supported.
* Existing QSC v1 packets remain readable.

Separate state objects may execute concurrently. A single mutable state object should not be mutated from multiple threads at the same time.

---

## QSC

**Qubit State Code (QSC)** is the checksummed binary format used to save and restore normal QSA registers.

QSC v1 stores the register size, component partition, Bloch/sparse/dense state, qubit membership, numerical configuration, state metadata, and an integrity checksum.

```python
packet = state.encode_qsc()
restored = QubitRegister.decode_qsc(packet)
```

The checksum is for corruption detection. QSC is not an authentication or encryption format.

---

## Validation

The release build is gated across Linux, macOS, and Windows. The suite includes native CTest targets, randomized dense-reference comparisons, QSC compatibility and hostile-input tests, structural and representation checks, stabilizer and phase-graph differential tests, exact tensor and estimator tests, parameter-sweep and gradient tests, causal-collapse cases, exact mathematical-language and factor-route tests, package installation tests, independent installed C/C++ consumers, and Linux ASan/UBSan.

The runtime evidence lane records matched numerical and quantum benchmarks together with numerical error, memory, route selection, and exactness boundaries. Its historical workflow filename still contains the 0.2.0 label; the release version is determined by the synchronized package metadata and source tree.

NumPy, SymPy, and Qiskit Aer are used as outside references in benchmark and differential work. They are not runtime dependencies of the QSA state engine.

Run the native suite with:

```bash
ctest --test-dir build --output-on-failure
```

---

## QSA, Qubit, and QELM

The three projects have different jobs.

**QSA** owns mathematical state representation and local exact execution. **Qubit** handles distributed temporary nodes and channel work. **QELM** owns the model, learning system, memory, orchestration, and long-lived intelligence.

```text
QELM Base
    |
    | state / observable / operation plan
    v
Qubit Network
    |
    | bounded temporary work
    v
QSA
    |
    | exact execution / propagation / measurement
    v
QSC or result returned to QELM Base
```

This separation lets a temporary Qubit node execute a bounded mathematical job without receiving QELM's complete model, long-term memory, conversation history, or private knowledge.

---

## Release history

| Release | Main work |
| --- | --- |
| **0.3.0** | Persistent exact representation/compiler fabric, exact mathematical language and rational algebra, Weyl/qutrit/typed relation/logic systems, exact rational linear and polynomial operations, source-bound translation contracts, bounded fermionic Gaussian covariance, structured phase coherence readout, exact factor max-sum decisions, persistent factor-message reuse, and matched exact-rational evidence. |
| **0.2.0** | Native fused numerical runtime, reusable exact tensor contraction/expectation, estimator and parameter-sweep execution, direct tensor rebinding, exact adjoint gradients, bounded scheduling, and causal/static gradient pruning. |
| **0.1.9** | Development bridge into 0.2.0. The accepted numerical, tensor, estimator, and differentiation work from this bridge was consolidated into 0.2.0 rather than maintained as a separate long-lived feature line. |
| **0.1.8** | Exact factorization reconstruction, batch-native weighted adjoints, deterministic row-parallel execution, exact sparse Pauli observables, and causal Pauli propagation. |
| **0.1.7** | Quantum-dot systems, structural acceleration, adaptive compaction, representation advice, stabilizer/phase-graph engines, component parallelism, zero-copy views, diagonal plans, and structured sparse kernels. |
| **0.1.6** | Installable CMake packages, stronger cross-platform CI, hostile-QSC checks, packaging validation, and release tooling. |
| **0.1.5** | Exact amplitude-class symmetry algebra and Hamming-weight states. |
| **0.1.4** | Exact compressed Grover execution. |
| **0.1.3** | Compiled plans, gate fusion, parameter binding, parallel register batches, and bulk readout. |
| **0.1.2** | Reworked component storage and specialized sparse/dense kernels. |
| **0.1.1** | Frozen C++, C, Python, and QSC v1 compatibility surface. |
| **0.1.0** | Initial Qubit State Algebra engine. |

The full release record is in [`CHANGELOG.md`](CHANGELOG.md).

---

## Documentation

Technical details are kept in the repository rather than repeated throughout this README:

* [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md)
* [`docs/QSC_FORMAT.md`](docs/QSC_FORMAT.md)
* [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)
* [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
* [`docs/THREADING.md`](docs/THREADING.md)
* [`docs/GROVER.md`](docs/GROVER.md)
* [`docs/SYMMETRY.md`](docs/SYMMETRY.md)
* [`docs/STABILIZER.md`](docs/STABILIZER.md)
* [`docs/SCALING_FAST_PATHS.md`](docs/SCALING_FAST_PATHS.md)
* [`docs/QUANTUM_DOTS.md`](docs/QUANTUM_DOTS.md)
* [`docs/CAUSAL_RUNTIME.md`](docs/CAUSAL_RUNTIME.md)
* [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)

---

## Contributing and security

QSA is an active source-available research and engineering project. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing a public API, QSC behavior, representation contract, or numerical tolerance.

Security reports should follow [`SECURITY.md`](SECURITY.md) rather than being posted as a public issue.

---

## Citation

Use the DOI at the top of this README or [`CITATION.cff`](CITATION.cff) when citing Qubit State Algebra. For results or features specific to this release, identify the software version as **QSA 0.3.0**.

---

## License

Qubit State Algebra is source-available under the **PolyForm Strict License 1.0.0**. Noncommercial use is permitted. Distribution, modification, derivative works, sublicensing, transfer, and commercial use require separate written permission.

See [`LICENSE`](LICENSE), [`LICENSE_HISTORY.md`](LICENSE_HISTORY.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md). Earlier versions lawfully obtained under MIT remain under their original MIT terms.