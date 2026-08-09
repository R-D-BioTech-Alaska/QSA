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
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](CHANGELOG.md)
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

## QSA 0.2.0

0.2.0 is the first release of the newer compiled numerical and tensor runtime developed through the 0.1.9 bridge. It keeps the established QSA state engine and compatibility surface, then adds reusable exact execution plans around it.

### Native numerical core

The new numerical layer provides persistent bounded CPU workers, deterministic reductions, fused real and complex arithmetic, dot and inner-product operations, and specialized small complex matrix batches. x86 builds can select AVX2/FMA kernels at runtime when the host supports them.

The point of this layer is not to reproduce a general array package. It exists to remove temporary allocations and repeated memory passes from the numerical shapes QSA actually uses.

### Reusable tensor execution

`TensorNetworkCircuit` can compile bounded-width exact contractions once and reuse the resulting plan and workspace across repeated queries. The same approach is used for exact Pauli expectations.

Compilation records the contraction structure, resource bounds, peak contraction width, and workspace requirements. `max_factors` and `max_contraction_entries` remain hard exactness/resource gates; no hidden truncation is introduced when those limits are exceeded.

This lets structured circuits reach logical widths that would be impossible to represent as a dense statevector when the contraction width stays bounded.

### Estimation and parameter sweeps

QSA 0.2.0 includes exact estimator planning for Pauli observables and a compile-once parameterized estimator for Rx, Ry, and Rz parameter sweeps.

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
| QSC v1 | Checksummed binary storage for normal QSA registers |

The representations are complementary. QSA does not force every calculation through the newest backend; the representation has to fit the mathematics of the workload.

---

## Measured results

The benchmark suite records both speed and numerical agreement. These numbers are **workload-specific measurements**, not a claim that QSA is universally faster than NumPy, Qiskit, or every dense simulator.

### QSA 0.2 runtime evidence

| Workload | Measured result |
| --- | ---: |
| Fused native real/complex numerical kernels and 2x2 complex batches | **4.52x to 7.96x faster than the matched NumPy runs** on the recorded hosted-CPU workloads |
| Repeated 18q exact tensor amplitude queries after compilation | **32.6x** faster than rebuilding/direct contraction |
| Repeated 100q bounded-width tensor amplitude queries after compilation | **36.4x** faster than rebuilding/direct contraction |
| 18q exact gradient, 24 observables / 6 parameters | **96.2x** best execution ratio vs matched Aer statevector parameter shift; **5.79x** setup-plus-first ratio |
| 100q exact gradient, 8 observables / 4 parameters | **364x** best execution ratio vs matched exact Aer MPS parameter shift; **4.51x** setup-plus-first ratio |
| Parameterized estimator sweep, 18q | roughly **29x** vs the matched Aer sweep in the recorded workload |
| Parameterized estimator sweep, 100q | roughly **161x to 239x** vs matched exact Aer MPS in the recorded workload |

The NumPy comparison uses the same numerical work and preallocated outputs; the recorded evidence compares QSA's four persistent workers with NumPy pinned to one backend thread. It is evidence for these fused QSA workloads, not a generic BLAS/GEMM comparison.

The Aer comparisons use the same circuit family, observables, parameters, and exact-result checks. The 100-qubit Aer route uses MPS with truncation disabled. These results demonstrate the value of preserving structure and causal locality; they do not imply the same multiplier for arbitrary circuits.

### Earlier structural results

QSA's older structural engines remain part of 0.2.0. Representative measurements include an approximately 1,744x gain on the recorded 18-qubit Clifford workload, approximately 120x to 138x for a 4,096-qubit stabilizer batch, a 100,000-qubit phase graph evolved in roughly 4.6 ms, and exact 50-qubit GHZ storage in a few KiB instead of a dense 16 PiB statevector.

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

Install QSA 0.2.0 from the release tag:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@v0.2.0"
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
* no accepted 0.2 path silently truncates amplitudes, Pauli terms, MPS bonds, or gradients;
* `QRegister` remains the general exact fallback where a specialized representation is not eligible.

Performance should be measured on the actual workload being run.

---

## Compatibility

QSA 0.2.0 keeps the established 0.1 compatibility surface while adding the newer runtime as additive C++ systems.

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

The release build is gated across Linux, macOS, and Windows. The suite includes native CTest targets, randomized dense-reference comparisons, QSC compatibility and hostile-input tests, structural and representation checks, stabilizer and phase-graph differential tests, exact tensor and estimator tests, parameter-sweep and gradient tests, causal-collapse cases, package installation tests, independent installed C/C++ consumers, and Linux ASan/UBSan.

The 0.2 runtime evidence lane additionally records matched numerical and quantum benchmarks together with numerical error, memory, route selection, and exactness boundaries.

NumPy and Qiskit Aer are used as outside references in benchmark and differential work. They are not runtime dependencies of the QSA state engine.

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

Use the DOI at the top of this README or [`CITATION.cff`](CITATION.cff) when citing Qubit State Algebra. For results or features specific to this release, identify the software version as **QSA 0.2.0**.

---

## License

Qubit State Algebra is source-available under the **PolyForm Strict License 1.0.0**. Noncommercial use is permitted. Distribution, modification, derivative works, sublicensing, transfer, and commercial use require separate written permission.

See [`LICENSE`](LICENSE), [`LICENSE_HISTORY.md`](LICENSE_HISTORY.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md). Earlier versions lawfully obtained under MIT remain under their original MIT terms.
