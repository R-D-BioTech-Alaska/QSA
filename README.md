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
[![Version](https://img.shields.io/badge/version-0.1.7-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-orange)](LICENSE)

### A from-scratch, qubit-native mathematical runtime for ordinary computers

**QSA builds, evolves, compresses, measures, and transports quantum states without using NumPy as its state engine.**

</div>

---

## Overview

**Qubit State Algebra (QSA)** is an exact pure-state engine built around the structure of a quantum state instead of forcing every register into one global statevector.

The numerical core is dependency-free C++20. It does not use NumPy, Qiskit, `std::complex`, BLAS, or full-system Kronecker matrices. Python controls the native engine through `ctypes`.

The repository contains two related systems:

* **QSA — Qubit State Algebra:** the in-memory state representation and execution engine.
* **QSC — Qubit State Code:** the checksummed binary format used to save and restore QSA registers.

QSA is being built as the native state engine for QELM, Qubit, and distributed quantum-channel work.

The long-term purpose is straightforward: move as much useful quantum computation as possible onto ordinary computers by exploiting exact mathematical structure before resorting to a global exponential representation. Physical quantum hardware remains useful, but QSA is designed so access to large parts of quantum computation does not have to depend on owning or renting a QPU.

---

## The Main Idea

A normal dense simulator allocates `2^n` complex amplitudes for an `n`-qubit register whether the state needs them or not.

QSA follows a different rule:

> **Do not build a global statevector unless the state actually requires one.**

A QSA register is stored as a collection of state components:

* Independent pure qubits remain geometric Bloch cells.
* Qubits merge only when an operation connects their components.
* Entangled components use sparse or dense amplitude storage as needed.
* Measurement collapses only the component being measured.
* Qubits can separate again when the state becomes factorable.
* Noise is handled through pure-state trajectories instead of a global density matrix.
* Specialized exact representations are used when stronger mathematical structure exists.
* Local observables can move backward through a circuit instead of forcing the complete state forward.
* Exact causal plans can skip gates outside an observable's backward light cone.

Examples:

* 10,000 independent qubits remain 10,000 small cells.
* 100 independent Bell pairs remain 100 separate two-qubit components.
* A 50-qubit GHZ state remains a sparse component with two nonzero amplitudes.
* A 60-qubit permutation-symmetric state can remain 61 Hamming-weight classes.
* A large Clifford circuit can remain an exact stabilizer tableau instead of becoming a statevector.
* A local Pauli observable can remain a small exact operator even when the logical register is extremely large.

QSA avoids exponential allocation when the state, circuit, or requested observable has exploitable structure. It does not claim that every possible quantum state can be compressed.

---

## Current Main After QSA 0.1.7

The released package remains QSA 0.1.7. Current `main` contains additional exact execution work that is being prepared for the next release.

### Exact component factorization repair

Component splitting now requires reconstruction of the original amplitudes before a candidate factorization is accepted.

A reduced-density determinant or numerical tolerance can be useful as a candidate test, but it is not enough to prove exact separability. The current path reconstructs the state and rejects the split if the amplitudes do not agree within the engine's exactness contract.

This preserves weak entanglement that would otherwise be at risk of being incorrectly removed by an overly permissive factorization threshold.

### Batch-native exact weighted adjoints

QSA now contains a native exact batch path for the weighted-adjoint workload used by the QELM Tripair training lane.

The path keeps the same mathematical operation while reducing bridge and per-sample overhead:

* contiguous batch input
* one native primal pass per step
* one native reverse pass per step
* exact parameter gradients
* direct NumPy pointer access at the Python boundary
* zero-copy CPU Torch/NumPy handoff where the caller already owns compatible storage

The existing scalar and general QRegister paths remain available.

### Deterministic row-parallel exact execution

Large independent row batches can be partitioned across CPU workers without changing per-row arithmetic.

The row-parallel path uses deterministic contiguous partitions and preserves the serial result exactly. The validated 65,040-row workload produced bitwise-identical forward values, reverse values, and gradients while keeping the immutable root state unchanged.

Parallel execution remains bounded and can be forced back to one worker when serial behavior is required.

### Exact sparse Pauli observables

QSA now includes `PauliObservable`, an exact sparse Heisenberg-picture observable engine.

Instead of always evolving an entire state forward, QSA can propagate a requested Pauli observable backward through a supported circuit.

The current exact route supports:

* X, Y, and Z
* H
* S and S-dagger
* T and T-dagger
* Rx, Ry, and Rz
* CNOT
* CZ
* SWAP

Non-Clifford rotations may expand one Pauli word into several exact terms. Identical words are merged exactly. No approximate term pruning is used.

A configurable exact term limit is fail-closed: if the operator becomes too large for the sparse route, QSA rejects that route instead of silently changing the mathematics.

### Exact causal Pauli propagation

`PauliPropagationPlan` adds circuit causality to sparse Pauli propagation.

The circuit is indexed by logical qubit once. A query then walks backward only through operations that touch the current observable support.

If an operation expands the support, the new qubits become part of the light cone. If the support stays local, unrelated gates are never visited.

The plan reports:

* source operation count
* visited operation count
* peak Pauli term count
* peak support width
* indexed references
* estimated plan memory

Trajectory-noise circuits are rejected by this exact route rather than partially interpreted.

The causal path is not assumed to remain cheap. Adversarial tests force the light cone to expand across a complete CNOT chain and verify that every mathematically relevant gate is then visited.

### Pauli-aware representation advice

The representation advisor can now rank an exact Pauli route when eligibility is derived from a real validated `PauliObservable` with matching register width.

Historical timing evidence may rank eligible representations, but it still cannot make an invalid representation eligible.

---

## What Changed in QSA 0.1.7

QSA 0.1.7 was a major expansion of the engine.

The original QSA register remains in place, but it is supported by several additional exact representations and execution systems.

### Exact structural gate acceleration

QSA can avoid merging disconnected components when the gate semantics prove that no entanglement can be created.

This includes exact fast paths for:

* CNOT with known control or target states
* CZ with known basis states
* SWAP between disconnected components
* Sparse X and Y permutations
* Diagonal operations that touch only a small number of components

The disconnected SWAP path is especially important. A SWAP does not create entanglement, so QSA can exchange logical qubit positions without constructing the tensor product of both components.

In a measured test involving two dense 12-qubit components, this changed the operation from roughly two seconds and 256 MiB to a few microseconds and about 129 KiB.

### Native quantum-dot systems

QSA includes a native quantum-dot pocket engine.

Each dot is represented as a four-level local system using two logical qubits. The engine supports:

* Pair-block topology
* Pair-plus-context topology
* Local charge and spin rotations
* Capacitive coupling
* Spin exchange
* Charge tunneling
* Second-order Trotter evolution
* Exact comparison against a normal `QRegister` reference path

Quantum-dot work remains opt-in and separate from the ordinary QSA register.

### Bayesian adaptive compaction

Separability checks are exact, but repeatedly scanning a persistently entangled component can waste time.

QSA 0.1.7 includes an optional Bayesian controller that learns whether another separability check is likely to succeed.

It does not change amplitudes, skip gates, prune states, or approximate the result. It only decides when an expensive representation-cleanup check is worth running.

Mandatory exact audits ensure that newly recovered separability is still detected.

Measured improvements on persistent dense entanglement were approximately 4.3x to 5.4x.

### Exact representation advisor

QSA has a representation advisor for higher-level schedulers.

It can rank eligible exact engines such as:

* `QRegister`
* `SymmetryState`
* `StabilizerState`
* `PhaseGraphState`
* `QuantumDotPocket`
* `PauliObservable` on current `main`

The advisor cannot make an invalid backend eligible. For example, a non-Clifford workload cannot be routed to the stabilizer engine simply because the stabilizer engine was previously fast.

### Component-aware symmetry discovery

Symmetry discovery does not have to begin by materializing the entire register.

QSA can inspect independent components directly and construct exact amplitude classes from their combined structure.

This allows some large structured states to be compressed without first paying the cost of a global `2^n` statevector.

### Exact stabilizer backend

QSA includes a packed binary symplectic tableau for Clifford circuits.

Supported operations include:

* X, Y, Z
* H
* S and S-dagger
* CNOT
* CZ
* SWAP
* Z measurement
* Whole-register measurement

A measured 18-qubit, 800-gate Clifford workload ran approximately 1,744x faster than the ordinary QSA register while using hundreds of bytes instead of several megabytes.

A 4,096-qubit, 100,000-gate Clifford workload can be represented using only a few megabytes.

### Word-parallel stabilizer batches

Long Clifford circuits can be executed through `StabilizerState::apply_batch()`.

The batch engine:

* Validates the complete batch before changing the state
* Finds the qubits touched by the circuit
* Transposes packed tableau columns
* Applies Clifford updates using word-level Boolean operations
* Restores the normal tableau layout afterward
* Enforces a configurable scratch-memory limit

For a 4,096-qubit, 100,000-gate workload, the batch path measured roughly 120x to 138x faster than scalar stabilizer execution.

### Exact phase-graph states

QSA includes a compact phase-graph backend for supported commuting phase circuits.

It stores local phases, edge phases, and global phase structure instead of allocating every basis amplitude.

A measured 100,000-qubit phase graph with 199,999 phase operations completed in roughly 4.6 milliseconds using about 4 MB.

### Dependency-aware execution

The operation planner can cancel or combine exact operations even when unrelated operations appear between them.

Examples include:

* X followed by X
* H followed by H
* CNOT followed by the same CNOT
* CZ followed by the same CZ
* SWAP followed by the same SWAP
* Compatible rotation fusion

A measured 200,000-operation cancellation workload reduced to zero remaining operations.

### Parallel independent-component execution

Operations on separate state components can be executed concurrently.

This preserves QSA's component boundaries while allowing additional CPU parallelism when a register contains independent work.

### Zero-copy read views

QSA can expose read-only views of component data without copying every amplitude into a new container.

This includes:

* Bloch-cell views
* Sparse support views
* Dense amplitude views
* Reusable probability output buffers
* Reusable measurement output buffers

The existing allocating methods remain available.

### Compiled diagonal plans

Repeated diagonal circuits can precompute their exact phase coefficients and reuse them.

The plan verifies the component layout and storage mode before execution. If the state no longer matches the compiled layout, the operation is rejected rather than applied incorrectly.

---

## Release History

| Release | Main work |
| ------- | --------- |
| **0.1.1** | Fixed the C++, C, Python, and QSC v1 compatibility surface. Added batch execution and QSC caching. |
| **0.1.2** | Reworked component storage and added specialized sparse and dense gate kernels. |
| **0.1.3** | Added compiled plans, gate fusion, parameter binding, parallel register batches, and bulk readout. |
| **0.1.4** | Added exact Grover operations and compressed Grover evolution. |
| **0.1.5** | Added `SymmetryState`, exact amplitude classes, and Hamming-weight states. |
| **0.1.6** | Added installable CMake packages, stronger CI, hostile-QSC testing, compatibility checks, security documentation, and release tooling. |
| **0.1.7** | Added quantum-dot systems, structural acceleration, Bayesian compaction, representation advice, component-aware symmetry discovery, stabilizer and phase-graph backends, parallel component execution, zero-copy views, compiled diagonal plans, structured sparse kernels, and word-parallel stabilizer batches. |
| **Current main** | Adds exact factorization repair, batch-native weighted adjoints, deterministic row-parallel execution, exact sparse Pauli observables, exact causal Pauli propagation, and Pauli-aware representation advice. |

The established QSA 0.1 interfaces remain available. New systems have been added through new methods and classes instead of replacing the original entry points.

---

## Current Capabilities

### State and operator systems

* Geometric single-qubit Bloch cells
* Sparse and dense local amplitude components
* Exact amplitude-class symmetry states
* Exact Grover compression
* Exact stabilizer tableaus
* Exact phase-graph states
* Native quantum-dot pockets
* Exact sparse Pauli observables
* Exact causal Pauli propagation plans
* QSC binary state storage

### Gates and evolution

* X, Y, Z, H, S, S-dagger, T, and T-dagger
* Rx, Ry, and Rz
* CNOT, CZ, and SWAP
* Arbitrary 2x2 and 4x4 C++ operators
* Exact structural gate variants
* Compiled operation plans
* Compiled diagonal plans
* Parameterized plans
* Pure-state noise trajectories
* Exact backward Pauli propagation for the supported Pauli gate family

### Measurement and readout

* Single-qubit measurement
* Sequential whole-register measurement
* Joint component measurement
* Bulk probability readout
* Zero-copy component views
* Reusable output buffers
* Exact Pauli expectation values over existing QRegister components

### Execution

* Sparse and dense specialized kernels
* Direct permutation and diagonal kernels
* Linear sparse X and Y paths
* Active-component diagonal execution
* Parallel independent registers
* Parallel independent components
* Stabilizer batch execution
* Batch-native exact weighted adjoints
* Deterministic row-parallel exact execution
* Adaptive compaction
* Representation advice
* Exact observable light-cone execution
* Optional native CPU optimization

### Interfaces

* C++ API
* Versioned C ABI
* NumPy-free Python package
* Installable CMake targets
* Linux, Windows, and macOS support
* Python 3.9, 3.12, and 3.13 package testing

---

## Selected Measured Results

These are workload-specific measurements, not one universal multiplier for every circuit.

| Workload | Result |
| -------- | -----: |
| 1,000 independent Bell pairs compared with QSA 0.1.0 | **32.12x faster** |
| 20 dense CNOT gates over 65,536 amplitudes | **107.18x faster** |
| 50,000 Python gate calls through one compiled plan | **629.61x faster** |
| Compressed 16-qubit Grover search | **160,665.80x faster than the dense exact path** |
| 20-qubit symmetry fast-forward | **130,750.18x faster than dense** |
| Disconnected SWAP between two dense 12-qubit components | approximately **745,000x faster** |
| Joint measurement of one dense 20-qubit component | approximately **489x faster** |
| Bayesian compaction on persistent entanglement | approximately **4.3x–5.4x faster** |
| Component-aware 20-qubit symmetry discovery | approximately **3,845x faster** |
| 18-qubit, 800-gate stabilizer workload | approximately **1,744x faster** |
| Sparse X over 262,144 support entries | up to approximately **23.3x faster** |
| Sparse Y over 262,144 support entries | up to approximately **15.4x faster** |
| 4,096-qubit, 100,000-gate stabilizer batch | approximately **120x–138x faster** |
| 100,000-qubit phase graph with 199,999 phase operations | approximately **4.6 ms** |
| 18-qubit entangled exact Pauli comparison on current main | **11,265.94x local measured ratio**, error `2.65121312491e-19` |
| 100,000-qubit, 100,001-gate exact causal Pauli query | **0.000692 ms** after plan build, `2 / 100,001` operations visited, error `0` |
| 65,040-row exact V10 parallel workload | approximately **2.2x–2.5x** on the measured four-core hosted runner with bitwise-identical results |

The largest results come from exact mathematical compression, operator compression, causal pruning, or avoiding work that the state does not require.

A fully general dense state still requires a fully general dense representation. A Pauli observable can also become exponentially large under adversarial non-Clifford evolution; QSA exposes that growth and rejects the bounded sparse route when its configured exact limit is crossed.

Benchmark code is available in [`benchmarks/`](benchmarks/).

---

## Structured-State Memory Results

Representative results:

```text
10,000 independent qubits
components=10000
engine-memory=1.03 MiB

50-qubit exact GHZ
components=1
nonzero-amplitudes=2
engine-memory=5.60 KiB

100 independent Bell pairs
components=100
engine-memory=25.88 KiB
```

Dense `complex128` comparison for a 50-qubit GHZ state:

```text
QSA sparse GHZ: approximately 5.60 KiB
Dense statevector: 16 PiB
```

Additional symmetry results:

```text
20-qubit amplitude-class state
logical-basis-states=1,048,576
QSA symmetry-memory=392 bytes
dense complex128 memory=16 MiB

60-qubit Hamming-weight state
logical-basis-states=1,152,921,504,606,846,976
amplitude-classes=61
QSA engine-memory=1,592 bytes
```

Current-main Pauli example:

```text
100,000 logical qubits
100,001 source operations
exact causal operations visited=2
peak Pauli terms=2
peak Pauli support=1
```

These results apply to states, circuits, and observables with the described structure. They are not compression ratios for arbitrary quantum workloads.

---

## Installation

Install the latest released QSA, 0.1.7, from GitHub:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@v0.1.7"
```

Install the current `main` branch:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@main"
```

Import QSA:

```python
from qsa import QubitRegister
```

---

## Quick Start

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

Expected nonzero amplitudes:

```text
|00> = 0.7071067811865476 + 0j
|11> = 0.7071067811865475 + 0j
```

Create a large sparse GHZ state:

```python
from qsa import QubitRegister

with QubitRegister(50) as state:
    state.h(0)

    for target in range(1, 50):
        state.cnot(0, target)

    print(state.describe())
    print("Nonzero amplitudes:", state.component_nonzero_count(0))
    print("Estimated memory:", state.estimated_bytes)
```

This creates an exact 50-qubit GHZ state without allocating a dense `2^50` statevector.

---

## Build From Source

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

For a local build optimized for the current machine:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQSTATE_NATIVE_ARCH=ON
```

Leave `QSTATE_NATIVE_ARCH` off when building portable packages or wheels.

---

## QSC

**Qubit State Code (QSC)** is the checksummed binary format used to save and restore normal QSA registers.

QSC v1 stores:

* Register size
* Component partition
* Bloch-cell state
* Sparse and dense component state
* Qubit membership and local ordering
* Numerical configuration
* State metadata
* Integrity checksum

```python
packet = state.encode_qsc()
restored = QubitRegister.decode_qsc(packet)
```

QSC v1 remains a compatibility contract. Existing QSC v1 packets remain readable by newer QSA 0.1 releases.

The checksum protects against accidental corruption. It is not authentication or encryption.

---

## QSA, Qubit, and QELM

QSA handles the mathematical state and exact local quantum execution.

**Qubit** handles distributed nodes and temporary channel work.

**QELM** holds the model, learning system, memory, and orchestration logic.

```text
QELM Base
    |
    | creates a state, observable, or operation plan
    v
Qubit Network
    |
    | leases bounded work to a temporary node
    v
QSA
    |
    | executes, propagates, measures, or updates the state
    v
QSC or result returned to QELM Base
```

A Qubit node does not need QELM's complete model, long-term memory, user history, or private knowledge. It can receive a temporary mathematical state and a bounded operation plan, then return only the result or updated state.

QSA supports this design by keeping independent, structured, and observable-local work compact.

---

## Validation

QSA is gated by the complete repository test matrix across Linux, macOS, and Windows.

Validation includes:

* Native CTest targets
* Randomized NumPy comparison operations
* Grover and symmetry comparison tests
* Quantum-dot differential, safety, stress, robustness, and concurrency tests
* Structural CNOT, CZ, and SWAP tests
* Bayesian compaction audit tests
* Representation-advisor eligibility tests
* Stabilizer comparison against `QRegister`
* Phase-graph differential validation
* Sparse X and Y permutation tests
* Structured diagonal tests
* Stabilizer scalar-versus-batch comparison
* Packed tableau boundary tests
* Invalid-batch atomicity
* Scratch-memory rejection
* Concurrent independent execution
* Exact component-factorization reconstruction checks
* Exact weighted-adjoint and V10 row-parallel comparison gates
* Direct QRegister versus exact sparse Pauli propagation
* Exact full-scan versus causal Pauli propagation
* Adversarial Pauli term-growth rejection
* Adversarial full causal-cone expansion
* ASan and UBSan
* Python 3.9, 3.12, and 3.13 package tests
* Wheel and source-distribution installation tests
* Independent installed C and C++ consumer builds

NumPy is used only as an outside reference in differential tests. It is not part of the QSA engine.

Run the native tests with:

```bash
ctest --test-dir build --output-on-failure
```

Run the independent NumPy comparison with:

```bash
python3 benchmarks/compare_numpy.py
```

The release checklist requires the default-branch `QSA Build and Test` workflow to be green before a release tag is published.

---

## Compatibility

The released QSA 0.1.7 interfaces remain the compatibility baseline while current `main` adds new exact paths.

* Existing C++ names remain available.
* Existing C ABI symbols remain available.
* Existing Python imports remain available.
* Existing gate and qubit ordering remain unchanged.
* Existing QSC v1 packets remain readable.
* Existing scalar execution paths remain available.
* New structured, Pauli, causal, batch, and parallel paths are additive.
* `QSA_NATIVE_LIB` is the preferred native-library override.
* `QUBIT_NATIVE_LIB` remains supported for older projects.

Separate state objects may run concurrently. Do not mutate one state object from multiple threads at the same time.

---

## Limits

QSA avoids exponential allocation when a state, circuit, or requested observable has exploitable structure.

It does not make every quantum state or every observable compact.

A fully general globally entangled state may still require exponentially many independent amplitudes. If that information is present, an exact simulator has to store it somewhere.

A Pauli observable under sufficiently broad non-Clifford evolution can also grow exponentially in exact term count. QSA's sparse observable route tracks that growth explicitly and fails closed when the configured bound is exceeded.

QSA is designed to delay, localize, transform, or avoid exponential cost when the mathematics permits:

* Independent qubits remain cells.
* Local entanglement remains local.
* Sparse states remain sparse.
* Equal-amplitude states can remain symmetry classes.
* Clifford circuits can remain stabilizer tableaus.
* Supported phase structures can remain phase graphs.
* Local Pauli observables can remain sparse operators.
* Causal queries can remain inside their backward light cones.
* Quantum-dot systems can remain bounded local components.
* Structured operations can skip unrelated components.
* Dense storage is used only where the state requires it.

Performance should be measured on the actual workload being run.

---

## Next Work

1. Automatic exact migration between QSA representations
2. Exact Schmidt-tree and tensor-linked components
3. Exact decision-diagram state representations
4. Expanded observable-space execution and gradients
5. Clifford-plus-magic hybrid execution
6. Phase-polynomial and wider commuting-circuit representations
7. Conserved-charge and particle-number sectors
8. Fermionic Gaussian and matchgate representations
9. Selective GPU execution for dense local components
10. Explicit SIMD kernels where they provide repeatable gains
11. Wider factor and separability detection
12. Larger trajectory and mixed-state workloads
13. QELM operation compiler and scheduler
14. Qubit node verification and state-destruction protocol
15. Authenticated QSC network transport
16. Android, iOS, and WebAssembly builds
17. Additional differential testing against independent simulators

---

## Documentation

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

## Contributing

QSA is an active source-available research and engineering project. Noncommercial use is permitted under PolyForm Strict 1.0.0; code changes require prior written permission.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing a public API, QSC behavior, representation contract, or numerical tolerance.

Security reports should follow [`SECURITY.md`](SECURITY.md) instead of being posted as a public issue.

---

## Citation

Use the DOI at the top of this README or the repository's [`CITATION.cff`](CITATION.cff) when citing Qubit State Algebra.

For the latest published software release, identify the version as **QSA 0.1.7**. When reporting results from current `main`, record the exact commit SHA because `main` contains post-0.1.7 execution work.

---

## License

Qubit State Algebra is source-available under the **PolyForm Strict License 1.0.0**. Noncommercial use is permitted. Distribution, modification, derivative works, sublicensing, transfer, and commercial use require separate written permission.

See [`LICENSE`](LICENSE), [`LICENSE_HISTORY.md`](LICENSE_HISTORY.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md). Earlier versions lawfully obtained under MIT remain under their original MIT terms.
