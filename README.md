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

Examples:

* 10,000 independent qubits remain 10,000 small cells.
* 100 independent Bell pairs remain 100 separate two-qubit components.
* A 50-qubit GHZ state remains a sparse component with two nonzero amplitudes.
* A 60-qubit permutation-symmetric state can remain 61 Hamming-weight classes.
* A large Clifford circuit can remain an exact stabilizer tableau instead of becoming a statevector.

QSA avoids exponential allocation when the state has exploitable structure. It does not claim that every possible quantum state can be compressed.

---

## What Changed in QSA 0.1.7

QSA 0.1.7 is a major expansion of the engine.

The original QSA register remains in place, but it is now supported by several additional exact representations and execution systems.

### Exact structural gate acceleration

QSA can now avoid merging disconnected components when the gate semantics prove that no entanglement can be created.

This includes exact fast paths for:

* CNOT with known control or target states
* CZ with known basis states
* SWAP between disconnected components
* Sparse X and Y permutations
* Diagonal operations that touch only a small number of components

The disconnected SWAP path is especially important. A SWAP does not create entanglement, so QSA can exchange logical qubit positions without constructing the tensor product of both components.

In a measured test involving two dense 12-qubit components, this changed the operation from roughly two seconds and 256 MiB to a few microseconds and about 129 KiB.

### Native quantum-dot systems

QSA now includes a native quantum-dot pocket engine.

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

QSA now has a representation advisor for higher-level schedulers.

It can rank eligible exact engines such as:

* `QRegister`
* `SymmetryState`
* `StabilizerState`
* `PhaseGraphState`
* `QuantumDotPocket`

The advisor cannot make an invalid backend eligible. For example, a non-Clifford workload cannot be routed to the stabilizer engine simply because the stabilizer engine was previously fast.

### Component-aware symmetry discovery

Symmetry discovery no longer has to begin by materializing the entire register.

QSA can inspect independent components directly and construct exact amplitude classes from their combined structure.

This allows some large structured states to be compressed without first paying the cost of a global `2^n` statevector.

### Exact stabilizer backend

QSA now includes a packed binary symplectic tableau for Clifford circuits.

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

Long Clifford circuits can now be executed through `StabilizerState::apply_batch()`.

The batch engine:

* Validates the complete batch before changing the state
* Finds the qubits touched by the circuit
* Transposes packed tableau columns
* Applies Clifford updates using word-level Boolean operations
* Restores the normal tableau layout afterward
* Enforces a configurable scratch-memory limit

For a 4,096-qubit, 100,000-gate workload, the batch path measured roughly 120x to 138x faster than scalar stabilizer execution.

### Exact phase-graph states

QSA now includes a compact phase-graph backend for supported commuting phase circuits.

It stores local phases, edge phases, and global phase structure instead of allocating every basis amplitude.

A measured 100,000-qubit phase graph with 199,999 phase operations completed in roughly 4.6 milliseconds using about 4 MB.

### Dependency-aware execution

The operation planner can now cancel or combine exact operations even when unrelated operations appear between them.

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

| Release   | Main work                                                                                                                                                                                                                                                                                                         |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0.1.1** | Fixed the C++, C, Python, and QSC v1 compatibility surface. Added batch execution and QSC caching.                                                                                                                                                                                                                |
| **0.1.2** | Reworked component storage and added specialized sparse and dense gate kernels.                                                                                                                                                                                                                                   |
| **0.1.3** | Added compiled plans, gate fusion, parameter binding, parallel register batches, and bulk readout.                                                                                                                                                                                                                |
| **0.1.4** | Added exact Grover operations and compressed Grover evolution.                                                                                                                                                                                                                                                    |
| **0.1.5** | Added `SymmetryState`, exact amplitude classes, and Hamming-weight states.                                                                                                                                                                                                                                        |
| **0.1.6** | Added installable CMake packages, stronger CI, hostile-QSC testing, compatibility checks, security documentation, and release tooling.                                                                                                                                                                            |
| **0.1.7** | Added quantum-dot systems, structural acceleration, Bayesian compaction, representation advice, component-aware symmetry discovery, stabilizer and phase-graph backends, parallel component execution, zero-copy views, compiled diagonal plans, structured sparse kernels, and word-parallel stabilizer batches. |

The established QSA 0.1 interfaces remain available. New systems have been added through new methods and classes instead of replacing the original entry points.

---

## Current Capabilities

### State systems

* Geometric single-qubit Bloch cells
* Sparse and dense local amplitude components
* Exact amplitude-class symmetry states
* Exact Grover compression
* Exact stabilizer tableaus
* Exact phase-graph states
* Native quantum-dot pockets
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

### Measurement and readout

* Single-qubit measurement
* Sequential whole-register measurement
* Joint component measurement
* Bulk probability readout
* Zero-copy component views
* Reusable output buffers

### Execution

* Sparse and dense specialized kernels
* Direct permutation and diagonal kernels
* Linear sparse X and Y paths
* Active-component diagonal execution
* Parallel independent registers
* Parallel independent components
* Stabilizer batch execution
* Adaptive compaction
* Representation advice
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

| Workload                                                |                                           Result |
| ------------------------------------------------------- | -----------------------------------------------: |
| 1,000 independent Bell pairs compared with QSA 0.1.0    |                                **32.12x faster** |
| 20 dense CNOT gates over 65,536 amplitudes              |                               **107.18x faster** |
| 50,000 Python gate calls through one compiled plan      |                               **629.61x faster** |
| Compressed 16-qubit Grover search                       | **160,665.80x faster than the dense exact path** |
| 20-qubit symmetry fast-forward                          |                **130,750.18x faster than dense** |
| Disconnected SWAP between two dense 12-qubit components |                approximately **745,000x faster** |
| Joint measurement of one dense 20-qubit component       |                    approximately **489x faster** |
| Bayesian compaction on persistent entanglement          |               approximately **4.3x–5.4x faster** |
| Component-aware 20-qubit symmetry discovery             |                  approximately **3,845x faster** |
| 18-qubit, 800-gate stabilizer workload                  |                  approximately **1,744x faster** |
| Sparse X over 262,144 support entries                   |             up to approximately **23.3x faster** |
| Sparse Y over 262,144 support entries                   |             up to approximately **15.4x faster** |
| 4,096-qubit, 100,000-gate stabilizer batch              |               approximately **120x–138x faster** |
| 100,000-qubit phase graph with 199,999 phase operations |                         approximately **4.6 ms** |

The largest results come from exact mathematical compression or from avoiding work that the state does not require.

A fully general dense state still requires a fully general dense representation.

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

These results apply to states with the described structure. They are not compression ratios for arbitrary states.

---

## Installation

Install QSA 0.1.7 from GitHub:

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

QSA handles the mathematical state.

**Qubit** handles distributed nodes and temporary channel work.

**QELM** holds the model, learning system, memory, and orchestration logic.

```text
QELM Base
    |
    | creates a state or operation plan
    v
Qubit Network
    |
    | leases bounded work to a temporary node
    v
QSA
    |
    | executes, measures, or updates the state
    v
QSC or result returned to QELM Base
```

A Qubit node does not need QELM's complete model, long-term memory, user history, or private knowledge. It can receive a temporary mathematical state and a bounded operation plan, then return only the result or updated state.

QSA supports this design by keeping independent and structured work compact.

---

## Validation

QSA 0.1.7 passed the complete repository test matrix across Linux, macOS, and Windows.

Validation includes:

* 29 complete CTest targets
* 14,400 randomized NumPy comparison operations
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

---

## Compatibility

QSA 0.1.7 keeps the established QSA 0.1 C++, C, Python, and QSC v1 interfaces in place.

* Existing C++ names remain available.
* Existing C ABI symbols remain available.
* Existing Python imports remain available.
* Existing gate and qubit ordering remain unchanged.
* Existing QSC v1 packets remain readable.
* Existing scalar execution paths remain available.
* New structured and specialized paths are additive.
* `QSA_NATIVE_LIB` is the preferred native-library override.
* `QUBIT_NATIVE_LIB` remains supported for older projects.

Separate state objects may run concurrently. Do not mutate one state object from multiple threads at the same time.

---

## Limits

QSA avoids exponential allocation when a state has exploitable structure.

It does not make every quantum state compact.

A fully general globally entangled state may still require exponentially many independent amplitudes. If that information is present, an exact simulator has to store it somewhere.

QSA is designed to delay, localize, or avoid that cost when the mathematics permits:

* Independent qubits remain cells.
* Local entanglement remains local.
* Sparse states remain sparse.
* Equal-amplitude states can remain symmetry classes.
* Clifford circuits can remain stabilizer tableaus.
* Supported phase structures can remain phase graphs.
* Quantum-dot systems can remain bounded local components.
* Structured operations can skip unrelated components.
* Dense storage is used only where the state requires it.

Performance should be measured on the actual workload being run.

---

## Next Work

1. Persistent compiled stabilizer plans
2. Automatic exact migration between QSA representations
3. Expanded Bayesian representation routing
4. Selective GPU execution for dense local components
5. Explicit SIMD kernels where they provide repeatable gains
6. Matrix-product and tensor-linked components
7. Wider factor and separability detection
8. Larger trajectory and mixed-state workloads
9. QELM operation compiler and scheduler
10. Qubit node verification and state-destruction protocol
11. Authenticated QSC network transport
12. Android, iOS, and WebAssembly builds
13. Additional differential testing against independent simulators

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
* [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)

---

## Contributing

QSA is an active source-available research and engineering project. Noncommercial use is permitted under PolyForm Strict 1.0.0; code changes require prior written permission.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing a public API, QSC behavior, representation contract, or numerical tolerance.

Security reports should follow [`SECURITY.md`](SECURITY.md) instead of being posted as a public issue.

---

## Citation

Use the DOI at the top of this README or the repository's [`CITATION.cff`](CITATION.cff) when citing Qubit State Algebra.

For results or features specific to this release, identify the software version as **QSA 0.1.7**.

---

## License

Qubit State Algebra is source-available under the **PolyForm Strict License 1.0.0**. Noncommercial use is permitted. Distribution, modification, derivative works, sublicensing, transfer, and commercial use require separate written permission.

See [`LICENSE`](LICENSE), [`LICENSE_HISTORY.md`](LICENSE_HISTORY.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md). Earlier versions lawfully obtained under MIT remain under their original MIT terms.
