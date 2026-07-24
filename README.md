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
[![Version](https://img.shields.io/badge/version-0.1.6-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

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

A register is kept as a set of state components:

* Independent pure qubits stay as geometric Bloch cells `(x, y, z)`.
* Qubits merge only when a gate connects their components.
* Entangled components use sparse or dense amplitude storage as needed.
* Gates operate directly on the affected amplitudes.
* Measurement collapses only the component being measured.
* Qubits that become separable can be split back out.
* Noise is handled through pure-state trajectories instead of a global density matrix.

Examples:

* 10,000 independent qubits remain 10,000 small cells.
* 100 independent Bell pairs remain 100 separate two-qubit components.
* A 50-qubit GHZ state remains a sparse component with two nonzero amplitudes.

QSA 0.1.5 also added exact amplitude-class states. If many basis states truly share the same amplitude, QSA can store one amplitude for the entire class instead of storing every basis amplitude separately.

---

## What Changed Since 0.1.0

| Release | Main work |
|---|---|
| **0.1.1** | Fixed and froze the C++, C, Python, and QSC v1 compatibility surface. Added batch execution and QSC caching. |
| **0.1.2** | Reworked component storage and added specialized sparse and dense gate kernels. |
| **0.1.3** | Added compiled operation plans, gate fusion, parameter binding, parallel register batches, and bulk readout. |
| **0.1.4** | Added exact Grover operations and a compressed Grover engine. |
| **0.1.5** | Generalized Grover compression into `SymmetryState`, an exact amplitude-class engine. |
| **0.1.6** | Added installable CMake packages, stronger CI, sanitizer runs, hostile-QSC tests, package checks, version checks, security documentation, and release tooling. |

The old interfaces remain available. New work has been added through new methods and symbols instead of replacing the original entry points.

---

## Current Capabilities

### State representation

* QSA-owned `QComplex` scalar
* Geometric single-qubit Bloch cells
* Sparse and dense entangled components
* Local component merging
* Singleton separability recovery
* Stable component storage without register-wide reindexing on each merge
* Exact amplitude queries without global materialization
* Optional materialization for small validation cases
* Exact amplitude-class symmetry states
* Ordered-range, explicit-label, count-only, and Hamming-weight class partitions
* Discovery of equal-amplitude classes from a `QubitRegister`
* Exact conversion from a symmetry state back into a normal register when the state can be materialized

### Gates and measurement

* X, Y, Z, H, S, S-dagger, T, and T-dagger
* Rx, Ry, and Rz
* CNOT, CZ, and SWAP
* Arbitrary 2x2 operators through C++
* Arbitrary 4x4 operators through C++
* Single-qubit measurement and collapse
* Whole-register measurement
* Bit-flip, phase-flip, depolarizing, and amplitude-damping trajectories

### Execution

* Specialized sparse and dense kernels
* Direct permutation and diagonal kernels
* Compiled operation plans
* Adjacent single-qubit gate fusion
* One-pass diagonal layers
* Parameterized plans
* Parallel execution over independent registers
* Bulk probability readout
* Mutation-aware QSC caching
* Optional native CPU optimization through `QSTATE_NATIVE_ARCH`

### Grover and symmetry

* Exact Grover oracle and diffusion on `QubitRegister`
* Compressed `GroverSearch` for ideal marked/unmarked amplitude evolution
* Fast-forward of repeated Grover iterations
* General class-space unitary evolution through `SymmetryState`
* Fast-forward of repeated class-space operations
* Exact Hamming-weight states for permutation-symmetric systems

### Interfaces and packaging

* C++ API
* Versioned C ABI
* NumPy-free Python package
* Supported import: `from qsa import ...`
* Legacy `qubit_native` import compatibility
* Installable CMake targets:
  * `QSA::qstate_core`
  * `QSA::qstate`
* Linux, Windows, and macOS CI
* Python 3.9, 3.12, and 3.13 package checks

---

## Measured Results

The numbers below are from specific benchmark workloads. They are not one universal multiplier for every circuit.

### Native engine compared with QSA 0.1.0

| Workload | QSA 0.1.0 | Accelerated engine | Change |
|---|---:|---:|---:|
| 1,000 independent Bell pairs | 10.675 ms | 0.332 ms | **32.12x faster** |
| 20,000 sparse Rz gates | 2.495 ms | 0.387 ms | **6.45x faster** |
| 2,000 sparse CNOT gates | 4.978 ms | 0.751 ms | **6.63x faster** |
| 2,000 sparse Ry gates | 220.335 ms | 26.908 ms | **8.19x faster** |
| 20 dense CNOT gates over 65,536 amplitudes | 447.406 ms | 4.174 ms | **107.18x faster** |
| 20 dense Ry gates over 65,536 amplitudes | 7.764 ms | 2.810 ms | **2.76x faster** |

### Plans and batched execution

| Workload | Result |
|---|---:|
| 20,002 adjacent single-qubit operations reduced to 3 native steps | **156.85x faster** |
| 640 diagonal operations reduced to one patch traversal | **31.23x faster** |
| 50,000 Python gate calls run through one optimized native plan | **629.61x faster** |
| Bulk probability readout for 4,096 structured qubits | **9.06x faster** |
| One plan applied across 256 independent registers | **2.99x faster** |

### Grover and symmetry states

| Workload | Result |
|---|---:|
| Compressed 16-qubit Grover search against the dense exact path | **160,665.80x faster** |
| Compressed 16-qubit Grover memory against a dense statevector | **10,922.67x smaller** |
| 20-qubit symmetry state, executing all 1,000 operations | **386.97x faster than dense** |
| Same 20-qubit symmetry operation fast-forwarded | **130,750.18x faster than dense** |
| 20-qubit symmetry memory against a dense statevector | **42,799.02x smaller** |
| 60-qubit Hamming-weight state | **61 classes in 1,592 bytes** |

The large Grover and symmetry results come from exact mathematical compression. They apply when the class structure is real and remains valid. If a circuit produces a general dense state with independent amplitudes, QSA still has to store and process those amplitudes.

Benchmark code and recorded results are in [`benchmarks/`](benchmarks/) and [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

---

## Structured-State Memory Results

A representative release build produced:

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

Dense `complex128` comparisons for the same GHZ structure:

```text
20 qubits
QSA: approximately 2.32 KiB
Dense statevector: 16 MiB

30 qubits
QSA: approximately 3.41 KiB
Dense statevector: 16 GiB

40 qubits
QSA: approximately 4.51 KiB
Dense statevector: 16 TiB

50 qubits
QSA: approximately 5.60 KiB
Dense statevector: 16 PiB
```

Additional symmetry results:

```text
20-qubit amplitude-class state
logical-basis-states=1,048,576
QSA symmetry-memory=392 bytes
dense complex128 memory=16 MiB
memory-reduction=42,799.02x

60-qubit Hamming-weight state
logical-basis-states=1,152,921,504,606,846,976
amplitude-classes=61
QSA engine-memory=1,592 bytes
dense complex128 equivalent=16 EiB
```

These are structured-state results. They are not compression ratios for arbitrary quantum states.

---

## Installation

### Install from GitHub

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@v0.1.6"
```

Install the current `main` branch:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@main"
```

Import QSA with:

```python
from qsa import QubitRegister
```

Upgrade an existing installation:

```bash
python -m pip install --upgrade --force-reinstall \
  "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@main"
```

Uninstall:

```bash
python -m pip uninstall qubit-state-algebra
```

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

For a local build that will only run on the same machine:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQSTATE_NATIVE_ARCH=ON
```

Leave `QSTATE_NATIVE_ARCH` off for portable packages and wheels.

The native build produces the shared library, test programs, examples, and benchmark executables for the active platform.

---

## Use QSA From Another CMake Project

Install QSA to a prefix:

```bash
cmake --install build --prefix "$PWD/install"
```

Use the C++ engine:

```cmake
find_package(QSA CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE QSA::qstate_core)
```

Use the stable C ABI shared library:

```cmake
find_package(QSA CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE QSA::qstate)
```

Set `CMAKE_PREFIX_PATH` when the install prefix is not in a normal system location.

---

## Python Examples

### Bell State

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

### Compiled Operation Plan

```python
from qsa import OperationPlan, QubitRegister

plan = OperationPlan([
    ("h", 0),
    ("cnot", 0, 1),
    ("rz", 1, 0.25),
    ("cnot", 1, 2),
])

with QubitRegister(3) as state:
    state.apply_plan(plan)
```

Plans are useful when the same circuit is run many times. QSA can fuse adjacent single-qubit work and combine compatible diagonal operations into fewer native passes.

### Parameterized Plan

```python
from qsa import Parameter, ParameterizedPlan, QubitRegister

theta = Parameter("theta")
phi = Parameter("phi")

plan = ParameterizedPlan([
    ("ry", 0, theta),
    ("rz", 0, phi),
    ("cnot", 0, 1),
    ("ry", 1, theta),
])

states = [QubitRegister(2) for _ in range(128)]
try:
    plan.apply_many(states, {"theta": 0.31, "phi": -0.22}, workers=0)
finally:
    for state in states:
        state.close()
```

### Grover Search

```python
from qsa import GroverSearch

with GroverSearch(40, [731]) as search:
    search.run_optimal()
    print(search.success_probability)
    print(search.sample())
```

`GroverSearch` stores the shared marked and unmarked amplitudes directly. It is exact for uniform-start Grover evolution with a fixed marked set.

Use `QubitRegister` when gate-level interoperability is needed:

```python
from qsa import QubitRegister

with QubitRegister(16) as state:
    for qubit in range(16):
        state.h(qubit)

    state.grover_iterations([48_731], 201)
```

The `QubitRegister` form is exact and interoperable with normal gates, but it promotes the register into a dense component.

### Symmetry State

```python
from qsa import SymmetryState

with SymmetryState.hamming_weight(60) as state:
    state.phases([-0.02 * weight for weight in range(61)])
    state.reflect()

    print(state.class_probability(30))
    print(state.estimated_bytes)
```

A 60-qubit Hamming-weight state has `2^60` basis states but only 61 Hamming-weight classes.

### Bulk Readout

```python
probabilities = state.probabilities_one()
```

This reads every qubit's `|1>` probability in one native call.

### QSC Save and Restore

```python
from qsa import QubitRegister

with QubitRegister(2) as state:
    state.h(0)
    state.cnot(0, 1)
    packet = state.encode_qsc()

restored = QubitRegister.decode_qsc(packet)
try:
    print(restored.describe())
finally:
    restored.close()
```

### Large Structured State

```python
from qsa import QubitRegister

with QubitRegister(50) as state:
    state.h(0)

    for target in range(1, 50):
        state.cnot(0, target)

    print(state.describe())
    print("Component size:", state.component_size(0))
    print("Nonzero amplitudes:", state.component_nonzero_count(0))
    print("Estimated memory:", state.estimated_bytes)
```

This creates an exact 50-qubit GHZ state without allocating a dense `2^50` vector.

---

## Source-Tree Python Use

Build the native library first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Linux:

```bash
export PYTHONPATH="$PWD/python"
export QSA_NATIVE_LIB="$PWD/build/libqstate.so"
python3 examples/bell.py
```

macOS:

```bash
export PYTHONPATH="$PWD/python"
export QSA_NATIVE_LIB="$PWD/build/libqstate.dylib"
python3 examples/bell.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\python"
$env:QSA_NATIVE_LIB = "$PWD\build\Release\qstate.dll"
py examples\bell.py
```

A normal package installation does not require these environment variables.

---

## C++ Example

```cpp
#include "qubit/qstate.hpp"

#include <cstdint>
#include <iostream>

int main() {
    qubit::QRegister state(50);

    state.apply_h(0);
    for (std::uint32_t target = 1; target < 50; ++target) {
        state.apply_cnot(0, target);
    }

    std::cout << state.describe();
    std::cout << "Support: " << state.component_nonzero_count(0) << '\n';
}
```

---

## QSC

**Qubit State Code (QSC)** is the binary format used to save and restore normal QSA registers.

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

QSC v1 is a compatibility contract. Existing packets remain readable by newer 0.1 releases.

The checksum is for accidental corruption. It is not authentication or encryption. Networked Qubit nodes must carry QSC inside an authenticated and encrypted transport.

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
    | leases work to a temporary node
    v
QSA
    |
    | executes, measures, or updates the state
    v
QSC or result returned to QELM Base
```

The node does not need QELM's full model, long-term memory, user history, or private knowledge. It can receive a temporary mathematical state, a bounded operation plan, and return only the result or updated state.

QSA helps this model by keeping work compact:

* Independent channels stay independent.
* Sparse states stay sparse.
* Symmetric states can be represented by amplitude classes.
* Dense work can be limited to the local component that actually needs it.
* Plans replace thousands of individual calls with one native workload.

---

## Validation

Run the native tests:

```bash
ctest --test-dir build --output-on-failure
```

Run the independent NumPy comparison:

```bash
python3 benchmarks/compare_numpy.py
```

NumPy is used only as an outside reference in the differential tests. It is not part of the QSA engine.

Validation covers:

* Random single- and two-qubit gates
* Bell and GHZ states
* Sparse and dense evolution
* Measurement and post-measurement collapse
* Component separation
* Noise trajectories
* QSC round trips and frozen QSC v1 fixtures
* Corrupt and hostile QSC input
* Plan optimizer equivalence
* Grover evolution against a dense exact reference
* Symmetry evolution against a dense exact reference
* Cross-version Python/native compatibility
* Portable, optimized, and sanitizer builds
* Installed C and C++ package consumers

The 0.1.6 release gates include:

* 12 native release tests
* 14,400 randomized NumPy comparison operations
* 200 Grover searches and 41,500 amplitude checks
* 20,000 symmetry operations and 39,272 amplitude checks
* ASan and UBSan builds
* Exact tagged-archive rebuild tests

See [`docs/VALIDATION_0_1_6.md`](docs/VALIDATION_0_1_6.md) and [`docs/VALIDATION_RESULTS_0_1_6.txt`](docs/VALIDATION_RESULTS_0_1_6.txt).

---

## Compatibility

QSA 0.1 keeps the established C++, C, Python, and QSC v1 interfaces in place.

* Existing C++ names remain available.
* Existing C ABI symbols remain available.
* Existing Python imports and method forms remain available.
* Existing gate and qubit ordering remain unchanged.
* Existing QSC v1 packets remain readable.
* `QSA_NATIVE_LIB` is the preferred native-library override.
* `QUBIT_NATIVE_LIB` remains supported for older projects.

See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

### Threading

Separate state handles may run at the same time. Immutable plans may be shared across separate registers. Do not mutate one register from multiple threads at once.

See [`docs/THREADING.md`](docs/THREADING.md).

---

## Limits

QSA avoids exponential allocation when a state has exploitable structure. It does not make every quantum state compact.

A fully general, globally entangled state may still require exponentially many independent amplitudes. If that information is present, an exact simulator has to store it somewhere.

QSA is designed to delay and localize that cost:

* Independent qubits remain cells.
* Local entanglement remains local.
* Sparse states remain sparse.
* Equal-amplitude basis states can remain in symmetry classes.
* Dense storage is used only where the state requires it.

Performance should be measured on the circuit being run, not inferred from one benchmark.

---

## Next Work

1. Selective GPU execution for dense local components
2. More explicit SIMD kernels
3. Automatic representation selection and migration
4. Stabilizer components for Clifford-heavy circuits
5. Symbolic-phase components
6. Matrix-product and tensor-linked components
7. Wider factor detection
8. Larger batched trajectory and mixed-state work
9. QELM operation compiler and scheduler
10. Qubit node leases, verification, and state-destruction protocol
11. Authenticated QSC network envelope
12. Android, iOS, and WebAssembly builds
13. Differential testing against Aer, QuEST, qsim, and other independent simulators

---

## Documentation

* [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md)
* [`docs/QSC_FORMAT.md`](docs/QSC_FORMAT.md)
* [`docs/REPO_INTEGRATION.md`](docs/REPO_INTEGRATION.md)
* [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)
* [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
* [`docs/THREADING.md`](docs/THREADING.md)
* [`docs/GROVER.md`](docs/GROVER.md)
* [`docs/SYMMETRY.md`](docs/SYMMETRY.md)
* [`docs/VALIDATION_0_1_6.md`](docs/VALIDATION_0_1_6.md)
* [`docs/VALIDATION_RESULTS_0_1_6.txt`](docs/VALIDATION_RESULTS_0_1_6.txt)
* [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)

---

## Contributing

QSA is an active research and engineering project.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing a public API, QSC behavior, or numerical tolerance.

Security reports should follow [`SECURITY.md`](SECURITY.md) instead of being posted as a public issue.

---

## Citation

Use the DOI at the top of this README or the repository's [`CITATION.cff`](CITATION.cff).

---

## License

Qubit State Algebra is released under the MIT License.

See [`LICENSE`](LICENSE)
