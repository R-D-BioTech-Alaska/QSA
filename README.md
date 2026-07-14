<p align="center">
  <a href="https://discord.gg/sr9QBj3k36">
    <img src="https://img.shields.io/badge/Discord-Join%20the%20Server-blue?style=for-the-badge"
         alt="Join our Discord" />
  </a>
</p>

<div align="center">

# Qubit State Algebra

[![DOI](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.19653.20965-blue)](https://doi.org/10.13140/RG.2.2.19653.20965)

### A from-scratch, qubit-native mathematical runtime for ordinary computers

**QSA builds, evolves, compresses, measures, and transports quantum states without relying on NumPy as its underlying state engine.**

</div>

---

## Overview

**Qubit State Algebra (QSA)** is a working first-generation engine for representing and evolving qubit states on ordinary computer hardware.

QSA does not use NumPy, Qiskit, `std::complex`, BLAS, or full-system Kronecker gate matrices in its numerical core. The engine is written in dependency-free C++20 and exposed to Python through a lightweight `ctypes` interface.

The project introduces two related systems:

* **QSA — Qubit State Algebra:** The adaptive mathematical representation used to construct and evolve quantum states in memory.
* **QSC — Qubit State Code:** A checksummed binary state format designed for storage, restoration, and future Qubit-node transport.

QSA is intended to serve as the native state engine for Qubit, QELM, and future distributed quantum-channel systems.

---

## What Makes QSA Different

A conventional dense simulator represents an `n`-qubit state as a single array containing `2^n` complex amplitudes.

QSA begins with a different rule:

> **Do not construct a global statevector unless the state itself requires a global representation.**

A QSA register is maintained as a product of independent state components:

* Independent pure qubits are stored as **geometric Bloch cells** represented by `(x, y, z)`.
* Interacting qubits are merged only into the **local entangled patch** that requires them.
* Entangled patches automatically select **sparse** or **dense** amplitude storage.
* Gates operate directly on affected amplitudes instead of embedding a full gate matrix across the register.
* Measurement collapses only the affected state component.
* Measured or separable qubits can be recovered as independent state cells.
* Rank-one reduced-state checks identify qubits that have become separable.
* Noise is represented through pure-state **quantum trajectories**, avoiding a global `4^n` density matrix.

This allows structured quantum states to remain compact.

Examples:

* 10,000 independent qubits remain 10,000 small geometric cells.
* 100 independent Bell pairs remain 100 separate two-qubit patches.
* A 50-qubit GHZ state remains a sparse patch containing only two nonzero amplitudes.

---

## Current Capabilities

* Custom QSA-owned `QComplex` scalar
* Geometric pure-qubit state cells
* Adaptive sparse and dense entangled patches
* Component merging
* Singleton factor recovery
* Direct single-qubit gate execution
* Direct two-qubit gate execution
* X, Y, Z, H, S, S†, T, and T†
* Rx, Ry, and Rz
* CNOT, CZ, and SWAP
* Arbitrary 2×2 operators through the C++ API
* Arbitrary 4×4 operators through the C++ API
* Measurement and state collapse
* Bit-flip trajectories
* Phase-flip trajectories
* Depolarizing trajectories
* Amplitude-damping trajectories
* Direct amplitude queries without global materialization
* Optional global state materialization for small validation cases
* QSC binary serialization
* QSC checksum validation
* Native C API
* NumPy-free Python package
* Bell-state tests
* GHZ-state tests
* Measurement-collapse tests
* Noise tests
* Component-compaction tests
* QSC round-trip tests
* Differential validation against an independent NumPy statevector reference

---

## Installation

### Install Directly From GitHub

QSA can be installed directly from the `main` branch:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@main"
```

After a tagged release is available, install that specific version instead:

```bash
python -m pip install "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@v0.1.0"
```

The supported Python import is:

```python
from qsa import QubitRegister
```

### Upgrade an Existing Installation

```bash
python -m pip install --upgrade --force-reinstall "qubit-state-algebra @ git+https://github.com/R-D-BioTech-Alaska/QSA.git@main"
```

### Uninstall

```bash
python -m pip uninstall qubit-state-algebra
```

---

## Build From Source

Requirements:

* CMake 3.20 or newer
* A C++20-compatible compiler
* Python 3.9 or newer for the optional Python package

Clone the repository:

```bash
git clone https://github.com/R-D-BioTech-Alaska/QSA.git
cd QSA
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The native build creates:

* `build/libqstate.so` on Linux
* `build/libqstate.dylib` on macOS
* `build/qstate.dll` on Windows
* `build/qstate_tests`
* `build/qstate_bell`
* `build/qstate_benchmark`

---

## Python Example

The Python package controls the native QSA engine and does not import NumPy.

```python
from qsa import QubitRegister

with QubitRegister(2) as state:
    state.h(0)
    state.cnot(0, 1)

    print(state.amplitude(0))  # |00>
    print(state.amplitude(3))  # |11>
    print(state.describe())

    packet = state.encode_qsc()
```

Expected Bell-state amplitudes are approximately:

```text
|00> = 0.7071067811865476 + 0j
|11> = 0.7071067811865475 + 0j
```

### Restore a State From QSC

```python
from qsa import QubitRegister

with QubitRegister(2) as state:
    state.h(0)
    state.cnot(0, 1)

    packet = state.encode_qsc()

restored = QubitRegister.decode_qsc(packet)

try:
    print(restored.describe())
    print(restored.amplitude(0))
    print(restored.amplitude(3))
finally:
    restored.close()
```

### Create a Large Structured State

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

This creates a 50-qubit GHZ state without allocating a dense `2^50` statevector.

---

## Running the Source-Tree Python Example

Build the native library first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

On Linux:

```bash
export PYTHONPATH="$PWD/python"
export QUBIT_NATIVE_LIB="$PWD/build/libqstate.so"
python3 examples/bell.py
```

On macOS:

```bash
export PYTHONPATH="$PWD/python"
export QUBIT_NATIVE_LIB="$PWD/build/libqstate.dylib"
python3 examples/bell.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\python"
$env:QUBIT_NATIVE_LIB = "$PWD\build\Release\qstate.dll"
py examples\bell.py
```

A normal package installation does not require manually setting `PYTHONPATH`.

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
    std::cout
        << "Support: "
        << state.component_nonzero_count(0)
        << '\n';

    return 0;
}
```

---

## QSC Node-State Format

**Qubit State Code (QSC)** is the binary representation used to serialize QSA registers.

```python
packet = state.encode_qsc()
restored = QubitRegister.decode_qsc(packet)
```

QSC preserves:

* Register size
* Component partition
* Geometric Bloch-cell state
* Sparse patch state
* Dense patch state
* Qubit membership
* Local qubit ordering
* Numerical configuration
* State metadata
* Integrity checksum

QSC is intended to become the first transport representation for leased quantum channels in the Qubit Network.

A future Qubit node can receive a QSC state fragment, execute QELM-assigned operations, measure or decode its result, and return an updated state packet or result packet to QELM Base.

---

## Validation

QSA includes native C++ tests and independent differential validation.

Run the native test suite:

```bash
ctest --test-dir build --output-on-failure
```

Run the benchmark:

```bash
./build/qstate_benchmark
```

Run the independent NumPy comparison:

```bash
python3 benchmarks/compare_numpy.py
```

The NumPy benchmark is used only as an external validation reference. NumPy is not used by the QSA engine itself.

Validation currently covers:

* Random single-qubit gates
* Random two-qubit gates
* Bell states
* GHZ states
* Sparse state evolution
* Dense patch promotion
* Measurement
* Post-measurement collapse
* Component separation
* QSC encoding
* QSC decoding
* State round trips
* Noise trajectories
* Fidelity comparison against a conventional statevector

---

## Initial Structured-State Results

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

Approximate dense `complex128` statevector comparisons for GHZ states:

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

These comparisons apply to the structured GHZ representation and should not be interpreted as universal compression ratios for arbitrary quantum states.

---

## Scientific Boundary

QSA removes unnecessary exponential allocation for quantum states that contain exploitable structure.

It does **not** claim that every possible quantum state can be represented compactly.

A sufficiently general and highly entangled state may still require exponentially many independent amplitudes. When that information is genuinely present in the state, no exact representation can simply discard it.

The purpose of QSA is to avoid paying that exponential cost before the actual state requires it.

Version 0.1 is:

* An exact pure-state engine within configured floating-point tolerance
* A research foundation
* An adaptive structured-state representation
* A native runtime for Qubit and QELM development

Version 0.1 is not yet proof that QSA is universally faster than every existing quantum simulator. Performance and memory claims should be evaluated circuit by circuit using reproducible benchmarks.

---

## Relationship to Qubit and QELM

QSA is the mathematical state engine.

**Qubit** is the broader distributed quantum-channel platform.

**QELM** is the intelligence and learning architecture that can control those channels.

The intended relationship is:

```text
QELM Base
    ↓
Creates and orchestrates quantum-channel logic
    ↓
Qubit Network
    ↓
Allocates temporary Qubit nodes and channel leases
    ↓
QSA
    ↓
Constructs, evolves, measures, and serializes qubit states
```

In this structure:

* QELM holds persistent intelligence and learning logic.
* Qubit provides the distributed channel and node system.
* QSA provides the native mathematical state representation.
* QSC transports QSA states between trusted systems and temporary nodes.

---

## Next Engineering Stages

1. SIMD kernels for dense local patches
2. GPU-backed patch execution
3. General multi-component factor detection
4. Stabilizer component mode
5. Symbolic-phase component mode
6. Matrix-product and tensor-linked component modes
7. Mixed-state ensembles
8. Batched quantum trajectories
9. QELM channel-operation compiler
10. Qubit node lease protocol
11. Qubit network transport protocol
12. Android builds
13. iOS builds
14. WebAssembly builds
15. Differential validation against Qiskit Aer
16. Differential validation against additional independent simulators

---

## Documentation

Additional technical documentation:

* [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md)
* [`docs/QSC_FORMAT.md`](docs/QSC_FORMAT.md)
* [`docs/REPO_INTEGRATION.md`](docs/REPO_INTEGRATION.md)
* [`docs/VALIDATION.md`](docs/VALIDATION.md)

---

## Contributing

QSA is an active research and engineering project.

Contributions are welcome in areas including:

* Quantum-state mathematics
* Sparse state structures
* Tensor-network representations
* Symbolic quantum arithmetic
* SIMD optimization
* GPU kernels
* Cross-platform builds
* Qubit-node transport
* QELM integration
* Testing and independent validation
* Documentation

Fork the repository, create a branch, and open a pull request with a clear description of the proposed change.

---

## License

Qubit State Algebra is released under the MIT License.

See [`LICENSE`](LICENSE).
