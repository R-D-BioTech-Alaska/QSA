# Qubit Native State Engine

**A from-scratch, qubit-native mathematical runtime for ordinary computers.**

This is a working first-generation engine for the new Qubit direction. It does not use NumPy, Qiskit, `std::complex`, BLAS, or full-system Kronecker gate matrices. The numerical core is written in dependency-free C++20 and is exposed to Python through a small `ctypes` bridge.

The engine introduces two related pieces:

- **QSA — Qubit State Algebra:** the adaptive mathematical representation used in memory.
- **QSC — Qubit State Code:** a checksummed binary state format suitable for storage and future Qubit-node transport.

## What is different

A conventional dense simulator treats an `n`-qubit state as one array of `2^n` complex values. This engine starts from a different rule:

> Do not build a global statevector unless the state itself requires a global representation.

The register is maintained as an exact product of independent state components:

- Independent pure qubits are stored as **geometric Bloch cells** `(x, y, z)`.
- Interacting qubits are merged only into the **local entangled patch** that needs them.
- Patches automatically choose **sparse** or **dense** amplitude storage.
- Gates operate directly on affected amplitude pairs or quartets; no full gate matrix is embedded across the register.
- Measurement collapses a patch and separates the measured qubit again.
- Exact rank-one checks recover qubits that become separable.
- Noise is supported through pure-state **quantum trajectories**, avoiding a `4^n` density matrix.

This means 10,000 independent qubits remain 10,000 small geometric cells. One hundred separate Bell pairs remain 100 two-qubit patches. A 50-qubit GHZ state remains one sparse patch with two nonzero amplitudes.

## Current capabilities

- Custom `QComplex` scalar owned by Qubit
- Geometric pure-qubit cells
- Adaptive sparse/dense entangled patches
- Exact component merging and singleton factor recovery
- X, Y, Z, H, S, S†, T, T†
- Rx, Ry, Rz
- CNOT, CZ, SWAP
- Arbitrary 2×2 and 4×4 operators in the C++ API
- Measurement and collapse
- Bit-flip, phase-flip, depolarizing, and amplitude-damping trajectories
- Direct amplitude queries without materializing the global state
- Optional global materialization for small validation cases
- QSC binary serialization with checksum validation
- C API and NumPy-free Python wrapper
- Automated Bell, GHZ, noise, collapse, compaction, and round-trip tests

## Build

Requirements:

- CMake 3.20+
- A C++20 compiler
- Python 3.9+ only for the optional wrapper

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The build creates:

- `build/libqstate.so` on Linux
- `build/libqstate.dylib` on macOS
- `build/qstate.dll` on Windows
- `build/qstate_tests`
- `build/qstate_bell`
- `build/qstate_benchmark`

## C++ example

```cpp
#include "qubit/qstate.hpp"
#include <iostream>

int main() {
    qubit::QRegister state(50);

    state.apply_h(0);
    for (std::uint32_t target = 1; target < 50; ++target) {
        state.apply_cnot(0, target);
    }

    std::cout << state.describe();
    std::cout << "support: " << state.component_nonzero_count(0) << '\n';
}
```

## Python example

The Python layer controls the native engine and does not import NumPy.

```python
from qubit_native import QubitRegister

with QubitRegister(2) as state:
    state.h(0).cnot(0, 1)

    print(state.amplitude(0))  # |00>
    print(state.amplitude(3))  # |11>
    print(state.describe())

    packet = state.encode_qsc()
```

Run from this project:

```bash
PYTHONPATH=python python3 examples/bell.py
```

Set `QUBIT_NATIVE_LIB` when the shared library is stored somewhere else.

## QSC node-state packet

```python
packet = state.encode_qsc()
restored = QubitRegister.decode_qsc(packet)
```

QSC preserves:

- Register size
- Component partition
- Bloch-cell state
- Sparse or dense patch state
- Qubit membership and local ordering
- Numeric configuration
- Integrity checksum

The format is designed as the first transport representation for future leased Qubit channels.

## Validation benchmark

A typical release build in the included container produced:

```text
10,000 independent qubits    components=10000  engine-memory=1.03 MiB
50-qubit exact GHZ           components=1      engine-memory=5.60 KiB
100 independent Bell pairs   components=100    engine-memory=25.88 KiB
```

Run it yourself:

```bash
./build/qstate_benchmark
python3 benchmarks/compare_numpy.py
```

## Scientific boundary

This engine removes unnecessary exponential allocation for structured states. It does **not** claim that arbitrary quantum states can always be compressed. A sufficiently general, highly entangled state can still require exponentially many independent amplitudes.

Version 0.1 is an exact pure-state engine within floating-point tolerance. It is a research foundation, not yet proof that the engine is universally faster than every existing simulator. Performance claims should be made circuit-by-circuit using reproducible benchmarks.

## Next engineering stages

1. SIMD kernels for dense local patches
2. GPU patch backend
3. General multi-component factor detection
4. Stabilizer and symbolic-phase component modes
5. Matrix-product/tensor-linked component mode
6. Mixed-state ensembles and batched trajectories
7. QELM channel-operation compiler
8. Qubit node lease and network protocol
9. Mobile builds for Android and iOS
10. Differential validation against Qiskit Aer and other independent simulators

See [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md), [`docs/QSC_FORMAT.md`](docs/QSC_FORMAT.md), and [`docs/REPO_INTEGRATION.md`](docs/REPO_INTEGRATION.md).
