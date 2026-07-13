# Validation Report — Qubit Native State Engine v0.1.0

Validation performed on July 13, 2026 in the build environment supplied for this implementation.

## Toolchain

- GCC 14.2.0
- CMake 3.31.6
- Python 3.13.5
- NumPy 2.3.5 used only as an independent differential reference

## Native unit tests

The compiled C++ test suite validates:

- Independent geometric cells
- H and rotation probabilities
- Bell-state amplitudes
- Entangled measurement collapse
- Recovery of independent cells after a non-entangling CNOT
- Exact 40-qubit sparse GHZ state
- QSC encode/decode amplitude preservation
- QSC corruption rejection
- Amplitude-damping trajectory
- Phase-flip trajectory
- Internal state invariants

Result:

```text
100% tests passed, 0 tests failed out of 1
```

## Sanitizer build

The full native test suite was rebuilt and run with:

```text
-fsanitize=address,undefined -fno-omit-frame-pointer
```

Result:

```text
100% tests passed, 0 tests failed out of 1
```

No AddressSanitizer or UndefinedBehaviorSanitizer errors were reported.

## Differential random-circuit validation

`tests/differential_numpy.py` executes the engine and an independent NumPy statevector reference side by side.

Validated gates:

- X, Y, Z, H, S, T
- Rx, Ry, Rz
- CNOT, CZ, SWAP
- Measurement and collapse

Test volume:

```text
120 circuits × 120 random gates = 14,400 random gate operations
```

Intermediate and final states were compared by global-phase-invariant fidelity. Measurements used identical samples and post-measurement states were compared again.

Result:

```text
Differential validation passed: 120 circuits x 120 random gates.
```

## Structured-state benchmark

Observed release-build results:

```text
10,000 independent qubits    components=10000  engine-memory=1.03 MiB
50-qubit exact GHZ           components=1      engine-memory=5.60 KiB
100 independent Bell pairs   components=100    engine-memory=25.88 KiB
```

GHZ comparison:

```text
20 qubits  QSA 2.32 KiB   dense complex128 16.00 MiB
30 qubits  QSA 3.41 KiB   dense complex128 16.00 GiB
40 qubits  QSA 4.51 KiB   dense complex128 16.00 TiB
50 qubits  QSA 5.60 KiB   dense complex128 16.00 PiB
```

These results demonstrate the benefit for states whose mathematical structure is compact. They are not a universal simulator ranking. Random, highly entangled patches can still become exponentially large.
