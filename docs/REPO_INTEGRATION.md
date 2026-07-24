# Adding the engine to the existing Qubit repository

The safest first integration keeps the current `Cubit.py`, `Qubit.py`, Discord badge, README history, and existing examples intact.

Recommended repository layout:

```text
Qubit/
├── Cubit.py
├── Qubit.py
├── QCopy.py
├── QubitStudio.py
├── README.md
├── images/
│   └── Qubit.png
└── native/
    ├── CMakeLists.txt
    ├── include/
    ├── src/
    ├── python/
    ├── tests/
    ├── examples/
    ├── benchmarks/
    └── docs/
```

Copy this project into `native/`.

Build from the repository root:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --parallel
ctest --test-dir native/build --output-on-failure
```

Use the Python bridge:

```python
import sys
sys.path.insert(0, "native/python")

from qsa import QubitRegister

state = QubitRegister(8)
state.h(0).cnot(0, 1)
```

Or set:

```bash
export PYTHONPATH="$PWD/native/python"
export QSA_NATIVE_LIB="$PWD/native/build/libqstate.so"
```

## Migration strategy

1. Keep `Cubit.py` as the current reference and compatibility implementation.
2. Add native-engine differential tests that execute the same small circuits in both engines.
3. Add a `NativeCubit` adapter that presents selected current method names over `QubitRegister`.
4. Route new QELM channel experiments to the native engine behind a feature flag.
5. Replace individual NumPy paths only after their outputs pass differential validation.
6. Preserve the old code until noise, subspace, qudit, and QELM behavior are explicitly covered.

This avoids rewriting the repository around an unvalidated core while still making the native algebra the forward architecture.


## Compatibility aliases

Existing projects do not need an immediate import migration. The installed package also provides:

```python
from qubit_native import QubitRegister
```

`QUBIT_NATIVE_LIB` remains accepted as a legacy alias for `QSA_NATIVE_LIB`. Both property-style and historical method-style queries remain valid, including `state.qubit_count` and `state.qubit_count()`.


## Symmetry integration for QELM and Qubit

QSA 0.1.5 adds a separate `SymmetryState` path for candidate spaces or quantum
channels whose basis states remain equivalent under the current operation:

```python
from qsa import SymmetryState

# Permutation-invariant 60-qubit channel: only 61 amplitudes.
state = SymmetryState.hamming_weight(60)
state.phases([weight * 0.01 for weight in range(61)])
state.reflect()
```

Use `SymmetryState.from_register(register, tolerance=0.0)` only as an explicit
compression checkpoint. Exact discovery may retain more classes when floating-
point paths differ by a few bits; a nonzero tolerance is opt-in and reports its
maximum amplitude change through `discovery_error`.

When a QELM or Qubit operation must distinguish members inside one class,
convert with `state.to_register(max_qubits=...)`. The conversion is exact for
ordered, explicit-label, and Hamming-weight states and leaves the symmetry
handle unchanged. Count-only states cannot be converted because they never
stored basis membership.
