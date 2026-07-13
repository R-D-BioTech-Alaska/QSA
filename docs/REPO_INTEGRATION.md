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

from qubit_native import QubitRegister

state = QubitRegister(8)
state.h(0).cnot(0, 1)
```

Or set:

```bash
export PYTHONPATH="$PWD/native/python"
export QUBIT_NATIVE_LIB="$PWD/native/build/libqstate.so"
```

## Migration strategy

1. Keep `Cubit.py` as the current reference and compatibility implementation.
2. Add native-engine differential tests that execute the same small circuits in both engines.
3. Add a `NativeCubit` adapter that presents selected current method names over `QubitRegister`.
4. Route new QELM channel experiments to the native engine behind a feature flag.
5. Replace individual NumPy paths only after their outputs pass differential validation.
6. Preserve the old code until noise, subspace, qudit, and QELM behavior are explicitly covered.

This avoids rewriting the repository around an unvalidated core while still making the native algebra the forward architecture.
