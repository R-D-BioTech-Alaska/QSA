# Contributing to QSA

QSA is a compatibility-sensitive research runtime. Contributions are welcome,
but changes must preserve the existing mathematical and integration contracts.

## Development build

Requirements:

- CMake 3.20 or newer
- A C++20 compiler
- Python 3.9 or newer
- NumPy only for independent differential validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run the independent differential suites after building:

```bash
export PYTHONPATH="$PWD/python"
export QSA_NATIVE_LIB="$PWD/build/libqstate.so"  # Linux
python tests/differential_numpy.py
python tests/differential_grover.py
python tests/differential_symmetry.py
```

Use the corresponding `.dylib` or `.dll` path on macOS or Windows.

## Compatibility rules

The following are persistent contracts and must not be changed destructively:

- Existing public C++ names and gate semantics
- Existing exported C ABI declarations
- `qsa` and `qubit_native` Python imports
- Qubit ordering and basis-index conventions
- QSC v1 bytes, field order, and decoder compatibility
- Existing method/property compatibility forms

New features should be additive. If an internal representation changes, frozen
QSC fixtures and downstream compatibility tests must continue to pass.

## Numerical changes

A numerical change should include:

1. A focused deterministic regression test.
2. Independent differential validation where practical.
3. A stated tolerance and its reason.
4. A benchmark if performance is part of the claim.
5. Confirmation that structured states do not materialize globally unless the
   operation mathematically requires it.

## Performance changes

Report medians from repeated runs and include the exact benchmark source. Do not
multiply unrelated speedups or present structure-specific acceleration as a
universal simulator multiplier.

## Pull requests

Keep each pull request focused. Include:

- What changed
- Why it is safe
- Which compatibility surfaces were exercised
- Test commands and results
- Before/after benchmark data for performance work

Do not commit build directories, compiled libraries, generated wheels, cache
files, or local QSC packets.
