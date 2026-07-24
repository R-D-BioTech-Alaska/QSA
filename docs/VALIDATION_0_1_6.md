# QSA 0.1.6 Validation

QSA 0.1.6 is a release-hardening update. It does not change state evolution,
component mathematics, QSC v1 serialization, Grover behavior, or symmetry
algebra behavior.

## Validation environment

- Linux
- GCC 14.2.0
- CMake 3.31.6
- Python 3.13.5
- NumPy 2.3.5 used only as an independent differential reference

## Release gates

- Portable optimized Release suite: 12/12 passed
- ASan/UBSan native suite: 6/6 passed
- CMake install and independent C/C++ consumer build: passed
- Lean wheel-equivalent native install: only the shared runtime was installed
- Canonical and legacy Python environment tests: passed
- Version synchronization gate: passed
- Compatibility manifest and frozen QSC v1 fixtures: passed
- Adversarial QSC truncation, corruption, version, limit, kind, and trailing-data tests: passed
- Current Python wrapper with QSA 0.1.0, 0.1.3, 0.1.4, and 0.1.5 native libraries: passed

## Differential validation

- General randomized circuits: 120 circuits × 120 gates = 14,400 operations
- Grover validation: 200 randomized searches and 41,500 amplitude checks
- Symmetry validation: 20,000 operations and 39,272 amplitude checks

All differential suites passed at the established tolerances.

## Packaging boundary

Python package builds explicitly disable native tests, examples, benchmarks,
headers, static archives, and CMake development metadata. Source CMake builds
retain their historical defaults. Native C and C++ consumers can install and
use `QSA::qstate` and `QSA::qstate_core` through `find_package(QSA CONFIG)`.

The local runner did not have `scikit-build-core`, `build`, or `wheel` installed,
so it could not produce a wheel locally. GitHub Actions now builds an sdist and
wheel, checks their metadata, installs the wheel, and runs a smoke test.
