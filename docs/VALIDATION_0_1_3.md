# QSA 0.1.3 Validation Report

QSA 0.1.3 was validated as an additive execution-layer release over the
0.1.2 accelerated state engine. The release preserves the QSA 0.1 C++, C,
Python, and QSC v1 compatibility contracts while adding optimized fixed plans,
parameterized plans, parallel register ensembles, diagonal layer fusion, and
bulk probability readout.

## Validation environment

- CMake 3.31.6
- GCC 14.2.0
- Python 3.13.5
- Linux x86-64
- Release validation performed in both portable and `QSTATE_NATIVE_ARCH=ON`
  configurations
- Native sanitizer validation performed with AddressSanitizer and
  UndefinedBehaviorSanitizer

## Release acceptance gates

### Native-architecture release build

The release build with `QSTATE_NATIVE_ARCH=ON` passed all six registered tests:

1. Core state-engine tests
2. Stable C ABI tests
3. Fixed and parameterized plan tests
4. Canonical `qsa` Python compatibility tests
5. Legacy `qubit_native` Python compatibility tests
6. Frozen compatibility-manifest and QSC fixture tests

### Portable release build

A separate portable Release build, without CPU-specific code generation, passed
the same six tests. This is the configuration intended for redistributable
packages and portable wheels.

### Sanitizer build

The AddressSanitizer and UndefinedBehaviorSanitizer build passed all three
native test executables:

- Core state-engine tests
- C ABI tests
- Plan and ensemble tests

No sanitizer failure was reported.

### Differential numerical validation

The independent NumPy reference comparison passed 120 circuits containing 120
random gates each, for 14,400 randomized gate operations at the established
QSA fidelity tolerance.

The plan optimizer also passed 250 randomized literal-versus-optimized trials,
with 80 operations per trial on four-qubit registers. This covers adjacent
matrix fusion, diagonal layer fusion, identity removal, and mixed one- and
two-qubit operation sequences.

### Serialization compatibility

Both frozen QSC v1 packets decode and re-encode byte-for-byte:

- `tests/fixtures/qsc_v1_sparse_exact.bin`
- `tests/fixtures/qsc_v1_component_order_exact.bin`

The 0.1.3 engine also reconstructs the multi-component ordering fixture from
its original operation sequence. QSC v1 magic, field ordering, numerical data,
component ordering, payload layouts, and FNV-1a checksum remain unchanged.

### Backward native-library compatibility

The QSA 0.1.3 Python wrapper was loaded against independently rebuilt native
libraries from QSA 0.1.2 and QSA 0.1.0.

- Original APIs continued to work.
- Fixed plans fell back to the historical operation path where native plan
  symbols were absent.
- Parameterized plans fell back to concrete historical operations.
- Both canonical and legacy package interfaces remained usable.

This confirms that the additive Python layer does not require a 0.1.3 native
library for the original QSA feature set.

### Installed-layout validation

The project was installed into a clean prefix, the Python packages were placed
in the installed layout, native-library override variables were removed, and
both canonical and legacy imports executed successfully. Parameterized parallel
plan execution also passed in that installed layout.

## Measured performance

Median results are recorded in `docs/PERFORMANCE_RESULTS_0_1_3.csv`.

| Workload | Literal or serial | Optimized or parallel | Speedup |
|---|---:|---:|---:|
| 20,002 adjacent single-qubit source operations compiled to 3 steps | 0.231510 ms | 0.001476 ms | 156.85x |
| 640 diagonal source operations compiled to one patch pass | 38.146483 ms | 1.221429 ms | 31.23x |
| One native plan across 256 independent registers with 5 workers | 54.198906 ms | 18.128704 ms | 2.99x |
| Python 50,000 scalar calls versus optimized native plan | 31.742000 ms | 0.051000 ms | 629.61x |
| Python all-qubit readout for 4,096 independent qubits | 60.929000 ms | 6.955000 ms | 9.06x |
| Python 300-run parameter sweep | 20.825000 ms | 10.074000 ms | 2.09x |
| Python plan across 256 independent registers | 19.654000 ms | 7.180000 ms | 2.76x |

These are workload-specific improvements. The 629.61x Python result combines
language-boundary elimination with native plan optimization and identity/fusion
reduction; it is not a universal state-engine multiplier. Parallel ensemble
results depend on register independence, available cores, plan size, and memory
bandwidth.

## Packaging boundary

Source builds, installed-layout execution, portable builds, native-architecture
builds, and sanitizer builds passed. A Python wheel was not produced in the
validation environment because `scikit-build-core>=0.10` was unavailable from
the isolated package index. The failure occurred before project compilation and
does not represent a QSA source or build failure.

## Result

QSA 0.1.3 satisfies the bounded release gates for forward implementation:
existing interfaces and QSC v1 remain compatible, numerical equivalence is
preserved, older native libraries remain usable through the current Python
wrapper, and the new execution paths provide measured acceleration for repeated
circuits, diagonal layers, parameter sweeps, structured readout, and independent
register ensembles.
