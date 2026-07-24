# QSA 0.1.4 Validation Report

QSA 0.1.4 adds Grover execution without changing the frozen QSA 0.1 C++, C,
Python, or QSC v1 compatibility contracts.

## Release suite

The optimized Release build passes eight bounded gates:

1. Native state-engine tests
2. Stable C ABI tests
3. Native operation-plan tests
4. Native Grover tests
5. Canonical `qsa` Python tests
6. Legacy `qubit_native` Python tests
7. Machine-readable QSA 0.1 contract-manifest validation
8. Independent randomized Grover differential validation

```text
100% tests passed, 0 tests failed out of 8
```

The portable Release build passes the same eight gates.

## Sanitizer validation

The native state, C ABI, operation-plan, and Grover test executables pass with
AddressSanitizer and UndefinedBehaviorSanitizer enabled, including leak
checking.

```text
100% tests passed, 0 tests failed out of 4
ASan/UBSan errors: 0
```

## General quantum-state differential validation

The established independent NumPy suite passed:

```text
120 circuits x 120 random gates = 14,400 randomized gate operations
Differential validation passed: 120 circuits x 120 random gates.
```

## Grover differential validation

The compressed Grover engine was compared against an independently evolved
dense NumPy amplitude vector across random search widths, marked sets, and
iteration counts. Small cases also compared the full `QRegister` Grover path.

```text
200 randomized searches
41,500 amplitude checks
Maximum accepted error: 2 x 10^-10
Grover differential validation passed.
```

Native tests additionally cover:

- exact single- and multiple-target search;
- optimal-iteration selection;
- 10,000-iteration fast-forward versus repeated oracle/diffusion operations;
- 60-qubit count-only search in constant memory;
- marked and unmarked basis sampling;
- exact `QRegister` Grover interoperability;
- dense Grover QSC v1 round trips.

## Persistence compatibility

The QSC v1 encoder and frozen fixture bytes remain unchanged. Exact dense
Grover states serialize through the existing dense-component record.

QSA 0.1.4 fixes an existing decoder defect caused by unspecified C++ argument
evaluation order when reading consecutive floating-point values. Bloch
coordinates and complex real/imaginary values are now read into named
temporaries in serialized order before object construction. This corrects
complex QSC restoration without changing the QSC v1 format.

## Older-library compatibility

The QSA 0.1.4 Python wrapper was loaded successfully against independently
built QSA 0.1.3 and QSA 0.1.0 native libraries. The original API remains
functional. Grover construction against an older library fails cleanly with an
ABI 1.4 requirement rather than affecting existing calls.

## Performance comparison

The measured complete 16-qubit, one-target optimal search used 201 Grover
iterations:

```text
Original QSA 0.1.0-era dense path: 20.682 ms
Prior QSA 0.1.3 dense path:        20.974 ms
QSA 0.1.4 exact QRegister path:    16.165 ms
QSA 0.1.4 compressed path:          0.0989 us
```

The compressed path is approximately 209,123x faster than the original
QSA-era dense path and 212,074x faster than the immediate prior path on this
specific symmetry-preserving workload. It is not a universal multiplier for
arbitrary circuits.
