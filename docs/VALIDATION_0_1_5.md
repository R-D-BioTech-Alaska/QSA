# QSA 0.1.5 Validation

QSA 0.1.5 retains the complete QSA 0.1 compatibility suite and adds symmetry-
algebra validation.

## Native and bridge gates

- Existing QRegister native tests
- Frozen C ABI tests plus ABI 1.5 symmetry calls
- Operation-plan and parameterized-plan tests
- Grover compressed and exact tests
- Symmetry class, unitary, refinement, discovery, Hamming-weight, and fallback tests
- Canonical `qsa` Python bridge tests
- Historical `qubit_native` Python bridge tests
- Frozen QSA 0.1 contract manifest
- Frozen QSC v1 byte fixtures

## Differential symmetry validation

`tests/differential_symmetry.py` compares the native amplitude-class engine with
an independent dense Python reference across:

- 250 randomized class partitions
- 20,000 randomized class operations
- phase operations
- weighted reflections
- embedded class-space rotations
- partition refinement
- exact fallback into QRegister
- basis-amplitude comparisons

The release run completed 20,000 operations and 39,272 amplitude
comparisons without exceeding a `2e-10` tolerance.

## Compatibility

The symmetry algebra has its own additive C handle and Python class. Existing
QRegister layouts, C symbols, operation plans, Grover handles, Python methods,
QSC v1 bytes, and default execution paths are unchanged. Symmetry materializes
through the existing QRegister dense/sparse representation only when explicitly
requested.
