# QSA Symmetry Algebra

QSA 0.1.5 generalizes the two-class Grover representation into an exact
amplitude-class engine. A `SymmetryState` stores one amplitude for every class
of basis states that are known to evolve identically.

For class `i`, QSA stores:

- its logical basis-state count `n_i`
- one shared complex amplitude `a_i`
- optional basis-membership information

Normalization is evaluated as:

```text
sum_i n_i |a_i|^2 = 1
```

The normalized class coefficient is `c_i = sqrt(n_i) a_i`. Arbitrary unitary
matrices can act directly on the compact class-coefficient vector. A logical
state containing `2^n` basis states therefore needs only `k` amplitudes when it
retains `k` equivalence classes.

## Membership modes

### Ordered ranges

Classes occupy consecutive basis-index ranges. This is compact and supports
amplitude queries, exact sampling, and fallback materialization without a
basis-label table.

```python
from qsa import SymmetryState

state = SymmetryState(20, [131072] * 8)
```

### Count-only symbolic

Only class sizes are retained. This mode supports class probabilities and all
class-space evolution over enormous logical spaces, but individual basis
membership is intentionally unavailable.

```python
state = SymmetryState.from_counts(60, [1, 7, (1 << 60) - 8])
```

### Explicit labels

A class label is supplied for each basis state. This supports arbitrary small
partitions and is used by automatic discovery when equivalent amplitudes are
not arranged in contiguous ranges.

```python
state = SymmetryState.from_labels(3, [0, 1, 1, 0, 2, 2, 2, 2])
```

### Hamming weight

Class `k` contains every basis state with exactly `k` set qubits. An `n`-qubit
permutation-invariant state therefore uses only `n + 1` amplitudes and `n + 1`
binomial class counts.

```python
state = SymmetryState.hamming_weight(60)
print(state.class_count)       # 61
print(state.class_size(30))    # C(60, 30)
```

This representation supports exact Dicke-state amplitudes, collective phase
operations, class-space unitaries, probability queries, and exact uniform
sampling inside a selected Hamming-weight class without storing `2^n` labels.

## Class-preserving operations

```python
state.phase(class_index, angle)
state.phases([angle_0, angle_1, ...])
state.reflect()
state.unitary(matrix)
state.iterate_unitary(matrix, count)
```

`reflect()` performs the weighted inversion-about-the-mean operation:

```text
a_i' = 2 * (sum_j n_j a_j / N) - a_i
```

`iterate_unitary()` exponentiates a class-space unitary by repeated squaring,
allowing very large repeated symmetric evolutions in logarithmic iteration
time.

## Partition refinement

A symbolic or explicitly known class can be split without materializing the
logical state. Both new classes initially retain the original amplitude.
Equivalent classes can later be merged again.

```python
new_class = state.split_class(class_index=0, first_count=10)
state.phase(new_class, 0.5)
removed = state.merge_equivalent()
```

Hamming-weight classes cannot be split or merged while retaining their symbolic
membership rule. They can still be materialized to a normal `QubitRegister`
when the requested size is safe.

## Automatic discovery

QSA can inspect a small existing `QubitRegister`, group equivalent amplitudes,
and return a symmetry state.

```python
symmetry = SymmetryState.from_register(
    register,
    max_qubits=24,
    tolerance=0.0,
)
```

A zero tolerance is bit-exact. A nonzero tolerance is explicit bounded
approximation. `symmetry.discovery_error` reports the maximum amplitude change
introduced by grouping and normalization.

Discovery is a one-time `O(2^n)` inspection. It is useful when the discovered
state will then undergo enough class-space evolution to repay that cost.

## Safe fallback

When a later operation no longer preserves the class partition, a state with
known basis membership can be converted exactly to the existing adaptive QSA
engine:

```python
register = symmetry.to_register(max_qubits=24)
```

This does not change QSC v1. The resulting `QubitRegister` uses the existing
Bloch, sparse, or dense component representation and can be serialized normally.
Count-only states cannot be materialized because basis membership was never
stored.

## Scientific boundary

The symmetry algebra is exact when all states in a class genuinely share one
amplitude. It does not compress arbitrary states with independently varying
amplitudes. Automatic tolerant discovery is explicitly approximate and reports
its measured amplitude error.

The largest gains occur in permutation-invariant systems, repeated amplitude
classes, symmetric search and walk processes, collective models, and QELM
candidate spaces where many logical states remain equivalent under the current
operations.
