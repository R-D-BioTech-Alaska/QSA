# Performance Architecture

## QSA 0.1.5 symmetry-algebra results

QSA 0.1.5 expands the two-amplitude Grover representation into a general
`k`-class state engine. The measured benchmark evolves eight equal amplitude
classes over a 20-qubit logical space containing 1,048,576 basis states.

Median results from nine native runs with `QSTATE_NATIVE_ARCH=ON`:

| Workload | Time | Relative to dense reference |
|---|---:|---:|
| Dense class reference, 1,000 iterations | 262.147 ms | 1.00x |
| Symmetry class unitary, 1,000 literal operations | 0.6774 ms | **386.97x faster** |
| Symmetry class unitary, 1,000-iteration fast-forward | 0.002005 ms | **130,750.18x faster** |

The dense vector used 16,777,216 bytes. The symmetry state used 392 engine
bytes, a **42,799.02x memory reduction** for this workload.

Automatic tolerant discovery compressed a 20-qubit uniform `QRegister` into
one amplitude class in a median 63.807 ms. Discovery is a one-time dense
inspection; subsequent repeated class evolution repays that cost quickly.

A 60-qubit Hamming-weight state retained all `2^60` logical basis states using
61 classes and 1,592 engine bytes. One hundred thousand class phase/reflection
steps completed in a median 9.329 ms. A dense complex128 representation of the
same logical state would require 16 EiB.

These gains apply only while the declared equivalence classes remain valid.
Exact fallback to `QRegister` remains available when an operation distinguishes
members inside a class. Raw medians are stored in
`docs/PERFORMANCE_RESULTS_0_1_5.csv`.

The isolated symmetry engine does not replace or slow the existing Grover path.
A nine-run QSA 0.1.4 versus 0.1.5 regression comparison measured the exact dense
path 0.5% faster and the exact `QRegister` path 5.2% faster in 0.1.5. The
sub-microsecond compressed paths differed by less than 2%, which is within the
normal timing sensitivity of nanosecond-scale benchmarks.

## Improvements in 0.1.3

QSA 0.1.3 adds a circuit-execution layer above the accelerated 0.1.2 state
kernels. Existing scalar calls remain unchanged, while repeated and batched
work can be compiled into immutable native plans.

### Fused operation plans

Adjacent single-qubit operations on the same qubit are multiplied into one
2x2 operator and applied with one state traversal. Runs that reduce to identity
within QSA's numerical tolerance are removed. Consecutive diagonal operations
such as Z, S, T, and Rz can be combined across multiple qubits and applied to a
local sparse or dense patch in one pass.

The optimizer is optional. `OperationPlan(..., optimize=False)` retains one
native step per source operation while still removing Python/C call overhead.

### Parallel register ensembles

One immutable plan can execute concurrently across independent `QRegister`
instances. This does not introduce locks into the state engine and does not
parallelize operations that share a register. It targets QELM minibatches,
parameter-shift evaluations, independent noise trajectories, and distributed
Qubit-node jobs where register independence is explicit.

### Parameterized native plans

`ParameterizedPlan` keeps circuit structure and named parameter slots native.
Each execution binds a compact numeric buffer, compiles the bound operations
once, and can apply that bound plan across an entire register ensemble. This
avoids rebuilding Python tuples, ctypes structures, and native plan handles for
each parameter update.

### Bulk probability readout

`probabilities_one()` traverses each component directly and returns all
single-qubit populations through one C/Python call. Sparse patches accumulate
set bits from their support entries; dense patches use contiguous half-block
sums. Scalar `probability_one()` was also changed to read native sparse/dense
storage directly without constructing a temporary entry vector.

### Measured 0.1.3 results

These are median results from seven native runs and five Python runs in the same
release build with `QSTATE_NATIVE_ARCH=ON`. They describe the listed workloads,
not a universal multiplier.

| Workload | Prior path | 0.1.3 path | Speedup |
|---|---:|---:|---:|
| Fuse 20,002 adjacent single-qubit operations into 3 steps | 0.232 ms | 0.00148 ms | **156.85x** |
| Fuse 640 diagonal operations into one dense-patch pass | 38.146 ms | 1.221 ms | **31.23x** |
| Execute one plan across 256 registers using 5 workers | 54.199 ms | 18.129 ms | **2.99x** |
| Python: 50,000 scalar calls versus an optimized native plan | 31.742 ms | 0.051 ms | **629.61x** |
| Python: read 4,096 structured-qubit probabilities | 60.929 ms | 6.955 ms | **9.06x** |
| Python: rebuild versus bind a 300-step parameter sweep | 20.825 ms | 10.074 ms | **2.09x** |
| Python: execute a plan across 256 registers | 19.654 ms | 7.180 ms | **2.76x** |

Raw medians are stored in `docs/PERFORMANCE_RESULTS_0_1_3.csv`. The native
benchmark is `qstate_plan_benchmark`; Python bridge and parameter benchmarks are
in `benchmarks/python_bridge.py`.

## Improvements in 0.1.2

QSA 0.1.2 moves optimization into the native state engine while preserving the
0.1 C++, C, Python, and QSC v1 contracts established in 0.1.1.

### Stable component storage

Component merges no longer erase from the middle of the component vector and
reindex every qubit in the register. Components are removed with swap-pop, and
only the moved component's qubits are remapped. A compact logical-order table
preserves historical `describe()` and QSC v1 component order.

The order table uses 32-bit keys and renumbers only after an effectively
unreachable four-billion-append boundary. This limits the metadata cost while
removing register-width work from local merges and splits.

### Specialized gate kernels

X, Y, Z, S, S-dagger, T, T-dagger, Rz, CNOT, CZ, and SWAP now use direct sparse
or dense kernels. Permutation and diagonal gates no longer rebuild states
through hash tables or renormalize unchanged support.

Sparse H, Rx, Ry, and arbitrary 2x2 operations use ordered linear pair merging
instead of an unordered accumulation table. Dense rebalance first counts
support and allocates sparse output only when conversion will actually occur.

### Targeted separability recovery

A two-qubit unitary can change singleton separability only for the two qubits
it acts on. QSA now checks those two partitions rather than scanning every
qubit in the component. Measurement and nonunitary trajectories retain full
component compaction because they can separate many qubits at once.

Dense singleton checks operate directly on dense amplitude pairs without first
materializing a sparse entry map. Sparse singleton checks use ordered linear
pair construction.

### Release optimization

Release builds enable interprocedural optimization when the compiler supports
it. Portable builds remain the default. Local QELM or Qubit installations can
add CPU-specific code generation explicitly:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQSTATE_NATIVE_ARCH=ON
```

`QSTATE_NATIVE_ARCH` is intentionally disabled for portable wheels.

## Measured native hot-path results

The table below compares commit `555a7b7` (QSA 0.1.1) with QSA 0.1.2. Both were
compiled directly with GCC 14.2.0 using `-O3 -DNDEBUG -std=c++20`. Each result
is the median of seven runs in the same environment.

| Workload | 0.1.1 | 0.1.2 | Speedup |
|---|---:|---:|---:|
| Construct 1,000 independent Bell pairs | 10.767 ms | 0.370 ms | **29.11x** |
| 20,000 Rz gates on a 50-qubit sparse GHZ patch | 2.525 ms | 0.505 ms | **5.00x** |
| 2,000 CNOT gates on a 50-qubit sparse patch | 4.686 ms | 0.766 ms | **6.11x** |
| 2,000 Ry gates on a 2,048-support sparse patch | 214.372 ms | 23.558 ms | **9.10x** |
| 20 CNOT gates on a 65,536-amplitude dense patch | 430.869 ms | 4.118 ms | **104.64x** |
| 20 Ry gates on a 65,536-amplitude dense patch | 8.402 ms | 2.702 ms | **3.11x** |

These are workload-specific engine measurements, not a universal simulator
multiplier. The reproducible benchmark is `qstate_hot_paths`; the raw median
values are stored in `docs/PERFORMANCE_RESULTS_0_1_2.csv`.

The existing 0.1.1 compiled operation-plan path remains available. A separate
median seven-run Python bridge result completed in 0.328 ms versus 35.073 ms for
individual calls, a 106.87x reduction in language-boundary overhead. Native
kernel gains and bridge gains affect different layers and should not be
multiplied into one universal claim.

## Structured-state memory

The stable-order metadata adds four bytes per live component. Representative
0.1.2 measurements are:

```text
10,000 independent qubits    1.07 MiB
50-qubit exact GHZ            5.83 KiB
100 independent Bell pairs   26.69 KiB
```

The increase is bounded metadata for eliminating register-wide reindexing; the
state remains compact and exact.

## Next implementation layers

1. Parallel layer execution across already-independent components inside one register.
2. Zero-copy bulk amplitude and measurement buffers.
3. Runtime-selected SIMD kernels for dense local patches.
4. Optional CUDA execution for dense patches without making CUDA a core dependency.
5. Additional exact structured component types, beginning with stabilizer and symbolic-phase forms.

## QSA 0.1.4 Grover results

The Grover benchmark uses a complete optimal single-target search over 65,536
logical states (16 qubits, 201 iterations). Each path produces the same marked-
state probability.

Representative median results from the same machine:

| Path | Time | Relative to prior QSA 0.1.3 path |
|---|---:|---:|
| QSA 0.1.3 materialize + external dense Grover | 20.974 ms | 1.00x |
| QSA 0.1.4 exact `QRegister` Grover | 16.165 ms | 1.30x faster |
| QSA 0.1.4 symmetry-compressed `GroverSearch` | 0.0989 us | 212,074x faster |

The compressed result is not a universal circuit speedup. It applies to ideal
Grover evolution whose marked and unmarked states retain two exact amplitude
classes. The exact `QRegister` mode remains available when arbitrary gate-level
interoperability is required.

Memory for the measured 16-qubit search fell from 1,048,576 bytes for a dense
complex statevector to 96 engine bytes for one explicit marked index, a
10,922.7x reduction. Count-only 60-qubit search used 88 engine bytes versus a
theoretical 16 EiB dense complex128 vector. A full create, 843,314,856-iteration
fast-forward, and probability read averaged approximately 0.357 microseconds on
the benchmark host.

Raw results are stored in `docs/PERFORMANCE_RESULTS_0_1_4.csv`.
