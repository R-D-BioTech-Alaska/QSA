# Read-only component views and output buffers

QSA provides additive C++ APIs for reading native component storage without copying amplitude arrays and for reusing caller-owned output buffers.

## Component read views

`component_read_view(qubit)` returns a read-only view of the component containing a selected qubit. `component_read_views()` returns one view per component in QSA's logical component order.

A view reports its component kind, qubit membership, dimension, and exactly one of:

- a `BlochCell` pointer;
- a dense amplitude span;
- a sparse amplitude-entry span.

The storage remains owned by `QRegister`. Every view is invalidated when the register is mutated or destroyed. The API does not permit state mutation through a view.

## Reusable outputs

- `probabilities_one_into(output)` writes all populations into a caller-owned span.
- `measure_all_into(seed, output)` preserves the existing sequential measurement and seed mapping.

The original allocating methods and the existing C ABI retain their historical behavior. The new methods are additive C++ paths for callers that want reusable storage.

The reproducible benchmark target is `qstate_read_views_benchmark`.
