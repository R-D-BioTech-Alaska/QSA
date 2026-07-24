# QSA Grover Execution

QSA 0.1.4 provides two deliberately separate Grover paths so Grover support is
useful without forcing every existing state or application into dense storage.

## Symmetry-compressed search

`qubit::GroverSearch` and Python `qsa.GroverSearch` represent ideal uniform-start
Grover evolution with two amplitude classes:

- one amplitude shared by all marked basis states;
- one amplitude shared by all unmarked basis states.

The representation is exact for a fixed marked set under the Grover phase
oracle, inversion about the mean, and complete Grover iterations. A search over
`2^60` logical states therefore remains under 100 engine bytes in count-only
mode instead of requiring a 16 EiB complex128 statevector.

A complete Grover iteration acts as a two-dimensional rotation. QSA raises that
rotation to an arbitrary iteration count by exponentiation by squaring, so
fast-forward cost is logarithmic in the number of requested iterations rather
than proportional to both iteration count and search-space size.

Explicit-index mode supports amplitude queries and concrete basis sampling.
Count-only mode supports probability evolution when only the number of marked
states is known.

## Exact QRegister execution

`QRegister::apply_grover_oracle`, `apply_grover_diffusion`, and
`apply_grover_iterations` apply Grover operations to the actual full register.
This mode interoperates with all other QSA gates and QSC v1 serialization.
Because a global basis-state oracle generally couples the whole register, this
path is explicitly allowed to promote the register to one dense component.
Configured dense-state and component-width limits remain enforced.

The Python equivalents are:

```python
state.grover_oracle([marked_index])
state.grover_diffusion()
state.grover_iterations([marked_index], count)
```

## Compatibility boundary

Grover support is additive. Existing gates, state layouts, C symbols, Python
methods, QSC v1 packets, component behavior, and default execution paths do not
invoke Grover code. The compressed representation has its own handle and is not
inserted into QSC v1, avoiding a serialization-format change. Exact QRegister
Grover states use the already-supported dense QSC v1 component record.

## Scientific boundary

The compressed engine does not make an arbitrary oracle or arbitrary quantum
state constant-size. It exploits the exact amplitude symmetry of standard
Grover evolution. Operations that distinguish individual marked states, assign
different phases, or otherwise break the two-class symmetry require the exact
QRegister path or another representation.

Classically simulating Grover does not create physical quantum query advantage.
The feature is useful for exact algorithm development, large logical-space
probability studies, QELM candidate-amplification experiments, oracle testing,
and future circuit/backend interoperability.
