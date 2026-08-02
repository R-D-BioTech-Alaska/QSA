# QSA causal state runtime

The causal runtime is the first QSA 0.2 execution surface for persistent Brain, QELM, and counterfactual state evolution.

## Why it exists

A persistent autoregressive quantum state should not be serialized to QSC and decoded again merely to explore each bounded candidate continuation. QSC remains the durable, checksummed storage and transport format. In-memory reasoning branches need a different contract.

`CausalState` provides that contract:

```cpp
CausalState response(QRegister(3));
auto candidates = response.fork_many(top_k);

candidates[0].mutate([](QRegister& state) {
    state.apply_ry(0, 0.25);
});

response.adopt(std::move(candidates[0]));
```

Fresh branches share one immutable `QRegister` in constant time. The first mutation detaches that branch through an exact native register copy. Adopting the selected branch transfers ownership without QSC encoding, parsing, or amplitude reconstruction.

This is the state-level copy-on-write foundation. A later QSA 0.2 phase will move the same ownership model into component pages so a mutation copies only affected components rather than the complete register.

## Component-wise observables

`PauliObservablePlan` evaluates exact Pauli words directly over QSA component read views.

```cpp
const std::vector<std::string> tripair{
    "XII", "YII", "ZII",
    "IXI", "IYI", "IZI",
    "IIX", "IIY", "IIZ",
    "ZZI", "IZZ", "ZIZ",
    "XXX", "YYY",
};

PauliObservablePlan closure(3, tripair);
auto values = closure.execute(response);
```

The plan follows QSA structure:

- Bloch cells return X, Y, and Z coordinates directly.
- Sparse components iterate only active support.
- Dense components evaluate only their local amplitude array.
- Independent component expectations are multiplied exactly.
- No global statevector is created solely to read observables.

A 10,000-qubit product state can therefore evaluate a bounded Pauli word while remaining 10,000 independent components. A conventional global statevector for the same logical width is not materializable.

## Exactness and boundaries

The implementation is exact for pure QSA states. It does not prune amplitudes, approximate phases, truncate entanglement, or change component topology.

The first test gate covers:

- constant-time shared forks;
- copy-on-write branch isolation;
- constant-time selected-branch adoption;
- Bell and GHZ Pauli identities;
- a 10,000-qubit factorized observable without component merging;
- all fourteen accepted Brain Tripair observables against a dense reference.

The benchmark target compares the previous in-memory QSC decode route with causal forks and reports the exact component-wise observable latency.

```bash
cmake --build build --target qstate_causal_runtime_benchmark
./build/qstate_causal_runtime_benchmark
```

This phase is infrastructure evidence. It does not by itself prove language improvement or universal quantum advantage. Brain capability still requires matched classical, phase-ablation, independent-seed, sealed-holdout, retention, and rollback gates.
