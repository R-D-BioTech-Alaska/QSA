from __future__ import annotations

import math

from qsa import SymmetryState


with SymmetryState.hamming_weight(60) as state:
    # Apply a phase based only on Hamming weight.
    state.phases([-0.02 * weight for weight in range(61)])
    state.reflect()

    print(state.describe())
    print("P(weight=30):", state.class_probability(30))
    print("Engine bytes:", state.estimated_bytes)


# Eight equal classes over 2^20 logical basis states.
counts = [1 << 17] * 8
with SymmetryState(20, counts) as state:
    angle = 0.0037
    c = math.cos(angle)
    s = math.sin(angle)
    unitary = [[0j] * 8 for _ in range(8)]
    for index in range(8):
        unitary[index][index] = 1
    unitary[0][0] = c
    unitary[0][1] = -s
    unitary[1][0] = s
    unitary[1][1] = c

    state.iterate_unitary(unitary, 1_000_000)
    print(state.class_probability(0), state.class_probability(1))
