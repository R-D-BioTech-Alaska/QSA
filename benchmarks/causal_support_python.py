from __future__ import annotations

import statistics
import time

from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_batch import fork_many, observe_many
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
BRANCHES = 128
REPEATS = 9
FIRST = 17
SECOND = 4_321
THIRD = 9_999


def supports():
    return (
        ((FIRST, "X"),),
        ((FIRST, "Y"),),
        ((FIRST, "Z"),),
        ((SECOND, "X"),),
        ((SECOND, "Y"),),
        ((SECOND, "Z"),),
        ((THIRD, "X"),),
        ((THIRD, "Y"),),
        ((THIRD, "Z"),),
        ((FIRST, "Z"), (SECOND, "Z")),
        ((SECOND, "Z"), (THIRD, "Z")),
        ((FIRST, "Z"), (THIRD, "Z")),
        ((FIRST, "X"), (SECOND, "X"), (THIRD, "X")),
        ((FIRST, "Y"), (SECOND, "Y"), (THIRD, "Y")),
    )


def words():
    encoded = []
    for observable in supports():
        word = ["I"] * QUBITS
        for qubit, axis in observable:
            word[qubit] = axis
        encoded.append("".join(word))
    return tuple(encoded)


def preparation_plan():
    return CausalParameterizedPlan(
        [
            ("ry", FIRST, 0.37),
            ("rz", FIRST, -0.21),
            ("ry", SECOND, -0.44),
            ("rz", SECOND, 0.19),
            ("ry", THIRD, 0.28),
            ("rz", THIRD, 0.33),
            ("cnot", FIRST, SECOND),
            ("cnot", SECOND, THIRD),
            ("cnot", THIRD, FIRST),
            ("ry", FIRST, -0.17),
            ("rz", SECOND, 0.14),
            ("ry", THIRD, 0.23),
        ]
    )


def maximum_error(first, second):
    return max(
        abs(left - right)
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    )


def main() -> None:
    prepare = preparation_plan()
    full_words = words()
    full = CausalPauliPlan(QUBITS, full_words)
    compact = CausalPauliSupportPlan(QUBITS, supports())

    full_samples = []
    compact_samples = []
    maximum = 0.0

    with CausalRegister(QUBITS) as root:
        root.apply(prepare, [])
        branches = fork_many(root, BRANCHES)
        try:
            observe_many(full, branches, workers=0)
            compact.execute_many(branches, workers=0)

            for _ in range(REPEATS):
                started = time.perf_counter()
                full_values = observe_many(full, branches, workers=0)
                full_samples.append((time.perf_counter() - started) * 1000.0)

                started = time.perf_counter()
                compact_values = compact.execute_many(branches, workers=0)
                compact_samples.append((time.perf_counter() - started) * 1000.0)

                maximum = max(
                    maximum,
                    maximum_error(full_values, compact_values),
                )
        finally:
            for branch in branches:
                branch.close()

    full_ms = statistics.median(full_samples)
    compact_ms = statistics.median(compact_samples)
    full_word_bytes = sum(len(word) for word in full_words)
    print(
        "causal_support"
        f" qubits={QUBITS}"
        f" branches={BRANCHES}"
        " observables=14"
        f" support_terms={compact.term_count}"
        f" full_word_bytes={full_word_bytes}"
        f" full_batch_ms={full_ms:.12f}"
        f" support_batch_ms={compact_ms:.12f}"
        f" speedup={full_ms / compact_ms:.12f}"
        f" max_error={maximum:.12g}"
    )

    compact.close()
    full.close()
    prepare.close()


if __name__ == "__main__":
    main()
