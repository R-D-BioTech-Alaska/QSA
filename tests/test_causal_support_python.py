from __future__ import annotations

from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_batch import fork_many, observe_many
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999


def tripair_support():
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


def tripair_words():
    words = []
    for observable in tripair_support():
        word = ["I"] * QUBITS
        for qubit, axis in observable:
            word[qubit] = axis
        words.append("".join(word))
    return tuple(words)


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
    full = CausalPauliPlan(QUBITS, tripair_words())
    compact = CausalPauliSupportPlan(QUBITS, tripair_support())
    identity = CausalPauliSupportPlan(QUBITS, ((),))

    assert compact.observable_count == 14
    assert compact.term_count == 21

    with CausalRegister(QUBITS) as root:
        root.apply(prepare, [])
        assert root.component_count == 9_998

        full_values = full.execute(root)
        compact_values = compact.execute(root)
        assert maximum_error((full_values,), (compact_values,)) <= 2.0e-12
        assert abs(identity.execute(root)[0] - 1.0) <= 2.0e-12

        branches = fork_many(root, 64)
        try:
            full_batch = observe_many(full, branches, workers=4)
            compact_batch = compact.execute_many(branches, workers=4)
            assert maximum_error(full_batch, compact_batch) <= 2.0e-12
            assert all(branch.validate() for branch in branches)
        finally:
            for branch in branches:
                branch.close()

    try:
        CausalPauliSupportPlan(2, (((0, "X"), (0, "Z")),))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate compact Pauli support was accepted")

    identity.close()
    compact.close()
    full.close()
    prepare.close()
    print("QSA compact Pauli Python tests passed.")


if __name__ == "__main__":
    main()
