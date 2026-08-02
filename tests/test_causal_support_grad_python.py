from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_grad import CausalParameterShift
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999


def plan():
    operations = []
    for qubit, name in (
        (FIRST, "first"),
        (SECOND, "second"),
        (THIRD, "third"),
    ):
        operations.append(("ry", qubit, Parameter(f"semantic_{name}_theta")))
        operations.append(("rz", qubit, Parameter(f"semantic_{name}_phi")))
    operations.extend(
        (
            ("cnot", FIRST, SECOND),
            ("cnot", SECOND, THIRD),
            ("cnot", THIRD, FIRST),
        )
    )
    for layer in range(2):
        for qubit, name in (
            (FIRST, "first"),
            (SECOND, "second"),
            (THIRD, "third"),
        ):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_{name}_first"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_{name}_second"))
            )
        operations.extend(
            (("cnot", FIRST, SECOND), ("cnot", SECOND, THIRD))
        )
    assert len(operations) == 25
    return CausalParameterizedPlan(operations)


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


def max_matrix_error(first, second):
    return max(
        abs(left - right)
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    )


def main() -> None:
    circuit = plan()
    full_observables = CausalPauliPlan(QUBITS, words())
    support_observables = CausalPauliSupportPlan(QUBITS, supports())
    full_gradient = CausalParameterShift(circuit, full_observables)
    support_gradient = CausalSupportParameterShift(circuit, support_observables)
    values = tuple(
        0.21 * math.sin((index + 1) * 0.29)
        for index in range(circuit.parameter_count)
    )

    with CausalRegister(QUBITS) as root:
        initial = root.encode_qsc()
        full = full_gradient.evaluate_and_jacobian(root, values, workers=0)
        compact = support_gradient.evaluate_and_jacobian(root, values, workers=0)

        assert len(compact.values) == 14
        assert len(compact.jacobian) == 14
        assert all(len(row) == 18 for row in compact.jacobian)
        assert max(abs(a - b) for a, b in zip(full.values, compact.values)) <= 2.0e-12
        assert max_matrix_error(full.jacobian, compact.jacobian) <= 2.0e-12

        cotangent = tuple(
            0.13 * math.cos((index + 1) * 0.47)
            for index in range(14)
        )
        tangent = tuple(
            0.11 * math.sin((index + 1) * 0.31)
            for index in range(18)
        )
        assert max(
            abs(a - b)
            for a, b in zip(full.vjp(cotangent), compact.vjp(cotangent))
        ) <= 2.0e-12
        assert max(
            abs(a - b)
            for a, b in zip(full.jvp(tangent), compact.jvp(tangent))
        ) <= 2.0e-12
        assert root.encode_qsc() == initial
        assert root.validate()

    support_observables.close()
    full_observables.close()
    circuit.close()
    print("QSA compact support gradient tests passed.")


if __name__ == "__main__":
    main()
