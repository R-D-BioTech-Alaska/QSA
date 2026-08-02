from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_component_grad import CausalComponentParameterShift
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999
EXTRA = 611


def circuit_plan():
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


def observable_plan():
    return CausalPauliSupportPlan(
        QUBITS,
        (
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
        ),
    )


def maximum_matrix_error(first, second):
    return max(
        abs(left - right)
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    )


def main() -> None:
    plan = circuit_plan()
    support = observable_plan()
    prepare = CausalParameterizedPlan(
        (("h", EXTRA), ("cnot", EXTRA, FIRST))
    )
    full_gradient = CausalSupportParameterShift(plan, support)
    component_gradient = CausalComponentParameterShift(
        plan,
        support,
        max_local_qubits=8,
    )
    values = tuple(
        0.21 * math.sin((index + 1) * 0.29)
        for index in range(plan.parameter_count)
    )

    with CausalRegister(QUBITS) as root:
        root.apply(prepare, ())
        initial = root.encode_qsc()
        full = full_gradient.evaluate_and_jacobian(root, values, workers=0)
        local = component_gradient.evaluate_and_jacobian(
            root,
            values,
            workers=0,
        )

        assert local.local_qubit_count == 4
        assert set(local.global_qubits) == {EXTRA, FIRST, SECOND, THIRD}
        assert max(
            abs(left - right)
            for left, right in zip(full.values, local.values)
        ) <= 2.0e-12
        assert maximum_matrix_error(full.jacobian, local.jacobian) <= 2.0e-12

        cotangent = tuple(
            0.13 * math.cos((index + 1) * 0.47)
            for index in range(14)
        )
        tangent = tuple(
            0.11 * math.sin((index + 1) * 0.31)
            for index in range(18)
        )
        assert max(
            abs(left - right)
            for left, right in zip(full.vjp(cotangent), local.vjp(cotangent))
        ) <= 2.0e-12
        assert max(
            abs(left - right)
            for left, right in zip(full.jvp(tangent), local.jvp(tangent))
        ) <= 2.0e-12
        assert root.encode_qsc() == initial
        assert root.validate()

    wide_qubits = tuple(range(100, 125))
    wide_plan = CausalParameterizedPlan(
        (("rz", wide_qubits[0], Parameter("theta")),)
    )
    wide_support = CausalPauliSupportPlan(
        QUBITS,
        (((wide_qubits[0], "Z"),),),
    )
    wide_prepare = CausalParameterizedPlan(
        (("h", wide_qubits[0]),)
        + tuple(
            ("cnot", wide_qubits[index - 1], wide_qubits[index])
            for index in range(1, len(wide_qubits))
        )
    )
    wide_gradient = CausalComponentParameterShift(
        wide_plan,
        wide_support,
        max_local_qubits=24,
    )
    with CausalRegister(QUBITS) as root:
        root.apply(wide_prepare, ())
        try:
            wide_gradient.evaluate_and_jacobian(
                root,
                {"theta": 0.1},
            )
        except CausalRuntimeError:
            pass
        else:
            raise AssertionError("oversized component gradient was accepted")

    wide_gradient.close()
    wide_prepare.close()
    wide_support.close()
    wide_plan.close()
    component_gradient.close()
    prepare.close()
    support.close()
    plan.close()
    print("QSA component-local gradient tests passed.")


if __name__ == "__main__":
    main()
