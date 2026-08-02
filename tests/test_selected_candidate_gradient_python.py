from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_component_candidates import CausalComponentCandidateRuntime
from qsa.causal_component_grad import CausalComponentParameterShift
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999
EXTRA = 611


def candidate_plan():
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


def candidate_rows(plan, count, step):
    return tuple(
        {
            name: 0.19
            * math.sin((step + 1) * (candidate + 1) * (slot + 1) * 0.017)
            for slot, name in enumerate(plan.parameter_names)
        }
        for candidate in range(count)
    )


def maximum_matrix_error(first, second):
    return max(
        abs(left - right)
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    )


def main() -> None:
    plan = candidate_plan()
    support = observable_plan()
    prepare = CausalParameterizedPlan(
        (("h", EXTRA), ("cnot", EXTRA, FIRST))
    )

    with CausalRegister(QUBITS) as initial:
        initial.apply(prepare, ())
        packet = initial.encode_qsc()

    standalone = CausalComponentParameterShift(
        plan,
        support,
        max_local_qubits=8,
    )
    with CausalComponentCandidateRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
        max_local_qubits=8,
    ) as runtime:
        for step in range(3):
            rows = candidate_rows(plan, 8, step)
            batch = runtime.evaluate(rows)
            selected = (step * 3 + 2) % len(rows)

            reused = runtime.selected_gradient(batch, selected, workers=4)
            independent = standalone.evaluate_and_jacobian(
                runtime.state,
                rows[selected],
                workers=4,
            )

            assert reused.local_qubit_count == 4
            assert set(reused.global_qubits) == {EXTRA, FIRST, SECOND, THIRD}
            assert max(
                abs(left - right)
                for left, right in zip(
                    reused.values,
                    batch.observations[selected],
                )
            ) <= 2.0e-12
            assert max(
                abs(left - right)
                for left, right in zip(reused.values, independent.values)
            ) <= 2.0e-12
            assert maximum_matrix_error(
                reused.jacobian,
                independent.jacobian,
            ) <= 2.0e-12

            cotangent = tuple(
                0.13 * math.cos((index + 1) * (step + 1) * 0.31)
                for index in range(14)
            )
            assert max(
                abs(left - right)
                for left, right in zip(
                    runtime.selected_vjp(
                        batch,
                        selected,
                        cotangent,
                        workers=4,
                    ),
                    independent.vjp(cotangent),
                )
            ) <= 2.0e-12

            runtime.commit(batch, selected)
            assert runtime.generation == step + 1
            assert runtime.state.validate()
            try:
                runtime.selected_gradient(batch, selected)
            except CausalRuntimeError:
                pass
            else:
                raise AssertionError("closed candidate batch accepted a gradient request")

    standalone.close()
    prepare.close()
    support.close()
    plan.close()
    print("QSA selected candidate gradient tests passed.")


if __name__ == "__main__":
    main()
