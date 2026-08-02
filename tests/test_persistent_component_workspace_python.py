from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_component_candidates import CausalComponentCandidateRuntime
from qsa.causal_persistent_workspace import CausalPersistentComponentRuntime
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


def rows(plan, count, step):
    return tuple(
        {
            name: 0.19
            * math.sin((step + 1) * (candidate + 1) * (slot + 1) * 0.017)
            for slot, name in enumerate(plan.parameter_names)
        }
        for candidate in range(count)
    )


def matrix_error(first, second):
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

    with CausalComponentCandidateRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
        max_local_qubits=8,
    ) as control, CausalPersistentComponentRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
        max_local_qubits=8,
    ) as persistent:
        assert not persistent.workspace_ready
        assert persistent.workspace_extractions == 0

        for step in range(3):
            candidate_rows = rows(plan, 8, step)
            control_batch = control.evaluate(candidate_rows)
            persistent_batch = persistent.evaluate(candidate_rows)
            assert persistent.workspace_ready
            assert persistent.workspace_qubit_count == 4
            assert persistent.workspace_extractions == 1
            assert set(persistent_batch.global_qubits) == {
                EXTRA,
                FIRST,
                SECOND,
                THIRD,
            }
            assert matrix_error(
                control_batch.observations,
                persistent_batch.observations,
            ) <= 2.0e-12

            try:
                persistent.refresh_workspace()
            except CausalRuntimeError:
                pass
            else:
                raise AssertionError("active workspace refresh was accepted")

            selected = (step * 3 + 2) % len(candidate_rows)
            control_gradient = control.selected_gradient(
                control_batch,
                selected,
                workers=4,
            )
            persistent_gradient = persistent.selected_gradient(
                persistent_batch,
                selected,
                workers=4,
            )
            assert max(
                abs(left - right)
                for left, right in zip(
                    control_gradient.values,
                    persistent_gradient.values,
                )
            ) <= 2.0e-12
            assert matrix_error(
                control_gradient.jacobian,
                persistent_gradient.jacobian,
            ) <= 2.0e-12

            control.commit(control_batch, selected)
            persistent.commit(persistent_batch, selected)
            assert persistent.generation == control.generation == step + 1
            assert persistent.workspace_extractions == 1
            assert max(
                abs(left - right)
                for left, right in zip(
                    support.execute(control.state),
                    support.execute(persistent.state),
                )
            ) <= 2.0e-12

        checkpoint = persistent.checkpoint()
        discarded = persistent.evaluate(rows(plan, 5, 7))
        persistent.discard(discarded)
        assert persistent.checkpoint() == checkpoint
        assert persistent.workspace_extractions == 1

        persistent.refresh_workspace()
        assert not persistent.workspace_ready
        refreshed = persistent.evaluate(rows(plan, 4, 8))
        assert persistent.workspace_extractions == 2
        persistent.discard(refreshed)
        assert persistent.state.validate()

    prepare.close()
    support.close()
    plan.close()
    print("QSA persistent component workspace tests passed.")


if __name__ == "__main__":
    main()
