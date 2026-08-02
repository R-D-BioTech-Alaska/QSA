from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_persistent_workspace import CausalPersistentComponentRuntime
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999
EXTRA = 611


def candidate_plan() -> CausalParameterizedPlan:
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


def observable_plan() -> CausalPauliSupportPlan:
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
            ((FIRST, "X"), (SECOND, "X")),
            ((FIRST, "Z"), (SECOND, "Z")),
            ((SECOND, "X"), (THIRD, "X")),
            ((SECOND, "Z"), (THIRD, "Z")),
            ((FIRST, "X"), (SECOND, "X"), (THIRD, "X")),
        ),
    )


def rows(plan: CausalParameterizedPlan, count: int, step: int):
    return tuple(
        {
            name: 0.19
            * math.sin((step + 1) * (candidate + 1) * (slot + 1) * 0.017)
            for slot, name in enumerate(plan.parameter_names)
        }
        for candidate in range(count)
    )


def cotangent(step: int) -> tuple[float, ...]:
    return tuple(
        0.23 * math.cos((step + 1) * (index + 1) * 0.071)
        for index in range(14)
    )


def maximum_error(first, second) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def test_selected_pullback_reuses_the_persistent_workspace() -> None:
    plan = candidate_plan()
    support = observable_plan()
    prepare = CausalParameterizedPlan((("h", EXTRA), ("cnot", EXTRA, FIRST)))

    with CausalRegister(QUBITS) as initial:
        initial.apply(prepare, ())
        packet = initial.encode_qsc()

    with CausalPersistentComponentRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
        max_local_qubits=8,
    ) as persistent:
        for step in range(3):
            batch = persistent.evaluate(rows(plan, 8, step))
            selected = (step * 3 + 2) % len(batch)
            weights = cotangent(step)
            root_before = persistent.state.encode_qsc()
            workspace_before = persistent._workspace.state.encode_qsc()

            reference = persistent.selected_gradient(batch, selected, workers=4)
            pullback = persistent.selected_pullback(batch, selected, weights)

            assert support.term_count == 20
            assert persistent.workspace_extractions == 1
            assert pullback.global_qubits == batch.global_qubits
            assert set(pullback.global_qubits) == {
                EXTRA,
                FIRST,
                SECOND,
                THIRD,
            }
            assert pullback.forward_sweeps == 1
            assert pullback.reverse_sweeps == 1
            assert pullback.shifted_evaluations == 0
            assert pullback.stored_forward_states == 0
            assert maximum_error(pullback.values, reference.values) <= 2.0e-12
            assert maximum_error(
                pullback.gradient,
                reference.vjp(weights),
            ) <= 2.0e-12
            assert maximum_error(
                persistent.selected_vjp(batch, selected, weights),
                pullback.gradient,
            ) <= 2.0e-12
            assert persistent.state.encode_qsc() == root_before
            assert persistent._workspace.state.encode_qsc() == workspace_before

            persistent.commit(batch, selected)
            assert persistent.generation == step + 1
            assert persistent.workspace_extractions == 1

        assert persistent.state.validate()

    prepare.close()
    support.close()
    plan.close()


def test_selected_pullback_rejects_foreign_and_closed_batches() -> None:
    plan = candidate_plan()
    support = observable_plan()

    with CausalPersistentComponentRuntime.create(
        QUBITS,
        plan,
        support,
        max_local_qubits=8,
    ) as first, CausalPersistentComponentRuntime.create(
        QUBITS,
        plan,
        support,
        max_local_qubits=8,
    ) as second:
        batch = first.evaluate(rows(plan, 3, 0))
        try:
            second.selected_pullback(batch, 0, cotangent(0))
        except ValueError:
            pass
        else:
            raise AssertionError("foreign persistent batch was accepted")

        first.discard(batch)
        try:
            first.selected_pullback(batch, 0, cotangent(0))
        except CausalRuntimeError:
            pass
        else:
            raise AssertionError("closed persistent batch was accepted")

    support.close()
    plan.close()


def main() -> None:
    test_selected_pullback_reuses_the_persistent_workspace()
    test_selected_pullback_rejects_foreign_and_closed_batches()
    print("QSA persistent weighted adjoint tests passed.")


if __name__ == "__main__":
    main()
