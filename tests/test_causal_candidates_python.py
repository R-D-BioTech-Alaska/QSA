from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_candidates import CausalCandidateRuntime
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999


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


def observables():
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
            name: 0.19 * math.sin((step + 1) * (candidate + 1) * (slot + 1) * 0.017)
            for slot, name in enumerate(plan.parameter_names)
        }
        for candidate in range(count)
    )


def close_values(first, second, tolerance=2.0e-12):
    if len(first) != len(second):
        raise AssertionError("observable widths differ")
    for left, right in zip(first, second):
        if abs(left - right) > tolerance:
            raise AssertionError(f"{left} differs from {right}")


def main() -> None:
    plan = candidate_plan()
    support = observables()

    with CausalCandidateRuntime.create(
        QUBITS,
        plan,
        support,
        workers=4,
    ) as runtime:
        initial = runtime.checkpoint()
        first_rows = rows(plan, 8, 0)
        first_batch = runtime.evaluate(first_rows)
        assert len(first_batch) == 8
        assert runtime.has_active_batch
        assert runtime.state.encode_qsc() == initial

        try:
            runtime.evaluate(first_rows)
        except CausalRuntimeError:
            pass
        else:
            raise AssertionError("runtime accepted a second active candidate batch")

        selected_first = 5
        with CausalRegister.decode_qsc(initial) as control:
            control.apply(plan, first_rows[selected_first])
            close_values(
                first_batch.observations[selected_first],
                support.execute(control),
            )
            runtime.commit(first_batch, selected_first)
            assert first_batch.closed
            assert not runtime.has_active_batch
            assert runtime.generation == 1
            close_values(support.execute(runtime.state), support.execute(control))

            committed = runtime.checkpoint()
            discard_batch = runtime.evaluate(rows(plan, 6, 1))
            try:
                runtime.checkpoint()
            except CausalRuntimeError:
                pass
            else:
                raise AssertionError("checkpoint accepted an active candidate batch")
            runtime.discard(discard_batch)
            assert runtime.checkpoint() == committed
            assert runtime.generation == 1

            second_rows = rows(plan, 7, 2)
            second_batch = runtime.evaluate(second_rows)
            selected_second = 3
            control.apply(plan, second_rows[selected_second])
            close_values(
                second_batch.observations[selected_second],
                support.execute(control),
            )
            runtime.commit(second_batch, selected_second)
            assert runtime.generation == 2
            close_values(support.execute(runtime.state), support.execute(control))
            assert runtime.state.validate()

    support.close()
    plan.close()
    print("QSA causal candidate transaction tests passed.")


if __name__ == "__main__":
    main()
