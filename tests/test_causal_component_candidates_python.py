from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_candidates import CausalCandidateRuntime
from qsa.causal_component_candidates import CausalComponentCandidateRuntime
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


def max_error(first, second):
    return max(
        abs(left - right)
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    )


def main() -> None:
    plan = candidate_plan()
    support = observables()
    entangle_extra = CausalParameterizedPlan(
        [("h", EXTRA), ("cnot", EXTRA, FIRST)]
    )

    with CausalRegister(QUBITS) as initial:
        initial.apply(entangle_extra, [])
        packet = initial.encode_qsc()

    with CausalCandidateRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
    ) as full, CausalComponentCandidateRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
        max_local_qubits=8,
    ) as local:
        for step in range(3):
            candidate_rows = rows(plan, 8, step)
            full_batch = full.evaluate(candidate_rows)
            local_batch = local.evaluate(candidate_rows)
            assert local_batch.local_qubit_count == 4
            assert set(local_batch.global_qubits) == {EXTRA, FIRST, SECOND, THIRD}
            assert max_error(
                full_batch.observations,
                local_batch.observations,
            ) <= 2.0e-12

            selected = (step * 3 + 2) % len(candidate_rows)
            full.commit(full_batch, selected)
            local.commit(local_batch, selected)
            assert full.generation == local.generation == step + 1
            assert max(
                abs(left - right)
                for left, right in zip(
                    support.execute(full.state),
                    support.execute(local.state),
                )
            ) <= 2.0e-12

        checkpoint = local.checkpoint()
        discarded = local.evaluate(rows(plan, 5, 7))
        local.discard(discarded)
        assert local.checkpoint() == checkpoint
        assert local.state.validate()

    wide_qubits = tuple(range(100, 125))
    wide_plan = CausalParameterizedPlan([("rz", wide_qubits[0], Parameter("theta"))])
    wide_support = CausalPauliSupportPlan(
        QUBITS,
        (((wide_qubits[0], "Z"),),),
    )
    wide_prepare = CausalParameterizedPlan(
        [("h", wide_qubits[0])]
        + [
            ("cnot", wide_qubits[index - 1], wide_qubits[index])
            for index in range(1, len(wide_qubits))
        ]
    )
    with CausalRegister(QUBITS) as state:
        state.apply(wide_prepare, [])
        with CausalComponentCandidateRuntime(
            state,
            wide_plan,
            wide_support,
            max_local_qubits=24,
        ) as bounded:
            try:
                bounded.evaluate(({"theta": 0.1},))
            except CausalRuntimeError:
                pass
            else:
                raise AssertionError("oversized component closure was accepted")

    wide_prepare.close()
    wide_support.close()
    wide_plan.close()
    entangle_extra.close()
    support.close()
    plan.close()
    print("QSA component-local candidate transaction tests passed.")


if __name__ == "__main__":
    main()
