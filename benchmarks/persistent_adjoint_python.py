from __future__ import annotations

import math
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_persistent_workspace import CausalPersistentComponentRuntime
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999
EXTRA = 611
STEPS = 6
CANDIDATES = 64


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


def rows(plan: CausalParameterizedPlan, step: int):
    return tuple(
        {
            name: 0.19
            * math.sin((step + 1) * (candidate + 1) * (slot + 1) * 0.017)
            for slot, name in enumerate(plan.parameter_names)
        }
        for candidate in range(CANDIDATES)
    )


def cotangent(step: int) -> tuple[float, ...]:
    return tuple(
        0.23 * math.cos((step + 1) * (index + 1) * 0.071)
        for index in range(14)
    )


def score(observations, step: int) -> tuple[float, ...]:
    weights = tuple(
        math.sin((step + 1) * (index + 1) * 0.13)
        for index in range(14)
    )
    return tuple(
        sum(float(value) * weights[index] for index, value in enumerate(row))
        for row in observations
    )


def maximum_error(first, second) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def matrix_error(first, second) -> float:
    return max(
        abs(float(left) - float(right))
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    )


def main() -> None:
    plan = candidate_plan()
    support = observable_plan()
    prepare = CausalParameterizedPlan((("h", EXTRA), ("cnot", EXTRA, FIRST)))

    with CausalRegister(QUBITS) as initial:
        initial.apply(prepare, ())
        packet = initial.encode_qsc()

    shift_evaluation_seconds = 0.0
    shift_pullback_seconds = 0.0
    shift_commit_seconds = 0.0
    adjoint_evaluation_seconds = 0.0
    adjoint_pullback_seconds = 0.0
    adjoint_commit_seconds = 0.0
    maximum_candidate_error = 0.0
    maximum_value_error = 0.0
    maximum_gradient_error = 0.0

    with CausalPersistentComponentRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
        max_local_qubits=8,
    ) as shifted, CausalPersistentComponentRuntime.decode_qsc(
        packet,
        plan,
        support,
        workers=4,
        max_local_qubits=8,
    ) as adjoint:
        for step in range(STEPS):
            parameter_rows = rows(plan, step)

            started = time.perf_counter()
            shifted_batch = shifted.evaluate(parameter_rows)
            shift_evaluation_seconds += time.perf_counter() - started

            started = time.perf_counter()
            adjoint_batch = adjoint.evaluate(parameter_rows)
            adjoint_evaluation_seconds += time.perf_counter() - started

            maximum_candidate_error = max(
                maximum_candidate_error,
                matrix_error(
                    shifted_batch.observations,
                    adjoint_batch.observations,
                ),
            )
            selected = max(
                range(CANDIDATES),
                key=lambda index: (score(shifted_batch.observations, step)[index], -index),
            )
            weights = cotangent(step)

            started = time.perf_counter()
            reference = shifted.selected_gradient(
                shifted_batch,
                selected,
                workers=4,
            )
            reference_vjp = reference.vjp(weights)
            shift_pullback_seconds += time.perf_counter() - started

            started = time.perf_counter()
            result = adjoint.selected_pullback(
                adjoint_batch,
                selected,
                weights,
            )
            adjoint_pullback_seconds += time.perf_counter() - started

            maximum_value_error = max(
                maximum_value_error,
                maximum_error(reference.values, result.values),
            )
            maximum_gradient_error = max(
                maximum_gradient_error,
                maximum_error(reference_vjp, result.gradient),
            )
            if result.shifted_evaluations != 0:
                raise RuntimeError("persistent adjoint used shifted evaluations")
            if result.stored_forward_states != 0:
                raise RuntimeError("persistent adjoint stored forward states")

            started = time.perf_counter()
            shifted.commit(shifted_batch, selected)
            shift_commit_seconds += time.perf_counter() - started

            started = time.perf_counter()
            adjoint.commit(adjoint_batch, selected)
            adjoint_commit_seconds += time.perf_counter() - started

        final_error = maximum_error(
            support.execute(shifted.state),
            support.execute(adjoint.state),
        )
        if shifted.workspace_extractions != 1 or adjoint.workspace_extractions != 1:
            raise RuntimeError("persistent workspace was extracted more than once")

    shift_complete_seconds = (
        shift_evaluation_seconds + shift_pullback_seconds + shift_commit_seconds
    )
    adjoint_complete_seconds = (
        adjoint_evaluation_seconds + adjoint_pullback_seconds + adjoint_commit_seconds
    )
    shift_pullback_ms = 1_000.0 * shift_pullback_seconds
    adjoint_pullback_ms = 1_000.0 * adjoint_pullback_seconds
    shift_complete_ms = 1_000.0 * shift_complete_seconds
    adjoint_complete_ms = 1_000.0 * adjoint_complete_seconds
    print(
        " ".join(
            (
                "persistent_adjoint",
                f"logical_qubits={QUBITS}",
                "local_qubits=4",
                f"steps={STEPS}",
                f"candidates_per_step={CANDIDATES}",
                "operations=25",
                "parameters=18",
                "observables=14",
                "support_terms=20",
                "shift_evaluations_per_step=36",
                "adjoint_forward_sweeps_per_step=1",
                "adjoint_reverse_sweeps_per_step=1",
                "stored_forward_states=0",
                f"shift_pullback_ms={shift_pullback_ms:.9f}",
                f"adjoint_pullback_ms={adjoint_pullback_ms:.9f}",
                f"pullback_speedup={shift_pullback_ms / adjoint_pullback_ms:.9f}",
                f"shift_complete_ms={shift_complete_ms:.9f}",
                f"adjoint_complete_ms={adjoint_complete_ms:.9f}",
                f"complete_speedup={shift_complete_ms / adjoint_complete_ms:.9f}",
                f"max_candidate_error={maximum_candidate_error:.17g}",
                f"max_value_error={maximum_value_error:.17g}",
                f"max_gradient_error={maximum_gradient_error:.17g}",
                f"max_final_error={final_error:.17g}",
            )
        ),
        flush=True,
    )

    prepare.close()
    support.close()
    plan.close()


if __name__ == "__main__":
    main()
