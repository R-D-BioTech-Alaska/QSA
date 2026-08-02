from __future__ import annotations

import math
import statistics
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan
from qsa.causal_component_candidates import CausalComponentCandidateRuntime
from qsa.causal_component_grad import CausalComponentParameterShift
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
CANDIDATES = 64
STEPS = 6
REPEATS = 7
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


def parameter_rows(plan):
    return tuple(
        tuple(
            {
                name: 0.17
                * math.sin((step + 1) * (candidate + 1) * (slot + 1) * 0.011)
                for slot, name in enumerate(plan.parameter_names)
            }
            for candidate in range(CANDIDATES)
        )
        for step in range(STEPS)
    )


def result_error(first, second):
    value_error = max(
        abs(left - right)
        for left, right in zip(first.values, second.values)
    )
    jacobian_error = max(
        abs(left - right)
        for first_row, second_row in zip(first.jacobian, second.jacobian)
        for left, right in zip(first_row, second_row)
    )
    return max(value_error, jacobian_error)


def run_separate(plan, support, all_rows):
    runtime = CausalComponentCandidateRuntime.create(
        QUBITS,
        plan,
        support,
        workers=0,
        max_local_qubits=8,
    )
    gradient = CausalComponentParameterShift(
        plan,
        support,
        max_local_qubits=8,
    )
    results = []
    started = time.perf_counter()
    try:
        for step, rows in enumerate(all_rows):
            batch = runtime.evaluate(rows)
            selected = (step * 11 + 7) % CANDIDATES
            result = gradient.evaluate_and_jacobian(
                runtime.state,
                rows[selected],
                workers=0,
            )
            results.append(result)
            runtime.commit(batch, selected)
        elapsed = (time.perf_counter() - started) * 1000.0
        return elapsed, tuple(results), support.execute(runtime.state)
    finally:
        gradient.close()
        runtime.close()


def run_reused(plan, support, all_rows):
    runtime = CausalComponentCandidateRuntime.create(
        QUBITS,
        plan,
        support,
        workers=0,
        max_local_qubits=8,
    )
    results = []
    started = time.perf_counter()
    try:
        for step, rows in enumerate(all_rows):
            batch = runtime.evaluate(rows)
            selected = (step * 11 + 7) % CANDIDATES
            result = runtime.selected_gradient(
                batch,
                selected,
                workers=0,
            )
            results.append(result)
            runtime.commit(batch, selected)
        elapsed = (time.perf_counter() - started) * 1000.0
        return elapsed, tuple(results), support.execute(runtime.state)
    finally:
        runtime.close()


def main() -> None:
    plan = candidate_plan()
    support = observable_plan()
    all_rows = parameter_rows(plan)

    separate_samples = []
    reused_samples = []
    maximum = 0.0

    for _ in range(REPEATS):
        separate_ms, separate_results, separate_final = run_separate(
            plan,
            support,
            all_rows,
        )
        reused_ms, reused_results, reused_final = run_reused(
            plan,
            support,
            all_rows,
        )
        separate_samples.append(separate_ms)
        reused_samples.append(reused_ms)
        maximum = max(
            maximum,
            max(
                result_error(first, second)
                for first, second in zip(separate_results, reused_results)
            ),
            max(
                abs(left - right)
                for left, right in zip(separate_final, reused_final)
            ),
        )

    separate_ms = statistics.median(separate_samples)
    reused_ms = statistics.median(reused_samples)
    print(
        "selected_candidate_gradient"
        f" qubits={QUBITS}"
        " local_qubits=3"
        f" candidates={CANDIDATES}"
        f" steps={STEPS}"
        " operations=25"
        " parameters=18"
        " observables=14"
        " separate_evaluations=37"
        " reused_shifted_evaluations=36"
        f" separate_ms={separate_ms:.12f}"
        f" reused_ms={reused_ms:.12f}"
        f" speedup={separate_ms / reused_ms:.12f}"
        f" max_error={maximum:.12g}"
    )

    support.close()
    plan.close()


if __name__ == "__main__":
    main()
