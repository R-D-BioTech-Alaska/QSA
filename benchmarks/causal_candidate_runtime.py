from __future__ import annotations

import math
import statistics
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_candidates import CausalCandidateRuntime
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
CANDIDATES = 64
STEPS = 6
REPEATS = 5
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


def maximum_error(first, second):
    return max(abs(left - right) for left, right in zip(first, second))


def run_qsc(plan, support, all_rows):
    state = CausalRegister(QUBITS)
    last_observations = None
    started = time.perf_counter()
    try:
        for step, rows in enumerate(all_rows):
            packet = state.encode_qsc()
            candidates = [
                CausalRegister.decode_qsc(packet)
                for _ in range(CANDIDATES)
            ]
            try:
                observations = []
                for candidate, values in zip(candidates, rows):
                    candidate.apply(plan, values)
                    observations.append(support.execute(candidate))
                selected = (step * 11 + 7) % CANDIDATES
                chosen = candidates[selected]
                candidates[selected] = None
                state.close()
                state = chosen
                last_observations = tuple(observations)
            finally:
                for candidate in candidates:
                    if candidate is not None:
                        candidate.close()
        elapsed = (time.perf_counter() - started) * 1000.0
        final_values = support.execute(state)
        return elapsed, final_values, last_observations
    finally:
        state.close()


def run_causal(plan, support, all_rows):
    runtime = CausalCandidateRuntime.create(
        QUBITS,
        plan,
        support,
        workers=0,
    )
    last_observations = None
    started = time.perf_counter()
    try:
        for step, rows in enumerate(all_rows):
            batch = runtime.evaluate(rows)
            selected = (step * 11 + 7) % CANDIDATES
            last_observations = batch.observations
            runtime.commit(batch, selected)
        elapsed = (time.perf_counter() - started) * 1000.0
        final_values = support.execute(runtime.state)
        generation = runtime.generation
        return elapsed, final_values, last_observations, generation
    finally:
        runtime.close()


def main() -> None:
    plan = candidate_plan()
    support = observables()
    all_rows = parameter_rows(plan)

    qsc_samples = []
    causal_samples = []
    maximum = 0.0
    generation = 0

    for _ in range(REPEATS):
        qsc_ms, qsc_values, qsc_last = run_qsc(plan, support, all_rows)
        causal_ms, causal_values, causal_last, generation = run_causal(
            plan,
            support,
            all_rows,
        )
        qsc_samples.append(qsc_ms)
        causal_samples.append(causal_ms)
        maximum = max(
            maximum,
            maximum_error(qsc_values, causal_values),
            max(
                maximum_error(first, second)
                for first, second in zip(qsc_last, causal_last)
            ),
        )

    qsc_ms = statistics.median(qsc_samples)
    causal_ms = statistics.median(causal_samples)
    print(
        "causal_candidate_runtime"
        f" qubits={QUBITS}"
        f" candidates={CANDIDATES}"
        f" steps={STEPS}"
        " operations=25"
        " parameters=18"
        " observables=14"
        f" generation={generation}"
        f" qsc_loop_ms={qsc_ms:.12f}"
        f" causal_loop_ms={causal_ms:.12f}"
        f" speedup={qsc_ms / causal_ms:.12f}"
        f" max_error={maximum:.12g}"
    )

    support.close()
    plan.close()


if __name__ == "__main__":
    main()
