from __future__ import annotations

import math
import statistics
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_batch import apply_many, fork_many, observe_many


BRANCHES = 128
REPEATS = 5


def build_plan() -> CausalParameterizedPlan:
    operations = []
    for qubit in range(3):
        operations.append(("ry", qubit, Parameter(f"semantic_theta_{qubit}")))
        operations.append(("rz", qubit, Parameter(f"semantic_phi_{qubit}")))
    operations.extend((
        ("cnot", 0, 1),
        ("cnot", 1, 2),
        ("cnot", 2, 0),
    ))
    for layer in range(2):
        for qubit in range(3):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_first"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_second"))
            )
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    if len(operations) != 25:
        raise AssertionError("Phase41A plan must contain 25 operations")
    return CausalParameterizedPlan(operations)


def rows(plan: CausalParameterizedPlan) -> list[dict[str, float]]:
    return [
        {
            name: 0.17 * math.sin((branch + 1) * (slot + 1) * 0.031)
            for slot, name in enumerate(plan.parameter_names)
        }
        for branch in range(BRANCHES)
    ]


def close_all(states: list[CausalRegister]) -> None:
    for state in states:
        state.close()


def max_error(first, second) -> float:
    return max(
        abs(a - b)
        for first_row, second_row in zip(first, second)
        for a, b in zip(first_row, second_row)
    )


def main() -> None:
    plan = build_plan()
    closure = CausalPauliPlan(
        3,
        [
            "XII", "YII", "ZII",
            "IXI", "IYI", "IZI",
            "IIX", "IIY", "IIZ",
            "ZZI", "IZZ", "ZIZ",
            "XXX", "YYY",
        ],
    )
    parameter_rows = rows(plan)

    qsc_samples = []
    scalar_samples = []
    batch_samples = []
    reference = None

    with CausalRegister(3) as root:
        root_qsc = root.encode_qsc()
        for _ in range(REPEATS):
            started = time.perf_counter()
            qsc_states = [CausalRegister.decode_qsc(root_qsc) for _ in range(BRANCHES)]
            try:
                for state, values in zip(qsc_states, parameter_rows):
                    state.apply(plan, values)
                qsc_values = tuple(closure.execute(state) for state in qsc_states)
            finally:
                close_all(qsc_states)
            qsc_samples.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            scalar_states = [root.fork() for _ in range(BRANCHES)]
            try:
                for state, values in zip(scalar_states, parameter_rows):
                    state.apply(plan, values)
                scalar_values = tuple(closure.execute(state) for state in scalar_states)
            finally:
                close_all(scalar_states)
            scalar_samples.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            batch_states = fork_many(root, BRANCHES)
            try:
                apply_many(plan, batch_states, parameter_rows, workers=0)
                batch_values = observe_many(closure, batch_states, workers=0)
            finally:
                close_all(batch_states)
            batch_samples.append((time.perf_counter() - started) * 1000.0)

            if reference is None:
                reference = qsc_values
            if max_error(qsc_values, scalar_values) > 1.0e-12:
                raise AssertionError("scalar causal results differ from QSC route")
            if max_error(qsc_values, batch_values) > 1.0e-12:
                raise AssertionError("batch causal results differ from QSC route")

    qsc_ms = statistics.median(qsc_samples)
    scalar_ms = statistics.median(scalar_samples)
    batch_ms = statistics.median(batch_samples)
    print(
        "causal_batch"
        f" branches={BRANCHES}"
        f" operations=25"
        f" parameters={plan.parameter_count}"
        f" observables={len(closure.words)}"
        f" qsc_ms={qsc_ms:.12f}"
        f" scalar_causal_ms={scalar_ms:.12f}"
        f" batch_causal_ms={batch_ms:.12f}"
        f" batch_vs_qsc={qsc_ms / batch_ms:.12f}"
        f" batch_vs_scalar={scalar_ms / batch_ms:.12f}"
        " max_error=0.0"
    )
    closure.close()
    plan.close()


if __name__ == "__main__":
    main()
