from __future__ import annotations

import math
import statistics
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_grad import CausalParameterShift, ParameterShiftResult


REPEATS = 7


def build_plan() -> CausalParameterizedPlan:
    operations = []
    for qubit in range(3):
        operations.append(("ry", qubit, Parameter(f"semantic_theta_{qubit}")))
        operations.append(("rz", qubit, Parameter(f"semantic_phi_{qubit}")))
    operations.extend(
        (
            ("cnot", 0, 1),
            ("cnot", 1, 2),
            ("cnot", 2, 0),
        )
    )
    for layer in range(2):
        for qubit in range(3):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_first"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_second"))
            )
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    return CausalParameterizedPlan(operations)


def build_observables() -> CausalPauliPlan:
    return CausalPauliPlan(
        3,
        [
            "XII",
            "YII",
            "ZII",
            "IXI",
            "IYI",
            "IZI",
            "IIX",
            "IIY",
            "IIZ",
            "ZZI",
            "IZZ",
            "ZIZ",
            "XXX",
            "YYY",
        ],
    )


def shifted_rows(values: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    rows = [values]
    for parameter in range(len(values)):
        positive = list(values)
        negative = list(values)
        positive[parameter] += math.pi / 2.0
        negative[parameter] -= math.pi / 2.0
        rows.append(tuple(positive))
        rows.append(tuple(negative))
    return tuple(rows)


def result_from_rows(
    measured: tuple[tuple[float, ...], ...],
    parameter_names: tuple[str, ...],
    observable_words: tuple[str, ...],
) -> ParameterShiftResult:
    values = tuple(measured[0])
    parameter_count = len(parameter_names)
    observable_count = len(values)
    jacobian = tuple(
        tuple(
            0.5
            * (
                measured[1 + 2 * parameter][observable]
                - measured[2 + 2 * parameter][observable]
            )
            for parameter in range(parameter_count)
        )
        for observable in range(observable_count)
    )
    return ParameterShiftResult(
        values=values,
        jacobian=jacobian,
        parameter_names=parameter_names,
        observable_words=observable_words,
    )


def scalar_result(
    root: CausalRegister,
    plan: CausalParameterizedPlan,
    observables: CausalPauliPlan,
    rows: tuple[tuple[float, ...], ...],
) -> ParameterShiftResult:
    measured = []
    for values in rows:
        branch = root.fork()
        try:
            branch.apply(plan, values)
            measured.append(observables.execute(branch))
        finally:
            branch.close()
    return result_from_rows(
        tuple(measured),
        tuple(plan.parameter_names),
        tuple(observables.words),
    )


def qsc_result(
    packet: bytes,
    plan: CausalParameterizedPlan,
    observables: CausalPauliPlan,
    rows: tuple[tuple[float, ...], ...],
) -> ParameterShiftResult:
    measured = []
    for values in rows:
        branch = CausalRegister.decode_qsc(packet)
        try:
            branch.apply(plan, values)
            measured.append(observables.execute(branch))
        finally:
            branch.close()
    return result_from_rows(
        tuple(measured),
        tuple(plan.parameter_names),
        tuple(observables.words),
    )


def max_error(first: ParameterShiftResult, second: ParameterShiftResult) -> float:
    value_error = max(abs(a - b) for a, b in zip(first.values, second.values))
    jacobian_error = max(
        abs(a - b)
        for first_row, second_row in zip(first.jacobian, second.jacobian)
        for a, b in zip(first_row, second_row)
    )
    return max(value_error, jacobian_error)


def main() -> None:
    plan = build_plan()
    observables = build_observables()
    gradient = CausalParameterShift(plan, observables)
    values = tuple(
        0.21 * math.sin((index + 1) * 0.29)
        for index in range(plan.parameter_count)
    )
    rows = shifted_rows(values)

    qsc_samples = []
    scalar_samples = []
    batch_samples = []
    maximum_error = 0.0

    with CausalRegister(3) as root:
        packet = root.encode_qsc()
        for _ in range(REPEATS):
            started = time.perf_counter()
            qsc = qsc_result(packet, plan, observables, rows)
            qsc_samples.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            scalar = scalar_result(root, plan, observables, rows)
            scalar_samples.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            batch = gradient.evaluate_and_jacobian(root, values, workers=0)
            batch_samples.append((time.perf_counter() - started) * 1000.0)

            maximum_error = max(
                maximum_error,
                max_error(qsc, scalar),
                max_error(qsc, batch),
            )

    qsc_ms = statistics.median(qsc_samples)
    scalar_ms = statistics.median(scalar_samples)
    batch_ms = statistics.median(batch_samples)
    print(
        "causal_gradient"
        " parameters=18"
        " observables=14"
        " evaluations=37"
        " operations=25"
        f" qsc_ms={qsc_ms:.12f}"
        f" scalar_causal_ms={scalar_ms:.12f}"
        f" batch_causal_ms={batch_ms:.12f}"
        f" batch_vs_qsc={qsc_ms / batch_ms:.12f}"
        f" batch_vs_scalar={scalar_ms / batch_ms:.12f}"
        f" max_error={maximum_error:.12g}"
    )

    observables.close()
    plan.close()


if __name__ == "__main__":
    main()
