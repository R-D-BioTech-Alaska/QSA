from __future__ import annotations

import math
import statistics
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_adjoint import CausalWeightedAdjoint
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift


def tripair_plan() -> CausalParameterizedPlan:
    operations = []
    for qubit in range(3):
        operations.append(("ry", qubit, Parameter(f"semantic_theta_{qubit}")))
        operations.append(("rz", qubit, Parameter(f"semantic_phi_{qubit}")))
    operations.extend((("cnot", 0, 1), ("cnot", 1, 2), ("cnot", 2, 0)))
    for layer in range(2):
        for qubit in range(3):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_first"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_second"))
            )
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    assert len(operations) == 25
    return CausalParameterizedPlan(operations)


def brain_support() -> CausalPauliSupportPlan:
    return CausalPauliSupportPlan(
        3,
        (
            ((0, "X"),),
            ((0, "Y"),),
            ((0, "Z"),),
            ((1, "X"),),
            ((1, "Y"),),
            ((1, "Z"),),
            ((2, "X"),),
            ((2, "Y"),),
            ((2, "Z"),),
            ((0, "X"), (1, "X")),
            ((0, "Z"), (1, "Z")),
            ((1, "X"), (2, "X")),
            ((1, "Z"), (2, "Z")),
            ((0, "X"), (1, "X"), (2, "X")),
        ),
    )


def maximum_error(first, second) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def main() -> None:
    plan = tripair_plan()
    support = brain_support()
    parameter_shift = CausalSupportParameterShift(plan, support)
    adjoint = CausalWeightedAdjoint(plan, support, max_qubits=8)
    values = tuple(
        0.23 * math.sin((index + 1) * 0.31)
        for index in range(plan.parameter_count)
    )
    cotangent = tuple(
        0.17 * math.cos((index + 1) * 0.43)
        for index in range(support.observable_count)
    )
    repeats = 25

    with CausalRegister(3) as root:
        initial = root.encode_qsc()
        shifted = parameter_shift.evaluate_and_jacobian(root, values, workers=1)
        exact = adjoint.evaluate(root, values, cotangent)
        value_error = maximum_error(shifted.values, exact.values)
        gradient_error = maximum_error(shifted.vjp(cotangent), exact.gradient)
        if root.encode_qsc() != initial:
            raise RuntimeError("adjoint benchmark mutated the causal root")

        shift_seconds = []
        adjoint_seconds = []
        for _ in range(repeats):
            started = time.perf_counter()
            parameter_shift.evaluate_and_jacobian(root, values, workers=1).vjp(
                cotangent
            )
            shift_seconds.append(time.perf_counter() - started)

            started = time.perf_counter()
            adjoint.evaluate(root, values, cotangent)
            adjoint_seconds.append(time.perf_counter() - started)

    parameter_shift_ms = 1_000.0 * statistics.median(shift_seconds)
    adjoint_ms = 1_000.0 * statistics.median(adjoint_seconds)
    speedup = parameter_shift_ms / adjoint_ms
    print(
        " ".join(
            (
                "causal_adjoint",
                "qubits=3",
                "operations=25",
                "parameters=18",
                "observables=14",
                "support_terms=20",
                "shift_evaluations=37",
                "adjoint_forward_sweeps=1",
                "adjoint_reverse_sweeps=1",
                f"stored_forward_states={exact.stored_forward_states}",
                f"parameter_shift_ms={parameter_shift_ms:.9f}",
                f"adjoint_ms={adjoint_ms:.9f}",
                f"speedup={speedup:.9f}",
                f"max_value_error={value_error:.17g}",
                f"max_gradient_error={gradient_error:.17g}",
            )
        ),
        flush=True,
    )
    support.close()
    plan.close()


if __name__ == "__main__":
    main()
