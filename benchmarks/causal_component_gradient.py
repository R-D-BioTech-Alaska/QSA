from __future__ import annotations

import math
import statistics
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_component_grad import CausalComponentParameterShift
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift


QUBITS = 10_000
REPEATS = 9
FIRST = 17
SECOND = 4_321
THIRD = 9_999


def circuit_plan():
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


def maximum_error(first, second):
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


def main() -> None:
    plan = circuit_plan()
    support = observable_plan()
    full_gradient = CausalSupportParameterShift(plan, support)
    component_gradient = CausalComponentParameterShift(
        plan,
        support,
        max_local_qubits=8,
    )
    values = tuple(
        0.21 * math.sin((index + 1) * 0.29)
        for index in range(plan.parameter_count)
    )

    full_samples = []
    component_samples = []
    maximum = 0.0
    local_qubits = 0

    with CausalRegister(QUBITS) as root:
        full_gradient.evaluate_and_jacobian(root, values, workers=0)
        component_gradient.evaluate_and_jacobian(root, values, workers=0)

        for _ in range(REPEATS):
            started = time.perf_counter()
            full = full_gradient.evaluate_and_jacobian(
                root,
                values,
                workers=0,
            )
            full_samples.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            component = component_gradient.evaluate_and_jacobian(
                root,
                values,
                workers=0,
            )
            component_samples.append((time.perf_counter() - started) * 1000.0)

            local_qubits = component.local_qubit_count
            maximum = max(maximum, maximum_error(full, component))

    full_ms = statistics.median(full_samples)
    component_ms = statistics.median(component_samples)
    print(
        "causal_component_gradient"
        f" qubits={QUBITS}"
        f" local_qubits={local_qubits}"
        " parameters=18"
        " observables=14"
        " evaluations=37"
        " operations=25"
        f" full_gradient_ms={full_ms:.12f}"
        f" component_gradient_ms={component_ms:.12f}"
        f" speedup={full_ms / component_ms:.12f}"
        f" max_error={maximum:.12g}"
    )

    component_gradient.close()
    support.close()
    plan.close()


if __name__ == "__main__":
    main()
