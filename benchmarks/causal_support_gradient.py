from __future__ import annotations

import math
import statistics
import time

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_grad import CausalParameterShift
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift


QUBITS = 10_000
REPEATS = 5
FIRST = 17
SECOND = 4_321
THIRD = 9_999


def plan():
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


def supports():
    return (
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
    )


def words():
    encoded = []
    for observable in supports():
        word = ["I"] * QUBITS
        for qubit, axis in observable:
            word[qubit] = axis
        encoded.append("".join(word))
    return tuple(encoded)


def maximum_error(first, second):
    value_error = max(abs(a - b) for a, b in zip(first.values, second.values))
    jacobian_error = max(
        abs(a - b)
        for first_row, second_row in zip(first.jacobian, second.jacobian)
        for a, b in zip(first_row, second_row)
    )
    return max(value_error, jacobian_error)


def main() -> None:
    circuit = plan()
    full_words = words()
    full_observables = CausalPauliPlan(QUBITS, full_words)
    support_observables = CausalPauliSupportPlan(QUBITS, supports())
    full_gradient = CausalParameterShift(circuit, full_observables)
    support_gradient = CausalSupportParameterShift(circuit, support_observables)
    values = tuple(
        0.21 * math.sin((index + 1) * 0.29)
        for index in range(circuit.parameter_count)
    )

    full_samples = []
    support_samples = []
    maximum = 0.0

    with CausalRegister(QUBITS) as root:
        full_gradient.evaluate_and_jacobian(root, values, workers=0)
        support_gradient.evaluate_and_jacobian(root, values, workers=0)

        for _ in range(REPEATS):
            started = time.perf_counter()
            full = full_gradient.evaluate_and_jacobian(root, values, workers=0)
            full_samples.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            compact = support_gradient.evaluate_and_jacobian(root, values, workers=0)
            support_samples.append((time.perf_counter() - started) * 1000.0)

            maximum = max(maximum, maximum_error(full, compact))

    full_ms = statistics.median(full_samples)
    support_ms = statistics.median(support_samples)
    print(
        "causal_support_gradient"
        f" qubits={QUBITS}"
        " parameters=18"
        " observables=14"
        " evaluations=37"
        " operations=25"
        f" support_terms={support_observables.term_count}"
        f" full_word_bytes={sum(len(word) for word in full_words)}"
        f" full_gradient_ms={full_ms:.12f}"
        f" support_gradient_ms={support_ms:.12f}"
        f" speedup={full_ms / support_ms:.12f}"
        f" max_error={maximum:.12g}"
    )

    support_observables.close()
    full_observables.close()
    circuit.close()


if __name__ == "__main__":
    main()
