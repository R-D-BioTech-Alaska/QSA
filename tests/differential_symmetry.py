"""Independent dense-reference validation for the QSA symmetry algebra."""

from __future__ import annotations

import cmath
import math
import random

from qsa import SymmetryState


def normalize(counts: list[int], amplitudes: list[complex]) -> list[complex]:
    norm = math.sqrt(sum(count * abs(value) ** 2 for count, value in zip(counts, amplitudes)))
    return [value / norm for value in amplitudes]


def class_for_basis(counts: list[int], basis: int) -> int:
    offset = 0
    for index, count in enumerate(counts):
        if basis < offset + count:
            return index
        offset += count
    raise AssertionError("basis outside dense reference")


def apply_rotation(
    counts: list[int], amplitudes: list[complex], first: int, second: int, angle: float
) -> list[complex]:
    coefficients = [
        amplitude * math.sqrt(count) for count, amplitude in zip(counts, amplitudes)
    ]
    c = math.cos(angle)
    s = math.sin(angle)
    old_first = coefficients[first]
    old_second = coefficients[second]
    coefficients[first] = c * old_first - s * old_second
    coefficients[second] = s * old_first + c * old_second
    return normalize(
        counts,
        [coefficient / math.sqrt(count) for count, coefficient in zip(counts, coefficients)],
    )


def rotation_matrix(size: int, first: int, second: int, angle: float) -> list[list[complex]]:
    matrix = [[0j for _ in range(size)] for _ in range(size)]
    for index in range(size):
        matrix[index][index] = 1 + 0j
    c = math.cos(angle)
    s = math.sin(angle)
    matrix[first][first] = c
    matrix[first][second] = -s
    matrix[second][first] = s
    matrix[second][second] = c
    return matrix


def run() -> None:
    rng = random.Random(0x51A1E7)
    amplitude_checks = 0
    operation_checks = 0

    for _trial in range(250):
        qubits = rng.randint(2, 8)
        dimension = 1 << qubits
        class_count = rng.randint(2, min(6, dimension))
        cuts = sorted(rng.sample(range(1, dimension), class_count - 1))
        bounds = [0, *cuts, dimension]
        counts = [bounds[index + 1] - bounds[index] for index in range(class_count)]
        amplitudes = [complex(1 / math.sqrt(dimension), 0.0) for _ in counts]

        with SymmetryState(qubits, counts) as state:
            for _step in range(80):
                operation = rng.randrange(4)
                if operation == 0:
                    selected = rng.randrange(len(counts))
                    angle = rng.uniform(-math.pi, math.pi)
                    state.phase(selected, angle)
                    amplitudes[selected] *= cmath.exp(1j * angle)
                elif operation == 1:
                    state.reflect()
                    mean = sum(
                        count * amplitude for count, amplitude in zip(counts, amplitudes)
                    ) / dimension
                    amplitudes = [2 * mean - amplitude for amplitude in amplitudes]
                elif operation == 2 and len(counts) >= 2:
                    first, second = rng.sample(range(len(counts)), 2)
                    angle = rng.uniform(-0.7, 0.7)
                    state.unitary(rotation_matrix(len(counts), first, second, angle))
                    amplitudes = apply_rotation(counts, amplitudes, first, second, angle)
                else:
                    splittable = [index for index, count in enumerate(counts) if count > 1]
                    if splittable:
                        selected = rng.choice(splittable)
                        first_count = rng.randint(1, counts[selected] - 1)
                        state.split_class(selected, first_count)
                        original = counts[selected]
                        counts[selected] = first_count
                        counts.insert(selected + 1, original - first_count)
                        amplitudes.insert(selected + 1, amplitudes[selected])
                operation_checks += 1

                assert state.validate()
                assert int(state.class_count) == len(counts)
                for class_index, expected in enumerate(amplitudes):
                    actual = state.class_amplitude(class_index)
                    assert abs(actual - expected) < 2e-10, (
                        class_index,
                        actual,
                        expected,
                    )

            fallback = state.to_register(max_qubits=qubits)
            try:
                for basis in range(dimension):
                    expected = amplitudes[class_for_basis(counts, basis)]
                    assert abs(state.amplitude(basis) - expected) < 2e-10
                    assert abs(fallback.amplitude(basis) - expected) < 2e-10
                    amplitude_checks += 2
            finally:
                fallback.close()

    print(
        "QSA symmetry differential validation passed: "
        f"{operation_checks} operations, {amplitude_checks} amplitude checks"
    )


if __name__ == "__main__":
    run()
