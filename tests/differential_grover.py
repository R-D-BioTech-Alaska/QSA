from __future__ import annotations

import math
import random
import numpy as np

from qsa import GroverSearch, QubitRegister

def dense_iteration(state: np.ndarray, marked: np.ndarray) -> None:
    state[marked] *= -1.0
    state[:] = 2.0 * state.mean() - state

def main() -> None:
    generator = random.Random(0x51534147)
    cases = 200
    amplitude_checks = 0

    for case in range(cases):
        qubits = generator.randint(2, 10)
        dimension = 1 << qubits
        marked_count = generator.randint(1, min(8, dimension - 1))
        marked = sorted(generator.sample(range(dimension), marked_count))
        iterations = generator.randint(0, 20)

        dense = np.full(dimension, 1.0 / math.sqrt(dimension), dtype=np.complex128)
        marked_array = np.asarray(marked, dtype=np.int64)
        for _ in range(iterations):
            dense_iteration(dense, marked_array)

        with GroverSearch(qubits, marked) as search:
            search.iterate(iterations)
            compressed = np.asarray(
                [search.amplitude(index) for index in range(dimension)],
                dtype=np.complex128,
            )
            amplitude_checks += dimension
            error = float(np.max(np.abs(compressed - dense)))
            if error > 2e-10:
                raise AssertionError(
                    f"Grover differential mismatch case={case} qubits={qubits} "
                    f"marked={marked_count} iterations={iterations} max_error={error}"
                )
            expected_success = float(np.sum(np.abs(dense[marked_array]) ** 2))
            if abs(search.success_probability - expected_success) > 2e-10:
                raise AssertionError("Grover success probability mismatch")

        if qubits <= 8 and case < 60:
            with QubitRegister(qubits) as exact:
                for qubit in range(qubits):
                    exact.h(qubit)
                exact.grover_iterations(marked, iterations)
                exact_values = np.asarray(
                    [exact.amplitude(index) for index in range(dimension)],
                    dtype=np.complex128,
                )
                error = float(np.max(np.abs(exact_values - dense)))
                if error > 2e-10:
                    raise AssertionError(
                        f"QRegister Grover mismatch case={case} max_error={error}"
                    )

    print(
        f"Grover differential validation passed: {cases} randomized searches, "
        f"{amplitude_checks} amplitude checks."
    )

if __name__ == "__main__":
    main()
