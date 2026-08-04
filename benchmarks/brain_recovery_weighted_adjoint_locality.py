from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_adjoint import CausalComponentWeightedAdjoint, CausalWeightedAdjoint
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift

SOURCE_COMMIT = "78cc67d6c52469486976bb75e9471b87da1fc7fd"
SOURCE_BLOBS = {
    "python/qsa/causal_adjoint.py": "030f09c057df5e052d8cadcec1abda960792e398",
    "python/qsa/causal_components.py": "5ae17971969a6ab7d7e64fd0cca3e6af4524a38e",
    "include/qubit/qadjoint.hpp": "13ddc3461d730e41c1bf58682a2fd3f0ae040191",
    "include/qubit/detail/qcausal_component_impl.hpp": "af962cf7382b5ae5ce5fedfaf31475db09e496c4",
    "tests/test_causal_adjoint_python.py": "0c57f6bb7674a0749df4f598c713bebbe25eab04",
    "benchmarks/causal_adjoint_python.py": "1ee96d13db7c2f1571a67e77f57a67de880d6f3e",
}

Operation = tuple[str, int, int | None]
Observable = tuple[tuple[int, str], ...]


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_identity() -> dict:
    actual = {path: git_output("hash-object", path) for path in SOURCE_BLOBS}
    main_ref = None
    for candidate in ("origin/main", "main"):
        try:
            git_output("rev-parse", "--verify", candidate)
        except subprocess.CalledProcessError:
            continue
        main_ref = candidate
        break
    merge_base = git_output("merge-base", "HEAD", main_ref) if main_ref else None
    try:
        git_output("cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}")
        source_commit_present = True
    except subprocess.CalledProcessError:
        source_commit_present = False
    return {
        "expected_base_commit": SOURCE_COMMIT,
        "head": git_output("rev-parse", "HEAD"),
        "main_ref": main_ref,
        "merge_base_with_main": merge_base,
        "source_commit_present": source_commit_present,
        "expected_blobs": SOURCE_BLOBS,
        "actual_blobs": actual,
        "passed": actual == SOURCE_BLOBS
        and source_commit_present
        and merge_base == SOURCE_COMMIT,
    }


def rotation_spec(parameter_count: int, qubits: int = 4) -> list[Operation]:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    operations: list[Operation] = []
    for slot in range(parameter_count):
        qubit = slot % qubits
        gate = "ry" if ((slot // qubits) + qubit) % 2 == 0 else "rz"
        operations.append((gate, qubit, slot))
        if (slot + 1) % qubits == 0:
            layer = (slot + 1) // qubits
            start = layer % qubits
            for offset in range(qubits - 1):
                control = (start + offset) % qubits
                target = (control + 1) % qubits
                operations.append(("cnot", control, target))
    return operations


def qsa_plan(spec: Sequence[Operation]) -> CausalParameterizedPlan:
    operations = []
    for name, first, third in spec:
        if name in ("ry", "rz"):
            assert third is not None
            operations.append((name, first, Parameter(f"theta_{third}")))
        elif name == "cnot":
            assert third is not None
            operations.append((name, first, third))
        else:
            raise ValueError(f"unsupported operation {name}")
    return CausalParameterizedPlan(tuple(operations))


def observables_for_width(
    qubit_count: int,
    active_qubits: int = 4,
) -> tuple[Observable, ...]:
    observables: list[Observable] = []
    for qubit in range(active_qubits):
        observables.append(((qubit, "X"),))
        observables.append(((qubit, "Z"),))
    for qubit in range(active_qubits - 1):
        observables.append(((qubit, "X"), (qubit + 1, "X")))
        observables.append(((qubit, "Z"), (qubit + 1, "Z")))
    if qubit_count < active_qubits:
        raise ValueError("qubit_count smaller than active support")
    return tuple(observables)


def parameter_values(parameter_count: int) -> tuple[float, ...]:
    return tuple(
        0.29 * math.sin((index + 1) * 0.37)
        + 0.11 * math.cos((index + 1) * 0.19)
        for index in range(parameter_count)
    )


def cotangent_values(observable_count: int) -> tuple[float, ...]:
    return tuple(
        0.23 * math.cos((index + 1) * 0.41)
        - 0.07 * math.sin((index + 1) * 0.17)
        for index in range(observable_count)
    )


def maximum_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def median_ms(function: Callable[[], object], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(statistics.median(samples))


def weighted_objective(
    root: CausalRegister,
    plan: CausalParameterizedPlan,
    support: CausalPauliSupportPlan,
    values: Sequence[float],
    cotangent: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    with root.fork() as branch:
        branch.apply(plan, values)
        observed = tuple(support.execute(branch))
    objective = sum(
        float(weight) * float(value)
        for weight, value in zip(cotangent, observed)
    )
    return observed, objective


def finite_difference_vjp(
    root: CausalRegister,
    plan: CausalParameterizedPlan,
    support: CausalPauliSupportPlan,
    values: Sequence[float],
    cotangent: Sequence[float],
    epsilon: float = 1.0e-6,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    base_values, _ = weighted_objective(root, plan, support, values, cotangent)
    gradient = []
    for index in range(len(values)):
        positive = list(values)
        negative = list(values)
        positive[index] += epsilon
        negative[index] -= epsilon
        _, positive_objective = weighted_objective(
            root, plan, support, positive, cotangent
        )
        _, negative_objective = weighted_objective(
            root, plan, support, negative, cotangent
        )
        gradient.append(
            (positive_objective - negative_objective) / (2.0 * epsilon)
        )
    return base_values, tuple(gradient)


def single_matrix(name: str, theta: float) -> np.ndarray:
    half = theta / 2.0
    if name == "ry":
        return np.array(
            [
                [math.cos(half), -math.sin(half)],
                [math.sin(half), math.cos(half)],
            ],
            dtype=np.complex128,
        )
    if name == "rz":
        return np.array(
            [
                [np.exp(-0.5j * theta), 0.0],
                [0.0, np.exp(0.5j * theta)],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"unsupported rotation {name}")


def derivative_matrix(name: str, theta: float) -> np.ndarray:
    half = theta / 2.0
    cosine = math.cos(half)
    sine = math.sin(half)
    if name == "ry":
        return np.array(
            [
                [-0.5 * sine, -0.5 * cosine],
                [0.5 * cosine, -0.5 * sine],
            ],
            dtype=np.complex128,
        )
    if name == "rz":
        zero = np.exp(-0.5j * theta)
        one = np.exp(0.5j * theta)
        return np.array(
            [[-0.5j * zero, 0.0], [0.0, 0.5j * one]],
            dtype=np.complex128,
        )
    raise ValueError(f"unsupported derivative {name}")


def apply_single(state: np.ndarray, qubit: int, matrix: np.ndarray) -> None:
    mask = 1 << qubit
    for base in range(state.size):
        if base & mask:
            continue
        one = base | mask
        zero_value = state[base]
        one_value = state[one]
        state[base] = matrix[0, 0] * zero_value + matrix[0, 1] * one_value
        state[one] = matrix[1, 0] * zero_value + matrix[1, 1] * one_value


def apply_cnot(state: np.ndarray, control: int, target: int) -> None:
    control_mask = 1 << control
    target_mask = 1 << target
    for basis in range(state.size):
        if not (basis & control_mask) or (basis & target_mask):
            continue
        paired = basis | target_mask
        state[basis], state[paired] = state[paired], state[basis]


def apply_operation(
    state: np.ndarray,
    operation: Operation,
    values: Sequence[float],
    reverse: bool,
) -> None:
    name, first, third = operation
    if name in ("ry", "rz"):
        assert third is not None
        matrix = single_matrix(name, float(values[third]))
        if reverse:
            matrix = matrix.conj().T
        apply_single(state, first, matrix)
        return
    if name == "cnot":
        assert third is not None
        apply_cnot(state, first, third)
        return
    raise ValueError(f"unsupported operation {name}")


def pauli_action(state: np.ndarray, observable: Observable) -> np.ndarray:
    output = np.zeros_like(state)
    for source in range(state.size):
        target = source
        coefficient = 1.0 + 0.0j
        for qubit, axis in observable:
            one = bool((source >> qubit) & 1)
            if axis == "X":
                target ^= 1 << qubit
            elif axis == "Y":
                target ^= 1 << qubit
                coefficient *= -1.0j if one else 1.0j
            elif axis == "Z":
                if one:
                    coefficient *= -1.0
            else:
                raise ValueError(f"invalid Pauli axis {axis}")
        output[target] += coefficient * state[source]
    return output


def dense_reverse_vjp(
    qubit_count: int,
    spec: Sequence[Operation],
    observables: Sequence[Observable],
    values: Sequence[float],
    cotangent: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    primal = np.zeros(1 << qubit_count, dtype=np.complex128)
    primal[0] = 1.0
    for operation in spec:
        apply_operation(primal, operation, values, reverse=False)

    observed = tuple(
        float(np.vdot(primal, pauli_action(primal, observable)).real)
        for observable in observables
    )
    adjoint = np.zeros_like(primal)
    for weight, observable in zip(cotangent, observables):
        if weight:
            adjoint += float(weight) * pauli_action(primal, observable)

    gradient = np.zeros(len(values), dtype=np.float64)
    for operation in reversed(spec):
        name, qubit, slot = operation
        apply_operation(primal, operation, values, reverse=True)
        if name in ("ry", "rz"):
            assert slot is not None
            derivative_state = primal.copy()
            apply_single(
                derivative_state,
                qubit,
                derivative_matrix(name, float(values[slot])),
            )
            gradient[slot] += 2.0 * float(
                np.vdot(adjoint, derivative_state).real
            )
        apply_operation(adjoint, operation, values, reverse=True)
    return (
        observed,
        tuple(float(value) for value in gradient),
        int(primal.nbytes + adjoint.nbytes),
    )


def gradient_sweep(repeats: int) -> list[dict]:
    rows = []
    for parameter_count in (4, 8, 16, 32):
        spec = rotation_spec(parameter_count)
        plan = qsa_plan(spec)
        observables = observables_for_width(4)
        support = CausalPauliSupportPlan(4, observables)
        shift = CausalSupportParameterShift(plan, support)
        adjoint = CausalWeightedAdjoint(plan, support, max_qubits=8)
        values = parameter_values(parameter_count)
        cotangent = cotangent_values(len(observables))

        with CausalRegister(4) as root:
            root_receipt = root.encode_qsc()
            shifted = shift.evaluate_and_jacobian(root, values, workers=1)
            exact = adjoint.evaluate(root, values, cotangent)
            finite_values, finite_gradient = finite_difference_vjp(
                root, plan, support, values, cotangent
            )
            dense_values, dense_gradient, dense_bytes = dense_reverse_vjp(
                4, spec, observables, values, cotangent
            )
            if root.encode_qsc() != root_receipt:
                raise RuntimeError("gradient sweep mutated the root state")

            shift_ms = median_ms(
                lambda: shift.evaluate_and_jacobian(
                    root, values, workers=1
                ).vjp(cotangent),
                repeats,
            )
            adjoint_ms = median_ms(
                lambda: adjoint.evaluate(root, values, cotangent), repeats
            )
            finite_ms = median_ms(
                lambda: finite_difference_vjp(
                    root, plan, support, values, cotangent
                ),
                max(3, repeats // 2),
            )
            dense_ms = median_ms(
                lambda: dense_reverse_vjp(
                    4, spec, observables, values, cotangent
                ),
                repeats,
            )

        shift_gradient = shifted.vjp(cotangent)
        row = {
            "parameters": parameter_count,
            "operations": len(spec),
            "observables": len(observables),
            "qsa_adjoint_ms": adjoint_ms,
            "qsa_parameter_shift_ms": shift_ms,
            "finite_difference_ms": finite_ms,
            "dense_reverse_reference_ms": dense_ms,
            "parameter_shift_over_adjoint": shift_ms / adjoint_ms,
            "finite_difference_over_adjoint": finite_ms / adjoint_ms,
            "dense_reverse_over_adjoint": dense_ms / adjoint_ms,
            "parameter_shift_evaluations": 1 + 2 * parameter_count,
            "finite_difference_evaluations": 1 + 2 * parameter_count,
            "adjoint_forward_sweeps": exact.forward_sweeps,
            "adjoint_reverse_sweeps": exact.reverse_sweeps,
            "adjoint_shifted_evaluations": exact.shifted_evaluations,
            "max_value_error_vs_parameter_shift": maximum_error(
                exact.values, shifted.values
            ),
            "max_gradient_error_vs_parameter_shift": maximum_error(
                exact.gradient, shift_gradient
            ),
            "max_value_error_vs_finite_difference": maximum_error(
                exact.values, finite_values
            ),
            "max_gradient_error_vs_finite_difference": maximum_error(
                exact.gradient, finite_gradient
            ),
            "max_value_error_vs_dense_reverse": maximum_error(
                exact.values, dense_values
            ),
            "max_gradient_error_vs_dense_reverse": maximum_error(
                exact.gradient, dense_gradient
            ),
            "dense_reverse_working_state_bytes": dense_bytes,
        }
        row["passed"] = (
            row["adjoint_forward_sweeps"] == 1
            and row["adjoint_reverse_sweeps"] == 1
            and row["adjoint_shifted_evaluations"] == 0
            and row["max_value_error_vs_parameter_shift"] <= 2.0e-12
            and row["max_gradient_error_vs_parameter_shift"] <= 2.0e-12
            and row["max_value_error_vs_finite_difference"] <= 2.0e-12
            and row["max_gradient_error_vs_finite_difference"] <= 3.0e-7
            and row["max_value_error_vs_dense_reverse"] <= 2.0e-12
            and row["max_gradient_error_vs_dense_reverse"] <= 2.0e-12
            and row["parameter_shift_over_adjoint"] > 1.0
            and row["finite_difference_over_adjoint"] > 1.0
        )
        rows.append(row)
        support.close()
        plan.close()
    return rows


def component_locality_sweep(repeats: int) -> tuple[list[dict], dict]:
    active_qubits = 4
    parameter_count = 16
    spec = rotation_spec(parameter_count, active_qubits)
    plan = qsa_plan(spec)
    values = parameter_values(parameter_count)
    rows = []

    for total_qubits in (8, 10, 12, 14, 16):
        observables = observables_for_width(total_qubits, active_qubits)
        support = CausalPauliSupportPlan(total_qubits, observables)
        cotangent = cotangent_values(len(observables))
        forced = CausalWeightedAdjoint(plan, support, max_qubits=16)
        component = CausalComponentWeightedAdjoint(
            plan, support, max_local_qubits=16
        )
        with CausalRegister(total_qubits) as root:
            initial = root.encode_qsc()
            forced_result = forced.evaluate(root, values, cotangent)
            component_result = component.evaluate(root, values, cotangent)
            if root.encode_qsc() != initial:
                raise RuntimeError("component sweep mutated the root state")
            forced_ms = median_ms(
                lambda: forced.evaluate(root, values, cotangent),
                max(3, repeats // 2),
            )
            component_ms = median_ms(
                lambda: component.evaluate(root, values, cotangent), repeats
            )
            root_qsc_bytes = len(initial)

        local_qubits = component_result.local_qubit_count
        forced_bytes = (
            2
            * (1 << total_qubits)
            * np.dtype(np.complex128).itemsize
        )
        local_bytes = (
            2 * (1 << local_qubits) * np.dtype(np.complex128).itemsize
        )
        row = {
            "total_qubits": total_qubits,
            "local_qubits": local_qubits,
            "global_qubits": list(component_result.global_qubits),
            "root_qsc_bytes": root_qsc_bytes,
            "forced_global_ms": forced_ms,
            "component_local_ms": component_ms,
            "forced_over_component": forced_ms / component_ms,
            "forced_dense_working_state_bytes": forced_bytes,
            "component_dense_working_state_bytes": local_bytes,
            "component_over_forced_working_state_ratio": (
                local_bytes / forced_bytes
            ),
            "max_value_error": maximum_error(
                forced_result.values, component_result.values
            ),
            "max_gradient_error": maximum_error(
                forced_result.gradient, component_result.gradient
            ),
        }
        row["passed"] = (
            row["local_qubits"] == active_qubits
            and row["max_value_error"] <= 2.0e-12
            and row["max_gradient_error"] <= 2.0e-12
            and row["component_over_forced_working_state_ratio"]
            == 2.0 ** (active_qubits - total_qubits)
        )
        rows.append(row)
        component.close()
        support.close()

    total_qubits = 12
    observables = observables_for_width(total_qubits, active_qubits)
    support = CausalPauliSupportPlan(total_qubits, observables)
    cotangent = cotangent_values(len(observables))
    component = CausalComponentWeightedAdjoint(
        plan, support, max_local_qubits=16
    )
    prepare = CausalParameterizedPlan(
        (("h", 0),)
        + tuple(
            ("cnot", index - 1, index)
            for index in range(1, total_qubits)
        )
    )
    with CausalRegister(total_qubits) as root:
        root.apply(prepare, ())
        result = component.evaluate(root, values, cotangent)
    fully_connected = {
        "total_qubits": total_qubits,
        "local_qubits": result.local_qubit_count,
        "global_qubits": list(result.global_qubits),
        "passed": result.local_qubit_count == total_qubits,
        "interpretation": (
            "The component-local path correctly recovers the full global "
            "width after the root state becomes one connected component."
        ),
    }
    component.close()
    prepare.close()
    support.close()
    plan.close()
    return rows, fully_connected


def build_receipt(repeats: int) -> dict:
    identity = source_identity()
    gradients = gradient_sweep(repeats)
    locality, fully_connected = component_locality_sweep(repeats)
    receipt = {
        "schema_version": (
            "qsa.brain-recovery-weighted-adjoint-locality.v1"
        ),
        "source_identity": identity,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "benchmark_policy": {
            "repeats": repeats,
            "timing_statistic": "median",
            "gpu_used": False,
            "complete_process_memory_claimed": False,
            "working_state_bytes_definition": (
                "Two complex128 dense vectors used by the reverse sweep; "
                "excludes allocator, plan, support, serialization, and "
                "process overhead."
            ),
            "dense_reverse_reference_limitation": (
                "Independent NumPy/Python exact reverse-mode reference; "
                "runtime is reported but is not an optimized "
                "classical-library performance claim."
            ),
        },
        "gradient_sweep": gradients,
        "component_locality_sweep": locality,
        "fully_connected_failure_boundary": fully_connected,
    }
    receipt["passed"] = (
        identity["passed"]
        and all(row["passed"] for row in gradients)
        and all(row["passed"] for row in locality)
        and fully_connected["passed"]
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-weighted-adjoint-locality.json"),
    )
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    receipt = build_receipt(args.repeats)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit(
            "weighted adjoint and component locality recovery gate failed"
        )


if __name__ == "__main__":
    main()
