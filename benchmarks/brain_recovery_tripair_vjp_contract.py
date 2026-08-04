from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_adjoint import (
    CausalComponentWeightedAdjoint,
    CausalWeightedAdjoint,
)
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift

from brain_recovery_weighted_adjoint_locality import (
    Operation,
    Observable,
    dense_reverse_vjp,
    finite_difference_vjp,
    maximum_error,
    median_ms,
    source_identity,
)

BRAIN_PARENT = "f9a183e7bdbffc3d838e8774da548a053f883152"
QSA_SOURCE = "78cc67d6c52469486976bb75e9471b87da1fc7fd"
TRIPAIR_INPUT_VALUES = 6
TRIPAIR_SHARED_VALUES = 12
TRIPAIR_TOTAL_VALUES = TRIPAIR_INPUT_VALUES + TRIPAIR_SHARED_VALUES
TRIPAIR_OBSERVABLE_NAMES = (
    "X0",
    "Y0",
    "Z0",
    "X1",
    "Y1",
    "Z1",
    "X2",
    "Y2",
    "Z2",
    "X0 X1",
    "Z0 Z1",
    "X1 X2",
    "Z1 Z2",
    "X0 X1 X2",
)
TRIPAIR_OBSERVABLES: tuple[Observable, ...] = (
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
)


def tripair_spec() -> tuple[Operation, ...]:
    operations: list[Operation] = []
    for qubit in range(3):
        operations.append(("ry", qubit, qubit))
        operations.append(("rz", qubit, 3 + qubit))
    operations.extend(
        (
            ("cnot", 0, 1),
            ("cnot", 1, 2),
            ("cnot", 2, 0),
        )
    )
    slot = TRIPAIR_INPUT_VALUES
    for _layer in range(2):
        for qubit in range(3):
            operations.append(("ry", qubit, slot))
            slot += 1
            operations.append(("rz", qubit, slot))
            slot += 1
        operations.extend(
            (
                ("cnot", 0, 1),
                ("cnot", 1, 2),
            )
        )
    if slot != TRIPAIR_TOTAL_VALUES:
        raise RuntimeError("Tripair parameter surface changed")
    return tuple(operations)


def tripair_plan(spec: Sequence[Operation]) -> CausalParameterizedPlan:
    operations = []
    parameters = [Parameter(f"tripair_{index}") for index in range(TRIPAIR_TOTAL_VALUES)]
    for name, first, third in spec:
        if name in ("ry", "rz"):
            if third is None:
                raise RuntimeError("parameterized operation lacks a slot")
            operations.append((name, first, parameters[int(third)]))
        elif name == "cnot":
            if third is None:
                raise RuntimeError("CNOT lacks a target")
            operations.append((name, first, int(third)))
        else:
            raise ValueError(name)
    return CausalParameterizedPlan(tuple(operations))


def shared_values() -> tuple[float, ...]:
    return tuple(
        0.17 * math.sin((index + 1) * 0.43)
        - 0.09 * math.cos((index + 1) * 0.27)
        for index in range(TRIPAIR_SHARED_VALUES)
    )


def sample_values(sample: int, shared: Sequence[float]) -> tuple[float, ...]:
    inputs = tuple(
        math.pi
        * math.tanh(
            0.31 * math.sin((sample + 1) * (index + 2) * 0.17)
            + 0.13 * math.cos((sample + 3) * (index + 1) * 0.11)
        )
        for index in range(TRIPAIR_INPUT_VALUES)
    )
    return inputs + tuple(float(value) for value in shared)


def sample_cotangent(sample: int) -> tuple[float, ...]:
    return tuple(
        0.19 * math.cos((sample + 1) * (index + 1) * 0.23)
        - 0.08 * math.sin((sample + 2) * (index + 1) * 0.13)
        for index in range(len(TRIPAIR_OBSERVABLES))
    )


def batch_inputs(batch_size: int) -> tuple[tuple[float, ...], ...]:
    shared = shared_values()
    return tuple(sample_values(sample, shared) for sample in range(batch_size))


def batch_cotangents(batch_size: int) -> tuple[tuple[float, ...], ...]:
    return tuple(sample_cotangent(sample) for sample in range(batch_size))


def combine_gradients(
    gradients: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    input_gradients = tuple(
        tuple(float(value) for value in gradient[:TRIPAIR_INPUT_VALUES])
        for gradient in gradients
    )
    shared_gradient = tuple(
        float(
            sum(
                gradient[TRIPAIR_INPUT_VALUES + index]
                for gradient in gradients
            )
        )
        for index in range(TRIPAIR_SHARED_VALUES)
    )
    return input_gradients, shared_gradient


def bundle_error(
    left: tuple[tuple[tuple[float, ...], ...], tuple[float, ...]],
    right: tuple[tuple[tuple[float, ...], ...], tuple[float, ...]],
) -> float:
    left_inputs, left_shared = left
    right_inputs, right_shared = right
    errors = [maximum_error(left_shared, right_shared)]
    errors.extend(
        maximum_error(left_row, right_row)
        for left_row, right_row in zip(left_inputs, right_inputs)
    )
    return max(errors)


def root_identity(root: CausalRegister) -> bytes:
    return bytes(root.encode_qsc())


def evaluate_adjoint_batch(
    root: CausalRegister,
    adjoint: CausalWeightedAdjoint,
    values: Sequence[Sequence[float]],
    cotangents: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...]],
    dict[str, int],
]:
    outputs = []
    gradients = []
    forward = reverse = shifted = 0
    for row_values, row_cotangent in zip(values, cotangents):
        result = adjoint.evaluate(root, row_values, row_cotangent)
        outputs.append(tuple(float(value) for value in result.values))
        gradients.append(tuple(float(value) for value in result.gradient))
        forward += int(result.forward_sweeps)
        reverse += int(result.reverse_sweeps)
        shifted += int(result.shifted_evaluations)
    return (
        tuple(outputs),
        combine_gradients(gradients),
        {
            "forward_sweeps": forward,
            "reverse_sweeps": reverse,
            "shifted_evaluations": shifted,
        },
    )


def evaluate_component_batch(
    root: CausalRegister,
    adjoint: CausalComponentWeightedAdjoint,
    values: Sequence[Sequence[float]],
    cotangents: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...]],
    dict[str, object],
]:
    outputs = []
    gradients = []
    local_widths: list[int] = []
    global_qubits: list[list[int]] = []
    forward = reverse = shifted = 0
    for row_values, row_cotangent in zip(values, cotangents):
        result = adjoint.evaluate(root, row_values, row_cotangent)
        outputs.append(tuple(float(value) for value in result.values))
        gradients.append(tuple(float(value) for value in result.gradient))
        local_widths.append(int(result.local_qubit_count))
        global_qubits.append([int(value) for value in result.global_qubits])
        forward += int(result.forward_sweeps)
        reverse += int(result.reverse_sweeps)
        shifted += int(result.shifted_evaluations)
    return (
        tuple(outputs),
        combine_gradients(gradients),
        {
            "forward_sweeps": forward,
            "reverse_sweeps": reverse,
            "shifted_evaluations": shifted,
            "local_widths": local_widths,
            "global_qubits": global_qubits,
        },
    )


def evaluate_shift_batch(
    root: CausalRegister,
    shift: CausalSupportParameterShift,
    values: Sequence[Sequence[float]],
    cotangents: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...]],
    int,
]:
    outputs = []
    gradients = []
    evaluations = 0
    for row_values, row_cotangent in zip(values, cotangents):
        result = shift.evaluate_and_jacobian(root, row_values, workers=1)
        outputs.append(tuple(float(value) for value in result.values))
        gradients.append(tuple(float(value) for value in result.vjp(row_cotangent)))
        evaluations += 1 + 2 * TRIPAIR_TOTAL_VALUES
    return tuple(outputs), combine_gradients(gradients), evaluations


def evaluate_finite_batch(
    root: CausalRegister,
    plan: CausalParameterizedPlan,
    support: CausalPauliSupportPlan,
    values: Sequence[Sequence[float]],
    cotangents: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...]],
    int,
]:
    outputs = []
    gradients = []
    evaluations = 0
    for row_values, row_cotangent in zip(values, cotangents):
        observed, gradient = finite_difference_vjp(
            root,
            plan,
            support,
            row_values,
            row_cotangent,
        )
        outputs.append(tuple(float(value) for value in observed))
        gradients.append(tuple(float(value) for value in gradient))
        evaluations += 1 + 2 * TRIPAIR_TOTAL_VALUES
    return tuple(outputs), combine_gradients(gradients), evaluations


def evaluate_dense_batch(
    spec: Sequence[Operation],
    values: Sequence[Sequence[float]],
    cotangents: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...]],
    int,
]:
    outputs = []
    gradients = []
    working_bytes = 0
    for row_values, row_cotangent in zip(values, cotangents):
        observed, gradient, local_bytes = dense_reverse_vjp(
            3,
            spec,
            TRIPAIR_OBSERVABLES,
            row_values,
            row_cotangent,
        )
        outputs.append(tuple(float(value) for value in observed))
        gradients.append(tuple(float(value) for value in gradient))
        working_bytes = max(working_bytes, int(local_bytes))
    return tuple(outputs), combine_gradients(gradients), working_bytes


def rebuild_component_batch(
    root: CausalRegister,
    spec: Sequence[Operation],
    values: Sequence[Sequence[float]],
    cotangents: Sequence[Sequence[float]],
) -> None:
    for row_values, row_cotangent in zip(values, cotangents):
        plan = tripair_plan(spec)
        support = CausalPauliSupportPlan(3, TRIPAIR_OBSERVABLES)
        component = CausalComponentWeightedAdjoint(
            plan,
            support,
            max_local_qubits=3,
        )
        try:
            component.evaluate(root, row_values, row_cotangent)
        finally:
            component.close()
            support.close()
            plan.close()


def compare_output_batches(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> float:
    return max(
        maximum_error(left_row, right_row)
        for left_row, right_row in zip(left, right)
    )


def batch_case(
    batch_size: int,
    repeats: int,
    *,
    root: CausalRegister,
    spec: Sequence[Operation],
    plan: CausalParameterizedPlan,
    support: CausalPauliSupportPlan,
    adjoint: CausalWeightedAdjoint,
    component: CausalComponentWeightedAdjoint,
    shift: CausalSupportParameterShift,
) -> dict[str, object]:
    values = batch_inputs(batch_size)
    cotangents = batch_cotangents(batch_size)
    identity_before = root_identity(root)

    direct_outputs, direct_bundle, direct_counts = evaluate_adjoint_batch(
        root, adjoint, values, cotangents
    )
    component_outputs, component_bundle, component_counts = evaluate_component_batch(
        root, component, values, cotangents
    )
    shift_outputs, shift_bundle, shift_evaluations = evaluate_shift_batch(
        root, shift, values, cotangents
    )
    finite_outputs, finite_bundle, finite_evaluations = evaluate_finite_batch(
        root, plan, support, values, cotangents
    )
    dense_outputs, dense_bundle, dense_working_bytes = evaluate_dense_batch(
        spec, values, cotangents
    )

    if root_identity(root) != identity_before:
        raise RuntimeError("Tripair VJP contract mutated the root register")

    direct_ms = median_ms(
        lambda: evaluate_adjoint_batch(root, adjoint, values, cotangents),
        repeats,
    )
    component_ms = median_ms(
        lambda: evaluate_component_batch(root, component, values, cotangents),
        repeats,
    )
    shift_ms = median_ms(
        lambda: evaluate_shift_batch(root, shift, values, cotangents),
        max(3, repeats // 2),
    )
    finite_ms = median_ms(
        lambda: evaluate_finite_batch(
            root, plan, support, values, cotangents
        ),
        max(3, repeats // 2),
    )
    dense_ms = median_ms(
        lambda: evaluate_dense_batch(spec, values, cotangents),
        repeats,
    )
    rebuild_ms = median_ms(
        lambda: rebuild_component_batch(root, spec, values, cotangents),
        max(3, repeats // 2),
    )

    errors = {
        "component_output_vs_direct": compare_output_batches(
            component_outputs, direct_outputs
        ),
        "shift_output_vs_direct": compare_output_batches(
            shift_outputs, direct_outputs
        ),
        "finite_output_vs_direct": compare_output_batches(
            finite_outputs, direct_outputs
        ),
        "dense_output_vs_direct": compare_output_batches(
            dense_outputs, direct_outputs
        ),
        "component_gradient_bundle_vs_direct": bundle_error(
            component_bundle, direct_bundle
        ),
        "shift_gradient_bundle_vs_direct": bundle_error(
            shift_bundle, direct_bundle
        ),
        "finite_gradient_bundle_vs_direct": bundle_error(
            finite_bundle, direct_bundle
        ),
        "dense_gradient_bundle_vs_direct": bundle_error(
            dense_bundle, direct_bundle
        ),
    }

    row: dict[str, object] = {
        "batch_size": batch_size,
        "logical_qubits_per_component": 3,
        "independent_component_count": batch_size,
        "input_values_per_sample": TRIPAIR_INPUT_VALUES,
        "shared_trainable_values": TRIPAIR_SHARED_VALUES,
        "observable_cotangents_per_sample": len(TRIPAIR_OBSERVABLES),
        "persistent_direct_adjoint_ms": direct_ms,
        "persistent_component_adjoint_ms": component_ms,
        "parameter_shift_ms": shift_ms,
        "finite_difference_ms": finite_ms,
        "independent_dense_reverse_ms": dense_ms,
        "rebuild_component_adjoint_ms": rebuild_ms,
        "parameter_shift_over_component_adjoint": shift_ms / component_ms,
        "finite_difference_over_component_adjoint": finite_ms / component_ms,
        "independent_dense_reverse_over_component_adjoint": dense_ms
        / component_ms,
        "rebuild_over_persistent_component_adjoint": rebuild_ms / component_ms,
        "component_adjoint_ms_per_sample": component_ms / batch_size,
        "parameter_shift_evaluations": shift_evaluations,
        "finite_difference_evaluations": finite_evaluations,
        "component_forward_sweeps": component_counts["forward_sweeps"],
        "component_reverse_sweeps": component_counts["reverse_sweeps"],
        "component_shifted_evaluations": component_counts[
            "shifted_evaluations"
        ],
        "component_local_widths": component_counts["local_widths"],
        "component_global_qubits": component_counts["global_qubits"],
        "direct_counts": direct_counts,
        "dense_reverse_working_state_bytes_per_sample": dense_working_bytes,
        "root_qsc_bytes": len(identity_before),
        "root_immutable": root_identity(root) == identity_before,
        "errors": errors,
    }
    row["passed"] = (
        row["root_immutable"] is True
        and component_counts["forward_sweeps"] == batch_size
        and component_counts["reverse_sweeps"] == batch_size
        and component_counts["shifted_evaluations"] == 0
        and component_counts["local_widths"] == [3] * batch_size
        and component_counts["global_qubits"] == [[0, 1, 2]] * batch_size
        and max(
            float(errors["component_output_vs_direct"]),
            float(errors["component_gradient_bundle_vs_direct"]),
        )
        <= 2.0e-12
        and max(
            float(errors["shift_output_vs_direct"]),
            float(errors["shift_gradient_bundle_vs_direct"]),
        )
        <= 2.0e-12
        and max(
            float(errors["finite_output_vs_direct"]),
        )
        <= 2.0e-12
        and float(errors["finite_gradient_bundle_vs_direct"]) <= 2.0e-7
        and max(
            float(errors["dense_output_vs_direct"]),
            float(errors["dense_gradient_bundle_vs_direct"]),
        )
        <= 2.0e-12
        and shift_ms > component_ms
        and finite_ms > component_ms
    )
    return row


def build_receipt(repeats: int) -> dict[str, object]:
    identity = source_identity()
    spec = tripair_spec()
    plan = tripair_plan(spec)
    support = CausalPauliSupportPlan(3, TRIPAIR_OBSERVABLES)
    adjoint = CausalWeightedAdjoint(plan, support, max_qubits=3)
    component = CausalComponentWeightedAdjoint(
        plan,
        support,
        max_local_qubits=3,
    )
    shift = CausalSupportParameterShift(plan, support)
    rows = []
    try:
        with CausalRegister(3) as root:
            for batch_size in (1, 4, 16, 64):
                rows.append(
                    batch_case(
                        batch_size,
                        repeats,
                        root=root,
                        spec=spec,
                        plan=plan,
                        support=support,
                        adjoint=adjoint,
                        component=component,
                        shift=shift,
                    )
                )
    finally:
        component.close()
        support.close()
        plan.close()

    receipt: dict[str, object] = {
        "schema_version": "qsa.brain-recovery-tripair-vjp-contract.v1",
        "source_identity": identity,
        "brain_contract": {
            "accepted_parent": BRAIN_PARENT,
            "qsa_source": QSA_SOURCE,
            "active_systems": ["language", "qelm", "brainq"],
            "feature_width": 256,
            "input_angle_values_per_sample": TRIPAIR_INPUT_VALUES,
            "shared_trainable_circuit_values": TRIPAIR_SHARED_VALUES,
            "total_qsa_values_per_sample": TRIPAIR_TOTAL_VALUES,
            "logical_qubits": 3,
            "trainable_layers": 2,
            "observable_names": list(TRIPAIR_OBSERVABLE_NAMES),
            "observable_count": len(TRIPAIR_OBSERVABLES),
            "component_policy": (
                "One independent fully connected three-qubit component per "
                "sample; components never merge across samples."
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "benchmark_policy": {
            "repeats": repeats,
            "timing_statistic": "median",
            "gpu_used": False,
            "plans_reused_in_candidate": True,
            "root_state_reused_and_verified_immutable": True,
            "batch_gradient_contract": (
                "Six input-angle gradients are retained per sample; the "
                "twelve shared circuit-angle gradients are summed "
                "deterministically across the batch."
            ),
            "dense_reverse_reference_limitation": (
                "Independent NumPy/Python exact reverse mode is an exact "
                "mathematical control, not an optimized library claim."
            ),
        },
        "batch_sweep": rows,
        "classifications_if_passed": [
            "REPRODUCED",
            "USEFUL_NONUNIQUE",
            "QUANTUM_NATIVE_STRUCTURAL",
        ],
        "not_established": [
            "complete AdamW optimizer-step advantage",
            "PyTorch custom-autograd integration",
            "Brain capability contribution",
            "quantum uniqueness",
            "physical-QPU advantage",
            "production readiness",
        ],
    }
    receipt["passed"] = identity["passed"] and all(
        bool(row["passed"]) for row in rows
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-vjp-contract.json"),
    )
    parser.add_argument("--repeats", type=int, default=5)
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
        raise SystemExit("Brain Tripair custom-VJP contract failed")


if __name__ == "__main__":
    main()
