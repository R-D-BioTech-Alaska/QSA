from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import brain_recovery_tripair_vjp_contract as v1


BRAIN_INPUT_ORDER = (
    "theta_0",
    "theta_1",
    "theta_2",
    "phi_0",
    "phi_1",
    "phi_2",
)
QSA_GATE_ORDER = (
    "theta_0",
    "phi_0",
    "theta_1",
    "phi_1",
    "theta_2",
    "phi_2",
)
BRAIN_TO_QSA = (0, 3, 1, 4, 2, 5)
QSA_TO_BRAIN = (0, 2, 4, 1, 3, 5)


def corrected_tripair_spec() -> tuple[v1.Operation, ...]:
    """Use sequential native slots in exact gate-execution order.

    Brain emits the six input angles as theta[0:3] followed by phi[0:3], while
    the circuit executes RY(theta_q) then RZ(phi_q) for each qubit. QSA assigns
    parameter slots by first occurrence. Version 1 incorrectly passed the Brain
    tensor order as though it were QSA's first-occurrence order. This successor
    makes the binding permutation explicit and keeps the physical gate sequence
    unchanged.
    """

    operations: list[v1.Operation] = []
    slot = 0
    for qubit in range(3):
        operations.append(("ry", qubit, slot))
        slot += 1
        operations.append(("rz", qubit, slot))
        slot += 1
    operations.extend(
        (("cnot", 0, 1), ("cnot", 1, 2), ("cnot", 2, 0))
    )
    for _layer in range(2):
        for qubit in range(3):
            operations.append(("ry", qubit, slot))
            slot += 1
            operations.append(("rz", qubit, slot))
            slot += 1
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    if slot != v1.TOTAL_VALUES:
        raise RuntimeError("Tripair parameter surface changed")
    return tuple(operations)


def corrected_values_for_sample(
    sample: int,
    shared: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    brain_values = tuple(
        math.pi
        * math.tanh(
            0.31 * math.sin((sample + 1) * (index + 2) * 0.17)
            + 0.13 * math.cos((sample + 3) * (index + 1) * 0.11)
        )
        for index in range(v1.INPUT_VALUES)
    )
    qsa_values = tuple(brain_values[index] for index in BRAIN_TO_QSA)
    return qsa_values + tuple(float(value) for value in shared)


def build_receipt(repeats: int) -> dict[str, object]:
    original_spec = v1.tripair_spec
    original_values = v1.values_for_sample
    v1.tripair_spec = corrected_tripair_spec
    v1.values_for_sample = corrected_values_for_sample
    try:
        receipt = v1.build_receipt(repeats)
    finally:
        v1.tripair_spec = original_spec
        v1.values_for_sample = original_values

    receipt["schema_version"] = "qsa.brain-recovery-tripair-vjp-contract.v2"
    receipt["iteration"] = {
        "predecessor": "qsa.brain-recovery-tripair-vjp-contract.v1",
        "predecessor_workflow_run": 30940091240,
        "predecessor_artifact_id": 8904753507,
        "predecessor_artifact_zip_sha256": (
            "69de005591f11c4033fd1a5cfb1664da6b28cf2f0a7e653c2e3c14cc9416e25b"
        ),
        "causal_failure": "parameter_binding_order_mismatch",
        "failure_detail": (
            "Version 1 built the exact interleaved Tripair gate sequence but "
            "passed the six Brain tensor values in theta-then-phi order. QSA "
            "correctly binds named plan parameters by first occurrence, so its "
            "native circuit and the independent dense reference received "
            "different values. QSA adjoint, component adjoint, parameter shift, "
            "and finite difference all agreed in version 1; only the differently "
            "bound dense reference disagreed."
        ),
        "successor_change": (
            "Preserve the physical circuit while explicitly permuting Brain's "
            "theta-then-phi tensor order into QSA first-occurrence gate order."
        ),
    }
    receipt["brain_contract"]["input_binding"] = {
        "brain_tensor_order": list(BRAIN_INPUT_ORDER),
        "qsa_first_occurrence_order": list(QSA_GATE_ORDER),
        "brain_to_qsa_indices": list(BRAIN_TO_QSA),
        "qsa_gradient_to_brain_indices": list(QSA_TO_BRAIN),
        "physical_gate_sequence_changed": False,
        "trainable_shared_angle_order_changed": False,
    }
    receipt["benchmark_policy"]["failed_iteration_preserved"] = True
    receipt["benchmark_policy"]["binding_permutation_predeclared"] = True
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-vjp-contract-v2.json"),
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
        raise SystemExit("Brain Tripair custom-VJP contract v2 failed")


if __name__ == "__main__":
    main()
