from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v2 as v2


def corrected_apply_pauli(
    state: v1.torch.Tensor,
    paulis: Iterable[tuple[int, str]],
) -> v1.torch.Tensor:
    """Apply supports in the repository's canonical (qubit, axis) order."""

    result = state
    for qubit, axis in paulis:
        bit = 1 << int(qubit)
        indices = v1.torch.arange(1 << v1.QUBITS, device=state.device)
        if axis == "Z":
            phase = v1.torch.where((indices & bit) == 0, 1.0, -1.0).to(
                result.dtype
            )
            result = result * phase.unsqueeze(0)
        elif axis == "X":
            result = result.index_select(1, indices ^ bit)
        elif axis == "Y":
            source = indices ^ bit
            source_bit = (source & bit) != 0
            negative = v1.torch.full(
                source.shape,
                -1j,
                dtype=result.dtype,
                device=result.device,
            )
            positive = v1.torch.full(
                source.shape,
                1j,
                dtype=result.dtype,
                device=result.device,
            )
            phase = v1.torch.where(source_bit, negative, positive)
            result = result.index_select(1, source) * phase.unsqueeze(0)
        else:
            raise ValueError(f"unsupported Pauli axis {axis}")
    return result


def build_receipt(steps: int) -> dict[str, object]:
    original_apply_pauli = v1.apply_pauli
    v1.apply_pauli = corrected_apply_pauli
    try:
        receipt = v2.build_receipt(steps)
    finally:
        v1.apply_pauli = original_apply_pauli

    receipt["schema_version"] = (
        "qsa.brain-recovery-tripair-optimizer-step.v3"
    )
    receipt["iteration"] = {
        "predecessor": "qsa.brain-recovery-tripair-optimizer-step.v2",
        "predecessor_workflow_run": 30941070265,
        "predecessor_job": 92099264484,
        "predecessor_artifact_id": 8905155402,
        "predecessor_artifact_zip_sha256": (
            "180eed92fcb7a82a9f7652d0cc6b2ea06244d2277e5256531ca06898bbab49e6"
        ),
        "predecessor_reached_complete_comparison": False,
        "causal_failure": "independent_control_pauli_tuple_order_reversed",
        "failure_detail": (
            "The repository's exact Pauli support contract stores entries as "
            "(qubit, axis). The predecessor independent PyTorch control unpacked "
            "them as (axis, qubit) and failed during its first observable pass."
        ),
        "successor_change": (
            "Correct only the independent PyTorch control's Pauli support "
            "unpacking and preserve all circuit, QSA, model, optimizer, batch, "
            "target, clipping, and step contracts."
        ),
        "model_math_changed": False,
        "qsa_candidate_changed": False,
        "optimizer_contract_changed": False,
        "comparison_budget_changed": False,
    }
    receipt["contract"]["pauli_support_order"] = {
        "canonical_entry": ["qubit", "axis"],
        "source": "QSA CausalPauliSupportPlan and Brain Tripair observable list",
        "physical_observables_changed": False,
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-optimizer-step-v3.json"),
    )
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 4:
        raise SystemExit("--steps must be at least 4")
    v1.torch.set_num_threads(1)
    v1.torch.set_num_interop_threads(1)
    receipt = build_receipt(args.steps)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit(
            "Brain Tripair complete optimizer-step contract v3 failed"
        )


if __name__ == "__main__":
    main()
