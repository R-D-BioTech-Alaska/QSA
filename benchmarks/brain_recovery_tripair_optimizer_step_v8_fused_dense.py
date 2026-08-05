from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsa.causal_adjoint_dense import CausalDenseBatchPrimal

import brain_recovery_tripair_optimizer_step_v7_batch_native as v7


v1 = v7.v1
v3 = v7.v3
v4 = v7.v4


class FusedDenseBatchTripairBridge(v7.BatchNativeSafeComponentTripairBridge):
    """Safe Tripair bridge with fused dense primal and reverse execution."""

    def __init__(self) -> None:
        super().__init__()
        self.dense_primal = CausalDenseBatchPrimal(
            self.plan,
            self.support,
            max_qubits=v1.QUBITS,
        )
        self.primal_state_materializations = 0
        self.primal_structural_branches = 0

    def forward(self, brain_angles, shared):
        import time

        started = time.perf_counter()
        rows = self.parameter_rows(brain_angles, shared)
        result = self.dense_primal.evaluate(self.root, rows)
        tensor = torch.tensor(
            result.values,
            dtype=brain_angles.dtype,
            device=brain_angles.device,
        )
        self.forward_seconds += time.perf_counter() - started
        self.forward_calls += 1
        self.forward_batch_calls += 1
        self.forward_core_native_calls += int(result.native_calls)
        self.forward_executions += len(rows)
        self.parameter_rows_bound += len(rows)
        self.observable_rows += len(rows)
        self.primal_state_materializations += int(result.state_materializations)
        self.primal_structural_branches += int(result.structural_branches)
        return tensor, v7.BatchNativeSafeBatch(rows, result.values)

    def state_receipt(self) -> dict[str, object]:
        receipt = super().state_receipt()
        receipt.update(
            {
                "mode": "safe_fused_dense_batch_forward_and_reverse",
                "forward_architecture": (
                    "one root materialization and direct dense row evolution"
                ),
                "reverse_architecture": (
                    "one root materialization, direct dense row reconstruction, "
                    "and deterministic weighted adjoints"
                ),
                "primal_state_materializations": self.primal_state_materializations,
                "primal_structural_branches": self.primal_structural_branches,
                "branch_state_constructions": 0,
                "branch_state_destructions": 0,
                "python_native_calls_per_step": 2,
            }
        )
        return receipt


def fused_case_passed(case: dict[str, object]) -> bool:
    qsa = case["candidate"]["qsa"]
    rollback = case["rollback"]
    expected_rows = int(case["batch_size"]) * int(case["steps"])
    expected_steps = int(case["steps"])
    return bool(
        case["active_trainable_parameters"] == v1.EXPECTED_PARAMETER_COUNT
        and case["all_gradients_finite_and_present"]
        and case["maximum_output_error"] <= 2.0e-12
        and case["maximum_observable_error"] <= 2.0e-12
        and case["maximum_loss_error"] <= 2.0e-13
        and case["maximum_gradient_error"] <= 2.0e-11
        and case["maximum_clip_norm_error"] <= 2.0e-11
        and case["maximum_parameter_error_after_step"] <= 2.0e-12
        and case["maximum_final_parameter_error"] <= 2.0e-12
        and qsa["root_immutable"]
        and not qsa["persistent_workspace_used"]
        and not qsa["retained_candidate_used"]
        and qsa["batch_native_forward"]
        and qsa["batch_native_reverse"]
        and qsa["forward_batch_calls"] == expected_steps
        and qsa["backward_batch_calls"] == expected_steps
        and qsa["forward_core_native_calls"] == expected_steps
        and qsa["backward_native_calls"] == expected_steps
        and qsa["forward_executions"] == expected_rows
        and qsa["forward_sweeps"] == expected_rows
        and qsa["reverse_sweeps"] == expected_rows
        and qsa["shifted_evaluations"] == 0
        and qsa["local_widths"] == [v1.QUBITS] * expected_rows
        and qsa["branch_state_constructions"] == 0
        and qsa["branch_state_destructions"] == 0
        and qsa["primal_state_materializations"] == expected_steps
        and qsa["primal_structural_branches"] == 0
        and qsa["parameter_rows_bound"] == expected_rows
        and qsa["observable_rows"] == expected_rows
        and qsa["maximum_backward_value_replay_error"] <= 2.0e-12
        and rollback["baseline_model_max_error"] == 0.0
        and rollback["candidate_model_max_error"] == 0.0
        and rollback["baseline_optimizer_exact"]
        and rollback["candidate_optimizer_exact"]
    )


def deterministic_signature(case: dict[str, object]) -> dict[str, object]:
    return {
        "batch_size": case["batch_size"],
        "steps": case["steps"],
        "baseline_initial_loss": case["baseline"]["initial_loss"],
        "baseline_final_loss": case["baseline"]["final_loss"],
        "candidate_initial_loss": case["candidate"]["initial_loss"],
        "candidate_final_loss": case["candidate"]["final_loss"],
        "maximum_output_error": case["maximum_output_error"],
        "maximum_observable_error": case["maximum_observable_error"],
        "maximum_loss_error": case["maximum_loss_error"],
        "maximum_gradient_error": case["maximum_gradient_error"],
        "maximum_clip_norm_error": case["maximum_clip_norm_error"],
        "maximum_parameter_error_after_step": (
            case["maximum_parameter_error_after_step"]
        ),
        "maximum_final_parameter_error": case["maximum_final_parameter_error"],
        "rollback": case["rollback"],
        "qsa_root_immutable": case["candidate"]["qsa"]["root_immutable"],
        "qsa_shifted_evaluations": case["candidate"]["qsa"][
            "shifted_evaluations"
        ],
        "qsa_replay_error": case["candidate"]["qsa"][
            "maximum_backward_value_replay_error"
        ],
    }


def build_receipt(steps: int) -> dict[str, object]:
    batch_sizes = (1, 4, 16, 64, 128)
    safe_cases = v7.run_cases(v4.SafeComponentTripairBridge, batch_sizes, steps)
    boundary_cases = v7.run_cases(
        v7.BatchNativeSafeComponentTripairBridge,
        batch_sizes,
        steps,
    )
    fused_cases = v7.run_cases(FusedDenseBatchTripairBridge, batch_sizes, steps)
    safe_by_batch = {int(case["batch_size"]): case for case in safe_cases}
    boundary_by_batch = {
        int(case["batch_size"]): case for case in boundary_cases
    }

    for case in fused_cases:
        batch_size = int(case["batch_size"])
        safe = safe_by_batch[batch_size]
        boundary = boundary_by_batch[batch_size]
        case["candidate"]["backend"] = (
            "QSA fused dense safe batch primal and weighted adjoint"
        )
        case["safe_per_sample_reference"] = {
            "passed": bool(v4.case_passed(safe)),
            "timing": safe["candidate"]["timing"],
            "qsa": safe["candidate"]["qsa"],
        }
        case["reduced_boundary_reference"] = {
            "passed": bool(v7.batch_case_passed(boundary)),
            "timing": boundary["candidate"]["timing"],
            "qsa": boundary["candidate"]["qsa"],
        }
        fused_total = float(case["candidate"]["timing"]["total"]["median_ms"])
        safe_total = float(safe["candidate"]["timing"]["total"]["median_ms"])
        boundary_total = float(
            boundary["candidate"]["timing"]["total"]["median_ms"]
        )
        case["safe_over_fused_total_speed"] = safe_total / fused_total
        case["boundary_over_fused_total_speed"] = boundary_total / fused_total
        case["fused_faster_than_safe"] = fused_total < safe_total
        case["fused_faster_than_boundary"] = fused_total < boundary_total
        case["passed"] = bool(
            fused_case_passed(case)
            and case["safe_per_sample_reference"]["passed"]
            and case["reduced_boundary_reference"]["passed"]
        )

    rerun = v7.run_cases(FusedDenseBatchTripairBridge, (64,), steps)[0]
    deterministic = {
        "batch_size": 64,
        "first": deterministic_signature(
            next(case for case in fused_cases if int(case["batch_size"]) == 64)
        ),
        "second": deterministic_signature(rerun),
    }
    deterministic["equal"] = deterministic["first"] == deterministic["second"]

    repository_root = Path(__file__).resolve().parents[1]
    receipt: dict[str, object] = {
        "schema_version": (
            "qsa.brain-recovery-tripair-optimizer-step.v8-fused-dense"
        ),
        "source": {
            "runtime_head": v7.git_head(repository_root),
            "runtime_parent_exactness_repair": (
                "748e4a40483de72ea5e9092d1db45a62ec6b4a16"
            ),
            "frozen_evidence_head": v7.git_head(v7.EVIDENCE_BENCHMARKS.parent),
            "accepted_brain_parent": v1.BRAIN_PARENT,
            "brain_quantum_source": v1.BRAIN_QUANTUM_SOURCE,
            "brain_quantum_blob": v1.BRAIN_QUANTUM_BLOB,
        },
        "environment": {
            "python": v7.sys.version,
            "platform": v7.platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64/complex128",
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "same_5395_trainable_parameters": True,
            "same_names_shapes_initialization_batches_targets_optimizer_clipping": True,
            "nonzero_learning_rate": 3.0e-4,
            "exact_rollback": True,
            "shifted_evaluations": 0,
            "persistent_candidate_reuse": False,
            "batch_sizes": list(batch_sizes),
            "steps_per_case": steps,
            "forward": (
                "one native call: materialize root once, apply shared schedule "
                "directly to dense row states, evaluate compact supports"
            ),
            "reverse": (
                "one native call: materialize root once, reconstruct each row "
                "in reverse, accumulate repeated slots deterministically"
            ),
            "stored_depth_trajectory": False,
            "python_native_calls_per_optimizer_step": 2,
        },
        "cases": fused_cases,
        "deterministic_rerun": deterministic,
        "classification_policy": {
            "correctness": (
                "Requires exact output, observable, loss, gradient, clipping, "
                "parameter trajectory, root immutability, deterministic rerun, "
                "and exact model/optimizer rollback."
            ),
            "complete_step_advantage": (
                "Established only where fused QSA complete AdamW median step is "
                "lower than PyTorch, reduced-boundary QSA, and safe per-sample QSA."
            ),
            "claim_boundary": (
                "A positive result is a QSA engineering and quantum-native "
                "structural execution advantage, not quantum uniqueness."
            ),
        },
        "not_established": [
            "held-out Brain-data learning preservation",
            "Brain language canonical or safety capability contribution",
            "persistent selected-pullback safety",
            "quantum uniqueness",
            "physical-QPU advantage",
            "production readiness",
        ],
    }
    receipt["passed"] = bool(
        all(bool(case["passed"]) for case in fused_cases)
        and deterministic["equal"]
    )
    receipt["fused_faster_than_safe_batch_sizes"] = [
        int(case["batch_size"])
        for case in fused_cases
        if bool(case["fused_faster_than_safe"])
    ]
    receipt["fused_faster_than_boundary_batch_sizes"] = [
        int(case["batch_size"])
        for case in fused_cases
        if bool(case["fused_faster_than_boundary"])
    ]
    receipt["fused_faster_than_pytorch_batch_sizes"] = [
        int(case["batch_size"])
        for case in fused_cases
        if bool(case["candidate_faster_than_pytorch"])
    ]
    receipt["complete_step_advantage_over_all_controls_batch_sizes"] = [
        int(case["batch_size"])
        for case in fused_cases
        if bool(case["fused_faster_than_safe"])
        and bool(case["fused_faster_than_boundary"])
        and bool(case["candidate_faster_than_pytorch"])
    ]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-tripair-optimizer-step-v8-fused-dense.json"
        ),
    )
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 4:
        raise SystemExit("--steps must be at least 4")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = build_receipt(args.steps)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("fused dense batch optimizer-step contract failed")


if __name__ == "__main__":
    main()
