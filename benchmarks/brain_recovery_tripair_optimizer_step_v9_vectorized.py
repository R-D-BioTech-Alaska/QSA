from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from qsa.causal_adjoint_vectorized import CausalVectorizedBatchAdjoint

import brain_recovery_tripair_optimizer_step_v8_fused_dense as v8


v7 = v8.v7
v1 = v8.v1
v4 = v8.v4


class VectorizedBatchTripairBridge(v8.FusedDenseBatchTripairBridge):
    """Safe Tripair bridge with operation-major amplitude-major batches."""

    def __init__(self) -> None:
        super().__init__()
        self.vectorized = CausalVectorizedBatchAdjoint(
            self.plan,
            self.support,
            max_qubits=v1.QUBITS,
        )
        self.vectorized_primal_materializations = 0
        self.vectorized_reverse_materializations = 0

    def forward(self, brain_angles, shared):
        started = time.perf_counter()
        rows = self.parameter_rows(brain_angles, shared)
        result = self.vectorized.values_many(self.root, rows)
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
        self.vectorized_primal_materializations += int(result.state_materializations)
        return tensor, v7.BatchNativeSafeBatch(rows, result.values)

    def backward(
        self,
        batch,
        cotangents,
        input_dtype,
        input_device,
        shared_shape,
    ):
        started = time.perf_counter()
        cotangent_rows = tuple(
            tuple(float(value) for value in row)
            for row in cotangents.detach().cpu().tolist()
        )
        result = self.vectorized.evaluate_many(
            self.root,
            batch.parameter_rows,
            cotangent_rows,
        )
        self.backward_native_calls += int(result.native_calls)
        self.backward_batch_calls += 1
        self.vectorized_reverse_materializations += int(result.native_calls)

        name_to_position = {
            name: position for position, name in enumerate(self.parameter_names)
        }
        input_gradients = torch.zeros(
            len(result.gradients),
            len(v1.BRAIN_INPUT_NAMES),
            dtype=input_dtype,
            device=input_device,
        )
        shared_gradient = torch.zeros(
            12,
            dtype=input_dtype,
            device=input_device,
        )
        for sample, gradient in enumerate(result.gradients):
            for brain_position, name in enumerate(v1.BRAIN_INPUT_NAMES):
                input_gradients[sample, brain_position] = gradient[
                    name_to_position[name]
                ]
            offset = 0
            for layer in range(v1.DEPTH):
                for qubit in range(v1.QUBITS):
                    for axis in ("ry", "rz"):
                        shared_gradient[offset] += gradient[
                            name_to_position[f"layer_{layer}_{qubit}_{axis}"]
                        ]
                        offset += 1

        for forward_values, replay_values in zip(batch.observations, result.values):
            self.maximum_backward_value_replay_error = max(
                self.maximum_backward_value_replay_error,
                max(
                    abs(float(left) - float(right))
                    for left, right in zip(forward_values, replay_values)
                ),
            )
        self.adjoint_forward_sweeps += int(result.forward_sweeps)
        self.adjoint_reverse_sweeps += int(result.reverse_sweeps)
        self.shifted_evaluations += int(result.shifted_evaluations)
        self.local_widths.extend(
            [int(result.local_qubit_count)] * len(result.gradients)
        )
        self.backward_seconds += time.perf_counter() - started
        self.backward_calls += 1
        return input_gradients, shared_gradient.reshape(shared_shape)

    def state_receipt(self) -> dict[str, object]:
        receipt = super().state_receipt()
        receipt.update(
            {
                "mode": "safe_operation_major_amplitude_major_batch",
                "forward_architecture": (
                    "amplitude-major batched state with operation-major gates"
                ),
                "reverse_architecture": (
                    "operation-major reverse reconstruction over contiguous rows"
                ),
                "component_closure_extractions": 0,
                "primal_state_materializations": 0,
                "vectorized_primal_materializations": (
                    self.vectorized_primal_materializations
                ),
                "vectorized_reverse_materializations": (
                    self.vectorized_reverse_materializations
                ),
                "primal_structural_branches": 0,
                "branch_state_constructions": 0,
                "branch_state_destructions": 0,
                "python_native_calls_per_step": 2,
                "state_layout": "amplitude-major contiguous rows",
            }
        )
        return receipt


def vectorized_case_passed(case: dict[str, object]) -> bool:
    qsa = case["candidate"]["qsa"]
    rollback = case["rollback"]
    expected_rows = int(case["batch_size"]) * int(case["steps"])
    expected_steps = int(case["steps"])
    return bool(
        case["active_trainable_parameters"] == v1.EXPECTED_PARAMETER_COUNT
        and case["all_gradients_finite_and_present"]
        and case["maximum_output_error"] <= 3.0e-12
        and case["maximum_observable_error"] <= 3.0e-12
        and case["maximum_loss_error"] <= 3.0e-13
        and case["maximum_gradient_error"] <= 3.0e-11
        and case["maximum_clip_norm_error"] <= 3.0e-11
        and case["maximum_parameter_error_after_step"] <= 3.0e-12
        and case["maximum_final_parameter_error"] <= 3.0e-12
        and qsa["root_immutable"]
        and not qsa["persistent_workspace_used"]
        and not qsa["retained_candidate_used"]
        and qsa["forward_batch_calls"] == expected_steps
        and qsa["backward_batch_calls"] == expected_steps
        and qsa["forward_core_native_calls"] == expected_steps
        and qsa["backward_native_calls"] == expected_steps
        and qsa["forward_executions"] == expected_rows
        and qsa["forward_sweeps"] == expected_rows
        and qsa["reverse_sweeps"] == expected_rows
        and qsa["shifted_evaluations"] == 0
        and qsa["local_widths"] == [v1.QUBITS] * expected_rows
        and qsa["component_closure_extractions"] == 0
        and qsa["branch_state_constructions"] == 0
        and qsa["branch_state_destructions"] == 0
        and qsa["vectorized_primal_materializations"] == expected_steps
        and qsa["vectorized_reverse_materializations"] == expected_steps
        and qsa["parameter_rows_bound"] == expected_rows
        and qsa["observable_rows"] == expected_rows
        and qsa["maximum_backward_value_replay_error"] <= 3.0e-12
        and rollback["baseline_model_max_error"] == 0.0
        and rollback["candidate_model_max_error"] == 0.0
        and rollback["baseline_optimizer_exact"]
        and rollback["candidate_optimizer_exact"]
    )


def deterministic_signature(case: dict[str, object]) -> dict[str, object]:
    return v8.deterministic_signature(case)


def build_receipt(steps: int) -> dict[str, object]:
    batch_sizes = (1, 4, 16, 64, 128)
    safe_cases = v7.run_cases(v4.SafeComponentTripairBridge, batch_sizes, steps)
    direct_cases = v7.run_cases(v8.FusedDenseBatchTripairBridge, batch_sizes, steps)
    vectorized_cases = v7.run_cases(VectorizedBatchTripairBridge, batch_sizes, steps)
    safe_by_batch = {int(case["batch_size"]): case for case in safe_cases}
    direct_by_batch = {int(case["batch_size"]): case for case in direct_cases}

    for case in vectorized_cases:
        batch_size = int(case["batch_size"])
        safe = safe_by_batch[batch_size]
        direct = direct_by_batch[batch_size]
        case["candidate"]["backend"] = (
            "QSA operation-major amplitude-major exact batch adjoint"
        )
        case["safe_per_sample_reference"] = {
            "passed": bool(v4.case_passed(safe)),
            "timing": safe["candidate"]["timing"],
            "qsa": safe["candidate"]["qsa"],
        }
        case["direct_dense_reference"] = {
            "passed": bool(v8.fused_case_passed(direct)),
            "timing": direct["candidate"]["timing"],
            "qsa": direct["candidate"]["qsa"],
        }
        vectorized_total = float(
            case["candidate"]["timing"]["total"]["median_ms"]
        )
        safe_total = float(safe["candidate"]["timing"]["total"]["median_ms"])
        direct_total = float(
            direct["candidate"]["timing"]["total"]["median_ms"]
        )
        case["safe_over_vectorized_total_speed"] = safe_total / vectorized_total
        case["direct_over_vectorized_total_speed"] = direct_total / vectorized_total
        case["vectorized_faster_than_safe"] = vectorized_total < safe_total
        case["vectorized_faster_than_direct"] = vectorized_total < direct_total
        case["passed"] = bool(
            vectorized_case_passed(case)
            and case["safe_per_sample_reference"]["passed"]
            and case["direct_dense_reference"]["passed"]
        )

    rerun = v7.run_cases(VectorizedBatchTripairBridge, (64,), steps)[0]
    deterministic = {
        "batch_size": 64,
        "first": deterministic_signature(
            next(
                case
                for case in vectorized_cases
                if int(case["batch_size"]) == 64
            )
        ),
        "second": deterministic_signature(rerun),
    }
    deterministic["equal"] = deterministic["first"] == deterministic["second"]

    repository_root = Path(__file__).resolve().parents[1]
    receipt: dict[str, object] = {
        "schema_version": (
            "qsa.brain-recovery-tripair-optimizer-step.v9-vectorized"
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
            "state_layout": "amplitude-major contiguous rows",
            "gate_schedule": "operation-major shared topology",
            "forward_native_calls_per_step": 1,
            "reverse_native_calls_per_step": 1,
            "stored_depth_trajectory": False,
            "deterministic_shared_gradient_order": (
                "sample-major then shared-parameter-major"
            ),
        },
        "cases": vectorized_cases,
        "deterministic_rerun": deterministic,
        "classification_policy": {
            "correctness": (
                "Requires exact output, observable, loss, gradient, clipping, "
                "parameter trajectory, root immutability, deterministic rerun, "
                "and exact model/optimizer rollback."
            ),
            "complete_step_advantage": (
                "Established only where vectorized QSA complete AdamW median "
                "step is lower than PyTorch, direct dense QSA, and safe per-sample QSA."
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
        all(bool(case["passed"]) for case in vectorized_cases)
        and deterministic["equal"]
    )
    receipt["vectorized_faster_than_safe_batch_sizes"] = [
        int(case["batch_size"])
        for case in vectorized_cases
        if bool(case["vectorized_faster_than_safe"])
    ]
    receipt["vectorized_faster_than_direct_batch_sizes"] = [
        int(case["batch_size"])
        for case in vectorized_cases
        if bool(case["vectorized_faster_than_direct"])
    ]
    receipt["vectorized_faster_than_pytorch_batch_sizes"] = [
        int(case["batch_size"])
        for case in vectorized_cases
        if bool(case["candidate_faster_than_pytorch"])
    ]
    receipt["complete_step_advantage_over_all_controls_batch_sizes"] = [
        int(case["batch_size"])
        for case in vectorized_cases
        if bool(case["vectorized_faster_than_safe"])
        and bool(case["vectorized_faster_than_direct"])
        and bool(case["candidate_faster_than_pytorch"])
    ]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-tripair-optimizer-step-v9-vectorized.json"
        ),
    )
    parser.add_argument("--steps", type=int, default=16)
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
        raise SystemExit("vectorized batch optimizer-step contract failed")


if __name__ == "__main__":
    main()
