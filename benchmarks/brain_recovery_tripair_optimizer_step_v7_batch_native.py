from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from qsa.causal_batch import apply_many, fork_many


EVIDENCE_BENCHMARKS = Path(
    os.environ.get(
        "QSA_RECOVERY_EVIDENCE_BENCHMARKS",
        str(Path(__file__).resolve().parent),
    )
).resolve()
if str(EVIDENCE_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_BENCHMARKS))

try:
    import brain_recovery_tripair_optimizer_step as v1
    import brain_recovery_tripair_optimizer_step_v3 as v3
    import brain_recovery_tripair_optimizer_step_v4_safe as v4
except ImportError as exc:
    raise SystemExit(
        "The frozen recovery evidence benchmark directory is required; set "
        "QSA_RECOVERY_EVIDENCE_BENCHMARKS to PR #35's benchmarks directory"
    ) from exc


@dataclass(frozen=True)
class BatchNativeSafeBatch:
    parameter_rows: tuple[dict[str, float], ...]
    observations: tuple[tuple[float, ...], ...]


class BatchNativeSafeComponentTripairBridge(v4.SafeComponentTripairBridge):
    """Exact safe Tripair bridge with batch forward and batch reverse calls.

    The forward path creates ephemeral branches, binds the shared plan to all
    rows in one native call, evaluates compact Pauli supports in one native
    call, and destroys every branch before returning. Backward extracts the
    component closure once and executes all weighted adjoints through one
    deterministic native call. No retained candidate or persistent pullback
    lifecycle is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self.forward_batch_calls = 0
        self.backward_batch_calls = 0
        self.forward_core_native_calls = 0
        self.backward_native_calls = 0
        self.branch_state_constructions = 0
        self.branch_state_destructions = 0
        self.parameter_rows_bound = 0
        self.observable_rows = 0
        self.maximum_backward_value_replay_error = 0.0

    def forward(self, brain_angles, shared):
        started = time.perf_counter()
        rows = self.parameter_rows(brain_angles, shared)
        branches = fork_many(self.root, len(rows))
        self.forward_core_native_calls += 1
        self.branch_state_constructions += len(branches)
        try:
            apply_many(self.plan, branches, rows, workers=1)
            self.forward_core_native_calls += 1
            observations = self.support.execute_many(branches, workers=1)
            self.forward_core_native_calls += 1
        finally:
            for branch in branches:
                branch.close()
                self.branch_state_destructions += 1
        tensor = torch.tensor(
            observations,
            dtype=brain_angles.dtype,
            device=brain_angles.device,
        )
        self.forward_seconds += time.perf_counter() - started
        self.forward_calls += 1
        self.forward_batch_calls += 1
        self.forward_executions += len(rows)
        self.parameter_rows_bound += len(rows)
        self.observable_rows += len(rows)
        return tensor, BatchNativeSafeBatch(rows, observations)

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
        result = self.adjoint.evaluate_many(
            self.root,
            batch.parameter_rows,
            cotangent_rows,
        )
        self.backward_native_calls += int(result.native_calls)
        self.backward_batch_calls += 1

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
                "mode": "safe_batch_native_forward_and_reverse",
                "batch_native_forward": True,
                "batch_native_reverse": True,
                "forward_batch_calls": self.forward_batch_calls,
                "backward_batch_calls": self.backward_batch_calls,
                "forward_core_native_calls": self.forward_core_native_calls,
                "backward_native_calls": self.backward_native_calls,
                "branch_state_constructions": self.branch_state_constructions,
                "branch_state_destructions": self.branch_state_destructions,
                "parameter_rows_bound": self.parameter_rows_bound,
                "observable_rows": self.observable_rows,
                "plan_compilations": 1,
                "component_closure_extractions": self.backward_batch_calls,
                "maximum_backward_value_replay_error": (
                    self.maximum_backward_value_replay_error
                ),
                "deterministic_reduction_order": "sample-major then parameter-major",
                "worker_count": 1,
            }
        )
        return receipt


def process_peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(usage)
    return int(usage) * 1024


def run_cases(bridge_type, batch_sizes, steps):
    original_bridge = v1.QSATripairBridge
    original_apply_pauli = v1.apply_pauli
    v1.QSATripairBridge = bridge_type
    v1.apply_pauli = v3.corrected_apply_pauli
    try:
        cases = []
        for batch_size in batch_sizes:
            case = v1.run_case(batch_size, steps)
            case["process_peak_rss_bytes_after_case"] = process_peak_rss_bytes()
            cases.append(case)
        return cases
    finally:
        v1.QSATripairBridge = original_bridge
        v1.apply_pauli = original_apply_pauli


def batch_case_passed(case: dict[str, object]) -> bool:
    qsa = case["candidate"]["qsa"]
    rollback = case["rollback"]
    expected_rows = int(case["batch_size"]) * int(case["steps"])
    expected_batches = int(case["steps"])
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
        and qsa["forward_batch_calls"] == expected_batches
        and qsa["backward_batch_calls"] == expected_batches
        and qsa["forward_core_native_calls"] == 3 * expected_batches
        and qsa["backward_native_calls"] == expected_batches
        and qsa["forward_executions"] == expected_rows
        and qsa["forward_sweeps"] == expected_rows
        and qsa["reverse_sweeps"] == expected_rows
        and qsa["shifted_evaluations"] == 0
        and qsa["local_widths"] == [v1.QUBITS] * expected_rows
        and qsa["branch_state_constructions"] == expected_rows
        and qsa["branch_state_destructions"] == expected_rows
        and qsa["parameter_rows_bound"] == expected_rows
        and qsa["observable_rows"] == expected_rows
        and qsa["maximum_backward_value_replay_error"] <= 2.0e-12
        and rollback["baseline_model_max_error"] == 0.0
        and rollback["candidate_model_max_error"] == 0.0
        and rollback["baseline_optimizer_exact"]
        and rollback["candidate_optimizer_exact"]
    )


def git_head(path: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_receipt(steps: int) -> dict[str, object]:
    batch_sizes = (1, 4, 16, 64, 128)
    safe_cases = run_cases(v4.SafeComponentTripairBridge, batch_sizes, steps)
    batch_cases = run_cases(BatchNativeSafeComponentTripairBridge, batch_sizes, steps)
    safe_by_batch = {int(case["batch_size"]): case for case in safe_cases}

    for case in batch_cases:
        batch_size = int(case["batch_size"])
        reference = safe_by_batch[batch_size]
        case["candidate"]["backend"] = (
            "QSA safe batch-native ephemeral forward and one-call batch adjoint"
        )
        case["current_safe_reference"] = {
            "passed": bool(v4.case_passed(reference)),
            "timing": reference["candidate"]["timing"],
            "qsa": reference["candidate"]["qsa"],
            "process_peak_rss_bytes_after_case": reference[
                "process_peak_rss_bytes_after_case"
            ],
        }
        safe_total = float(
            reference["candidate"]["timing"]["total"]["median_ms"]
        )
        batch_total = float(case["candidate"]["timing"]["total"]["median_ms"])
        case["safe_over_batch_native_total_speed"] = safe_total / batch_total
        case["batch_native_faster_than_current_safe"] = batch_total < safe_total
        case["passed"] = bool(
            batch_case_passed(case) and case["current_safe_reference"]["passed"]
        )

    repository_root = Path(__file__).resolve().parents[1]
    receipt: dict[str, object] = {
        "schema_version": (
            "qsa.brain-recovery-tripair-optimizer-step.v7-batch-native"
        ),
        "source": {
            "runtime_head": git_head(repository_root),
            "runtime_parent_exactness_repair": (
                "748e4a40483de72ea5e9092d1db45a62ec6b4a16"
            ),
            "frozen_evidence_head": git_head(EVIDENCE_BENCHMARKS.parent),
            "accepted_brain_parent": v1.BRAIN_PARENT,
            "brain_quantum_source": v1.BRAIN_QUANTUM_SOURCE,
            "brain_quantum_blob": v1.BRAIN_QUANTUM_BLOB,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
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
                "fork-many + parameterized-plan-execute-many + compact-support-"
                "execute-many with ephemeral branches"
            ),
            "reverse": (
                "one component extraction and one deterministic native batch "
                "weighted-adjoint call per optimizer step"
            ),
        },
        "cases": batch_cases,
        "classification_policy": {
            "correctness": (
                "Requires model-output, observable, loss, gradient, clipping, "
                "parameter-trajectory, root-immutability, replay, and rollback parity."
            ),
            "batch_native_advantage": (
                "Established only where complete AdamW median step time is lower "
                "than both the current safe per-sample QSA bridge and PyTorch."
            ),
            "claim_boundary": (
                "This is an exact QSA engineering and quantum-native structural "
                "execution result; it does not establish quantum uniqueness."
            ),
        },
        "not_established": [
            "bounded real Brain-data learning preservation",
            "Brain language canonical or safety capability contribution",
            "persistent selected-pullback safety",
            "quantum uniqueness",
            "physical-QPU advantage",
            "production readiness",
        ],
    }
    receipt["passed"] = all(bool(case["passed"]) for case in batch_cases)
    receipt["batch_native_faster_than_safe_batch_sizes"] = [
        int(case["batch_size"])
        for case in batch_cases
        if bool(case["batch_native_faster_than_current_safe"])
    ]
    receipt["batch_native_faster_than_pytorch_batch_sizes"] = [
        int(case["batch_size"])
        for case in batch_cases
        if bool(case["candidate_faster_than_pytorch"])
    ]
    receipt["complete_step_advantage_over_both_batch_sizes"] = [
        int(case["batch_size"])
        for case in batch_cases
        if bool(case["batch_native_faster_than_current_safe"])
        and bool(case["candidate_faster_than_pytorch"])
    ]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-tripair-optimizer-step-v7-batch-native.json"
        ),
    )
    parser.add_argument("--steps", type=int, default=8)
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
        raise SystemExit("batch-native safe optimizer-step contract failed")


if __name__ == "__main__":
    main()
