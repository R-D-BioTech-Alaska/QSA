from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import psutil
import torch
from torch.nn import functional as F

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v3 as v3
import brain_recovery_tripair_optimizer_step_v4_safe as v4

TOLERANCES = {
    "output": 2.0e-12,
    "observable": 2.0e-12,
    "loss": 2.0e-13,
    "gradient": 2.0e-11,
    "clip_norm": 2.0e-11,
    "parameter": 2.0e-12,
    "optimizer": 2.0e-12,
}
BATCH_SIZES = (1, 4, 16, 64)
STEP_COUNTS = (1, 5, 20)


@dataclass
class DetailedStep:
    output: torch.Tensor
    observables: torch.Tensor
    loss: torch.Tensor
    raw_gradients: dict[str, torch.Tensor]
    clipped_gradients: dict[str, torch.Tensor]
    gradient_norm: float
    timing: dict[str, float]


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def first_cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_bytes(state: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def state_sha256(state: object) -> str:
    return hashlib.sha256(state_bytes(state)).hexdigest()


def gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        result[name] = parameter.grad.detach().clone()
    return result


def maximum_mapping_error(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> tuple[float, dict[str, float]]:
    per_name = {
        name: float(torch.max(torch.abs(first[name] - second[name])))
        for name in first
    }
    return max(per_name.values(), default=0.0), per_name


def optimizer_numeric_error(first: dict, second: dict) -> tuple[float, dict[str, float]]:
    if first["param_groups"] != second["param_groups"]:
        return float("inf"), {"param_groups": float("inf")}
    errors: dict[str, float] = {}
    first_keys = list(first["state"].keys())
    second_keys = list(second["state"].keys())
    if len(first_keys) != len(second_keys):
        return float("inf"), {"state_width": float("inf")}
    for position, (left_key, right_key) in enumerate(zip(first_keys, second_keys)):
        left = first["state"][left_key]
        right = second["state"][right_key]
        if left.keys() != right.keys():
            return float("inf"), {f"state_{position}_keys": float("inf")}
        for name in left:
            label = f"state_{position}.{name}"
            if isinstance(left[name], torch.Tensor):
                errors[label] = float(torch.max(torch.abs(left[name] - right[name])))
            else:
                errors[label] = 0.0 if left[name] == right[name] else float("inf")
    return max(errors.values(), default=0.0), errors


def execute_step(
    model: v1.TripairCircuitModule,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    target: torch.Tensor,
    backend: str,
    bridge: object | None,
    clip_norm: float,
    *,
    apply_update: bool,
) -> DetailedStep:
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    output, observables = model(features, backend, bridge)
    loss = F.mse_loss(output, target)
    forward_done = time.perf_counter()
    loss.backward()
    backward_done = time.perf_counter()
    raw = gradients(model)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm))
    clipped = gradients(model)
    clip_done = time.perf_counter()
    if apply_update:
        optimizer.step()
    optimizer_done = time.perf_counter()
    return DetailedStep(
        output=output.detach(),
        observables=observables.detach(),
        loss=loss.detach(),
        raw_gradients=raw,
        clipped_gradients=clipped,
        gradient_norm=norm,
        timing={
            "forward_ms": 1000.0 * (forward_done - started),
            "backward_ms": 1000.0 * (backward_done - forward_done),
            "clip_ms": 1000.0 * (clip_done - backward_done),
            "optimizer_ms": 1000.0 * (optimizer_done - clip_done),
            "total_ms": 1000.0 * (optimizer_done - started),
        },
    )


def tensor_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first - second)))


def summarize_timings(rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for name in ("forward_ms", "backward_ms", "clip_ms", "optimizer_ms", "total_ms"):
        values = [float(row[name]) for row in rows]
        result[name] = {
            "first": values[0],
            "median": statistics.median(values),
            "maximum": max(values),
        }
    return result


def first_failure(steps: Sequence[dict[str, object]], section: str) -> dict[str, object] | None:
    for row in steps:
        values = row[section]
        checks = {
            "output": values["output_error"] <= TOLERANCES["output"],
            "observable": values["observable_error"] <= TOLERANCES["observable"],
            "loss": values["loss_error"] <= TOLERANCES["loss"],
            "raw_gradient": values["raw_gradient_error"] <= TOLERANCES["gradient"],
            "clipped_gradient": values["clipped_gradient_error"] <= TOLERANCES["gradient"],
            "clip_norm": values["clip_norm_error"] <= TOLERANCES["clip_norm"],
        }
        if section == "trajectory":
            checks["parameter"] = values["parameter_error"] <= TOLERANCES["parameter"]
            checks["optimizer"] = values["optimizer_error"] <= TOLERANCES["optimizer"]
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            return {"step": row["step"], "failed_checks": failed, "values": values}
    return None


def run_case(batch_size: int, steps: int, learning_rate: float) -> dict[str, object]:
    torch.manual_seed(370038 + batch_size)
    baseline = v1.TripairCircuitModule()
    candidate = copy.deepcopy(baseline)
    if v1.count_parameters(baseline) != v1.EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Tripair circuit parameter contract changed")

    initial_baseline = v1.clone_state(baseline.state_dict())
    initial_candidate = v1.clone_state(candidate.state_dict())
    baseline_optimizer = torch.optim.AdamW(
        baseline.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    initial_baseline_optimizer = copy.deepcopy(baseline_optimizer.state_dict())
    initial_candidate_optimizer = copy.deepcopy(candidate_optimizer.state_dict())

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    bridge = v4.SafeComponentTripairBridge()
    step_rows: list[dict[str, object]] = []
    baseline_timings: list[dict[str, float]] = []
    candidate_timings: list[dict[str, float]] = []
    shadow_timings: list[dict[str, float]] = []
    all_finite = True
    original_apply_pauli = v1.apply_pauli
    v1.apply_pauli = v3.corrected_apply_pauli
    try:
        for index, (features, target) in enumerate(v1.training_batches(batch_size, steps), start=1):
            baseline_step = execute_step(
                baseline,
                baseline_optimizer,
                features,
                target,
                "torch",
                None,
                1.0,
                apply_update=True,
            )

            shadow = copy.deepcopy(candidate)
            shadow_optimizer = torch.optim.AdamW(
                shadow.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                weight_decay=0.01,
            )
            shadow_optimizer.load_state_dict(copy.deepcopy(candidate_optimizer.state_dict()))
            shadow_step = execute_step(
                shadow,
                shadow_optimizer,
                features,
                target,
                "torch",
                None,
                1.0,
                apply_update=False,
            )

            candidate_step = execute_step(
                candidate,
                candidate_optimizer,
                features,
                target,
                "qsa",
                bridge,
                1.0,
                apply_update=True,
            )

            same_raw, same_raw_by_name = maximum_mapping_error(
                shadow_step.raw_gradients, candidate_step.raw_gradients
            )
            same_clipped, same_clipped_by_name = maximum_mapping_error(
                shadow_step.clipped_gradients, candidate_step.clipped_gradients
            )
            trajectory_raw, trajectory_raw_by_name = maximum_mapping_error(
                baseline_step.raw_gradients, candidate_step.raw_gradients
            )
            trajectory_clipped, trajectory_clipped_by_name = maximum_mapping_error(
                baseline_step.clipped_gradients, candidate_step.clipped_gradients
            )
            optimizer_error, optimizer_by_state = optimizer_numeric_error(
                baseline_optimizer.state_dict(), candidate_optimizer.state_dict()
            )
            parameter_error = v1.maximum_state_error(
                baseline.state_dict(), candidate.state_dict()
            )

            step_rows.append(
                {
                    "step": index,
                    "same_state_backend": {
                        "output_error": tensor_error(shadow_step.output, candidate_step.output),
                        "observable_error": tensor_error(
                            shadow_step.observables, candidate_step.observables
                        ),
                        "loss_error": abs(float(shadow_step.loss) - float(candidate_step.loss)),
                        "raw_gradient_error": same_raw,
                        "raw_gradient_error_by_parameter": same_raw_by_name,
                        "clipped_gradient_error": same_clipped,
                        "clipped_gradient_error_by_parameter": same_clipped_by_name,
                        "clip_norm_error": abs(
                            shadow_step.gradient_norm - candidate_step.gradient_norm
                        ),
                    },
                    "trajectory": {
                        "output_error": tensor_error(
                            baseline_step.output, candidate_step.output
                        ),
                        "observable_error": tensor_error(
                            baseline_step.observables, candidate_step.observables
                        ),
                        "loss_error": abs(
                            float(baseline_step.loss) - float(candidate_step.loss)
                        ),
                        "raw_gradient_error": trajectory_raw,
                        "raw_gradient_error_by_parameter": trajectory_raw_by_name,
                        "clipped_gradient_error": trajectory_clipped,
                        "clipped_gradient_error_by_parameter": trajectory_clipped_by_name,
                        "clip_norm_error": abs(
                            baseline_step.gradient_norm - candidate_step.gradient_norm
                        ),
                        "parameter_error": parameter_error,
                        "optimizer_error": optimizer_error,
                        "optimizer_error_by_state": optimizer_by_state,
                    },
                }
            )
            baseline_timings.append(baseline_step.timing)
            candidate_timings.append(candidate_step.timing)
            shadow_timings.append(shadow_step.timing)
            peak_rss = max(peak_rss, process.memory_info().rss)
            all_finite = all_finite and v1.all_finite_gradients(baseline)
            all_finite = all_finite and v1.all_finite_gradients(candidate)

        final_baseline_hash = state_sha256(v1.clone_state(baseline.state_dict()))
        final_candidate_hash = state_sha256(v1.clone_state(candidate.state_dict()))
        final_baseline_optimizer_hash = state_sha256(baseline_optimizer.state_dict())
        final_candidate_optimizer_hash = state_sha256(candidate_optimizer.state_dict())
        final_parameter_error = v1.maximum_state_error(
            baseline.state_dict(), candidate.state_dict()
        )
        final_optimizer_error, _ = optimizer_numeric_error(
            baseline_optimizer.state_dict(), candidate_optimizer.state_dict()
        )
        qsa_state = bridge.state_receipt()
        qsa_state["root_qsc_bytes"] = len(bridge.root.encode_qsc())
        qsa_state["plan_parameter_count"] = len(bridge.plan.parameter_names)

        baseline.load_state_dict(initial_baseline)
        candidate.load_state_dict(initial_candidate)
        baseline_optimizer.load_state_dict(initial_baseline_optimizer)
        candidate_optimizer.load_state_dict(initial_candidate_optimizer)
        rollback = {
            "baseline_model_error": v1.maximum_state_error(
                baseline.state_dict(), initial_baseline
            ),
            "candidate_model_error": v1.maximum_state_error(
                candidate.state_dict(), initial_candidate
            ),
            "baseline_optimizer_exact": v1.optimizer_state_equal(
                baseline_optimizer.state_dict(), initial_baseline_optimizer
            ),
            "candidate_optimizer_exact": v1.optimizer_state_equal(
                candidate_optimizer.state_dict(), initial_candidate_optimizer
            ),
        }
    finally:
        v1.apply_pauli = original_apply_pauli
        bridge.close()

    backend_failure = first_failure(step_rows, "same_state_backend")
    trajectory_failure = first_failure(step_rows, "trajectory")
    passed = bool(
        all_finite
        and backend_failure is None
        and trajectory_failure is None
        and final_parameter_error <= TOLERANCES["parameter"]
        and final_optimizer_error <= TOLERANCES["optimizer"]
        and qsa_state["root_immutable"]
        and qsa_state["shifted_evaluations"] == 0
        and rollback["baseline_model_error"] == 0.0
        and rollback["candidate_model_error"] == 0.0
        and rollback["baseline_optimizer_exact"]
        and rollback["candidate_optimizer_exact"]
    )
    return {
        "batch_size": batch_size,
        "steps": steps,
        "learning_rate": learning_rate,
        "passed": passed,
        "all_gradients_finite": all_finite,
        "first_same_state_backend_failure": backend_failure,
        "first_trajectory_failure": trajectory_failure,
        "final_parameter_error": final_parameter_error,
        "final_optimizer_error": final_optimizer_error,
        "final_hashes": {
            "baseline_model": final_baseline_hash,
            "candidate_model": final_candidate_hash,
            "baseline_optimizer": final_baseline_optimizer_hash,
            "candidate_optimizer": final_candidate_optimizer_hash,
        },
        "timing": {
            "baseline": summarize_timings(baseline_timings),
            "candidate": summarize_timings(candidate_timings),
            "same_state_torch_shadow": summarize_timings(shadow_timings),
            "baseline_over_candidate_total_median": (
                summarize_timings(baseline_timings)["total_ms"]["median"]
                / summarize_timings(candidate_timings)["total_ms"]["median"]
            ),
        },
        "memory": {
            "peak_process_rss_bytes": peak_rss,
            "process_rss_after_bytes": process.memory_info().rss,
            "resource_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "model_parameter_bytes_each": sum(
                parameter.numel() * parameter.element_size()
                for parameter in baseline.parameters()
            ),
            "baseline_optimizer_serialized_bytes": len(
                state_bytes(baseline_optimizer.state_dict())
            ),
            "candidate_optimizer_serialized_bytes": len(
                state_bytes(candidate_optimizer.state_dict())
            ),
        },
        "qsa": qsa_state,
        "rollback": rollback,
        "trajectory": step_rows,
    }


def deterministic_summary(case: dict[str, object]) -> dict[str, object]:
    return {
        "passed": case["passed"],
        "first_same_state_backend_failure": case["first_same_state_backend_failure"],
        "first_trajectory_failure": case["first_trajectory_failure"],
        "final_parameter_error": case["final_parameter_error"],
        "final_optimizer_error": case["final_optimizer_error"],
        "final_hashes": case["final_hashes"],
        "trajectory": case["trajectory"],
        "rollback": case["rollback"],
    }


def source_identity() -> dict[str, object]:
    paths = [
        Path("benchmarks/brain_recovery_tripair_optimizer_step.py"),
        Path("benchmarks/brain_recovery_tripair_optimizer_step_v3.py"),
        Path("benchmarks/brain_recovery_tripair_optimizer_step_v4_safe.py"),
        Path("benchmarks/brain_recovery_tripair_optimizer_step_v5_trajectory.py"),
        Path(".github/workflows/brain-recovery-tripair-optimizer-step-v5-trajectory.yml"),
    ]
    return {
        "checkout_head": run_git("rev-parse", "HEAD"),
        "pull_request_head": run_git("rev-parse", "HEAD^2"),
        "base": run_git("rev-parse", "HEAD^1"),
        "files": {
            str(path): {
                "git_blob": run_git("hash-object", str(path)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
    }


def build_receipt() -> dict[str, object]:
    started = time.perf_counter()
    cases = [
        run_case(batch_size, steps, 3.0e-4)
        for steps in STEP_COUNTS
        for batch_size in BATCH_SIZES
    ]
    zero_lr = run_case(64, 20, 0.0)
    deterministic_first = run_case(64, 20, 3.0e-4)
    deterministic_second = run_case(64, 20, 3.0e-4)
    deterministic_equal = (
        deterministic_summary(deterministic_first)
        == deterministic_summary(deterministic_second)
    )
    expected_pairs = {
        (batch_size, steps) for steps in STEP_COUNTS for batch_size in BATCH_SIZES
    }
    actual_pairs = {(case["batch_size"], case["steps"]) for case in cases}
    diagnostic_completed = bool(
        actual_pairs == expected_pairs
        and all(case["rollback"]["baseline_model_error"] == 0.0 for case in cases)
        and all(case["rollback"]["candidate_model_error"] == 0.0 for case in cases)
        and all(case["rollback"]["baseline_optimizer_exact"] for case in cases)
        and all(case["rollback"]["candidate_optimizer_exact"] for case in cases)
        and zero_lr["rollback"]["baseline_model_error"] == 0.0
        and zero_lr["rollback"]["candidate_model_error"] == 0.0
        and deterministic_equal
    )
    return {
        "schema_version": "qsa.brain-recovery-tripair-optimizer-step.v5-trajectory",
        "purpose": "Separate same-state QSA/PyTorch backend parity from accumulated matched-optimizer trajectory divergence.",
        "source": source_identity(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": first_cpu_model_name(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "compiler": subprocess.run(
                ("c++", "--version"), capture_output=True, text=True, check=True
            ).stdout.splitlines()[0],
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "active_trainable_parameters": v1.EXPECTED_PARAMETER_COUNT,
            "batch_sizes": list(BATCH_SIZES),
            "step_counts": list(STEP_COUNTS),
            "learning_rate": 3.0e-4,
            "zero_learning_rate_control": {"batch_size": 64, "steps": 20},
            "deterministic_rerun": {"batch_size": 64, "steps": 20},
            "candidate": "safe ephemeral QSA forward plus component weighted adjoint backward",
            "baseline": "PyTorch exact complex reverse-mode autograd",
            "same_state_shadow": "PyTorch exact reverse mode evaluated on the candidate's pre-update parameters",
            "persistent_candidate_reuse": False,
            "runtime_source_changed": False,
            "tolerances": TOLERANCES,
        },
        "cases": cases,
        "zero_learning_rate_control": zero_lr,
        "deterministic_rerun": {
            "equal": deterministic_equal,
            "first": deterministic_summary(deterministic_first),
            "second": deterministic_summary(deterministic_second),
        },
        "all_cases_passed": all(case["passed"] for case in cases),
        "diagnostic_completed": diagnostic_completed,
        "elapsed_seconds_including_all_diagnostic_arms": time.perf_counter() - started,
        "classification_policy": {
            "same_state_backend_failure": "A QSA forward or VJP implementation discrepancy at identical parameters.",
            "trajectory_only_failure": "An accumulated numerical reduction or optimizer-state divergence despite same-state backend parity.",
            "zero_lr_pass": "Parameter updates are required for the divergence.",
            "zero_lr_fail": "The divergence exists without parameter updates and remains backend or harness facing.",
        },
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = build_receipt()
    output = Path("brain-recovery-tripair-optimizer-step-v5-trajectory.json")
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["diagnostic_completed"]:
        raise SystemExit("trajectory diagnostic did not complete its declared contract")


if __name__ == "__main__":
    main()
