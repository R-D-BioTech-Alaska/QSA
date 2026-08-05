from __future__ import annotations

import argparse
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
from typing import Callable

import numpy as np
import psutil
import torch
from torch.nn import functional as F

import brain_recovery_tripair_optimizer_step_v10_contiguous as v10


v9 = v10.v9
v8 = v10.v8
v7 = v10.v7
v1 = v10.v1
v3 = v9.v8.v7.v3
v4 = v10.v4
import brain_recovery_tripair_optimizer_step_v5_trajectory as v5


BATCH_SIZE = 64
TRAIN_STEPS = 64
TRAIN_BATCH_COUNT = 8
HELDOUT_BATCH_COUNT = 4
CHECKPOINT_STEPS = (1, 4, 16, 32, 64)
LEARNING_RATE = 3.0e-4
CLIP_NORM = 1.0
TOLERANCES = {
    "output": 4.0e-12,
    "observable": 4.0e-12,
    "loss": 4.0e-13,
    "gradient": 4.0e-11,
    "parameter": 4.0e-12,
    "optimizer": 4.0e-12,
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    backend: str
    bridge_type: type | None
    learning_rate: float
    gradients_enabled: bool


@dataclass
class ArmRun:
    receipt: dict[str, object]
    final_model: dict[str, torch.Tensor]
    final_optimizer: dict[str, object]
    checkpoints: dict[int, dict[str, object]]


def first_cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def state_bytes(state: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def state_sha256(state: object) -> str:
    return hashlib.sha256(state_bytes(state)).hexdigest()


def fixed_batches(
    seed: int,
    batch_count: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for batch_index in range(batch_count):
        features = torch.randn(
            BATCH_SIZE,
            v1.FEATURE_WIDTH,
            generator=generator,
            dtype=torch.float64,
        )
        target = 0.18 * torch.tanh(
            torch.roll(features, shifts=batch_index % 11 + 1, dims=1)
            + 0.37 * torch.roll(features, shifts=7, dims=1)
        )
        batches.append((features, target))
    return tuple(batches)


def dataset_receipt(batches) -> dict[str, object]:
    digest = hashlib.sha256()
    for features, target in batches:
        digest.update(features.numpy().tobytes(order="C"))
        digest.update(target.numpy().tobytes(order="C"))
    return {
        "batch_count": len(batches),
        "batch_size": BATCH_SIZE,
        "examples": len(batches) * BATCH_SIZE,
        "feature_width": v1.FEATURE_WIDTH,
        "dtype": "float64",
        "sha256": digest.hexdigest(),
    }


def evaluation(
    model: torch.nn.Module,
    batches,
) -> dict[str, object]:
    previous = model.training
    model.eval()
    losses = []
    output_norms = []
    observable_norms = []
    with torch.no_grad():
        for features, target in batches:
            output, observables = model(features, "torch", None)
            losses.append(float(F.mse_loss(output, target)))
            output_norms.append(float(torch.linalg.vector_norm(output)))
            observable_norms.append(float(torch.linalg.vector_norm(observables)))
    model.train(previous)
    return {
        "mean_loss": statistics.fmean(losses),
        "losses": losses,
        "mean_output_norm": statistics.fmean(output_norms),
        "mean_observable_norm": statistics.fmean(observable_norms),
    }


def parameter_delta(
    current: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
) -> dict[str, object]:
    maximum = 0.0
    squared = 0.0
    changed = 0
    total = 0
    per_parameter = {}
    for name, value in current.items():
        difference = value.detach() - initial[name]
        absolute = torch.abs(difference)
        local_maximum = float(torch.max(absolute))
        local_l2 = float(torch.linalg.vector_norm(difference))
        per_parameter[name] = {
            "maximum": local_maximum,
            "l2": local_l2,
        }
        maximum = max(maximum, local_maximum)
        squared += local_l2 * local_l2
        changed += int(torch.count_nonzero(absolute).item())
        total += difference.numel()
    return {
        "maximum": maximum,
        "l2": squared ** 0.5,
        "changed_scalars": changed,
        "total_scalars": total,
        "per_parameter": per_parameter,
    }


def gradients_receipt(model: torch.nn.Module) -> tuple[bool, dict[str, float]]:
    norms = {}
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            norms[name] = float("nan")
            finite = False
            continue
        norms[name] = float(torch.linalg.vector_norm(parameter.grad))
        finite = finite and bool(torch.all(torch.isfinite(parameter.grad)))
    return finite, norms


def checkpoint_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    model_state = v1.clone_state(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return {
        "model": model_state,
        "optimizer": optimizer_state,
        "model_sha256": state_sha256(model_state),
        "optimizer_sha256": state_sha256(optimizer_state),
    }


def run_arm(
    spec: ArmSpec,
    initial_model: dict[str, torch.Tensor],
    train_batches,
    heldout_batches,
) -> ArmRun:
    model = v1.TripairCircuitModule()
    model.load_state_dict(initial_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_model_local = v1.clone_state(model.state_dict())
    bridge = spec.bridge_type() if spec.bridge_type is not None else None
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    initial_train = evaluation(model, train_batches)
    initial_heldout = evaluation(model, heldout_batches)
    losses = []
    gradient_norms = []
    raw_gradient_norms = []
    step_seconds = []
    checkpoints: dict[int, dict[str, object]] = {}
    all_finite = True
    no_gradient_steps = 0
    original_apply_pauli = v1.apply_pauli
    v1.apply_pauli = v3.corrected_apply_pauli
    try:
        for step in range(1, TRAIN_STEPS + 1):
            features, target = train_batches[(step - 1) % len(train_batches)]
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            if spec.gradients_enabled:
                output, observables = model(features, spec.backend, bridge)
                loss = F.mse_loss(output, target)
                forward_done = time.perf_counter()
                loss.backward()
                backward_done = time.perf_counter()
                finite, per_parameter_norms = gradients_receipt(model)
                all_finite = all_finite and finite
                raw_norm = float(
                    torch.sqrt(
                        sum(
                            torch.sum(parameter.grad.detach() ** 2)
                            for parameter in model.parameters()
                        )
                    )
                )
                clipped_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                )
                optimizer.step()
                finished = time.perf_counter()
                gradient_norms.append(clipped_norm)
                raw_gradient_norms.append(raw_norm)
            else:
                with torch.no_grad():
                    output, observables = model(features, spec.backend, bridge)
                    loss = F.mse_loss(output, target)
                forward_done = time.perf_counter()
                backward_done = forward_done
                finished = forward_done
                per_parameter_norms = {}
                no_gradient_steps += int(
                    all(parameter.grad is None for parameter in model.parameters())
                )
            losses.append(float(loss))
            step_seconds.append(finished - started)
            peak_rss = max(peak_rss, process.memory_info().rss)
            if step in CHECKPOINT_STEPS:
                snapshot = checkpoint_snapshot(model, optimizer)
                snapshot.update(
                    {
                        "step": step,
                        "loss": float(loss),
                        "gradient_norm": (
                            gradient_norms[-1] if gradient_norms else None
                        ),
                        "raw_gradient_norm": (
                            raw_gradient_norms[-1] if raw_gradient_norms else None
                        ),
                        "per_parameter_gradient_norm": per_parameter_norms,
                        "parameter_delta": parameter_delta(
                            model.state_dict(), initial_model_local
                        ),
                    }
                )
                checkpoints[step] = snapshot

        final_train = evaluation(model, train_batches)
        final_heldout = evaluation(model, heldout_batches)
        final_model = v1.clone_state(model.state_dict())
        final_optimizer = copy.deepcopy(optimizer.state_dict())
        movement = parameter_delta(final_model, initial_model_local)
        bridge_state = bridge.state_receipt() if bridge is not None else None
        elapsed = sum(step_seconds)
        receipt = {
            "name": spec.name,
            "backend": spec.backend,
            "bridge": (
                spec.bridge_type.__name__ if spec.bridge_type is not None else None
            ),
            "learning_rate": spec.learning_rate,
            "gradients_enabled": spec.gradients_enabled,
            "active_trainable_parameters": v1.count_parameters(model),
            "parameter_names": list(dict(model.named_parameters()).keys()),
            "parameter_shapes": {
                name: list(parameter.shape)
                for name, parameter in model.named_parameters()
            },
            "initial_train": initial_train,
            "final_train": final_train,
            "initial_heldout": initial_heldout,
            "final_heldout": final_heldout,
            "train_loss_change": (
                final_train["mean_loss"] - initial_train["mean_loss"]
            ),
            "heldout_loss_change": (
                final_heldout["mean_loss"] - initial_heldout["mean_loss"]
            ),
            "step_losses": losses,
            "gradient_norms": gradient_norms,
            "raw_gradient_norms": raw_gradient_norms,
            "all_gradients_finite": all_finite,
            "no_gradient_steps": no_gradient_steps,
            "parameter_movement": movement,
            "final_model_sha256": state_sha256(final_model),
            "final_optimizer_sha256": state_sha256(final_optimizer),
            "initial_optimizer_sha256": state_sha256(initial_optimizer),
            "timing": {
                "elapsed_seconds": elapsed,
                "median_step_ms": 1000.0 * statistics.median(step_seconds),
                "p95_step_ms": 1000.0 * sorted(step_seconds)[
                    min(len(step_seconds) - 1, round(0.95 * (len(step_seconds) - 1)))
                ],
                "examples_per_second": (
                    BATCH_SIZE * TRAIN_STEPS / elapsed if elapsed > 0.0 else 0.0
                ),
            },
            "memory": {
                "peak_process_rss_bytes": peak_rss,
                "process_rss_after_bytes": process.memory_info().rss,
                "resource_maxrss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            },
            "qsa": bridge_state,
            "checkpoints": {
                str(step): {
                    key: value
                    for key, value in snapshot.items()
                    if key not in ("model", "optimizer")
                }
                for step, snapshot in checkpoints.items()
            },
        }

        model.load_state_dict(initial_model_local)
        optimizer.load_state_dict(initial_optimizer)
        receipt["rollback"] = {
            "model_max_error": v1.maximum_state_error(
                model.state_dict(), initial_model_local
            ),
            "optimizer_exact": v1.optimizer_state_equal(
                optimizer.state_dict(), initial_optimizer
            ),
        }
    finally:
        v1.apply_pauli = original_apply_pauli
        if bridge is not None:
            bridge.close()

    return ArmRun(
        receipt=receipt,
        final_model=final_model,
        final_optimizer=final_optimizer,
        checkpoints=checkpoints,
    )


def compare_runs(reference: ArmRun, candidate: ArmRun) -> dict[str, object]:
    checkpoint_rows = {}
    maximum_parameter_error = 0.0
    maximum_optimizer_error = 0.0
    for step in CHECKPOINT_STEPS:
        left = reference.checkpoints[step]
        right = candidate.checkpoints[step]
        parameter_error = v1.maximum_state_error(left["model"], right["model"])
        optimizer_error, _ = v5.optimizer_numeric_error(
            left["optimizer"], right["optimizer"]
        )
        maximum_parameter_error = max(maximum_parameter_error, parameter_error)
        maximum_optimizer_error = max(maximum_optimizer_error, optimizer_error)
        checkpoint_rows[str(step)] = {
            "parameter_error": parameter_error,
            "optimizer_error": optimizer_error,
            "loss_error": abs(float(left["loss"]) - float(right["loss"])),
            "gradient_norm_error": abs(
                float(left["gradient_norm"]) - float(right["gradient_norm"])
            ),
        }
    final_parameter_error = v1.maximum_state_error(
        reference.final_model, candidate.final_model
    )
    final_optimizer_error, final_optimizer_by_state = v5.optimizer_numeric_error(
        reference.final_optimizer, candidate.final_optimizer
    )
    losses = zip(
        reference.receipt["step_losses"],
        candidate.receipt["step_losses"],
    )
    maximum_loss_error = max(
        abs(float(left) - float(right)) for left, right in losses
    )
    return {
        "checkpoint_errors": checkpoint_rows,
        "maximum_checkpoint_parameter_error": maximum_parameter_error,
        "maximum_checkpoint_optimizer_error": maximum_optimizer_error,
        "final_parameter_error": final_parameter_error,
        "final_optimizer_error": final_optimizer_error,
        "final_optimizer_error_by_state": final_optimizer_by_state,
        "maximum_loss_trajectory_error": maximum_loss_error,
        "heldout_mean_loss_error": abs(
            float(reference.receipt["final_heldout"]["mean_loss"])
            - float(candidate.receipt["final_heldout"]["mean_loss"])
        ),
        "exact_final_model_hash": (
            reference.receipt["final_model_sha256"]
            == candidate.receipt["final_model_sha256"]
        ),
        "exact_final_optimizer_hash": (
            reference.receipt["final_optimizer_sha256"]
            == candidate.receipt["final_optimizer_sha256"]
        ),
    }


def deterministic_summary(run: ArmRun) -> dict[str, object]:
    return {
        "final_model_sha256": run.receipt["final_model_sha256"],
        "final_optimizer_sha256": run.receipt["final_optimizer_sha256"],
        "step_losses": run.receipt["step_losses"],
        "gradient_norms": run.receipt["gradient_norms"],
        "raw_gradient_norms": run.receipt["raw_gradient_norms"],
        "final_train": run.receipt["final_train"],
        "final_heldout": run.receipt["final_heldout"],
        "parameter_movement": run.receipt["parameter_movement"],
        "checkpoints": run.receipt["checkpoints"],
        "rollback": run.receipt["rollback"],
    }


def git_head(path: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_receipt() -> dict[str, object]:
    torch.manual_seed(370011)
    template = v1.TripairCircuitModule()
    if v1.count_parameters(template) != v1.EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Tripair parameter contract changed")
    initial_model = v1.clone_state(template.state_dict())
    train_batches = fixed_batches(2026080411, TRAIN_BATCH_COUNT)
    heldout_batches = fixed_batches(2026080499, HELDOUT_BATCH_COUNT)

    specs = (
        ArmSpec("pytorch_control", "torch", None, LEARNING_RATE, True),
        ArmSpec(
            "safe_qsa_per_sample",
            "qsa",
            v4.SafeComponentTripairBridge,
            LEARNING_RATE,
            True,
        ),
        ArmSpec(
            "contiguous_qsa_v10",
            "qsa",
            v10.ContiguousVectorizedTripairBridge,
            LEARNING_RATE,
            True,
        ),
        ArmSpec(
            "contiguous_qsa_zero_lr",
            "qsa",
            v10.ContiguousVectorizedTripairBridge,
            0.0,
            True,
        ),
        ArmSpec(
            "contiguous_qsa_gradient_disabled",
            "qsa",
            v10.ContiguousVectorizedTripairBridge,
            LEARNING_RATE,
            False,
        ),
    )
    runs = {
        spec.name: run_arm(
            spec,
            initial_model,
            train_batches,
            heldout_batches,
        )
        for spec in specs
    }
    deterministic_repeat = run_arm(
        specs[2],
        initial_model,
        train_batches,
        heldout_batches,
    )

    pytorch = runs["pytorch_control"]
    safe = runs["safe_qsa_per_sample"]
    contiguous = runs["contiguous_qsa_v10"]
    zero_lr = runs["contiguous_qsa_zero_lr"]
    disabled = runs["contiguous_qsa_gradient_disabled"]
    safe_comparison = compare_runs(pytorch, safe)
    contiguous_comparison = compare_runs(pytorch, contiguous)
    deterministic = {
        "equal": (
            deterministic_summary(contiguous)
            == deterministic_summary(deterministic_repeat)
        ),
        "first": deterministic_summary(contiguous),
        "second": deterministic_summary(deterministic_repeat),
    }

    learning_arms = (pytorch, safe, contiguous)
    control_arms = (zero_lr, disabled)
    common_contract = bool(
        all(
            run.receipt["active_trainable_parameters"]
            == v1.EXPECTED_PARAMETER_COUNT
            for run in runs.values()
        )
        and all(
            run.receipt["parameter_names"]
            == pytorch.receipt["parameter_names"]
            for run in runs.values()
        )
        and all(
            run.receipt["parameter_shapes"]
            == pytorch.receipt["parameter_shapes"]
            for run in runs.values()
        )
    )
    learning_preserved = bool(
        all(run.receipt["all_gradients_finite"] for run in learning_arms)
        and all(
            run.receipt["parameter_movement"]["maximum"] > 0.0
            for run in learning_arms
        )
        and all(
            run.receipt["final_train"]["mean_loss"]
            < run.receipt["initial_train"]["mean_loss"]
            for run in learning_arms
        )
        and safe_comparison["final_parameter_error"] <= TOLERANCES["parameter"]
        and safe_comparison["final_optimizer_error"] <= TOLERANCES["optimizer"]
        and contiguous_comparison["final_parameter_error"]
        <= TOLERANCES["parameter"]
        and contiguous_comparison["final_optimizer_error"]
        <= TOLERANCES["optimizer"]
        and contiguous_comparison["maximum_loss_trajectory_error"]
        <= TOLERANCES["loss"]
    )
    controls_passed = bool(
        zero_lr.receipt["all_gradients_finite"]
        and zero_lr.receipt["parameter_movement"]["maximum"] == 0.0
        and disabled.receipt["parameter_movement"]["maximum"] == 0.0
        and disabled.receipt["no_gradient_steps"] == TRAIN_STEPS
        and all(
            run.receipt["rollback"]["model_max_error"] == 0.0
            and run.receipt["rollback"]["optimizer_exact"]
            for run in runs.values()
        )
    )
    qsa_contract = bool(
        safe.receipt["qsa"]["shifted_evaluations"] == 0
        and contiguous.receipt["qsa"]["shifted_evaluations"] == 0
        and zero_lr.receipt["qsa"]["shifted_evaluations"] == 0
        and disabled.receipt["qsa"]["shifted_evaluations"] == 0
        and safe.receipt["qsa"]["root_immutable"]
        and contiguous.receipt["qsa"]["root_immutable"]
        and zero_lr.receipt["qsa"]["root_immutable"]
        and disabled.receipt["qsa"]["root_immutable"]
    )

    repository_root = Path(__file__).resolve().parents[1]
    receipt: dict[str, object] = {
        "schema_version": "qsa.brain-recovery-tripair-learning-pilot.v11",
        "source": {
            "runtime_head": git_head(repository_root),
            "runtime_parent_exactness_repair": (
                "748e4a40483de72ea5e9092d1db45a62ec6b4a16"
            ),
            "frozen_evidence_head": git_head(v7.EVIDENCE_BENCHMARKS.parent),
            "accepted_brain_parent": v1.BRAIN_PARENT,
            "brain_quantum_source": v1.BRAIN_QUANTUM_SOURCE,
            "brain_quantum_blob": v1.BRAIN_QUANTUM_BLOB,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": first_cpu_model_name(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "active_trainable_parameters": v1.EXPECTED_PARAMETER_COUNT,
            "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS,
            "train_batch_count": TRAIN_BATCH_COUNT,
            "heldout_batch_count": HELDOUT_BATCH_COUNT,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "learning_rate": LEARNING_RATE,
            "gradient_clip_norm": CLIP_NORM,
            "optimizer": "AdamW",
            "identical_initial_model": True,
            "identical_batch_order": True,
            "identical_targets": True,
            "identical_parameter_names_shapes": common_contract,
            "persistent_candidate_reuse": False,
            "shifted_evaluations": 0,
            "runtime_source_changed": True,
            "tolerances": TOLERANCES,
        },
        "data": {
            "train": dataset_receipt(train_batches),
            "heldout": dataset_receipt(heldout_batches),
            "data_classification": (
                "deterministic source-bound Tripair learning workload; not "
                "Brain language, canonical, or safety data"
            ),
        },
        "arms": {name: run.receipt for name, run in runs.items()},
        "comparisons": {
            "safe_qsa_vs_pytorch": safe_comparison,
            "contiguous_qsa_vs_pytorch": contiguous_comparison,
            "contiguous_vs_safe_speed": (
                safe.receipt["timing"]["median_step_ms"]
                / contiguous.receipt["timing"]["median_step_ms"]
            ),
            "pytorch_vs_contiguous_speed": (
                pytorch.receipt["timing"]["median_step_ms"]
                / contiguous.receipt["timing"]["median_step_ms"]
            ),
        },
        "deterministic_contiguous_rerun": deterministic,
        "common_contract_passed": common_contract,
        "learning_preserved": learning_preserved,
        "controls_passed": controls_passed,
        "qsa_contract_passed": qsa_contract,
        "classification_policy": {
            "safe_path_exact": (
                "Same loss, parameter, optimizer, and held-out trajectories as "
                "PyTorch within declared tolerances."
            ),
            "real_learning_preservation": (
                "Nonzero learning-rate parameter movement and lower final mean "
                "training loss, with zero-LR and gradient-disabled controls stationary."
            ),
            "claim_boundary": (
                "This establishes bounded learning preservation for the exact "
                "source-bound Tripair workload. It does not establish Brain "
                "language/canonical/safety capability contribution or quantum uniqueness."
            ),
        },
        "not_established": [
            "Brain language learning preservation",
            "Brain canonical or structured-safety retention",
            "Brain capability contribution",
            "phase-specific contribution",
            "persistent selected-pullback safety",
            "quantum uniqueness",
            "physical-QPU advantage",
            "production readiness",
        ],
    }
    receipt["passed"] = bool(
        common_contract
        and learning_preserved
        and controls_passed
        and qsa_contract
        and deterministic["equal"]
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-learning-pilot-v11.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = build_receipt()
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("bounded Tripair learning pilot failed")


if __name__ == "__main__":
    main()
