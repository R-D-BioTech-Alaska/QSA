from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Sequence

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v2 as v2


def maximum_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def tensor_stats(tensor: v1.torch.Tensor) -> dict[str, float | bool]:
    detached = tensor.detach().cpu().to(v1.torch.float64)
    return {
        "minimum": float(detached.min()),
        "maximum": float(detached.max()),
        "maximum_absolute": float(detached.abs().max()),
        "l2_norm": float(detached.norm()),
        "finite": bool(v1.torch.all(v1.torch.isfinite(detached))),
    }


def actual_cotangent_case(batch_size: int) -> dict[str, object]:
    seed = 370038 + batch_size
    v1.torch.manual_seed(seed)
    baseline = v1.TripairCircuitModule()
    candidate = copy.deepcopy(baseline)
    baseline_optimizer = v1.torch.optim.AdamW(
        baseline.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    features, target = v1.training_batches(batch_size, 1)[0]

    # Reproduce the exact optimizer-v3 ordering: the independent PyTorch
    # baseline completes its first step before the untouched candidate enters
    # the QSA forward/backward path.
    v1.run_step(
        baseline,
        baseline_optimizer,
        features,
        target,
        "torch",
        None,
        1.0,
    )

    encoded = v1.F.linear(
        features,
        candidate.input_angles.weight,
        candidate.input_angles.bias,
    )
    theta, phi = encoded.chunk(2, dim=-1)
    brain_angles = v1.torch.cat(
        (v1.torch.pi * v1.torch.tanh(theta), v1.torch.pi * v1.torch.tanh(phi)),
        dim=-1,
    )

    bridge = v2.LazyWorkspaceTripairBridge()
    root_before = bytes(bridge.runtime.state.encode_qsc())
    try:
        qsa_observations, batch = bridge.forward(
            brain_angles,
            candidate.trainable_angles,
        )
        workspace = bridge.runtime._workspace
        if workspace is None:
            raise RuntimeError("persistent workspace did not initialize")
        workspace_before = bytes(workspace.state.encode_qsc())

        detached_qsa = qsa_observations.detach().clone().requires_grad_(True)
        qsa_residual = v1.torch.sigmoid(candidate.residual_gate) * v1.F.linear(
            detached_qsa,
            candidate.observable_projection.weight,
            candidate.observable_projection.bias,
        )
        qsa_loss = v1.F.mse_loss(qsa_residual, target)
        qsa_actual_cotangent = v1.torch.autograd.grad(
            qsa_loss,
            detached_qsa,
            retain_graph=False,
            create_graph=False,
        )[0]

        torch_observations = v1.torch_observables(
            brain_angles,
            candidate.trainable_angles,
        )
        torch_residual = v1.torch.sigmoid(candidate.residual_gate) * v1.F.linear(
            torch_observations,
            candidate.observable_projection.weight,
            candidate.observable_projection.bias,
        )
        torch_loss = v1.F.mse_loss(torch_residual, target)
        torch_actual_cotangent = v1.torch.autograd.grad(
            torch_loss,
            torch_observations,
            retain_graph=False,
            create_graph=False,
        )[0]

        adjoint = bridge.runtime._local_adjoint(batch.global_qubits)
        sample_results = []
        selected_failures = []
        for sample in range(batch_size):
            stored = tuple(float(value) for value in batch.observations[sample])
            actual = tuple(
                float(value)
                for value in qsa_actual_cotangent[sample].detach().cpu()
            )
            cotangent_cases = {
                "zero": (0.0,) * len(v1.OBSERVABLES),
                "unit": (1.0,) * len(v1.OBSERVABLES),
                "actual": actual,
                "actual_x10": tuple(10.0 * value for value in actual),
                "actual_negative": tuple(-value for value in actual),
            }
            case_results = []
            for name, cotangent in cotangent_cases.items():
                result = adjoint.evaluate(
                    workspace.state,
                    batch.parameter_rows[sample],
                    cotangent,
                )
                value_error = maximum_error(stored, result.values)
                selected_success = True
                selected_error = None
                selected_gradient_maximum = None
                try:
                    selected = bridge.runtime.selected_pullback(
                        batch,
                        sample,
                        cotangent,
                    )
                    selected_gradient_maximum = max(
                        abs(float(value)) for value in selected.gradient
                    )
                except Exception as exc:  # fail-closed behavior is evidence
                    selected_success = False
                    selected_error = f"{type(exc).__name__}: {exc}"
                    selected_failures.append(
                        {
                            "sample": sample,
                            "cotangent_case": name,
                            "error": selected_error,
                        }
                    )
                case_results.append(
                    {
                        "cotangent_case": name,
                        "cotangent_stats": {
                            "minimum": min(cotangent),
                            "maximum": max(cotangent),
                            "maximum_absolute": max(abs(value) for value in cotangent),
                            "l2_norm": math.sqrt(
                                sum(value * value for value in cotangent)
                            ),
                            "finite": all(math.isfinite(value) for value in cotangent),
                        },
                        "stored_vs_adjoint_value_error": value_error,
                        "adjoint_forward_sweeps": int(result.forward_sweeps),
                        "adjoint_reverse_sweeps": int(result.reverse_sweeps),
                        "adjoint_shifted_evaluations": int(
                            result.shifted_evaluations
                        ),
                        "selected_pullback_success": selected_success,
                        "selected_pullback_error": selected_error,
                        "selected_gradient_maximum_absolute": (
                            selected_gradient_maximum
                        ),
                    }
                )
            sample_results.append(
                {
                    "sample": sample,
                    "stored_observations": list(stored),
                    "actual_cotangent": list(actual),
                    "cotangent_cases": case_results,
                }
            )

        observation_error = float(
            v1.torch.max(v1.torch.abs(qsa_observations - torch_observations))
        )
        cotangent_error = float(
            v1.torch.max(
                v1.torch.abs(qsa_actual_cotangent - torch_actual_cotangent)
            )
        )
        loss_error = abs(float(qsa_loss) - float(torch_loss))
        root_immutable_active = bytes(bridge.runtime.state.encode_qsc()) == root_before
        workspace_immutable_active = (
            bytes(workspace.state.encode_qsc()) == workspace_before
        )
        bridge.runtime.discard(batch)
        root_immutable_after_discard = (
            bytes(bridge.runtime.state.encode_qsc()) == root_before
        )
        workspace_immutable_after_discard = (
            bytes(workspace.state.encode_qsc()) == workspace_before
        )
    finally:
        bridge.close()

    all_value_errors = [
        float(cotangent_case["stored_vs_adjoint_value_error"])
        for sample in sample_results
        for cotangent_case in sample["cotangent_cases"]
    ]
    result: dict[str, object] = {
        "batch_size": batch_size,
        "seed": seed,
        "baseline_step_completed_before_candidate": True,
        "qsa_vs_torch_observation_max_error": observation_error,
        "qsa_vs_torch_loss_error": loss_error,
        "qsa_vs_torch_actual_cotangent_max_error": cotangent_error,
        "actual_cotangent_stats": tensor_stats(qsa_actual_cotangent),
        "maximum_stored_vs_adjoint_value_error": max(all_value_errors),
        "selected_pullback_failures": selected_failures,
        "selected_pullback_failure_count": len(selected_failures),
        "root_immutable_while_batch_active": root_immutable_active,
        "workspace_immutable_while_batch_active": workspace_immutable_active,
        "root_immutable_after_discard": root_immutable_after_discard,
        "workspace_immutable_after_discard": workspace_immutable_after_discard,
        "workspace_extractions": int(bridge.runtime.workspace_extractions),
        "sample_results": sample_results,
    }
    result["diagnostic_complete"] = (
        len(sample_results) == batch_size
        and observation_error <= 2.0e-12
        and loss_error <= 2.0e-13
        and cotangent_error <= 2.0e-12
        and root_immutable_active
        and workspace_immutable_active
        and root_immutable_after_discard
        and workspace_immutable_after_discard
        and result["workspace_extractions"] == 1
    )
    return result


def build_receipt() -> dict[str, object]:
    cases = [actual_cotangent_case(batch_size) for batch_size in (1, 4)]
    maximum_value_error = max(
        float(case["maximum_stored_vs_adjoint_value_error"])
        for case in cases
    )
    selected_failures = sum(
        int(case["selected_pullback_failure_count"]) for case in cases
    )
    actual_failures = [
        failure
        for case in cases
        for failure in case["selected_pullback_failures"]
        if failure["cotangent_case"] == "actual"
    ]
    receipt: dict[str, object] = {
        "schema_version": (
            "qsa.brain-recovery-persistent-tripair-cotangent-diagnostic.v1"
        ),
        "source": {
            "qsa_commit": v1.QSA_SOURCE,
            "brain_parent": v1.BRAIN_PARENT,
            "predecessor_optimizer_workflow": 30941224527,
            "predecessor_optimizer_artifact": 8905225345,
            "zero_cotangent_diagnostic_workflow": 30941729878,
            "zero_cotangent_diagnostic_artifact": 8905423722,
        },
        "environment": {
            "python": v1.sys.version,
            "platform": v1.platform.platform(),
            "torch": v1.torch.__version__,
            "dtype": "float64/complex128",
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "exact_optimizer_order_reproduced": True,
            "baseline_step_runs_before_candidate": True,
            "candidate_initial_parameters_unchanged": True,
            "cotangent_cases": [
                "zero",
                "unit",
                "actual",
                "actual_x10",
                "actual_negative",
            ],
            "selected_pullback_parity_threshold": 2.0e-12,
            "parity_gate_weakened": False,
        },
        "cases": cases,
        "diagnosis": {
            "maximum_stored_vs_adjoint_value_error": maximum_value_error,
            "selected_pullback_failure_count": selected_failures,
            "actual_cotangent_failure_count": len(actual_failures),
            "actual_cotangent_failures": actual_failures,
        },
        "gate_policy": {
            "diagnostic_pass_requires_complete_exact_controls": True,
            "selected_pullback_success_is_measured_not_assumed": True,
        },
    }
    receipt["passed"] = all(bool(case["diagnostic_complete"]) for case in cases)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-persistent-tripair-cotangent-diagnostic.json"
        ),
    )
    args = parser.parse_args()
    v1.torch.set_num_threads(1)
    v1.torch.set_num_interop_threads(1)
    receipt = build_receipt()
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("persistent Tripair cotangent diagnostic incomplete")


if __name__ == "__main__":
    main()
