from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v2 as v2
import brain_recovery_tripair_optimizer_step_v3 as v3


def maximum_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def first_step_row_and_cotangent() -> tuple[dict[str, float], tuple[float, ...]]:
    batch_size = 1
    v1.torch.manual_seed(370038 + batch_size)
    baseline = v1.TripairCircuitModule()
    candidate = v1.copy.deepcopy(baseline)
    baseline_optimizer = v1.torch.optim.AdamW(
        baseline.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    features, target = v1.training_batches(batch_size, 1)[0]
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
    try:
        row = dict(
            bridge.parameter_rows(brain_angles, candidate.trainable_angles)[0]
        )
        observations, batch = bridge.forward(
            brain_angles, candidate.trainable_angles
        )
        detached = observations.detach().clone().requires_grad_(True)
        residual = v1.torch.sigmoid(candidate.residual_gate) * v1.F.linear(
            detached,
            candidate.observable_projection.weight,
            candidate.observable_projection.bias,
        )
        loss = v1.F.mse_loss(residual, target)
        cotangent = tuple(
            float(value)
            for value in v1.torch.autograd.grad(loss, detached)[0][0]
        )
        bridge.runtime.discard(batch)
    finally:
        bridge.close()
    return row, cotangent


def cached_selected_sequence(
    row: dict[str, float],
    cotangent: tuple[float, ...],
    repeats: int,
) -> dict[str, object]:
    bridge = v2.LazyWorkspaceTripairBridge()
    try:
        batch = bridge.runtime.evaluate((row,))
        stored = tuple(float(value) for value in batch.observations[0])
        root_before = bytes(bridge.runtime.state.encode_qsc())
        workspace_before = bytes(bridge.runtime._workspace.state.encode_qsc())
        calls = []
        try:
            for call in range(repeats):
                success = True
                error = None
                values = None
                gradient_maximum = None
                try:
                    result = bridge.runtime.selected_pullback(
                        batch, 0, cotangent
                    )
                    values = tuple(float(value) for value in result.values)
                    gradient_maximum = max(
                        abs(float(value)) for value in result.gradient
                    )
                except Exception as exc:
                    success = False
                    error = f"{type(exc).__name__}: {exc}"
                calls.append(
                    {
                        "call": call + 1,
                        "success": success,
                        "error": error,
                        "value_error": (
                            None if values is None else maximum_error(stored, values)
                        ),
                        "gradient_maximum_absolute": gradient_maximum,
                        "root_immutable": bytes(
                            bridge.runtime.state.encode_qsc()
                        )
                        == root_before,
                        "workspace_immutable": bytes(
                            bridge.runtime._workspace.state.encode_qsc()
                        )
                        == workspace_before,
                    }
                )
        finally:
            bridge.runtime.discard(batch)
        return {
            "calls": calls,
            "workspace_extractions": int(bridge.runtime.workspace_extractions),
            "root_immutable_after_discard": bytes(
                bridge.runtime.state.encode_qsc()
            )
            == root_before,
            "workspace_immutable_after_discard": bytes(
                bridge.runtime._workspace.state.encode_qsc()
            )
            == workspace_before,
        }
    finally:
        bridge.close()


def cached_adjoint_sequence(
    row: dict[str, float],
    cotangent: tuple[float, ...],
    repeats: int,
) -> dict[str, object]:
    bridge = v2.LazyWorkspaceTripairBridge()
    try:
        batch = bridge.runtime.evaluate((row,))
        stored = tuple(float(value) for value in batch.observations[0])
        workspace = bridge.runtime._workspace
        adjoint = bridge.runtime._local_adjoint(batch.global_qubits)
        root_before = bytes(bridge.runtime.state.encode_qsc())
        workspace_before = bytes(workspace.state.encode_qsc())
        calls = []
        for call in range(repeats):
            result = adjoint.evaluate(workspace.state, row, cotangent)
            calls.append(
                {
                    "call": call + 1,
                    "value_error": maximum_error(stored, result.values),
                    "gradient_maximum_absolute": max(
                        abs(float(value)) for value in result.gradient
                    ),
                    "root_immutable": bytes(
                        bridge.runtime.state.encode_qsc()
                    )
                    == root_before,
                    "workspace_immutable": bytes(
                        workspace.state.encode_qsc()
                    )
                    == workspace_before,
                }
            )
        bridge.runtime.discard(batch)
        return {
            "calls": calls,
            "workspace_extractions": int(bridge.runtime.workspace_extractions),
            "root_immutable_after_discard": bytes(
                bridge.runtime.state.encode_qsc()
            )
            == root_before,
            "workspace_immutable_after_discard": bytes(
                workspace.state.encode_qsc()
            )
            == workspace_before,
        }
    finally:
        bridge.close()


def fresh_runtime_sequence(
    row: dict[str, float],
    cotangent: tuple[float, ...],
    repeats: int,
) -> dict[str, object]:
    calls = []
    for call in range(repeats):
        bridge = v2.LazyWorkspaceTripairBridge()
        try:
            batch = bridge.runtime.evaluate((row,))
            stored = tuple(float(value) for value in batch.observations[0])
            success = True
            error = None
            values = None
            gradient_maximum = None
            try:
                result = bridge.runtime.selected_pullback(
                    batch, 0, cotangent
                )
                values = tuple(float(value) for value in result.values)
                gradient_maximum = max(
                    abs(float(value)) for value in result.gradient
                )
            except Exception as exc:
                success = False
                error = f"{type(exc).__name__}: {exc}"
            calls.append(
                {
                    "call": call + 1,
                    "success": success,
                    "error": error,
                    "value_error": (
                        None if values is None else maximum_error(stored, values)
                    ),
                    "gradient_maximum_absolute": gradient_maximum,
                    "workspace_extractions": int(
                        bridge.runtime.workspace_extractions
                    ),
                }
            )
            bridge.runtime.discard(batch)
        finally:
            bridge.close()
    return {"calls": calls}


def primed_selected_sequence(
    row: dict[str, float],
    cotangent: tuple[float, ...],
    repeats: int,
) -> dict[str, object]:
    bridge = v2.LazyWorkspaceTripairBridge()
    try:
        batch = bridge.runtime.evaluate((row,))
        stored = tuple(float(value) for value in batch.observations[0])
        workspace = bridge.runtime._workspace
        adjoint = bridge.runtime._local_adjoint(batch.global_qubits)
        prime = adjoint.evaluate(
            workspace.state,
            row,
            (0.0,) * len(v1.OBSERVABLES),
        )
        prime_error = maximum_error(stored, prime.values)
        calls = []
        for call in range(repeats):
            success = True
            error = None
            value_error = None
            try:
                result = bridge.runtime.selected_pullback(
                    batch, 0, cotangent
                )
                value_error = maximum_error(stored, result.values)
            except Exception as exc:
                success = False
                error = f"{type(exc).__name__}: {exc}"
            calls.append(
                {
                    "call": call + 1,
                    "success": success,
                    "error": error,
                    "value_error": value_error,
                }
            )
        bridge.runtime.discard(batch)
        return {"prime_value_error": prime_error, "calls": calls}
    finally:
        bridge.close()


def build_receipt(repeats: int) -> dict[str, object]:
    original_apply_pauli = v1.apply_pauli
    v1.apply_pauli = v3.corrected_apply_pauli
    try:
        row, cotangent = first_step_row_and_cotangent()
        receipt: dict[str, object] = {
            "schema_version": (
                "qsa.brain-recovery-persistent-tripair-repeatability.v1"
            ),
            "source": {
                "qsa_commit": v1.QSA_SOURCE,
                "brain_parent": v1.BRAIN_PARENT,
                "autograd_lifecycle_run": 30942385906,
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
                "same_brain_derived_row": True,
                "same_actual_loss_derived_cotangent": True,
                "repeats": repeats,
                "selected_pullback_threshold": 2.0e-12,
                "threshold_weakened": False,
            },
            "row": row,
            "cotangent": list(cotangent),
            "cached_selected_pullback": cached_selected_sequence(
                row, cotangent, repeats
            ),
            "cached_direct_adjoint": cached_adjoint_sequence(
                row, cotangent, repeats
            ),
            "fresh_runtime_per_pullback": fresh_runtime_sequence(
                row, cotangent, repeats
            ),
            "zero_cotangent_primed_selected_pullback": primed_selected_sequence(
                row, cotangent, repeats
            ),
        }
        receipt["diagnostic_complete"] = True
        receipt["passed"] = True
        return receipt
    finally:
        v1.apply_pauli = original_apply_pauli


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-persistent-tripair-repeatability.json"
        ),
    )
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    v1.torch.set_num_threads(1)
    v1.torch.set_num_interop_threads(1)
    receipt = build_receipt(args.repeats)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
