from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Mapping, Sequence

from qsa.causal import CausalRegister

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v2 as v2
import brain_recovery_tripair_optimizer_step_v3 as v3


def maximum_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def stats(values: Sequence[float]) -> dict[str, float | bool]:
    return {
        "minimum": min(float(value) for value in values),
        "maximum": max(float(value) for value in values),
        "maximum_absolute": max(abs(float(value)) for value in values),
        "l2_norm": math.sqrt(sum(float(value) ** 2 for value in values)),
        "finite": all(math.isfinite(float(value)) for value in values),
    }


def direct_global_values(
    bridge: v2.LazyWorkspaceTripairBridge,
    row: Mapping[str, float] | Sequence[float],
) -> tuple[float, ...]:
    with CausalRegister(v1.QUBITS) as state:
        state.apply(bridge.plan, row)
        return tuple(float(value) for value in bridge.support.execute(state))


class LifecycleTracingBridge(v2.LazyWorkspaceTripairBridge):
    def __init__(self) -> None:
        super().__init__()
        self.last_batch = None
        self.last_brain_angles = None
        self.last_shared = None
        self.trace: dict[str, object] = {}

    def forward(self, brain_angles, shared):
        observations, batch = super().forward(brain_angles, shared)
        self.last_batch = batch
        self.last_brain_angles = brain_angles.detach().cpu().clone()
        self.last_shared = shared.detach().cpu().clone()
        self.trace["forward"] = {
            "batch_size": int(brain_angles.shape[0]),
            "brain_angles": self.last_brain_angles.tolist(),
            "shared_angles": self.last_shared.reshape(-1).tolist(),
            "observations": observations.detach().cpu().tolist(),
            "batch_parameter_rows": [list(row) for row in batch.parameter_rows],
            "batch_global_qubits": list(batch.global_qubits),
            "batch_local_qubits": int(batch.local_qubit_count),
            "root_qsc_bytes": len(self.runtime.state.encode_qsc()),
            "workspace_qsc_bytes": len(self.runtime._workspace.state.encode_qsc()),
            "workspace_extractions": int(self.runtime.workspace_extractions),
        }
        return observations, batch

    def _sample_trace(
        self,
        batch,
        sample: int,
        cotangent: Sequence[float],
    ) -> dict[str, object]:
        workspace = self.runtime._workspace
        if workspace is None:
            raise RuntimeError("persistent workspace is missing")
        local_plan, local_observables = self.runtime._local_runtime(
            batch.global_qubits
        )
        row = batch.parameter_rows[sample]
        stored = tuple(float(value) for value in batch.observations[sample])
        retained = batch._branches[sample]
        if retained is None:
            raise RuntimeError("retained candidate branch is missing")
        retained_values = tuple(
            float(value) for value in local_observables.execute(retained)
        )
        with workspace.state.fork() as branch:
            branch.apply(local_plan, row)
            single_values = tuple(
                float(value) for value in local_observables.execute(branch)
            )
        adjoint = self.runtime._local_adjoint(batch.global_qubits)
        zero_result = adjoint.evaluate(
            workspace.state,
            row,
            (0.0,) * len(v1.OBSERVABLES),
        )
        actual_result = adjoint.evaluate(
            workspace.state,
            row,
            tuple(float(value) for value in cotangent),
        )
        global_values = direct_global_values(self, row)
        selected_result = None
        selected_error = None
        try:
            selected_result = self.runtime.selected_pullback(
                batch,
                sample,
                tuple(float(value) for value in cotangent),
            )
        except Exception as exc:
            selected_error = f"{type(exc).__name__}: {exc}"
        return {
            "sample": sample,
            "parameter_row": list(row),
            "cotangent": [float(value) for value in cotangent],
            "cotangent_stats": stats(cotangent),
            "stored": list(stored),
            "retained": list(retained_values),
            "single": list(single_values),
            "adjoint_zero": list(zero_result.values),
            "adjoint_actual": list(actual_result.values),
            "global": list(global_values),
            "errors": {
                "stored_vs_retained": maximum_error(stored, retained_values),
                "stored_vs_single": maximum_error(stored, single_values),
                "stored_vs_adjoint_zero": maximum_error(
                    stored, zero_result.values
                ),
                "stored_vs_adjoint_actual": maximum_error(
                    stored, actual_result.values
                ),
                "stored_vs_global": maximum_error(stored, global_values),
                "single_vs_adjoint_actual": maximum_error(
                    single_values, actual_result.values
                ),
            },
            "zero_gradient_maximum_absolute": max(
                abs(float(value)) for value in zero_result.gradient
            ),
            "actual_gradient_maximum_absolute": max(
                abs(float(value)) for value in actual_result.gradient
            ),
            "selected_pullback_success": selected_result is not None,
            "selected_pullback_error": selected_error,
            "selected_gradient_maximum_absolute": (
                None
                if selected_result is None
                else max(abs(float(value)) for value in selected_result.gradient)
            ),
        }

    def backward(
        self,
        batch,
        cotangents,
        input_dtype,
        input_device,
        shared_shape,
    ):
        cotangent_rows = cotangents.detach().cpu().to(v1.torch.float64).tolist()
        self.trace["backward_entry"] = {
            "shape": list(cotangents.shape),
            "dtype": str(cotangents.dtype),
            "device": str(cotangents.device),
            "contiguous": bool(cotangents.is_contiguous()),
            "requires_grad": bool(cotangents.requires_grad),
            "rows": cotangent_rows,
            "row_stats": [stats(row) for row in cotangent_rows],
            "root_immutable": bytes(self.runtime.state.encode_qsc())
            == self._root_before,
            "workspace_immutable": bytes(
                self.runtime._workspace.state.encode_qsc()
            )
            == self._workspace_before,
        }
        self.trace["samples"] = [
            self._sample_trace(batch, sample, row)
            for sample, row in enumerate(cotangent_rows)
        ]
        try:
            result = super().backward(
                batch,
                cotangents,
                input_dtype,
                input_device,
                shared_shape,
            )
            self.trace["super_backward"] = {
                "success": True,
                "input_gradient_maximum_absolute": float(
                    result[0].detach().abs().max()
                ),
                "shared_gradient_maximum_absolute": float(
                    result[1].detach().abs().max()
                ),
            }
            return result
        except Exception as exc:
            self.trace["super_backward"] = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            raise


def build_receipt() -> dict[str, object]:
    original_apply_pauli = v1.apply_pauli
    original_bridge = v1.QSATripairBridge
    v1.apply_pauli = v3.corrected_apply_pauli
    v1.QSATripairBridge = LifecycleTracingBridge
    batch_size = 1
    seed = 370038 + batch_size
    v1.torch.manual_seed(seed)
    baseline = v1.TripairCircuitModule()
    candidate = v1.copy.deepcopy(baseline)
    baseline_optimizer = v1.torch.optim.AdamW(
        baseline.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    candidate_optimizer = v1.torch.optim.AdamW(
        candidate.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    bridge = LifecycleTracingBridge()
    features, target = v1.training_batches(batch_size, 1)[0]
    baseline_result = None
    candidate_result = None
    candidate_error = None
    try:
        baseline_result = v1.run_step(
            baseline,
            baseline_optimizer,
            features,
            target,
            "torch",
            None,
            1.0,
        )
        try:
            candidate_result = v1.run_step(
                candidate,
                candidate_optimizer,
                features,
                target,
                "qsa",
                bridge,
                1.0,
            )
        except Exception as exc:
            candidate_error = f"{type(exc).__name__}: {exc}"
        receipt: dict[str, object] = {
            "schema_version": (
                "qsa.brain-recovery-tripair-autograd-lifecycle-diagnostic.v1"
            ),
            "source": {
                "qsa_commit": v1.QSA_SOURCE,
                "brain_parent": v1.BRAIN_PARENT,
                "optimizer_v3_run": 30942022060,
                "optimizer_v3_artifact": 8905551051,
                "cotangent_diagnostic_run": 30942021694,
                "cotangent_diagnostic_artifact": 8905538441,
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
                "batch_size": batch_size,
                "seed": seed,
                "baseline_step_runs_first": True,
                "candidate_initial_parameters_match_baseline": True,
                "same_batches_targets_adamw_and_clipping": True,
                "uses_real_torch_autograd_function": True,
                "selected_pullback_threshold": 2.0e-12,
                "threshold_weakened": False,
            },
            "baseline": {
                "completed": baseline_result is not None,
                "loss": None if baseline_result is None else float(baseline_result[2]),
            },
            "candidate": {
                "completed": candidate_result is not None,
                "error": candidate_error,
                "loss": None if candidate_result is None else float(candidate_result[2]),
            },
            "trace": bridge.trace,
            "state": bridge.state_receipt(),
        }
        samples = bridge.trace.get("samples", [])
        receipt["diagnostic_complete"] = bool(samples) and all(
            "errors" in sample for sample in samples
        )
        receipt["selected_pullback_reproduced_failure"] = any(
            not bool(sample["selected_pullback_success"])
            for sample in samples
        )
        receipt["passed"] = bool(receipt["diagnostic_complete"])
        return receipt
    finally:
        bridge.close()
        v1.apply_pauli = original_apply_pauli
        v1.QSATripairBridge = original_bridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-tripair-autograd-lifecycle-diagnostic.json"
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
        raise SystemExit("Tripair autograd lifecycle diagnostic incomplete")


if __name__ == "__main__":
    main()
