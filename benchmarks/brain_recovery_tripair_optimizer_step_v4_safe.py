from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from qsa.causal import CausalRegister
from qsa.causal_adjoint import CausalComponentWeightedAdjoint
from qsa.causal_support import CausalPauliSupportPlan

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v3 as v3


@dataclass(frozen=True)
class SafeBatch:
    parameter_rows: tuple[dict[str, float], ...]


class SafeComponentTripairBridge(v1.QSATripairBridge):
    """Exact component-local bridge without retained persistent candidates.

    Forward executes the bound QSA plan on an ephemeral branch. Backward uses
    the already-reproduced component-local weighted adjoint. No persistent
    batch branch or cached selected-pullback path is used.
    """

    def __init__(self) -> None:
        self.plan = v1.qsa_plan()
        self.support = CausalPauliSupportPlan(v1.QUBITS, v1.OBSERVABLES)
        self.root = CausalRegister(v1.QUBITS)
        self.adjoint = CausalComponentWeightedAdjoint(
            self.plan,
            self.support,
            max_local_qubits=v1.QUBITS,
        )
        self.forward_seconds = 0.0
        self.backward_seconds = 0.0
        self.forward_calls = 0
        self.backward_calls = 0
        self.forward_executions = 0
        self.adjoint_forward_sweeps = 0
        self.adjoint_reverse_sweeps = 0
        self.shifted_evaluations = 0
        self.local_widths: list[int] = []
        self._root_before = bytes(self.root.encode_qsc())

    def forward(self, brain_angles, shared):
        started = time.perf_counter()
        rows = self.parameter_rows(brain_angles, shared)
        observations = []
        for row in rows:
            with self.root.fork() as branch:
                branch.apply(self.plan, row)
                observations.append(
                    tuple(float(value) for value in self.support.execute(branch))
                )
            self.forward_executions += 1
        tensor = torch.tensor(
            observations,
            dtype=brain_angles.dtype,
            device=brain_angles.device,
        )
        self.forward_seconds += time.perf_counter() - started
        self.forward_calls += 1
        return tensor, SafeBatch(parameter_rows=rows)

    def backward(
        self,
        batch,
        cotangents,
        input_dtype,
        input_device,
        shared_shape,
    ):
        started = time.perf_counter()
        input_gradients = torch.zeros(
            len(batch.parameter_rows),
            len(v1.BRAIN_INPUT_NAMES),
            dtype=input_dtype,
            device=input_device,
        )
        shared_gradient = torch.zeros(
            12,
            dtype=input_dtype,
            device=input_device,
        )
        name_to_position = {
            name: position for position, name in enumerate(self.parameter_names)
        }
        for sample, (row, cotangent) in enumerate(
            zip(batch.parameter_rows, cotangents.detach().cpu().tolist())
        ):
            result = self.adjoint.evaluate(self.root, row, cotangent)
            for brain_position, name in enumerate(v1.BRAIN_INPUT_NAMES):
                input_gradients[sample, brain_position] = result.gradient[
                    name_to_position[name]
                ]
            offset = 0
            for layer in range(v1.DEPTH):
                for qubit in range(v1.QUBITS):
                    for axis in ("ry", "rz"):
                        shared_gradient[offset] += result.gradient[
                            name_to_position[f"layer_{layer}_{qubit}_{axis}"]
                        ]
                        offset += 1
            self.adjoint_forward_sweeps += int(result.forward_sweeps)
            self.adjoint_reverse_sweeps += int(result.reverse_sweeps)
            self.shifted_evaluations += int(result.shifted_evaluations)
            self.local_widths.append(int(result.local_qubit_count))
        self.backward_seconds += time.perf_counter() - started
        self.backward_calls += 1
        return input_gradients, shared_gradient.reshape(shared_shape)

    def state_receipt(self) -> dict[str, object]:
        return {
            "mode": "safe_component_local_no_persistent_candidate",
            "persistent_workspace_used": False,
            "retained_candidate_used": False,
            "root_immutable": bytes(self.root.encode_qsc()) == self._root_before,
            "workspace_immutable": True,
            "workspace_extractions": 0,
            "generation": 0,
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "forward_executions": self.forward_executions,
            "forward_sweeps": self.adjoint_forward_sweeps,
            "reverse_sweeps": self.adjoint_reverse_sweeps,
            "shifted_evaluations": self.shifted_evaluations,
            "local_widths": self.local_widths,
            "forward_seconds": self.forward_seconds,
            "backward_seconds": self.backward_seconds,
        }

    def close(self) -> None:
        self.adjoint.close()
        self.root.close()
        self.support.close()
        self.plan.close()


def case_passed(case: dict[str, object]) -> bool:
    qsa = case["candidate"]["qsa"]
    rollback = case["rollback"]
    expected = int(case["batch_size"]) * int(case["steps"])
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
        and qsa["forward_executions"] == expected
        and qsa["forward_sweeps"] == expected
        and qsa["reverse_sweeps"] == expected
        and qsa["shifted_evaluations"] == 0
        and qsa["local_widths"] == [v1.QUBITS] * expected
        and rollback["baseline_model_max_error"] == 0.0
        and rollback["candidate_model_max_error"] == 0.0
        and rollback["baseline_optimizer_exact"]
        and rollback["candidate_optimizer_exact"]
    )


def build_receipt(steps: int) -> dict[str, object]:
    original_bridge = v1.QSATripairBridge
    original_apply_pauli = v1.apply_pauli
    v1.QSATripairBridge = SafeComponentTripairBridge
    v1.apply_pauli = v3.corrected_apply_pauli
    try:
        receipt = v1.build_receipt(steps)
    finally:
        v1.QSATripairBridge = original_bridge
        v1.apply_pauli = original_apply_pauli

    receipt["schema_version"] = (
        "qsa.brain-recovery-tripair-optimizer-step.v4-safe-component"
    )
    receipt["iteration"] = {
        "predecessor": "qsa.brain-recovery-tripair-optimizer-step.v3",
        "predecessor_failure": "persistent repeated selected-pullback parity defect",
        "causally_different_successor": (
            "Remove retained persistent candidate reuse. Use ephemeral QSA "
            "forward branches and the reproduced component-local weighted "
            "adjoint for backward."
        ),
        "qsa_runtime_source_changed": False,
        "model_parameters_changed": False,
        "optimizer_contract_changed": False,
        "comparison_budget_changed": False,
        "parity_thresholds_weakened": False,
    }
    receipt["contract"]["candidate_backend"] = {
        "forward": "ephemeral QSA component-local plan execution",
        "backward": "exact CausalComponentWeightedAdjoint",
        "persistent_candidate_reuse": False,
        "forward_sweeps_per_sample": 2,
        "reverse_sweeps_per_sample": 1,
        "shifted_evaluations": 0,
    }
    for case in receipt["cases"]:
        case["passed"] = case_passed(case)
    receipt["passed"] = all(bool(case["passed"]) for case in receipt["cases"])
    receipt["candidate_faster_batch_sizes"] = [
        int(case["batch_size"])
        for case in receipt["cases"]
        if bool(case["candidate_faster_than_pytorch"])
    ]
    receipt["not_established"] = [
        "complete 203539-parameter Tripair attachment advantage",
        "Brain language canonical or safety capability contribution",
        "persistent selected-pullback safety",
        "GPU advantage",
        "quantum uniqueness",
        "physical-QPU advantage",
        "production readiness",
    ]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-optimizer-step-v4-safe.json"),
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
        raise SystemExit("safe component-local optimizer-step contract failed")


if __name__ == "__main__":
    main()
