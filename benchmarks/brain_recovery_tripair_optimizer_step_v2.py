from __future__ import annotations

import argparse
import json
from pathlib import Path

import brain_recovery_tripair_optimizer_step as v1


OriginalBridge = v1.QSATripairBridge


class LazyWorkspaceTripairBridge(OriginalBridge):
    """Preserve QSA's lazy persistent-workspace lifecycle.

    The predecessor harness tried to hash the component workspace during
    construction. QSA intentionally creates that workspace on the first
    evaluate() call, so `_workspace` is correctly None at construction time.
    This successor captures the workspace identity immediately after the first
    successful forward evaluation and then applies the unchanged immutability
    gate for the rest of the optimizer-step experiment.
    """

    def __init__(self) -> None:
        self.plan = v1.qsa_plan()
        self.support = v1.CausalPauliSupportPlan(v1.QUBITS, v1.OBSERVABLES)
        self.runtime = v1.CausalPersistentComponentRuntime.create(
            v1.QUBITS,
            self.plan,
            self.support,
            workers=1,
            max_local_qubits=v1.QUBITS,
        )
        self.forward_seconds = 0.0
        self.backward_seconds = 0.0
        self.forward_calls = 0
        self.backward_calls = 0
        self.forward_sweeps = 0
        self.reverse_sweeps = 0
        self.shifted_evaluations = 0
        self._root_before = bytes(self.runtime.state.encode_qsc())
        self._workspace_before: bytes | None = None

    def forward(self, brain_angles, shared):
        observations, batch = OriginalBridge.forward(self, brain_angles, shared)
        if self._workspace_before is None:
            workspace = self.runtime._workspace
            if workspace is None:
                raise RuntimeError(
                    "persistent component workspace did not initialize on evaluate"
                )
            self._workspace_before = bytes(workspace.state.encode_qsc())
        return observations, batch

    def state_receipt(self) -> dict[str, object]:
        workspace = self.runtime._workspace
        workspace_initialized = (
            workspace is not None and self._workspace_before is not None
        )
        receipt = {
            "root_immutable": bytes(self.runtime.state.encode_qsc())
            == self._root_before,
            "workspace_initialized": workspace_initialized,
            "workspace_immutable": bool(
                workspace_initialized
                and bytes(workspace.state.encode_qsc()) == self._workspace_before
            ),
            "workspace_extractions": int(self.runtime.workspace_extractions),
            "generation": int(self.runtime.generation),
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "forward_sweeps": self.forward_sweeps,
            "reverse_sweeps": self.reverse_sweeps,
            "shifted_evaluations": self.shifted_evaluations,
            "forward_seconds": self.forward_seconds,
            "backward_seconds": self.backward_seconds,
        }
        return receipt


def build_receipt(steps: int) -> dict[str, object]:
    original = v1.QSATripairBridge
    v1.QSATripairBridge = LazyWorkspaceTripairBridge
    try:
        receipt = v1.build_receipt(steps)
    finally:
        v1.QSATripairBridge = original

    receipt["schema_version"] = (
        "qsa.brain-recovery-tripair-optimizer-step.v2"
    )
    receipt["iteration"] = {
        "predecessor": "qsa.brain-recovery-tripair-optimizer-step.v1",
        "predecessor_workflow_run": 30940846373,
        "predecessor_job": 92098507241,
        "predecessor_artifact_id": 8905070310,
        "predecessor_artifact_zip_sha256": (
            "bd2359047f6723a5a3e6916e518f18ffb627c29b6afa87067893d84e45a8e493"
        ),
        "predecessor_reached_scientific_execution": False,
        "causal_failure": "lazy_workspace_inspected_before_first_evaluate",
        "failure_detail": (
            "CausalPersistentComponentRuntime intentionally initializes its "
            "selected component workspace on the first evaluate() call. The "
            "predecessor harness attempted to hash runtime._workspace.state in "
            "the bridge constructor while runtime._workspace was correctly None."
        ),
        "successor_change": (
            "Capture the persistent workspace QSC identity immediately after "
            "the first successful evaluate() call, then preserve the unchanged "
            "workspace immutability and single-extraction gates."
        ),
        "model_math_changed": False,
        "optimizer_contract_changed": False,
        "comparison_budget_changed": False,
    }
    receipt["contract"]["persistent_workspace_lifecycle"] = {
        "construction_state": "not_materialized",
        "initialization_event": "first evaluate",
        "identity_capture": "immediately after first evaluate",
        "subsequent_immutability_required": True,
        "workspace_extractions_required": 1,
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-optimizer-step-v2.json"),
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
            "Brain Tripair complete optimizer-step contract v2 failed"
        )


if __name__ == "__main__":
    main()
