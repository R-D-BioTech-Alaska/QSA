from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v2 as v2


def maximum_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def direct_global_values(
    bridge: v2.LazyWorkspaceTripairBridge,
    row: Mapping[str, float] | Sequence[float],
) -> tuple[float, ...]:
    with v1.CausalRegister(v1.QUBITS) as state:
        state.apply(bridge.plan, row)
        return tuple(float(value) for value in bridge.support.execute(state))


def diagnose_active_batch(
    bridge: v2.LazyWorkspaceTripairBridge,
    row: Mapping[str, float] | Sequence[float],
    row_kind: str,
    scale: float,
) -> dict[str, object]:
    root_before = bytes(bridge.runtime.state.encode_qsc())
    batch = bridge.runtime.evaluate((row,))
    try:
        workspace = bridge.runtime._workspace
        if workspace is None:
            raise RuntimeError("persistent workspace did not initialize")
        workspace_before = bytes(workspace.state.encode_qsc())
        local_plan, local_observables = bridge.runtime._local_runtime(
            batch.global_qubits
        )
        stored = tuple(float(value) for value in batch.observations[0])
        retained_branch = batch._branches[0]
        if retained_branch is None:
            raise RuntimeError("retained candidate branch is missing")
        retained_values = tuple(
            float(value) for value in local_observables.execute(retained_branch)
        )

        with workspace.state.fork() as single_branch:
            single_branch.apply(local_plan, row)
            single_values = tuple(
                float(value) for value in local_observables.execute(single_branch)
            )

        adjoint = bridge.runtime._local_adjoint(batch.global_qubits)
        adjoint_result = adjoint.evaluate(
            workspace.state,
            row,
            (0.0,) * len(v1.OBSERVABLES),
        )
        adjoint_values = tuple(float(value) for value in adjoint_result.values)
        global_values = direct_global_values(bridge, row)

        bound_original = tuple(float(value) for value in bridge.plan._values(row))
        bound_local = tuple(float(value) for value in local_plan._values(row))
        receipt = {
            "row_kind": row_kind,
            "scale": float(scale),
            "parameter_names": list(bridge.plan.parameter_names),
            "bound_original": list(bound_original),
            "bound_local": list(bound_local),
            "bound_original_vs_local_max_error": maximum_error(
                bound_original, bound_local
            ),
            "stored_values": list(stored),
            "retained_branch_values": list(retained_values),
            "single_execute_values": list(single_values),
            "adjoint_values": list(adjoint_values),
            "global_execute_values": list(global_values),
            "errors": {
                "stored_vs_retained_branch": maximum_error(
                    stored, retained_values
                ),
                "stored_vs_single_execute": maximum_error(
                    stored, single_values
                ),
                "stored_vs_adjoint": maximum_error(stored, adjoint_values),
                "stored_vs_global_execute": maximum_error(
                    stored, global_values
                ),
                "retained_branch_vs_single_execute": maximum_error(
                    retained_values, single_values
                ),
                "single_execute_vs_adjoint": maximum_error(
                    single_values, adjoint_values
                ),
                "single_execute_vs_global_execute": maximum_error(
                    single_values, global_values
                ),
            },
            "workspace_global_qubits": list(batch.global_qubits),
            "workspace_local_qubits": int(batch.local_qubit_count),
            "root_immutable_while_batch_active": bytes(
                bridge.runtime.state.encode_qsc()
            )
            == root_before,
            "workspace_immutable_while_batch_active": bytes(
                workspace.state.encode_qsc()
            )
            == workspace_before,
            "adjoint_forward_sweeps": int(adjoint_result.forward_sweeps),
            "adjoint_reverse_sweeps": int(adjoint_result.reverse_sweeps),
            "adjoint_shifted_evaluations": int(
                adjoint_result.shifted_evaluations
            ),
        }
    finally:
        bridge.runtime.discard(batch)
    receipt["root_immutable_after_discard"] = bytes(
        bridge.runtime.state.encode_qsc()
    ) == root_before
    return receipt


def brain_derived_row() -> tuple[dict[str, float], dict[str, object]]:
    v1.torch.manual_seed(370039)
    model = v1.TripairCircuitModule()
    features, _target = v1.training_batches(1, 1)[0]
    encoded = v1.F.linear(
        features,
        model.input_angles.weight,
        model.input_angles.bias,
    )
    theta, phi = encoded.chunk(2, dim=-1)
    brain_angles = v1.torch.cat(
        (v1.torch.pi * v1.torch.tanh(theta), v1.torch.pi * v1.torch.tanh(phi)),
        dim=-1,
    )
    bridge = v2.LazyWorkspaceTripairBridge()
    try:
        row = dict(
            bridge.parameter_rows(brain_angles, model.trainable_angles)[0]
        )
    finally:
        bridge.close()
    metadata = {
        "seed": 370039,
        "feature_batch_seed": 20260805,
        "brain_angles": [float(value) for value in brain_angles[0]],
        "shared_angles": [
            float(value) for value in model.trainable_angles.detach().reshape(-1)
        ],
    }
    return row, metadata


def scaled_mapping(row: Mapping[str, float], scale: float) -> dict[str, float]:
    return {name: float(value) * float(scale) for name, value in row.items()}


def build_receipt() -> dict[str, object]:
    row, row_metadata = brain_derived_row()
    bridge = v2.LazyWorkspaceTripairBridge()
    cases = []
    try:
        for scale in (0.05, 0.25, 0.5, 1.0, 2.0):
            mapping_row = scaled_mapping(row, scale)
            cases.append(
                diagnose_active_batch(
                    bridge,
                    mapping_row,
                    "mapping",
                    scale,
                )
            )
            sequence_row = tuple(
                float(value) for value in bridge.plan._values(mapping_row)
            )
            cases.append(
                diagnose_active_batch(
                    bridge,
                    sequence_row,
                    "sequence",
                    scale,
                )
            )
        runtime_receipt = bridge.state_receipt()
    finally:
        bridge.close()

    stored_vs_single = [
        float(case["errors"]["stored_vs_single_execute"])
        for case in cases
    ]
    single_vs_adjoint = [
        float(case["errors"]["single_execute_vs_adjoint"])
        for case in cases
    ]
    mapping_cases = [case for case in cases if case["row_kind"] == "mapping"]
    sequence_cases = [case for case in cases if case["row_kind"] == "sequence"]
    diagnosis = {
        "maximum_stored_vs_single_execute": max(stored_vs_single),
        "maximum_single_execute_vs_adjoint": max(single_vs_adjoint),
        "mapping_failure_scales": [
            float(case["scale"])
            for case in mapping_cases
            if float(case["errors"]["stored_vs_adjoint"]) > 2.0e-12
        ],
        "sequence_failure_scales": [
            float(case["scale"])
            for case in sequence_cases
            if float(case["errors"]["stored_vs_adjoint"]) > 2.0e-12
        ],
    }
    if diagnosis["maximum_stored_vs_single_execute"] > 2.0e-12:
        diagnosis["primary_boundary"] = "native_batch_apply_or_observe_path"
    elif diagnosis["maximum_single_execute_vs_adjoint"] > 2.0e-12:
        diagnosis["primary_boundary"] = "weighted_adjoint_forward_replay"
    else:
        diagnosis["primary_boundary"] = "not_reproduced_in_diagnostic"

    receipt: dict[str, object] = {
        "schema_version": "qsa.brain-recovery-persistent-tripair-value-diagnostic.v1",
        "source": {
            "qsa_commit": v1.QSA_SOURCE,
            "brain_parent": v1.BRAIN_PARENT,
            "predecessor_optimizer_workflow": 30941224527,
            "predecessor_optimizer_artifact": 8905225345,
        },
        "environment": {
            "python": v1.sys.version,
            "platform": v1.platform.platform(),
            "torch": v1.torch.__version__,
            "dtype": "float64/complex128",
            "device": "cpu",
            "gpu_used": False,
        },
        "brain_derived_row": row_metadata,
        "cases": cases,
        "runtime": runtime_receipt,
        "diagnosis": diagnosis,
        "gate_policy": {
            "diagnostic_execution_passes_when_receipt_is_complete": True,
            "selected_pullback_parity_threshold_remains": 2.0e-12,
            "parity_gate_weakened": False,
        },
    }
    receipt["passed"] = (
        len(cases) == 10
        and all(bool(case["root_immutable_after_discard"]) for case in cases)
        and runtime_receipt["root_immutable"]
        and runtime_receipt["workspace_immutable"]
        and runtime_receipt["workspace_extractions"] == 1
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-persistent-tripair-value-diagnostic.json"
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
        raise SystemExit("persistent Tripair value diagnostic did not complete")


if __name__ == "__main__":
    main()
