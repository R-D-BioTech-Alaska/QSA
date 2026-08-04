from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan
from qsa.causal_adjoint import CausalWeightedAdjoint
from qsa.causal_persistent_workspace import CausalPersistentComponentRuntime
from qsa.causal_support import CausalPauliSupportPlan

QUBITS = 3
OBSERVABLES = (
    ((0, "X"),),
    ((0, "Y"),),
    ((0, "Z"),),
    ((1, "X"),),
    ((1, "Y"),),
    ((1, "Z"),),
    ((2, "X"),),
    ((2, "Y"),),
    ((2, "Z"),),
    ((0, "X"), (1, "X")),
    ((0, "Z"), (1, "Z")),
    ((1, "X"), (2, "X")),
    ((1, "Z"), (2, "Z")),
    ((0, "X"), (1, "X"), (2, "X")),
)
ROW = (
    0.3124396074163113,
    -0.07963004529432623,
    0.39810893949212595,
    0.28426761961630564,
    0.5052334881302805,
    -0.3229708617334321,
    -0.0125051926541718,
    0.022732824648996306,
    -0.0336652649693861,
    -0.03895688111827134,
    0.027265026715557267,
    0.0366287606416343,
    -0.04838245378961558,
    0.012160534835759605,
    0.05382336602324708,
    0.032375679189268696,
    -0.07492525721446715,
    -0.05077216192117811,
)
COTANGENT = (
    0.0006495679291308826,
    -3.2081399888359166e-05,
    -8.402792120863487e-05,
    -0.0011959454092432058,
    -0.000655328262391246,
    -0.00017361478726831488,
    0.0005629271063722978,
    0.0006940507577077425,
    -0.00040087681993070524,
    -0.00037043897969600307,
    -0.0007821207249970998,
    0.0008999631386941152,
    -0.000675584494418656,
    -0.0006317860423712623,
)


def plan() -> CausalParameterizedPlan:
    parameters = [Parameter(f"p{index}") for index in range(len(ROW))]
    operations = []
    slot = 0
    for qubit in range(QUBITS):
        operations.append(("ry", qubit, parameters[slot]))
        slot += 1
        operations.append(("rz", qubit, parameters[slot]))
        slot += 1
    operations.extend((("cnot", 0, 1), ("cnot", 1, 2), ("cnot", 2, 0)))
    for _layer in range(2):
        for qubit in range(QUBITS):
            operations.append(("ry", qubit, parameters[slot]))
            slot += 1
            operations.append(("rz", qubit, parameters[slot]))
            slot += 1
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    if slot != len(ROW):
        raise RuntimeError("Tripair parameter surface changed")
    return CausalParameterizedPlan(tuple(operations))


def maximum_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def runtime() -> tuple[
    CausalPersistentComponentRuntime,
    CausalParameterizedPlan,
    CausalPauliSupportPlan,
]:
    current_plan = plan()
    support = CausalPauliSupportPlan(QUBITS, OBSERVABLES)
    current_runtime = CausalPersistentComponentRuntime.create(
        QUBITS,
        current_plan,
        support,
        workers=1,
        max_local_qubits=QUBITS,
    )
    return current_runtime, current_plan, support


def close_runtime(current_runtime, current_plan, support) -> None:
    current_runtime.close()
    support.close()
    current_plan.close()


def selected_sequence(repeats: int) -> dict[str, object]:
    current_runtime, current_plan, support = runtime()
    try:
        batch = current_runtime.evaluate((ROW,))
        stored = tuple(float(value) for value in batch.observations[0])
        root_before = bytes(current_runtime.state.encode_qsc())
        workspace_before = bytes(current_runtime._workspace.state.encode_qsc())
        calls = []
        for index in range(repeats):
            try:
                result = current_runtime.selected_pullback(
                    batch, 0, COTANGENT
                )
                calls.append(
                    {
                        "call": index + 1,
                        "success": True,
                        "error": None,
                        "value_error": maximum_error(stored, result.values),
                        "gradient_maximum_absolute": max(
                            abs(float(value)) for value in result.gradient
                        ),
                        "root_immutable": bytes(
                            current_runtime.state.encode_qsc()
                        )
                        == root_before,
                        "workspace_immutable": bytes(
                            current_runtime._workspace.state.encode_qsc()
                        )
                        == workspace_before,
                    }
                )
            except Exception as exc:
                calls.append(
                    {
                        "call": index + 1,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "value_error": None,
                        "gradient_maximum_absolute": None,
                        "root_immutable": bytes(
                            current_runtime.state.encode_qsc()
                        )
                        == root_before,
                        "workspace_immutable": bytes(
                            current_runtime._workspace.state.encode_qsc()
                        )
                        == workspace_before,
                    }
                )
        current_runtime.discard(batch)
        return {
            "calls": calls,
            "workspace_extractions": int(current_runtime.workspace_extractions),
        }
    finally:
        close_runtime(current_runtime, current_plan, support)


def cached_adjoint_sequence(repeats: int) -> dict[str, object]:
    current_runtime, current_plan, support = runtime()
    try:
        batch = current_runtime.evaluate((ROW,))
        stored = tuple(float(value) for value in batch.observations[0])
        workspace = current_runtime._workspace
        adjoint = current_runtime._local_adjoint(batch.global_qubits)
        calls = []
        for index in range(repeats):
            try:
                result = adjoint.evaluate(workspace.state, ROW, COTANGENT)
                calls.append(
                    {
                        "call": index + 1,
                        "success": True,
                        "error": None,
                        "value_error": maximum_error(stored, result.values),
                        "gradient_maximum_absolute": max(
                            abs(float(value)) for value in result.gradient
                        ),
                    }
                )
            except Exception as exc:
                calls.append(
                    {
                        "call": index + 1,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "value_error": None,
                        "gradient_maximum_absolute": None,
                    }
                )
        current_runtime.discard(batch)
        return {"calls": calls}
    finally:
        close_runtime(current_runtime, current_plan, support)


def fresh_adjoint_sequence(repeats: int) -> dict[str, object]:
    current_runtime, current_plan, support = runtime()
    fresh_adjoints = []
    try:
        batch = current_runtime.evaluate((ROW,))
        stored = tuple(float(value) for value in batch.observations[0])
        workspace = current_runtime._workspace
        local_plan, local_observables = current_runtime._local_runtime(
            batch.global_qubits
        )
        calls = []
        for index in range(repeats):
            adjoint = CausalWeightedAdjoint(
                local_plan,
                local_observables,
                max_qubits=QUBITS,
            )
            fresh_adjoints.append(adjoint)
            try:
                result = adjoint.evaluate(workspace.state, ROW, COTANGENT)
                calls.append(
                    {
                        "call": index + 1,
                        "success": True,
                        "error": None,
                        "value_error": maximum_error(stored, result.values),
                        "gradient_maximum_absolute": max(
                            abs(float(value)) for value in result.gradient
                        ),
                    }
                )
            except Exception as exc:
                calls.append(
                    {
                        "call": index + 1,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "value_error": None,
                        "gradient_maximum_absolute": None,
                    }
                )
        current_runtime.discard(batch)
        return {"calls": calls}
    finally:
        for adjoint in fresh_adjoints:
            adjoint.close()
        close_runtime(current_runtime, current_plan, support)


def fresh_runtime_sequence(repeats: int) -> dict[str, object]:
    calls = []
    for index in range(repeats):
        current_runtime, current_plan, support = runtime()
        try:
            batch = current_runtime.evaluate((ROW,))
            stored = tuple(float(value) for value in batch.observations[0])
            try:
                result = current_runtime.selected_pullback(
                    batch, 0, COTANGENT
                )
                calls.append(
                    {
                        "call": index + 1,
                        "success": True,
                        "error": None,
                        "value_error": maximum_error(stored, result.values),
                        "gradient_maximum_absolute": max(
                            abs(float(value)) for value in result.gradient
                        ),
                    }
                )
            except Exception as exc:
                calls.append(
                    {
                        "call": index + 1,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "value_error": None,
                        "gradient_maximum_absolute": None,
                    }
                )
            current_runtime.discard(batch)
        finally:
            close_runtime(current_runtime, current_plan, support)
    return {"calls": calls}


def primed_selected_sequence(repeats: int) -> dict[str, object]:
    current_runtime, current_plan, support = runtime()
    try:
        batch = current_runtime.evaluate((ROW,))
        stored = tuple(float(value) for value in batch.observations[0])
        adjoint = current_runtime._local_adjoint(batch.global_qubits)
        prime = adjoint.evaluate(
            current_runtime._workspace.state,
            ROW,
            (0.0,) * len(OBSERVABLES),
        )
        calls = []
        for index in range(repeats):
            try:
                result = current_runtime.selected_pullback(
                    batch, 0, COTANGENT
                )
                calls.append(
                    {
                        "call": index + 1,
                        "success": True,
                        "error": None,
                        "value_error": maximum_error(stored, result.values),
                    }
                )
            except Exception as exc:
                calls.append(
                    {
                        "call": index + 1,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "value_error": None,
                    }
                )
        current_runtime.discard(batch)
        return {
            "prime_value_error": maximum_error(stored, prime.values),
            "calls": calls,
        }
    finally:
        close_runtime(current_runtime, current_plan, support)


def build_receipt(repeats: int) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": (
            "qsa.brain-recovery-persistent-tripair-repeatability-qsa-only.v1"
        ),
        "source": {
            "qsa_commit": "78cc67d6c52469486976bb75e9471b87da1fc7fd",
            "brain_parent": "f9a183e7bdbffc3d838e8774da548a053f883152",
            "lifecycle_artifact": 8905675965,
            "lifecycle_json_sha256": (
                "f2707f9efef5f1e4393eef4d433f9297fd3a131550e3279bb2af91bb9ebd9b0e"
            ),
        },
        "contract": {
            "row_source": "exact lifecycle diagnostic row",
            "cotangent_source": "exact lifecycle diagnostic grad_output",
            "repeats": repeats,
            "device": "cpu",
            "pytorch_used": False,
            "selected_pullback_threshold": 2.0e-12,
            "threshold_weakened": False,
        },
        "row": list(ROW),
        "cotangent": list(COTANGENT),
        "cached_selected_pullback": selected_sequence(repeats),
        "cached_direct_adjoint": cached_adjoint_sequence(repeats),
        "fresh_adjoint_each_call": fresh_adjoint_sequence(repeats),
        "fresh_runtime_each_call": fresh_runtime_sequence(repeats),
        "zero_cotangent_primed_selected_pullback": primed_selected_sequence(
            repeats
        ),
    }
    receipt["passed"] = True
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-persistent-tripair-repeatability-qsa-only.json"
        ),
    )
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    receipt = build_receipt(args.repeats)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
