from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_persistent_workspace import CausalPersistentComponentRuntime
from qsa.causal_support import CausalPauliSupportPlan

QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999
EXTRA = 611
TOLERANCE = 2.0e-12


def candidate_plan() -> CausalParameterizedPlan:
    operations = []
    for qubit, name in ((FIRST, "first"), (SECOND, "second"), (THIRD, "third")):
        operations.append(("ry", qubit, Parameter(f"semantic_{name}_theta")))
        operations.append(("rz", qubit, Parameter(f"semantic_{name}_phi")))
    operations.extend((("cnot", FIRST, SECOND), ("cnot", SECOND, THIRD), ("cnot", THIRD, FIRST)))
    for layer in range(2):
        for qubit, name in ((FIRST, "first"), (SECOND, "second"), (THIRD, "third")):
            operations.append(("ry", qubit, Parameter(f"layer_{layer}_{name}_first")))
            operations.append(("rz", qubit, Parameter(f"layer_{layer}_{name}_second")))
        operations.extend((("cnot", FIRST, SECOND), ("cnot", SECOND, THIRD)))
    return CausalParameterizedPlan(operations)


def observable_plan() -> CausalPauliSupportPlan:
    return CausalPauliSupportPlan(
        QUBITS,
        (
            ((FIRST, "X"),), ((FIRST, "Y"),), ((FIRST, "Z"),),
            ((SECOND, "X"),), ((SECOND, "Y"),), ((SECOND, "Z"),),
            ((THIRD, "X"),), ((THIRD, "Y"),), ((THIRD, "Z"),),
            ((FIRST, "X"), (SECOND, "X")),
            ((FIRST, "Z"), (SECOND, "Z")),
            ((SECOND, "X"), (THIRD, "X")),
            ((SECOND, "Z"), (THIRD, "Z")),
            ((FIRST, "X"), (SECOND, "X"), (THIRD, "X")),
        ),
    )


def parameter_rows(plan: CausalParameterizedPlan, generation: int, count: int = 8):
    return tuple(
        {
            name: 0.19 * math.sin((generation + 1) * (row + 1) * (slot + 1) * 0.017)
            for slot, name in enumerate(plan.parameter_names)
        }
        for row in range(count)
    )


def cotangent(index: int) -> tuple[float, ...]:
    return tuple(0.23 * math.cos((index + 1) * (term + 1) * 0.071) for term in range(14))


def maximum_error(left, right) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_pullback(runtime, batch, selected: int, weights) -> dict[str, object]:
    shifted = runtime.selected_gradient(batch, selected, workers=1)
    expected_gradient = shifted.vjp(weights)
    result = runtime.selected_pullback(batch, selected, weights)
    return {
        "value_error": maximum_error(result.values, batch.observations[selected]),
        "gradient_error": maximum_error(result.gradient, expected_gradient),
        "gradient_maximum": max(abs(float(value)) for value in result.gradient),
    }


def verify_active_batch(runtime, batch, selected: int, generation: int) -> dict[str, object]:
    root_before = runtime.state.encode_qsc()
    workspace_before = runtime._workspace.state.encode_qsc()
    base = cotangent(generation)
    alternate = cotangent(generation + 1)
    zero = tuple(0.0 for _ in base)
    negative = tuple(-value for value in base)
    sequence = [
        ("same_1", selected, base),
        ("same_2", selected, base),
        ("same_3", selected, base),
        ("different", selected, alternate),
        ("zero", selected, zero),
        ("negative", selected, negative),
        ("new_selected", (selected + 1) % len(batch), base),
    ]
    results = {
        label: verify_pullback(runtime, batch, index, weights)
        for label, index, weights in sequence
    }
    root_after = runtime.state.encode_qsc()
    workspace_after = runtime._workspace.state.encode_qsc()
    passed = bool(
        all(
            value["value_error"] <= TOLERANCE
            and value["gradient_error"] <= TOLERANCE
            for value in results.values()
        )
        and results["zero"]["gradient_maximum"] <= TOLERANCE
        and root_before == root_after
        and workspace_before == workspace_after
    )
    return {
        "generation": runtime.generation,
        "selected": selected,
        "results": results,
        "root_qsc_equal": root_before == root_after,
        "workspace_qsc_equal": workspace_before == workspace_after,
        "root_sha256": sha256(root_after),
        "workspace_sha256": sha256(workspace_after),
        "passed": passed,
    }


def git_head(path: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_receipt() -> dict[str, object]:
    plan = candidate_plan()
    support = observable_plan()
    prepare = CausalParameterizedPlan((("h", EXTRA), ("cnot", EXTRA, FIRST)))
    with CausalRegister(QUBITS) as initial:
        initial.apply(prepare, ())
        initial_qsc = initial.encode_qsc()

    lifecycle: dict[str, object] = {}
    with CausalPersistentComponentRuntime.decode_qsc(
        initial_qsc, plan, support, workers=1, max_local_qubits=8
    ) as runtime:
        checkpoint_0 = runtime.checkpoint()
        batch_0 = runtime.evaluate(parameter_rows(plan, 0))
        lifecycle["generation_0"] = verify_active_batch(runtime, batch_0, 2, 0)
        runtime.discard(batch_0)
        try:
            runtime.selected_pullback(batch_0, 2, cotangent(0))
        except Exception as error:
            lifecycle["discard_rejected"] = {
                "passed": True,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            lifecycle["discard_rejected"] = {"passed": False}

        same_generation = runtime.evaluate(parameter_rows(plan, 0))
        lifecycle["new_batch_same_generation"] = verify_active_batch(
            runtime, same_generation, 2, 0
        )
        runtime.commit(same_generation, 2)
        lifecycle["generation_after_commit"] = runtime.generation

        batch_1 = runtime.evaluate(parameter_rows(plan, 1))
        lifecycle["generation_1"] = verify_active_batch(runtime, batch_1, 5, 1)
        runtime.discard(batch_1)
        runtime.refresh_workspace()
        lifecycle["workspace_ready_after_refresh"] = runtime.workspace_ready

        refreshed = runtime.evaluate(parameter_rows(plan, 1))
        lifecycle["after_refresh"] = verify_active_batch(runtime, refreshed, 5, 1)
        runtime.discard(refreshed)
        checkpoint_1 = runtime.checkpoint()

    with CausalPersistentComponentRuntime.decode_qsc(
        checkpoint_0, plan, support, workers=1, max_local_qubits=8
    ) as rollback:
        rollback_batch = rollback.evaluate(parameter_rows(plan, 0))
        lifecycle["decoded_rollback"] = verify_active_batch(rollback, rollback_batch, 2, 0)
        rollback.discard(rollback_batch)
        lifecycle["rollback_root_exact"] = rollback.checkpoint() == checkpoint_0

    named = (
        "generation_0", "new_batch_same_generation", "generation_1",
        "after_refresh", "decoded_rollback",
    )
    passed = bool(
        all(lifecycle[name]["passed"] for name in named)
        and lifecycle["discard_rejected"]["passed"]
        and lifecycle["generation_after_commit"] == 1
        and lifecycle["workspace_ready_after_refresh"] is False
        and lifecycle["rollback_root_exact"]
    )
    receipt = {
        "schema_version": "qsa.persistent-pullback-lifecycle-regression.v2",
        "source": {
            "runtime_head": git_head(Path(__file__).resolve().parents[1]),
            "base_exactness_repair": "748e4a40483de72ea5e9092d1db45a62ec6b4a16",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "logical_qubits": QUBITS,
            "local_qubits": 4,
            "candidate_rows": 8,
            "parameters": plan.parameter_count,
            "observables": support.observable_count,
            "tolerance": TOLERANCE,
            "historical_reported_mismatch": 0.2868323271,
            "historical_red_must_reproduce_now": False,
        },
        "lifecycle": lifecycle,
        "checkpoint_0_sha256": sha256(checkpoint_0),
        "checkpoint_1_sha256": sha256(checkpoint_1),
        "passed": passed,
        "interpretation": (
            "The historical repeated-pullback failure is not reproduced by the current "
            "baseline. This regression does not attribute the correction to PR #36; "
            "the full original optimizer harness is evaluated separately."
        ),
    }
    prepare.close()
    support.close()
    plan.close()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("persistent-pullback-lifecycle-regression-v2.json"),
    )
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("persistent pullback lifecycle regression failed")


if __name__ == "__main__":
    main()
