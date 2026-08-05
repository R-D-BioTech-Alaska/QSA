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
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_adjoint import CausalWeightedAdjoint
from qsa.causal_persistent_workspace import CausalPersistentComponentRuntime
from qsa.causal_support import CausalPauliSupportPlan


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999
EXTRA = 611


def candidate_plan() -> CausalParameterizedPlan:
    operations = []
    for qubit, name in (
        (FIRST, "first"),
        (SECOND, "second"),
        (THIRD, "third"),
    ):
        operations.append(("ry", qubit, Parameter(f"semantic_{name}_theta")))
        operations.append(("rz", qubit, Parameter(f"semantic_{name}_phi")))
    operations.extend(
        (
            ("cnot", FIRST, SECOND),
            ("cnot", SECOND, THIRD),
            ("cnot", THIRD, FIRST),
        )
    )
    for layer in range(2):
        for qubit, name in (
            (FIRST, "first"),
            (SECOND, "second"),
            (THIRD, "third"),
        ):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_{name}_first"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_{name}_second"))
            )
        operations.extend(
            (("cnot", FIRST, SECOND), ("cnot", SECOND, THIRD))
        )
    return CausalParameterizedPlan(operations)


def observable_plan() -> CausalPauliSupportPlan:
    return CausalPauliSupportPlan(
        QUBITS,
        (
            ((FIRST, "X"),),
            ((FIRST, "Y"),),
            ((FIRST, "Z"),),
            ((SECOND, "X"),),
            ((SECOND, "Y"),),
            ((SECOND, "Z"),),
            ((THIRD, "X"),),
            ((THIRD, "Y"),),
            ((THIRD, "Z"),),
            ((FIRST, "X"), (SECOND, "X")),
            ((FIRST, "Z"), (SECOND, "Z")),
            ((SECOND, "X"), (THIRD, "X")),
            ((SECOND, "Z"), (THIRD, "Z")),
            ((FIRST, "X"), (SECOND, "X"), (THIRD, "X")),
        ),
    )


def rows(plan: CausalParameterizedPlan, count: int, step: int):
    return tuple(
        {
            name: 0.19
            * math.sin((step + 1) * (candidate + 1) * (slot + 1) * 0.017)
            for slot, name in enumerate(plan.parameter_names)
        }
        for candidate in range(count)
    )


def cotangent(step: int) -> tuple[float, ...]:
    return tuple(
        0.23 * math.cos((step + 1) * (index + 1) * 0.071)
        for index in range(14)
    )


def maximum_error(first, second) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def attempt_pullback(
    label: str,
    callback,
    expected_values,
    expected_gradient,
) -> dict[str, object]:
    try:
        result = callback()
        return {
            "label": label,
            "succeeded": True,
            "error": None,
            "maximum_value_error": maximum_error(result.values, expected_values),
            "maximum_gradient_error": maximum_error(
                result.gradient,
                expected_gradient,
            ),
            "values": list(result.values),
            "gradient": list(result.gradient),
        }
    except Exception as error:
        return {
            "label": label,
            "succeeded": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def fresh_adjoint_result(persistent, batch, selected, weights):
    local_plan, local_observables = persistent._local_runtime(batch.global_qubits)
    return CausalWeightedAdjoint(
        local_plan,
        local_observables,
        max_qubits=persistent.max_local_qubits,
    ).evaluate(
        persistent._workspace.state,
        batch.parameter_row(selected),
        weights,
    )


def decoded_workspace_result(persistent, batch, selected, weights):
    local_plan, local_observables = persistent._local_runtime(batch.global_qubits)
    packet = persistent._workspace.state.encode_qsc()
    with CausalRegister.decode_qsc(packet) as decoded:
        return CausalWeightedAdjoint(
            local_plan,
            local_observables,
            max_qubits=persistent.max_local_qubits,
        ).evaluate(decoded, batch.parameter_row(selected), weights)


def exercise_active_batch(persistent, batch, selected, weights) -> dict[str, object]:
    root_before = persistent.state.encode_qsc()
    workspace_before = persistent._workspace.state.encode_qsc()
    parameter_shift = persistent.selected_gradient(batch, selected, workers=1)
    expected_values = tuple(batch.observations[selected])
    expected_gradient = parameter_shift.vjp(weights)

    cached = [
        attempt_pullback(
            f"cached_{index + 1}",
            lambda: persistent.selected_pullback(batch, selected, weights),
            expected_values,
            expected_gradient,
        )
        for index in range(3)
    ]
    fresh_adjoint = [
        attempt_pullback(
            f"fresh_adjoint_{index + 1}",
            lambda: fresh_adjoint_result(
                persistent,
                batch,
                selected,
                weights,
            ),
            expected_values,
            expected_gradient,
        )
        for index in range(3)
    ]
    decoded = [
        attempt_pullback(
            f"decoded_workspace_{index + 1}",
            lambda: decoded_workspace_result(
                persistent,
                batch,
                selected,
                weights,
            ),
            expected_values,
            expected_gradient,
        )
        for index in range(3)
    ]

    alternate_weights = cotangent(1)
    alternate_reference = parameter_shift.vjp(alternate_weights)
    zero_weights = tuple(0.0 for _ in weights)
    negative_weights = tuple(-value for value in weights)
    cotangent_variants = [
        attempt_pullback(
            "different_cotangent",
            lambda: persistent.selected_pullback(
                batch,
                selected,
                alternate_weights,
            ),
            expected_values,
            alternate_reference,
        ),
        attempt_pullback(
            "zero_cotangent",
            lambda: persistent.selected_pullback(
                batch,
                selected,
                zero_weights,
            ),
            expected_values,
            tuple(0.0 for _ in expected_gradient),
        ),
        attempt_pullback(
            "negative_cotangent",
            lambda: persistent.selected_pullback(
                batch,
                selected,
                negative_weights,
            ),
            expected_values,
            tuple(-value for value in expected_gradient),
        ),
    ]

    alternate_index = (selected + 1) % len(batch)
    alternate_parameter_shift = persistent.selected_gradient(
        batch,
        alternate_index,
        workers=1,
    )
    selected_variant = attempt_pullback(
        "new_selected_index",
        lambda: persistent.selected_pullback(
            batch,
            alternate_index,
            weights,
        ),
        tuple(batch.observations[alternate_index]),
        alternate_parameter_shift.vjp(weights),
    )
    root_after = persistent.state.encode_qsc()
    workspace_after = persistent._workspace.state.encode_qsc()
    return {
        "generation": persistent.generation,
        "selected_index": selected,
        "cached": cached,
        "fresh_adjoint": fresh_adjoint,
        "decoded_workspace": decoded,
        "cotangent_variants": cotangent_variants,
        "selected_variant": selected_variant,
        "root_qsc_equal": root_before == root_after,
        "workspace_qsc_equal": workspace_before == workspace_after,
        "root_sha256_before": sha256(root_before),
        "root_sha256_after": sha256(root_after),
        "workspace_sha256_before": sha256(workspace_before),
        "workspace_sha256_after": sha256(workspace_after),
    }


def build_receipt() -> dict[str, object]:
    plan = candidate_plan()
    support = observable_plan()
    prepare = CausalParameterizedPlan((("h", EXTRA), ("cnot", EXTRA, FIRST)))
    with CausalRegister(QUBITS) as initial:
        initial.apply(prepare, ())
        root_packet = initial.encode_qsc()

    lifecycle = {}
    with CausalPersistentComponentRuntime.decode_qsc(
        root_packet,
        plan,
        support,
        workers=1,
        max_local_qubits=8,
    ) as persistent:
        initial_checkpoint = persistent.checkpoint()
        batch = persistent.evaluate(rows(plan, 8, 0))
        lifecycle["same_batch_generation_0"] = exercise_active_batch(
            persistent,
            batch,
            2,
            cotangent(0),
        )
        persistent.discard(batch)
        try:
            persistent.selected_pullback(batch, 2, cotangent(0))
        except Exception as error:
            lifecycle["discard_rejection"] = {
                "rejected": True,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            lifecycle["discard_rejection"] = {"rejected": False}

        batch_after_discard = persistent.evaluate(rows(plan, 8, 0))
        lifecycle["new_batch_same_generation"] = exercise_active_batch(
            persistent,
            batch_after_discard,
            2,
            cotangent(0),
        )
        persistent.commit(batch_after_discard, 2)
        lifecycle["generation_after_commit"] = persistent.generation

        batch_generation_1 = persistent.evaluate(rows(plan, 8, 1))
        lifecycle["same_batch_generation_1"] = exercise_active_batch(
            persistent,
            batch_generation_1,
            5,
            cotangent(1),
        )
        persistent.discard(batch_generation_1)
        persistent.refresh_workspace()
        lifecycle["workspace_ready_after_refresh"] = persistent.workspace_ready

        batch_after_refresh = persistent.evaluate(rows(plan, 8, 1))
        lifecycle["after_workspace_refresh"] = exercise_active_batch(
            persistent,
            batch_after_refresh,
            5,
            cotangent(1),
        )
        persistent.discard(batch_after_refresh)
        final_checkpoint = persistent.checkpoint()

    with CausalPersistentComponentRuntime.decode_qsc(
        initial_checkpoint,
        plan,
        support,
        workers=1,
        max_local_qubits=8,
    ) as rolled_back:
        rollback_batch = rolled_back.evaluate(rows(plan, 8, 0))
        lifecycle["decoded_rollback"] = exercise_active_batch(
            rolled_back,
            rollback_batch,
            2,
            cotangent(0),
        )
        rolled_back.discard(rollback_batch)
        lifecycle["rollback_root_exact"] = (
            rolled_back.checkpoint() == initial_checkpoint
        )

    first_case = lifecycle["same_batch_generation_0"]
    cached = first_case["cached"]
    fresh = first_case["fresh_adjoint"]
    decoded = first_case["decoded_workspace"]
    red_reproduced = bool(
        cached[0]["succeeded"]
        and any(not result["succeeded"] for result in cached[1:])
        and all(result["succeeded"] for result in fresh)
        and all(result["succeeded"] for result in decoded)
        and first_case["root_qsc_equal"]
        and first_case["workspace_qsc_equal"]
    )

    repository_root = Path(__file__).resolve().parents[1]
    receipt = {
        "schema_version": "qsa.persistent-pullback-lifecycle-diagnostic.v1",
        "source": {
            "runtime_head": git_head(repository_root),
            "base_exactness_repair": "748e4a40483de72ea5e9092d1db45a62ec6b4a16",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "qubits": QUBITS,
            "local_qubits": 4,
            "candidate_rows": 8,
            "parameters": len(plan.parameter_names),
            "observables": support.observable_count,
            "same_cotangent_repeats": 3,
            "root_qsc_must_remain_exact": True,
            "workspace_qsc_must_remain_exact": True,
        },
        "lifecycle": lifecycle,
        "initial_checkpoint_sha256": sha256(initial_checkpoint),
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "red_reproduced": red_reproduced,
        "hypothesis": (
            "Cached CausalWeightedAdjoint/native plan lifecycle mutates hidden "
            "state not represented in root or workspace QSC bytes when the same "
            "retained batch is pulled back repeatedly."
        ),
        "not_a_fix": True,
    }
    prepare.close()
    support.close()
    plan.close()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("persistent-pullback-lifecycle-diagnostic-v1.json"),
    )
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["red_reproduced"]:
        raise SystemExit("persistent repeated-pullback red defect was not reproduced")


if __name__ == "__main__":
    main()
