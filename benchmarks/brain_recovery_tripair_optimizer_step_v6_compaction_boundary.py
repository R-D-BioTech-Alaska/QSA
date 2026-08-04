from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_adjoint import CausalComponentWeightedAdjoint
from qsa.causal_support import CausalPauliSupportPlan

import brain_recovery_tripair_optimizer_step as v1
import brain_recovery_tripair_optimizer_step_v3 as v3
import brain_recovery_tripair_optimizer_step_v4_safe as v4
import brain_recovery_tripair_optimizer_step_v5_trajectory as v5

BATCH_SIZE = 64
TARGET_STEP = 5
DEFAULT_FACTOR_TOLERANCE = 1.0e-10


def run_git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_state(brain_angles: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    batch = brain_angles.shape[0]
    state = torch.zeros(batch, 1 << v1.QUBITS, dtype=torch.complex128)
    state[:, 0] = 1.0 + 0.0j
    theta = brain_angles[:, : v1.QUBITS]
    phi = brain_angles[:, v1.QUBITS :]
    for qubit in range(v1.QUBITS):
        state = v1.apply_ry(state, theta[:, qubit], qubit)
        state = v1.apply_rz(state, phi[:, qubit], qubit)
    state = v1.apply_cnot(state, 0, 1)
    state = v1.apply_cnot(state, 1, 2)
    state = v1.apply_cnot(state, 2, 0)
    angles = shared.reshape(v1.DEPTH, v1.QUBITS, 2)
    for layer in range(v1.DEPTH):
        for qubit in range(v1.QUBITS):
            state = v1.apply_ry(
                state, angles[layer, qubit, 0].expand(batch), qubit
            )
            state = v1.apply_rz(
                state, angles[layer, qubit, 1].expand(batch), qubit
            )
        state = v1.apply_cnot(state, 0, 1)
        state = v1.apply_cnot(state, 1, 2)
    return state


def exact_observables(state: torch.Tensor) -> torch.Tensor:
    values = []
    for observable in v1.OBSERVABLES:
        operated = v3.corrected_apply_pauli(state, observable)
        values.append(torch.sum(torch.conj(state) * operated, dim=-1).real)
    return torch.stack(values, dim=-1)


def singleton_determinants(state: torch.Tensor) -> list[float]:
    if state.shape[0] != 1:
        raise ValueError("singleton determinant helper requires one state")
    result = []
    for qubit in range(v1.QUBITS):
        zero_raw, one_raw = v1.basis_indices(qubit)
        zero = state[0, torch.tensor(zero_raw)]
        one = state[0, torch.tensor(one_raw)]
        rho00 = torch.sum(torch.abs(zero) ** 2).real
        rho11 = torch.sum(torch.abs(one) ** 2).real
        rho01 = torch.sum(zero * torch.conj(one))
        determinant = rho00 * rho11 - torch.abs(rho01) ** 2
        result.append(float(determinant))
    return result


def plan_operations() -> tuple[object, ...]:
    parameters = {name: Parameter(name) for name in v1.QSA_INPUT_NAMES}
    operations: list[object] = []
    for qubit in range(v1.QUBITS):
        operations.append(("ry", qubit, parameters[f"theta_{qubit}"]))
        operations.append(("rz", qubit, parameters[f"phi_{qubit}"]))
    operations.extend((("cnot", 0, 1), ("cnot", 1, 2), ("cnot", 2, 0)))
    for layer in range(v1.DEPTH):
        for qubit in range(v1.QUBITS):
            operations.append(("ry", qubit, Parameter(f"layer_{layer}_{qubit}_ry")))
            operations.append(("rz", qubit, Parameter(f"layer_{layer}_{qubit}_rz")))
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    return tuple(operations)


def bound_operations(row: Mapping[str, float]) -> list[tuple[object, ...]]:
    result: list[tuple[object, ...]] = []
    for operation in plan_operations():
        if operation[0] in ("ry", "rz"):
            parameter = operation[2]
            name = str(parameter.name)
            result.append((operation[0], int(operation[1]), float(row[name])))
        else:
            result.append(tuple(operation))
    return result


def apply_torch_operation(
    state: torch.Tensor, operation: tuple[object, ...]
) -> torch.Tensor:
    name = str(operation[0])
    if name == "ry":
        return v1.apply_ry(
            state,
            torch.tensor([float(operation[2])], dtype=torch.float64),
            int(operation[1]),
        )
    if name == "rz":
        return v1.apply_rz(
            state,
            torch.tensor([float(operation[2])], dtype=torch.float64),
            int(operation[1]),
        )
    if name == "cnot":
        return v1.apply_cnot(state, int(operation[1]), int(operation[2]))
    raise ValueError(operation)


def aligned_amplitude_error(branch: CausalRegister, exact: torch.Tensor) -> float:
    qsa = torch.tensor(
        [branch.amplitude(index) for index in range(1 << v1.QUBITS)],
        dtype=torch.complex128,
    )
    target = exact[0]
    overlap = torch.sum(torch.conj(target) * qsa)
    if float(torch.abs(overlap)) > 0.0:
        qsa = qsa * torch.conj(overlap) / torch.abs(overlap)
    return float(torch.max(torch.abs(qsa - target)))


def execute_structural(
    plan: CausalParameterizedPlan,
    support: CausalPauliSupportPlan,
    root: CausalRegister,
    row: Mapping[str, float],
) -> tuple[list[float], int, int, str]:
    with root.fork() as branch:
        branch.apply(plan, row)
        values = list(support.execute(branch))
        return (
            values,
            int(branch.component_count),
            int(branch.estimated_bytes),
            hashlib.sha256(branch.encode_qsc()).hexdigest(),
        )


def replay_to_target() -> tuple[
    v1.TripairCircuitModule,
    torch.optim.Optimizer,
    tuple[torch.Tensor, torch.Tensor],
    v4.SafeComponentTripairBridge,
]:
    torch.manual_seed(370038 + BATCH_SIZE)
    candidate = v1.TripairCircuitModule()
    optimizer = torch.optim.AdamW(
        candidate.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    batches = tuple(v1.training_batches(BATCH_SIZE, TARGET_STEP))
    bridge = v4.SafeComponentTripairBridge()
    original_apply_pauli = v1.apply_pauli
    v1.apply_pauli = v3.corrected_apply_pauli
    try:
        for features, target in batches[: TARGET_STEP - 1]:
            v5.execute_step(
                candidate,
                optimizer,
                features,
                target,
                "qsa",
                bridge,
                1.0,
                apply_update=True,
            )
    finally:
        v1.apply_pauli = original_apply_pauli
    return candidate, optimizer, batches[TARGET_STEP - 1], bridge


def build_receipt() -> dict[str, object]:
    candidate, optimizer, (features, target), bridge = replay_to_target()
    del optimizer, target
    original_apply_pauli = v1.apply_pauli
    v1.apply_pauli = v3.corrected_apply_pauli
    support = CausalPauliSupportPlan(v1.QUBITS, v1.OBSERVABLES)
    optimized_plan = v1.qsa_plan()
    unoptimized_plan = CausalParameterizedPlan(plan_operations(), optimize=False)
    dense_adjoint = CausalComponentWeightedAdjoint(
        optimized_plan,
        support,
        max_local_qubits=v1.QUBITS,
    )
    root = CausalRegister(v1.QUBITS)
    try:
        encoded = F.linear(
            features, candidate.input_angles.weight, candidate.input_angles.bias
        )
        theta, phi = encoded.chunk(2, dim=-1)
        brain_angles = torch.cat(
            (torch.pi * torch.tanh(theta), torch.pi * torch.tanh(phi)), dim=-1
        )
        rows = bridge.parameter_rows(brain_angles, candidate.trainable_angles)
        torch_states = exact_state(brain_angles, candidate.trainable_angles)
        torch_values = exact_observables(torch_states)

        samples = []
        maximum = {
            "optimized_vs_torch": (-1.0, -1, -1),
            "unoptimized_vs_torch": (-1.0, -1, -1),
            "dense_adjoint_vs_torch": (-1.0, -1, -1),
        }
        structural_values: list[list[float]] = []
        for sample, row in enumerate(rows):
            optimized_values, components, estimated_bytes, qsc_sha = execute_structural(
                optimized_plan, support, root, row
            )
            unoptimized_values, unoptimized_components, _, _ = execute_structural(
                unoptimized_plan, support, root, row
            )
            dense_result = dense_adjoint.evaluate(
                root,
                row,
                [0.0] * len(v1.OBSERVABLES),
            )
            dense_values = list(dense_result.values)
            exact_values = [float(value) for value in torch_values[sample]]
            structural_values.append(optimized_values)
            errors = {}
            for label, values in (
                ("optimized_vs_torch", optimized_values),
                ("unoptimized_vs_torch", unoptimized_values),
                ("dense_adjoint_vs_torch", dense_values),
            ):
                per_observable = [
                    abs(left - right) for left, right in zip(values, exact_values)
                ]
                error = max(per_observable)
                observable = per_observable.index(error)
                errors[label] = {
                    "maximum": error,
                    "observable_index": observable,
                    "observable_name": v1.OBSERVABLE_NAMES[observable],
                }
                if error > maximum[label][0]:
                    maximum[label] = (error, sample, observable)
            samples.append(
                {
                    "sample": sample,
                    "component_count_optimized": components,
                    "component_count_unoptimized": unoptimized_components,
                    "estimated_bytes_optimized": estimated_bytes,
                    "qsc_sha256_optimized": qsc_sha,
                    "final_exact_singleton_determinants": singleton_determinants(
                        torch_states[sample : sample + 1]
                    ),
                    "errors": errors,
                }
            )

        offender = maximum["optimized_vs_torch"][1]
        offender_row = rows[offender]
        operations = bound_operations(offender_row)
        prefix_state = torch.zeros(1, 1 << v1.QUBITS, dtype=torch.complex128)
        prefix_state[:, 0] = 1.0 + 0.0j
        prefix_records = []
        with root.fork() as branch:
            previous_components = int(branch.component_count)
            for index, operation in enumerate(operations, start=1):
                one = CausalParameterizedPlan((operation,), optimize=False)
                try:
                    branch.apply(one, ())
                finally:
                    one.close()
                prefix_state = apply_torch_operation(prefix_state, operation)
                qsa_values = list(support.execute(branch))
                exact_values = [float(value) for value in exact_observables(prefix_state)[0]]
                per_observable = [
                    abs(left - right) for left, right in zip(qsa_values, exact_values)
                ]
                components = int(branch.component_count)
                determinants = singleton_determinants(prefix_state)
                prefix_records.append(
                    {
                        "operation_index": index,
                        "operation": list(operation),
                        "component_count_before": previous_components,
                        "component_count_after": components,
                        "component_count_increased": components > previous_components,
                        "exact_singleton_determinants": determinants,
                        "minimum_exact_singleton_determinant": min(determinants),
                        "minimum_determinant_over_factor_tolerance": (
                            min(determinants) / DEFAULT_FACTOR_TOLERANCE
                        ),
                        "maximum_observable_error": max(per_observable),
                        "maximum_observable_error_index": per_observable.index(
                            max(per_observable)
                        ),
                        "aligned_amplitude_error": aligned_amplitude_error(
                            branch, prefix_state
                        ),
                        "estimated_bytes": int(branch.estimated_bytes),
                        "qsc_sha256": hashlib.sha256(
                            branch.encode_qsc()
                        ).hexdigest(),
                    }
                )
                previous_components = components

        first_component_increase = next(
            (
                row
                for row in prefix_records
                if row["component_count_increased"]
                and row["component_count_before"] == 1
            ),
            None,
        )
        first_material_error = next(
            (
                row
                for row in prefix_records
                if row["maximum_observable_error"] > 2.0e-12
            ),
            None,
        )
        optimized_error, optimized_sample, optimized_observable = maximum[
            "optimized_vs_torch"
        ]
        unoptimized_error, _, _ = maximum["unoptimized_vs_torch"]
        dense_error, _, _ = maximum["dense_adjoint_vs_torch"]
        hypothesis_supported = bool(
            optimized_error > 2.0e-12
            and abs(optimized_error - unoptimized_error) <= 1.0e-12
            and dense_error <= 2.0e-12
            and first_component_increase is not None
            and first_material_error is not None
            and first_component_increase["operation_index"]
            == first_material_error["operation_index"]
            and first_component_increase["minimum_exact_singleton_determinant"]
            <= DEFAULT_FACTOR_TOLERANCE
        )
        receipt = {
            "schema_version": "qsa.brain-recovery-tripair-optimizer-step.v6-compaction-boundary",
            "purpose": "Identify the exact batch-64 step-5 sample, observable, operation, and QSA representation transition responsible for the transient safe-forward mismatch.",
            "source": {
                "base": run_git("rev-parse", "HEAD^1"),
                "pull_request_head": run_git("rev-parse", "HEAD^2"),
                "checkout_head": run_git("rev-parse", "HEAD"),
                "files": {
                    str(path): {
                        "git_blob": run_git("hash-object", str(path)),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in (
                        Path("benchmarks/brain_recovery_tripair_optimizer_step.py"),
                        Path("benchmarks/brain_recovery_tripair_optimizer_step_v3.py"),
                        Path("benchmarks/brain_recovery_tripair_optimizer_step_v4_safe.py"),
                        Path("benchmarks/brain_recovery_tripair_optimizer_step_v5_trajectory.py"),
                        Path("benchmarks/brain_recovery_tripair_optimizer_step_v6_compaction_boundary.py"),
                        Path(".github/workflows/brain-recovery-tripair-optimizer-step-v6-compaction-boundary.yml"),
                        Path("include/qubit/qstate.hpp"),
                        Path("src/qstate.cpp"),
                    )
                },
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "device": "cpu",
                "gpu_used": False,
            },
            "contract": {
                "batch_size": BATCH_SIZE,
                "pre_update_step": TARGET_STEP,
                "prior_updates": TARGET_STEP - 1,
                "active_trainable_parameters": v1.EXPECTED_PARAMETER_COUNT,
                "default_factor_tolerance": DEFAULT_FACTOR_TOLERANCE,
                "structural_plan_optimized_and_unoptimized_compared": True,
                "dense_component_adjoint_forward_control": True,
                "runtime_source_changed": False,
            },
            "maximum_errors": {
                label: {
                    "maximum": value[0],
                    "sample": value[1],
                    "observable_index": value[2],
                    "observable_name": v1.OBSERVABLE_NAMES[value[2]],
                }
                for label, value in maximum.items()
            },
            "offending_sample": optimized_sample,
            "offending_observable": {
                "index": optimized_observable,
                "name": v1.OBSERVABLE_NAMES[optimized_observable],
            },
            "offending_parameter_row": {
                name: float(value) for name, value in offender_row.items()
            },
            "samples": samples,
            "offending_prefix_trace": prefix_records,
            "first_component_increase_from_one": first_component_increase,
            "first_material_observable_error": first_material_error,
            "hypothesis": {
                "statement": "Default near-separable singleton factorization at factor_tolerance=1e-10 performs an approximate structural split during a CNOT, while dense weighted-adjoint forward values remain exact.",
                "supported": hypothesis_supported,
                "plan_optimizer_excluded": abs(optimized_error - unoptimized_error)
                <= 1.0e-12,
                "dense_adjoint_exact": dense_error <= 2.0e-12,
            },
            "diagnostic_completed": True,
        }
        return receipt
    finally:
        v1.apply_pauli = original_apply_pauli
        dense_adjoint.close()
        root.close()
        unoptimized_plan.close()
        optimized_plan.close()
        support.close()
        bridge.close()


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = build_receipt()
    output = Path("brain-recovery-tripair-optimizer-step-v6-compaction-boundary.json")
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["diagnostic_completed"]:
        raise SystemExit("compaction-boundary diagnostic did not complete")


if __name__ == "__main__":
    main()
