from __future__ import annotations

import cmath
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from qsa.causal import CausalParameterizedPlan, CausalRegister

QUBITS = 3
AMPLITUDE_TOLERANCE = 2.0e-12
NEAR_SEPARABLE_ANGLES = {
    "theta_0": -4.997844046742442e-06,
    "phi_0": 0.3015519695643022,
    "theta_1": 2.8978373361297964,
    "phi_1": -0.9642484754042258,
    "theta_2": -1.9121148716137508,
    "phi_2": -1.9542263333534304,
}


def run_git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    ).stdout.strip()


def apply_ry(state: list[complex], angle: float, qubit: int) -> list[complex]:
    result = list(state)
    cosine = math.cos(angle / 2.0)
    sine = math.sin(angle / 2.0)
    bit = 1 << qubit
    for base in range(1 << QUBITS):
        if base & bit:
            continue
        one = base | bit
        zero_value = state[base]
        one_value = state[one]
        result[base] = cosine * zero_value - sine * one_value
        result[one] = sine * zero_value + cosine * one_value
    return result


def apply_rz(state: list[complex], angle: float, qubit: int) -> list[complex]:
    negative = cmath.exp(-0.5j * angle)
    positive = cmath.exp(0.5j * angle)
    bit = 1 << qubit
    return [
        value * (positive if index & bit else negative)
        for index, value in enumerate(state)
    ]


def apply_cnot(state: list[complex], control: int, target: int) -> list[complex]:
    result = [0.0j] * len(state)
    control_bit = 1 << control
    target_bit = 1 << target
    for index, value in enumerate(state):
        destination = index ^ target_bit if index & control_bit else index
        result[destination] = value
    return result


def exact_state(angles: dict[str, float]) -> list[complex]:
    state = [0.0j] * (1 << QUBITS)
    state[0] = 1.0 + 0.0j
    for qubit in range(QUBITS):
        state = apply_ry(state, angles[f"theta_{qubit}"], qubit)
        state = apply_rz(state, angles[f"phi_{qubit}"], qubit)
    return apply_cnot(state, 0, 1)


def qsa_state(angles: dict[str, float]) -> tuple[list[complex], int, int, str]:
    operations = []
    for qubit in range(QUBITS):
        operations.append(("ry", qubit, angles[f"theta_{qubit}"]))
        operations.append(("rz", qubit, angles[f"phi_{qubit}"]))
    operations.append(("cnot", 0, 1))
    plan = CausalParameterizedPlan(tuple(operations), optimize=False)
    register = CausalRegister(QUBITS)
    try:
        register.apply(plan, ())
        amplitudes = [register.amplitude(index) for index in range(1 << QUBITS)]
        return (
            amplitudes,
            int(register.component_count),
            int(register.estimated_bytes),
            hashlib.sha256(register.encode_qsc()).hexdigest(),
        )
    finally:
        register.close()
        plan.close()


def aligned_maximum_error(candidate: list[complex], reference: list[complex]) -> float:
    overlap = sum(left.conjugate() * right for left, right in zip(reference, candidate))
    if abs(overlap) > 0.0:
        phase = overlap.conjugate() / abs(overlap)
        candidate = [value * phase for value in candidate]
    return max(abs(left - right) for left, right in zip(candidate, reference))


def singleton_determinant(state: list[complex], qubit: int) -> float:
    bit = 1 << qubit
    rho00 = 0.0
    rho11 = 0.0
    rho01 = 0.0j
    for base in range(1 << QUBITS):
        if base & bit:
            continue
        one = base | bit
        zero_value = state[base]
        one_value = state[one]
        rho00 += abs(zero_value) ** 2
        rho11 += abs(one_value) ** 2
        rho01 += zero_value * one_value.conjugate()
    return rho00 * rho11 - abs(rho01) ** 2


def run_case(name: str, angles: dict[str, float], expected_components: int) -> dict[str, object]:
    reference = exact_state(angles)
    candidate, components, estimated_bytes, qsc_sha256 = qsa_state(angles)
    error = aligned_maximum_error(candidate, reference)
    determinants = [singleton_determinant(reference, qubit) for qubit in range(QUBITS)]
    return {
        "name": name,
        "angles": angles,
        "expected_component_count": expected_components,
        "actual_component_count": components,
        "maximum_aligned_amplitude_error": error,
        "amplitude_tolerance": AMPLITUDE_TOLERANCE,
        "exact_singleton_determinants": determinants,
        "estimated_bytes": estimated_bytes,
        "qsc_sha256": qsc_sha256,
        "passed": components == expected_components and error <= AMPLITUDE_TOLERANCE,
    }


def build_receipt() -> dict[str, object]:
    exact_control = dict(NEAR_SEPARABLE_ANGLES)
    exact_control["theta_0"] = 0.0
    near = run_case("weak_but_real_entanglement", dict(NEAR_SEPARABLE_ANGLES), 2)
    control = run_case("exactly_separable_control", exact_control, 3)
    source = Path(__file__)
    return {
        "schema_version": "qsa.exact-component-factorization-regression.v1",
        "purpose": (
            "Fail closed when a weakly entangled state is only near-separable; "
            "preserve exact compaction for a truly separable control."
        ),
        "repository": "R-D-BioTech-Alaska/QSA",
        "base": run_git("rev-parse", "HEAD^1") if len(run_git("rev-list", "--parents", "-n", "1", "HEAD").split()) > 2 else run_git("rev-parse", "HEAD~1"),
        "head": run_git("rev-parse", "HEAD"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
            "gpu_used": False,
        },
        "source": {
            "path": str(source),
            "git_blob": run_git("hash-object", str(source)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bytes": source.stat().st_size,
        },
        "cases": [near, control],
        "passed": near["passed"] and control["passed"],
        "expected_pre_repair_result": {
            "weak_case": "fails because factor_tolerance accepts a nonzero determinant and reconstructs an approximate product state",
            "exact_control": "passes and remains compact",
        },
    }


def main() -> None:
    receipt = build_receipt()
    output = Path("brain-recovery-exact-component-factorization-regression.json")
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("exact component-factorization regression failed")


if __name__ == "__main__":
    main()
