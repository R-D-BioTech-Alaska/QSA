from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def normalized_amplitudes(magnitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
    amplitudes = magnitudes.astype(np.complex128) * np.exp(1j * phases)
    norm = float(np.linalg.norm(amplitudes))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid route amplitudes")
    return amplitudes / norm


def coherent_recombination_probability(amplitudes: np.ndarray) -> float:
    routes = amplitudes.size
    overlap = np.sum(amplitudes) / math.sqrt(routes)
    return float(np.abs(overlap) ** 2)


def dephased_recombination_probability(amplitudes: np.ndarray) -> float:
    routes = amplitudes.size
    return float(np.sum(np.abs(amplitudes) ** 2) / routes)


def paired_phase_assay(seed: int = 20260811, pairs: int = 2048, routes: int = 8) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    coherent_gaps = []
    dephased_gaps = []
    constructive = []
    destructive = []
    for _ in range(pairs):
        magnitudes = rng.uniform(0.2, 1.0, size=routes)
        base_phases = rng.uniform(-math.pi, math.pi, size=routes)
        first = normalized_amplitudes(magnitudes, base_phases)
        second_phases = base_phases.copy()
        second_phases[routes // 2 :] += math.pi
        second = normalized_amplitudes(magnitudes, second_phases)

        coherent_first = coherent_recombination_probability(first)
        coherent_second = coherent_recombination_probability(second)
        dephased_first = dephased_recombination_probability(first)
        dephased_second = dephased_recombination_probability(second)

        coherent_gaps.append(abs(coherent_first - coherent_second))
        dephased_gaps.append(abs(dephased_first - dephased_second))
        constructive.append(coherent_first)
        destructive.append(coherent_second)

    coherent_arr = np.asarray(coherent_gaps)
    dephased_arr = np.asarray(dephased_gaps)
    return {
        "pairs": pairs,
        "routes": routes,
        "mean_coherent_pair_gap": float(np.mean(coherent_arr)),
        "median_coherent_pair_gap": float(np.median(coherent_arr)),
        "max_coherent_pair_gap": float(np.max(coherent_arr)),
        "mean_dephased_pair_gap": float(np.mean(dephased_arr)),
        "max_dephased_pair_gap": float(np.max(dephased_arr)),
        "mean_first_probability": float(np.mean(constructive)),
        "mean_phase_flipped_probability": float(np.mean(destructive)),
        "dephased_identity_error": float(np.max(dephased_arr)),
    }


def orthogonal_phase_code_assay(routes: int = 16) -> dict[str, object]:
    if routes & (routes - 1):
        raise ValueError("routes must be a power of two")
    uniform = np.full(routes, 1.0 / math.sqrt(routes), dtype=np.float64)
    coherent_matrix = np.empty((routes, routes), dtype=np.float64)
    dephased_matrix = np.empty((routes, routes), dtype=np.float64)
    phase_codes = []
    for code in range(routes):
        phases = np.asarray(
            [math.pi * (bin(code & index).count("1") & 1) for index in range(routes)],
            dtype=np.float64,
        )
        phase_codes.append(normalized_amplitudes(uniform, phases))

    for code, amplitudes in enumerate(phase_codes):
        for probe, probe_state in enumerate(phase_codes):
            coherent_matrix[code, probe] = float(abs(np.vdot(probe_state, amplitudes)) ** 2)
            dephased_matrix[code, probe] = float(
                np.sum(np.abs(probe_state) ** 2 * np.abs(amplitudes) ** 2)
            )

    coherent_diagonal = np.diag(coherent_matrix)
    coherent_off = coherent_matrix[~np.eye(routes, dtype=bool)]
    dephased_span = float(np.max(dephased_matrix) - np.min(dephased_matrix))
    return {
        "routes": routes,
        "coherent_code_count": routes,
        "coherent_diagonal_min": float(np.min(coherent_diagonal)),
        "coherent_off_diagonal_max": float(np.max(coherent_off)),
        "dephased_output_span": dephased_span,
        "dephased_matrix_value": float(dephased_matrix[0, 0]),
    }


def scaling_assay(seed: int = 20260811) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for routes in (8, 32, 128, 512, 2048, 8192, 32768, 131072):
        magnitudes = rng.uniform(0.1, 1.0, size=routes)
        phases = rng.uniform(-math.pi, math.pi, size=routes)
        amplitudes = normalized_amplitudes(magnitudes, phases)
        started = time.perf_counter()
        repeats = max(8, min(2048, 1_048_576 // routes))
        value = 0.0
        for _ in range(repeats):
            value += coherent_recombination_probability(amplitudes)
        elapsed = time.perf_counter() - started
        vector_bytes = int(amplitudes.nbytes)
        dense_density_bytes = 16 * routes * routes
        rows.append(
            {
                "routes": routes,
                "route_qubits_equivalent": math.log2(routes),
                "repeats": repeats,
                "elapsed_ms": elapsed * 1000.0,
                "microseconds_per_recombination": elapsed * 1.0e6 / repeats,
                "amplitude_vector_bytes": vector_bytes,
                "dense_density_matrix_bytes": str(dense_density_bytes),
                "density_to_vector_storage_ratio": dense_density_bytes / vector_bytes,
                "finite_accumulator": math.isfinite(value),
            }
        )
    return rows


def run() -> dict[str, object]:
    paired = paired_phase_assay()
    code = orthogonal_phase_code_assay()
    scaling = scaling_assay()
    passed = (
        paired["mean_coherent_pair_gap"] > 1.0e-3
        and paired["dephased_identity_error"] < 1.0e-14
        and code["coherent_diagonal_min"] > 1.0 - 1.0e-12
        and code["coherent_off_diagonal_max"] < 1.0e-12
        and code["dephased_output_span"] < 1.0e-14
        and all(row["finite_accumulator"] for row in scaling)
    )
    return {
        "schema": "qsa.frontier.coherent-router.v1",
        "purpose": (
            "Causal assay for a previously unused coherent route-recombination mechanism. "
            "The dephased control preserves route populations and removes only relative-phase coherence."
        ),
        "paired_phase_assay": paired,
        "orthogonal_phase_code_assay": code,
        "scaling": scaling,
        "interpretation_boundary": (
            "This establishes representational information available only before dephasing and measures bounded recombination cost. "
            "It does not by itself establish a Brain capability gain; the next gate is replay on frozen Brain routing traces with an identical dephased control."
        ),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("frontier-coherent-router-v1.json"))
    args = parser.parse_args()
    receipt = run()
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
