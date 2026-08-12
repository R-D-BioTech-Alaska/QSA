from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def tt_svd(vector: np.ndarray, levels: int, tolerance: float) -> tuple[list[np.ndarray], list[int]]:
    tensor = np.asarray(vector).reshape((2,) * levels)
    cores = []
    ranks = [1]
    unfolding = tensor
    left_rank = 1
    for _ in range(levels - 1):
        unfolding = unfolding.reshape(left_rank * 2, -1)
        u, singular, vh = np.linalg.svd(unfolding, full_matrices=False)
        squared = singular * singular
        tail = np.cumsum(squared[::-1])[::-1]
        threshold = tolerance * tolerance * max(float(np.sum(squared)), 1.0)
        rank = singular.size
        for candidate in range(1, singular.size + 1):
            residual = float(tail[candidate]) if candidate < singular.size else 0.0
            if residual <= threshold:
                rank = candidate
                break
        u = u[:, :rank]
        singular = singular[:rank]
        vh = vh[:rank, :]
        cores.append(u.reshape(left_rank, 2, rank))
        unfolding = singular[:, None] * vh
        left_rank = rank
        ranks.append(rank)
    cores.append(unfolding.reshape(left_rank, 2, 1))
    ranks.append(1)
    return cores, ranks


def tt_reconstruct(cores: list[np.ndarray]) -> np.ndarray:
    value = cores[0]
    for core in cores[1:]:
        value = np.tensordot(value, core, axes=([-1], [0]))
    return np.asarray(value).reshape(-1)


def tt_scalars(cores: list[np.ndarray]) -> int:
    return int(sum(core.size for core in cores))


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected))
    if denominator == 0.0:
        return float(np.linalg.norm(actual - expected))
    return float(np.linalg.norm(actual - expected) / denominator)


def poisson_periodic(density: np.ndarray, mu: float, alpha: float) -> tuple[np.ndarray, float]:
    size = density.size
    centered = density - float(np.mean(density))
    frequencies = 2.0 * math.pi * np.fft.fftfreq(size)
    laplace = 4.0 * np.sin(0.5 * frequencies) ** 2
    spectrum = np.fft.fft(centered)
    denominator = laplace + mu
    potential_spectrum = alpha * spectrum / denominator
    potential = np.fft.ifft(potential_spectrum).real
    residual = np.fft.ifft(denominator * np.fft.fft(potential) - alpha * spectrum)
    residual_norm = float(np.linalg.norm(residual) / max(np.linalg.norm(alpha * centered), 1e-30))
    return potential, residual_norm


def evolve_split_step(psi: np.ndarray, potential: np.ndarray, dt: float, kappa: float) -> np.ndarray:
    size = psi.size
    phase = np.exp(-0.5j * dt * potential)
    work = phase * psi
    frequencies = 2.0 * math.pi * np.fft.fftfreq(size)
    laplace = 4.0 * np.sin(0.5 * frequencies) ** 2
    work = np.fft.ifft(np.exp(-1j * dt * kappa * laplace) * np.fft.fft(work))
    return phase * work


def structured_state(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) / float(size)
    envelope = 1.0 + 0.19 * np.cos(2.0 * math.pi * 3.0 * x) + 0.07 * np.sin(2.0 * math.pi * 5.0 * x)
    phase = 0.31 * np.sin(2.0 * math.pi * 2.0 * x) + 0.11 * np.cos(2.0 * math.pi * 7.0 * x)
    psi = envelope * np.exp(1j * phase)
    return psi / np.linalg.norm(psi)


def rank_receipt(vector: np.ndarray, levels: int, tolerance: float) -> dict[str, object]:
    real_cores, real_ranks = tt_svd(vector.real, levels, tolerance)
    imag_cores, imag_ranks = tt_svd(vector.imag, levels, tolerance)
    reconstruction = tt_reconstruct(real_cores) + 1j * tt_reconstruct(imag_cores)
    scalars = tt_scalars(real_cores) + tt_scalars(imag_cores)
    dense_bytes = int(vector.nbytes)
    tt_bytes = scalars * 8
    return {
        "max_rank": max(max(real_ranks), max(imag_ranks)),
        "real_ranks": real_ranks,
        "imag_ranks": imag_ranks,
        "relative_error": relative_error(reconstruction, vector),
        "tt_scalars": scalars,
        "tt_bytes": tt_bytes,
        "dense_bytes": dense_bytes,
        "storage_ratio": dense_bytes / tt_bytes if tt_bytes else None,
    }


def run_case(levels: int, tolerance: float, rank_budget: int) -> dict[str, object]:
    size = 1 << levels
    psi = structured_state(size)
    density = np.abs(psi) ** 2
    started = time.perf_counter()
    potential, residual = poisson_periodic(density, mu=0.025, alpha=0.7)
    evolved = evolve_split_step(psi, potential, dt=0.04, kappa=0.12)
    elapsed = time.perf_counter() - started
    initial_norm = float(np.vdot(psi, psi).real)
    evolved_norm = float(np.vdot(evolved, evolved).real)
    fields = {
        "psi": rank_receipt(psi, levels, tolerance),
        "density": rank_receipt(density.astype(np.complex128), levels, tolerance),
        "potential": rank_receipt(potential.astype(np.complex128), levels, tolerance),
        "evolved": rank_receipt(evolved, levels, tolerance),
    }
    maximum_rank = max(int(field["max_rank"]) for field in fields.values())
    maximum_error = max(float(field["relative_error"]) for field in fields.values())
    return {
        "levels": levels,
        "grid_points": size,
        "elapsed_ms": elapsed * 1000.0,
        "poisson_relative_residual": residual,
        "norm_drift": abs(evolved_norm - initial_norm),
        "max_rank": maximum_rank,
        "max_reconstruction_error": maximum_error,
        "rank_budget": rank_budget,
        "fields": fields,
        "passed": residual < 1e-10 and abs(evolved_norm - initial_norm) < 1e-10 and maximum_error < 5e-9 and maximum_rank <= rank_budget,
    }


def run() -> dict[str, object]:
    tolerance = 1e-11
    rank_budget = 32
    cases = [run_case(levels, tolerance, rank_budget) for levels in (8, 10, 12, 14, 16)]
    return {
        "schema": "qsa.frontier.schrodinger-poisson-qtt.v1",
        "purpose": "Eligibility assay for structured Schrodinger-Poisson fields on dyadic grids before a native QTT solver is attempted.",
        "qtt_tolerance": tolerance,
        "rank_budget": rank_budget,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "interpretation_boundary": (
            "The assay materializes each one-dimensional field to certify its binary TT ranks. "
            "A pass demonstrates bounded-rank eligibility for this structured family, not an end-to-end non-materializing QTT speedup. "
            "The next gate is a native QTT Laplacian/Poisson/evolution path whose cost is measured without dense construction."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("frontier-schrodinger-poisson-qtt-v1.json"))
    args = parser.parse_args()
    receipt = run()
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
