from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GaussianState:
    modes: int

    def __post_init__(self) -> None:
        if self.modes < 1:
            raise ValueError("modes must be positive")
        self.mean = np.zeros(2 * self.modes, dtype=np.float64)
        self.cov = 0.5 * np.eye(2 * self.modes, dtype=np.float64)

    def _apply_local(self, indices: tuple[int, ...], transform: np.ndarray) -> None:
        idx = np.asarray(indices, dtype=np.int64)
        if transform.shape != (idx.size, idx.size):
            raise ValueError("local transform shape mismatch")
        self.mean[idx] = transform @ self.mean[idx]
        self.cov[idx, :] = transform @ self.cov[idx, :]
        self.cov[:, idx] = self.cov[:, idx] @ transform.T

    def displace(self, mode: int, x: float, p: float) -> None:
        self.mean[2 * mode] += float(x)
        self.mean[2 * mode + 1] += float(p)

    def rotate(self, mode: int, phi: float) -> None:
        c = math.cos(phi)
        s = math.sin(phi)
        self._apply_local(
            (2 * mode, 2 * mode + 1),
            np.asarray(((c, -s), (s, c)), dtype=np.float64),
        )

    def squeeze(self, mode: int, r: float) -> None:
        self._apply_local(
            (2 * mode, 2 * mode + 1),
            np.asarray(((math.exp(-r), 0.0), (0.0, math.exp(r))), dtype=np.float64),
        )

    def beam_splitter(self, first: int, second: int, theta: float) -> None:
        if first == second:
            raise ValueError("beam splitter modes must differ")
        c = math.cos(theta)
        s = math.sin(theta)
        transform = np.asarray(
            (
                (c, 0.0, s, 0.0),
                (0.0, c, 0.0, s),
                (-s, 0.0, c, 0.0),
                (0.0, -s, 0.0, c),
            ),
            dtype=np.float64,
        )
        self._apply_local(
            (2 * first, 2 * first + 1, 2 * second, 2 * second + 1),
            transform,
        )

    def two_mode_squeeze(self, first: int, second: int, r: float) -> None:
        if first == second:
            raise ValueError("two-mode squeeze modes must differ")
        c = math.cosh(r)
        s = math.sinh(r)
        transform = np.asarray(
            (
                (c, 0.0, s, 0.0),
                (0.0, c, 0.0, -s),
                (s, 0.0, c, 0.0),
                (0.0, -s, 0.0, c),
            ),
            dtype=np.float64,
        )
        self._apply_local(
            (2 * first, 2 * first + 1, 2 * second, 2 * second + 1),
            transform,
        )

    def mean_photon_number(self, mode: int) -> float:
        i = 2 * mode
        return float(
            0.5
            * (
                self.cov[i, i]
                + self.cov[i + 1, i + 1]
                + self.mean[i] ** 2
                + self.mean[i + 1] ** 2
                - 1.0
            )
        )

    def purity_error(self) -> float:
        omega = np.zeros_like(self.cov)
        for mode in range(self.modes):
            i = 2 * mode
            omega[i, i + 1] = 1.0
            omega[i + 1, i] = -1.0
        eig = np.linalg.eigvals(1j * omega @ self.cov)
        symplectic = np.sort(np.abs(np.real_if_close(eig)))
        target = np.full(symplectic.shape, 0.5, dtype=np.float64)
        return float(np.max(np.abs(symplectic - target)))

    @property
    def storage_bytes(self) -> int:
        return int(self.mean.nbytes + self.cov.nbytes)


def analytic_checks() -> dict[str, object]:
    single_r = 0.73
    single = GaussianState(1)
    single.squeeze(0, single_r)
    expected_single = math.sinh(single_r) ** 2
    single_error = abs(single.mean_photon_number(0) - expected_single)

    pair_r = 0.41
    pair = GaussianState(2)
    pair.two_mode_squeeze(0, 1, pair_r)
    expected_pair = math.sinh(pair_r) ** 2
    pair_errors = [
        abs(pair.mean_photon_number(0) - expected_pair),
        abs(pair.mean_photon_number(1) - expected_pair),
    ]

    displaced = GaussianState(1)
    displaced.displace(0, 0.6, -0.8)
    expected_displaced = 0.5 * (0.6**2 + 0.8**2)
    displaced_error = abs(displaced.mean_photon_number(0) - expected_displaced)

    return {
        "single_mode_squeezed_photon_error": single_error,
        "two_mode_squeezed_photon_error": max(pair_errors),
        "coherent_displacement_photon_error": displaced_error,
        "single_mode_purity_error": single.purity_error(),
        "two_mode_purity_error": pair.purity_error(),
    }


def scaling_receipt(cutoff: int = 32) -> list[dict[str, object]]:
    rows = []
    for modes in (2, 4, 8, 16, 32, 64, 128):
        state = GaussianState(modes)
        dense_dimension = cutoff**modes
        dense_bytes = 16 * dense_dimension
        ratio = dense_bytes / state.storage_bytes
        rows.append(
            {
                "modes": modes,
                "fock_cutoff_per_mode": cutoff,
                "dense_fock_dimension": str(dense_dimension),
                "gaussian_storage_bytes": state.storage_bytes,
                "dense_statevector_bytes": str(dense_bytes),
                "storage_ratio": ratio if math.isfinite(ratio) else None,
                "log10_storage_ratio": math.log10(dense_bytes) - math.log10(state.storage_bytes),
            }
        )
    return rows


def local_operation_scaling(seed: int = 20260811) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    operations = 256
    for modes in (8, 16, 32, 64, 128, 256):
        state = GaussianState(modes)
        started = time.perf_counter()
        for step in range(operations):
            mode = int(rng.integers(0, modes))
            selector = step % 4
            if selector == 0:
                state.rotate(mode, float(rng.uniform(-0.3, 0.3)))
            elif selector == 1:
                state.squeeze(mode, float(rng.uniform(-0.08, 0.08)))
            elif selector == 2:
                state.displace(
                    mode,
                    float(rng.uniform(-0.05, 0.05)),
                    float(rng.uniform(-0.05, 0.05)),
                )
            else:
                second = (mode + 1) % modes
                state.beam_splitter(mode, second, float(rng.uniform(-0.2, 0.2)))
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "modes": modes,
                "operations": operations,
                "elapsed_ms": elapsed * 1000.0,
                "microseconds_per_operation": elapsed * 1.0e6 / operations,
                "storage_bytes": state.storage_bytes,
                "finite": bool(np.all(np.isfinite(state.mean)) and np.all(np.isfinite(state.cov))),
                "symmetry_error": float(np.max(np.abs(state.cov - state.cov.T))),
            }
        )
    return rows


def fixed_particle_scaling() -> list[dict[str, object]]:
    rows = []
    for modes, particles in ((16, 8), (32, 12), (64, 16), (128, 24), (256, 32)):
        dimension = math.comb(particles + modes - 1, particles)
        occupation_bytes = modes * 4
        dense_bytes = dimension * 16
        rows.append(
            {
                "modes": modes,
                "particles": particles,
                "logical_fock_dimension": str(dimension),
                "single_occupation_state_bytes": occupation_bytes,
                "dense_statevector_bytes": str(dense_bytes),
                "log10_storage_ratio": math.log10(dense_bytes) - math.log10(occupation_bytes),
            }
        )
    return rows


def run() -> dict[str, object]:
    checks = analytic_checks()
    local = local_operation_scaling()
    passed = (
        max(checks.values()) < 1.0e-10
        and all(row["finite"] for row in local)
        and all(row["symmetry_error"] < 1.0e-11 for row in local)
    )
    return {
        "schema": "qsa.frontier.gaussian-bosonic.v1",
        "purpose": "Bounded exact-family assay for an unused Gaussian/Fock QSA representation lane.",
        "convention": "hbar=1; vacuum covariance=I/2",
        "analytic_checks": checks,
        "gaussian_scaling": scaling_receipt(),
        "local_operation_scaling": local,
        "fixed_particle_fock_scaling": fixed_particle_scaling(),
        "interpretation_boundary": (
            "Scaling ratios apply only while the state remains in the certified Gaussian family or an explicitly sparse/fixed-particle Fock representation. "
            "They are not a claim that arbitrary non-Gaussian states avoid exponential growth."
        ),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("frontier-gaussian-bosonic-v1.json"))
    args = parser.parse_args()
    receipt = run()
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
