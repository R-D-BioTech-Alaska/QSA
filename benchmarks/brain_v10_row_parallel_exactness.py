from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import brain_recovery_tripair_optimizer_step_v10_contiguous as v10


WORKER_ENV = "QSA_CAUSAL_VECTORIZED_WORKERS"


def _timed(function, repetitions: int) -> tuple[float, object]:
    samples: list[float] = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - started)
    assert result is not None
    return statistics.median(samples), result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_receipt(rows: int, repetitions: int, workers: int) -> dict[str, object]:
    bridge = v10.ContiguousVectorizedTripairBridge()
    vectorized = bridge.vectorized
    root = bridge.root
    parameter_count = len(vectorized.parameter_names)
    observable_count = int(vectorized.observables.observable_count)

    parameter_index = np.arange(rows * parameter_count, dtype=np.float64).reshape(
        rows, parameter_count
    )
    parameters = np.ascontiguousarray(
        0.75 * np.sin(parameter_index * 0.0013)
        + 0.125 * np.cos(parameter_index * 0.00017),
        dtype=np.float64,
    )
    cotangent_index = np.arange(
        rows * observable_count, dtype=np.float64
    ).reshape(rows, observable_count)
    cotangents = np.ascontiguousarray(
        0.5 * np.cos(cotangent_index * 0.0011)
        - 0.2 * np.sin(cotangent_index * 0.00031),
        dtype=np.float64,
    )

    root_before = root.encode_qsc()
    previous = os.environ.get(WORKER_ENV)
    try:
        os.environ[WORKER_ENV] = "1"
        vectorized.values_numpy(root, parameters)
        vectorized.evaluate_numpy(root, parameters, cotangents)
        serial_forward_seconds, serial_values = _timed(
            lambda: vectorized.values_numpy(root, parameters), repetitions
        )
        serial_reverse_seconds, serial_result = _timed(
            lambda: vectorized.evaluate_numpy(root, parameters, cotangents),
            repetitions,
        )

        os.environ[WORKER_ENV] = str(workers)
        vectorized.values_numpy(root, parameters)
        vectorized.evaluate_numpy(root, parameters, cotangents)
        parallel_forward_seconds, parallel_values = _timed(
            lambda: vectorized.values_numpy(root, parameters), repetitions
        )
        parallel_reverse_seconds, parallel_result = _timed(
            lambda: vectorized.evaluate_numpy(root, parameters, cotangents),
            repetitions,
        )
    finally:
        if previous is None:
            os.environ.pop(WORKER_ENV, None)
        else:
            os.environ[WORKER_ENV] = previous

    root_after = root.encode_qsc()
    forward_exact = bool(np.array_equal(serial_values, parallel_values))
    reverse_values_exact = bool(
        np.array_equal(serial_result.values, parallel_result.values)
    )
    gradients_exact = bool(
        np.array_equal(serial_result.gradients, parallel_result.gradients)
    )
    serial_replay_exact = bool(np.array_equal(serial_values, serial_result.values))
    parallel_replay_exact = bool(
        np.array_equal(parallel_values, parallel_result.values)
    )
    root_exact = root_before == root_after

    receipt: dict[str, object] = {
        "schema_version": "qsa.brain-v10-row-parallel-exactness.v1",
        "rows": rows,
        "parameter_count": parameter_count,
        "observable_count": observable_count,
        "repetitions": repetitions,
        "requested_parallel_workers": workers,
        "cpu_count": os.cpu_count(),
        "worker_environment": WORKER_ENV,
        "root_qsc_sha256_before": _sha256(root_before),
        "root_qsc_sha256_after": _sha256(root_after),
        "root_byte_identical": root_exact,
        "serial_parallel_forward_torch_style_exact": forward_exact,
        "serial_parallel_reverse_values_exact": reverse_values_exact,
        "serial_parallel_gradients_exact": gradients_exact,
        "serial_forward_reverse_replay_exact": serial_replay_exact,
        "parallel_forward_reverse_replay_exact": parallel_replay_exact,
        "maximum_forward_absolute_error": float(
            np.max(np.abs(serial_values - parallel_values))
        ),
        "maximum_reverse_value_absolute_error": float(
            np.max(np.abs(serial_result.values - parallel_result.values))
        ),
        "maximum_gradient_absolute_error": float(
            np.max(np.abs(serial_result.gradients - parallel_result.gradients))
        ),
        "timing": {
            "serial_forward_seconds": serial_forward_seconds,
            "parallel_forward_seconds": parallel_forward_seconds,
            "forward_speedup": serial_forward_seconds / parallel_forward_seconds,
            "serial_weighted_adjoint_seconds": serial_reverse_seconds,
            "parallel_weighted_adjoint_seconds": parallel_reverse_seconds,
            "weighted_adjoint_speedup": serial_reverse_seconds
            / parallel_reverse_seconds,
        },
        "contract": {
            "same_vectorized_kernel_per_row": True,
            "same_circuit": True,
            "same_parameter_rows": True,
            "same_cotangent_rows": True,
            "same_per_row_arithmetic_order": True,
            "disjoint_output_slices": True,
            "persistent_candidate_reuse": False,
            "root_mutation": False,
            "production_authority": False,
        },
    }
    receipt["passed"] = bool(
        forward_exact
        and reverse_values_exact
        and gradients_exact
        and serial_replay_exact
        and parallel_replay_exact
        and root_exact
        and receipt["maximum_forward_absolute_error"] == 0.0
        and receipt["maximum_reverse_value_absolute_error"] == 0.0
        and receipt["maximum_gradient_absolute_error"] == 0.0
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=65040)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-v10-row-parallel-exactness.json"),
    )
    args = parser.parse_args()
    if args.rows < 4096:
        raise SystemExit("--rows must be at least the parallel threshold (4096)")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.workers < 2:
        raise SystemExit("--workers must be at least 2")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = build_receipt(args.rows, args.repetitions, args.workers)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("row-parallel exactness contract failed")


if __name__ == "__main__":
    main()
