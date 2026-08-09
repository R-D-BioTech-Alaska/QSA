from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np


def best_ms(function, repetitions: int) -> float:
    best = math.inf
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        best = min(best, elapsed)
    return best


def real_data(size: int, shift: float) -> np.ndarray:
    x = np.arange(1, size + 1, dtype=np.float64)
    return np.sin(0.00017 * x + shift) + 0.35 * np.cos(0.00011 * x - shift)


def complex_data(size: int, shift: float) -> np.ndarray:
    x = np.arange(1, size + 1, dtype=np.float64)
    real = np.sin(0.00013 * x + shift)
    imag = 0.8 * np.cos(0.00009 * x - shift)
    return real + 1j * imag


def parse_qsa_output(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = raw_line.partition("=")
        if not separator:
            continue
        try:
            values[key.strip()] = float(raw_value.strip())
        except ValueError:
            continue
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-elements", type=int, default=2_000_000)
    parser.add_argument("--complex-elements", type=int, default=1_000_000)
    parser.add_argument("--matrix2-vectors", type=int, default=500_000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--qsa-output", type=Path)
    args = parser.parse_args()

    print(f"numpy_version={np.__version__}")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        print(f"numpy_env_{variable.lower()}={os.environ.get(variable, 'unset')}")

    first = real_data(args.real_elements, 0.17)
    second = real_data(args.real_elements, -0.29)
    temporary = np.empty_like(first)
    output = np.empty_like(first)
    real_norm = 0.0

    def real_work() -> None:
        nonlocal real_norm
        np.multiply(second, -0.375, out=temporary)
        np.add(temporary, 0.0625, out=temporary)
        np.multiply(first, 1.25, out=output)
        np.add(output, temporary, out=output)
        real_norm = float(np.dot(output, output))

    real_work()
    real_ms = best_ms(real_work, args.repetitions)
    print(f"numpy_real_elements={args.real_elements}")
    print(f"numpy_real_ms={real_ms:.12g}")
    print(f"numpy_real_norm2={real_norm:.17g}")

    complex_first = complex_data(args.complex_elements, 0.23)
    complex_second = complex_data(args.complex_elements, -0.31)
    complex_temporary = np.empty_like(complex_first)
    complex_output = np.empty_like(complex_first)
    complex_norm = 0.0
    first_scale = np.complex128(0.75 - 0.25j)
    second_scale = np.complex128(-0.125 + 0.5j)
    bias = np.complex128(0.03125 - 0.0625j)

    def complex_work() -> None:
        nonlocal complex_norm
        np.multiply(complex_second, second_scale, out=complex_temporary)
        np.add(complex_temporary, bias, out=complex_temporary)
        np.multiply(complex_first, first_scale, out=complex_output)
        np.add(complex_output, complex_temporary, out=complex_output)
        complex_norm = float(np.vdot(complex_output, complex_output).real)

    complex_work()
    complex_ms = best_ms(complex_work, args.repetitions)
    print(f"numpy_complex_elements={args.complex_elements}")
    print(f"numpy_complex_ms={complex_ms:.12g}")
    print(f"numpy_complex_norm2={complex_norm:.17g}")

    matrix = np.asarray(
        [[0.5 + 0.25j, -0.125 + 0.75j],
         [0.625 - 0.375j, 0.25 + 0.125j]],
        dtype=np.complex128,
    )
    matrix_input = complex_data(args.matrix2_vectors * 2, 0.41).reshape(-1, 2)
    matrix_output = np.empty_like(matrix_input)

    def matrix_work() -> None:
        np.matmul(matrix_input, matrix.T, out=matrix_output)

    matrix_work()
    matrix_ms = best_ms(matrix_work, args.repetitions)
    matrix_checksum = float(np.vdot(matrix_output.ravel(), matrix_output.ravel()).real)
    print(f"numpy_matrix2_vectors={args.matrix2_vectors}")
    print(f"numpy_matrix2_ms={matrix_ms:.12g}")
    print(f"numpy_matrix2_checksum={matrix_checksum:.17g}")

    if args.qsa_output is not None:
        qsa = parse_qsa_output(args.qsa_output)
        comparisons = (
            ("real", "real_fused_parallel_ms", real_ms),
            ("complex", "complex_fused_parallel_ms", complex_ms),
            ("matrix2", "matrix2_parallel_ms", matrix_ms),
        )
        for name, qsa_key, numpy_ms in comparisons:
            qsa_ms = qsa.get(qsa_key)
            if qsa_ms is not None and qsa_ms > 0.0:
                print(f"qsa_{name}_vs_numpy={numpy_ms / qsa_ms:.12g}")

        norm_pairs = (
            ("real", "real_parallel_norm2", real_norm),
            ("complex", "complex_parallel_norm2", complex_norm),
            ("matrix2", "matrix2_checksum", matrix_checksum),
        )
        for name, qsa_key, numpy_value in norm_pairs:
            qsa_value = qsa.get(qsa_key)
            if qsa_value is None:
                continue
            scale = max(1.0, abs(qsa_value), abs(numpy_value))
            print(f"qsa_{name}_numpy_relative_error={abs(qsa_value - numpy_value) / scale:.12g}")


if __name__ == "__main__":
    main()
