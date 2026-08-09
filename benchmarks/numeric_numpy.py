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


def structured_dense(sites: int) -> np.ndarray:
    first = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    second = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    for site in range(sites):
        angle = 0.01 * float(site + 1)
        phase = 0.013 * float(site + 1)
        first = np.kron(
            first,
            np.asarray([math.cos(angle), math.sin(angle)], dtype=np.complex128),
        )
        second = np.kron(
            second,
            np.asarray([math.cos(phase), 1j * math.sin(phase)], dtype=np.complex128),
        )
    return (0.75 + 0.125j) * first + (-0.2 + 0.05j) * second


def structured_operator(sites: int) -> list[tuple[complex, list[np.ndarray]]]:
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for site in range(sites):
        angle = 0.017 * float(site + 1)
        phase = 0.011 * float(site + 1)
        ca = math.cos(angle)
        sa = math.sin(angle)
        cp = math.cos(phase)
        sp = math.sin(phase)
        first.append(
            np.asarray([[ca, sa], [sa, -ca]], dtype=np.complex128)
        )
        second.append(
            np.asarray([[cp, -1j * sp], [1j * sp, -cp]], dtype=np.complex128)
        )
    return [(0.625 + 0.0j, first), (-0.275 + 0.0j, second)]


def apply_product_operator(
    state: np.ndarray,
    factors: list[np.ndarray],
) -> np.ndarray:
    tensor = state.reshape((2,) * len(factors))
    for axis, matrix in enumerate(factors):
        moved = np.moveaxis(tensor, axis, 0)
        transformed = np.tensordot(matrix, moved, axes=(1, 0))
        tensor = np.moveaxis(transformed, 0, axis)
    return tensor.reshape(-1)


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

    kron20 = structured_dense(20)
    kron20_norm = 0.0

    def kron20_work() -> None:
        nonlocal kron20_norm
        kron20_norm = float(np.vdot(kron20, kron20).real)

    kron20_work()
    kron20_ms = best_ms(kron20_work, max(5, args.repetitions))
    print("numpy_kron20_sites=20")
    print(f"numpy_kron20_logical_elements={kron20.size}")
    print(f"numpy_kron20_norm_ms={kron20_ms:.12g}")
    print(f"numpy_kron20_norm2={kron20_norm:.17g}")
    print(f"numpy_kron20_bytes={kron20.nbytes}")

    kron20_operator = structured_operator(20)
    kron20_expectation = 0.0 + 0.0j
    kron20_applied = np.empty_like(kron20)

    def kron20_expectation_work() -> None:
        nonlocal kron20_expectation, kron20_applied
        kron20_applied = np.zeros_like(kron20)
        for coefficient, factors in kron20_operator:
            kron20_applied += coefficient * apply_product_operator(kron20, factors)
        kron20_expectation = np.vdot(kron20, kron20_applied)

    kron20_expectation_work()
    kron20_expectation_ms = best_ms(kron20_expectation_work, 3)
    operator_local_bytes = sum(
        matrix.nbytes
        for _, factors in kron20_operator
        for matrix in factors
    )
    print(f"numpy_kron20_expectation_ms={kron20_expectation_ms:.12g}")
    print(f"numpy_kron20_expectation_real={kron20_expectation.real:.17g}")
    print(f"numpy_kron20_expectation_imag={kron20_expectation.imag:.17g}")
    print(f"numpy_kron20_operator_local_bytes={operator_local_bytes}")
    print(f"numpy_kron20_retained_working_bytes={kron20.nbytes + kron20_applied.nbytes + operator_local_bytes}")
    print(f"numpy_kron100_dense_vector_bytes_estimate={16 * (1 << 100)}")
    print(f"numpy_kron100_dense_matrix_bytes_estimate={16 * (1 << 200)}")

    if args.qsa_output is not None:
        qsa = parse_qsa_output(args.qsa_output)
        comparisons = (
            ("real", "real_fused_serial_ms", "real_fused_parallel_ms", real_ms),
            ("complex", "complex_fused_serial_ms", "complex_fused_parallel_ms", complex_ms),
            ("matrix2", "matrix2_serial_ms", "matrix2_parallel_ms", matrix_ms),
        )
        for name, serial_key, parallel_key, numpy_ms in comparisons:
            serial_ms = qsa.get(serial_key)
            if serial_ms is not None and serial_ms > 0.0:
                print(f"qsa_{name}_serial_vs_numpy={numpy_ms / serial_ms:.12g}")
            parallel_ms = qsa.get(parallel_key)
            if parallel_ms is not None and parallel_ms > 0.0:
                print(f"qsa_{name}_vs_numpy={numpy_ms / parallel_ms:.12g}")

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

        qsa_kron_ms = qsa.get("kron20_norm_ms")
        if qsa_kron_ms is not None and qsa_kron_ms > 0.0:
            print(f"qsa_kron20_vs_numpy={kron20_ms / qsa_kron_ms:.12g}")
        qsa_kron_norm = qsa.get("kron20_norm2")
        if qsa_kron_norm is not None:
            scale = max(1.0, abs(qsa_kron_norm), abs(kron20_norm))
            print(f"qsa_kron20_numpy_relative_error={abs(qsa_kron_norm - kron20_norm) / scale:.12g}")
        qsa_kron_bytes = qsa.get("kron20_bytes")
        if qsa_kron_bytes is not None and qsa_kron_bytes > 0.0:
            print(f"qsa_kron20_memory_ratio={kron20.nbytes / qsa_kron_bytes:.12g}")

        qsa_expectation_ms = qsa.get("kron20_expectation_ms")
        if qsa_expectation_ms is not None and qsa_expectation_ms > 0.0:
            print(f"qsa_kron20_expectation_vs_numpy={kron20_expectation_ms / qsa_expectation_ms:.12g}")
        qsa_expectation_real = qsa.get("kron20_expectation_real")
        qsa_expectation_imag = qsa.get("kron20_expectation_imag")
        if qsa_expectation_real is not None and qsa_expectation_imag is not None:
            qsa_expectation = complex(qsa_expectation_real, qsa_expectation_imag)
            scale = max(1.0, abs(qsa_expectation), abs(kron20_expectation))
            print(
                "qsa_kron20_expectation_numpy_relative_error="
                f"{abs(qsa_expectation - kron20_expectation) / scale:.12g}"
            )
        qsa_operator_bytes = qsa.get("kron20_operator_bytes")
        if (
            qsa_kron_bytes is not None
            and qsa_operator_bytes is not None
            and qsa_kron_bytes + qsa_operator_bytes > 0.0
        ):
            qsa_retained = qsa_kron_bytes + qsa_operator_bytes
            numpy_retained = kron20.nbytes + kron20_applied.nbytes + operator_local_bytes
            print(f"qsa_kron20_operator_memory_ratio={numpy_retained / qsa_retained:.12g}")


if __name__ == "__main__":
    main()
