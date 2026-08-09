from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def timed(function):
    start = time.perf_counter_ns()
    value = function()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return elapsed_ms, value


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


def brickwork(qubits: int, layers: int) -> QuantumCircuit:
    circuit = QuantumCircuit(qubits)
    for layer in range(layers):
        for qubit in range(qubits):
            circuit.ry(0.007 * float((layer + 1) * (qubit + 3)), qubit)
            circuit.rz(-0.005 * float((layer + 2) * (qubit + 1)), qubit)
        start = layer & 1
        for qubit in range(start, qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)
    return circuit


def pauli_observables(qubits: int, count: int) -> list[SparsePauliOp]:
    axes = ("I", "X", "Y", "Z")
    result: list[SparsePauliOp] = []
    for index in range(count):
        first = (index * 7 + 3) % qubits
        second = (first + 1 + index % 3) % qubits
        third = (second + 2) % qubits
        labels = ["I"] * qubits
        labels[qubits - 1 - first] = axes[1 + index % 3]
        labels[qubits - 1 - second] = axes[1 + (index + 1) % 3]
        if index % 2 != 0 and third != first and third != second:
            labels[qubits - 1 - third] = axes[1 + (index + 2) % 3]
        result.append(SparsePauliOp.from_list([("".join(labels), 1.0)]))
    return result


def attach_observables(
    circuit: QuantumCircuit,
    observables: list[SparsePauliOp],
) -> QuantumCircuit:
    measured = circuit.copy()
    qubits = list(range(circuit.num_qubits))
    for index, observable in enumerate(observables):
        measured.save_expectation_value(
            observable,
            qubits,
            label=f"obs_{index}",
        )
    return measured


def execute(backend: AerSimulator, circuit: QuantumCircuit):
    return backend.run(circuit).result()


def result_values(result, count: int) -> list[complex]:
    data = result.data(0)
    return [complex(data[f"obs_{index}"]) for index in range(count)]


def run_case(
    qubits: int,
    query_count: int,
    method: str,
    qsa: dict[str, float],
    repetitions: int,
) -> None:
    prefix = f"aer{qubits}"
    build_ms, circuit = timed(lambda: brickwork(qubits, 5))
    observables = pauli_observables(qubits, query_count)
    attach_ms, measured = timed(lambda: attach_observables(circuit, observables))

    options = {
        "method": method,
        "device": "CPU",
        "precision": "double",
        "max_parallel_threads": 4,
        "max_parallel_experiments": 1,
        "max_parallel_shots": 1,
        "zero_threshold": 0.0,
    }
    if method == "matrix_product_state":
        options.update(
            matrix_product_state_truncation_threshold=0.0,
            chop_threshold=0.0,
            mps_omp_threads=4,
        )
    backend = AerSimulator(**options)

    transpile_ms, compiled = timed(
        lambda: transpile(measured, backend, optimization_level=0)
    )

    run_times: list[float] = []
    result = None
    for _ in range(max(1, repetitions)):
        run_ms, candidate = timed(lambda: execute(backend, compiled))
        run_times.append(run_ms)
        if result is None:
            result = candidate
    values = result_values(result, query_count)
    checksum = sum(values, 0.0 + 0.0j)

    first_run_ms = run_times[0]
    best_run_ms = min(run_times)
    setup_ms = build_ms + attach_ms + transpile_ms

    print(f"{prefix}_method={method}")
    print(f"{prefix}_qubits={qubits}")
    print(f"{prefix}_operations={len(circuit.data)}")
    print(f"{prefix}_queries={query_count}")
    print(f"{prefix}_circuit_build_ms={build_ms:.12g}")
    print(f"{prefix}_observable_attach_ms={attach_ms:.12g}")
    print(f"{prefix}_transpile_ms={transpile_ms:.12g}")
    print(f"{prefix}_first_run_ms={first_run_ms:.12g}")
    print(f"{prefix}_best_run_ms={best_run_ms:.12g}")
    print(f"{prefix}_setup_plus_first_run_ms={setup_ms + first_run_ms:.12g}")
    print(f"{prefix}_checksum_real={checksum.real:.17g}")
    print(f"{prefix}_checksum_imag={checksum.imag:.17g}")

    qsa_query = qsa.get(f"est{qubits}_query_ms")
    if qsa_query is not None and qsa_query > 0.0:
        print(f"qsa_est{qubits}_vs_aer_first={first_run_ms / qsa_query:.12g}")
        print(f"qsa_est{qubits}_vs_aer_best={best_run_ms / qsa_query:.12g}")

    qsa_compile = qsa.get(f"est{qubits}_observable_compile_ms", 0.0)
    qsa_circuit_compile = qsa.get(f"est{qubits}_circuit_compile_ms", 0.0)
    if qsa_query is not None:
        qsa_total = qsa_circuit_compile + qsa_compile + qsa_query
        if qsa_total > 0.0:
            print(
                f"qsa_est{qubits}_end_to_end_vs_aer="
                f"{(setup_ms + first_run_ms) / qsa_total:.12g}"
            )

    qsa_checksum_real = qsa.get(f"est{qubits}_checksum_real")
    qsa_checksum_imag = qsa.get(f"est{qubits}_checksum_imag")
    if qsa_checksum_real is not None and qsa_checksum_imag is not None:
        qsa_checksum = complex(qsa_checksum_real, qsa_checksum_imag)
        scale = max(1.0, abs(qsa_checksum), abs(checksum))
        print(
            f"qsa_est{qubits}_aer_checksum_relative_error="
            f"{abs(qsa_checksum - checksum) / scale:.12g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsa-output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")

    qsa = parse_qsa_output(args.qsa_output)
    print(f"qiskit_version={qiskit.__version__}")
    print(f"qiskit_aer_version={qiskit_aer.__version__}")
    print("aer_cpu_thread_ceiling=4")
    print("aer_transpile_optimization_level=0")
    print("aer_mps_truncation_threshold=0")

    run_case(18, 24, "statevector", qsa, args.repetitions)
    run_case(100, 8, "matrix_product_state", qsa, args.repetitions)


if __name__ == "__main__":
    main()
