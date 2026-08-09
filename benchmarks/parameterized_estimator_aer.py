from __future__ import annotations

import argparse
import time
from pathlib import Path

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

VALUE_TOLERANCE = 5.0e-9


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


def parameterized_brickwork(
    qubits: int,
    layers: int,
    parameter_count: int,
) -> tuple[QuantumCircuit, list[Parameter]]:
    symbols = [Parameter(f"theta_{index:02d}") for index in range(parameter_count)]
    circuit = QuantumCircuit(qubits)
    next_parameter = 0
    for layer in range(layers):
        for qubit in range(qubits):
            ry_angle = 0.007 * float((layer + 1) * (qubit + 3))
            if next_parameter < parameter_count:
                ry_angle = symbols[next_parameter]
                next_parameter += 1
            circuit.ry(ry_angle, qubit)

            rz_angle = -0.005 * float((layer + 2) * (qubit + 1))
            if next_parameter < parameter_count:
                rz_angle = symbols[next_parameter]
                next_parameter += 1
            circuit.rz(rz_angle, qubit)

        start = layer & 1
        for qubit in range(start, qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)

    if next_parameter != parameter_count:
        raise RuntimeError("not all benchmark parameters were assigned")
    return circuit, symbols


def parameter_points(point_count: int, parameter_count: int) -> list[list[float]]:
    return [
        [
            -0.47 + 0.021 * float(point + 1) + 0.031 * float(parameter + 1)
            for parameter in range(parameter_count)
        ]
        for point in range(point_count)
    ]


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
        measured.save_expectation_value(observable, qubits, label=f"obs_{index}")
    return measured


def compile_circuit(
    circuit: QuantumCircuit,
    backend: AerSimulator,
) -> tuple[str, QuantumCircuit]:
    backend_width = backend.target.num_qubits
    if backend_width is None or circuit.num_qubits <= backend_width:
        return "backend_target", transpile(circuit, backend, optimization_level=0)
    return (
        "wide_standard_basis",
        transpile(
            circuit,
            basis_gates=["rx", "ry", "rz", "cx"],
            optimization_level=0,
        ),
    )


def result_values(result, experiment: int, count: int) -> list[complex]:
    data = result.data(experiment)
    return [complex(data[f"obs_{index}"]) for index in range(count)]


def execute_sweep(
    backend: AerSimulator,
    compiled: QuantumCircuit,
    symbols: list[Parameter],
    points: list[list[float]],
    observable_count: int,
) -> tuple[list[list[complex]], float, float, float]:
    wall_start = time.perf_counter_ns()

    def bind_all():
        circuits = []
        for point in points:
            bindings = dict(zip(symbols, point, strict=True))
            circuits.append(compiled.assign_parameters(bindings, inplace=False))
        return circuits

    bind_ms, circuits = timed(bind_all)
    run_ms, result = timed(lambda: backend.run(circuits).result())
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
    values = [
        result_values(result, experiment, observable_count)
        for experiment in range(len(circuits))
    ]
    return values, wall_ms, bind_ms, run_ms


def qsa_value(qsa: dict[str, float], prefix: str, point: int, observable: int) -> complex:
    return complex(
        qsa[f"{prefix}_value_{point}_{observable}_real"],
        qsa[f"{prefix}_value_{point}_{observable}_imag"],
    )


def run_case(
    qubits: int,
    observable_count: int,
    parameter_count: int,
    point_count: int,
    method: str,
    qsa: dict[str, float],
    repetitions: int,
) -> None:
    prefix = f"aer_pest{qubits}"
    qsa_prefix = f"pest{qubits}"

    options = {
        "method": method,
        "device": "CPU",
        "precision": "double",
        "max_parallel_threads": 4,
        "max_parallel_experiments": 4,
        "max_parallel_shots": 1,
        "zero_threshold": 0.0,
    }
    if method == "matrix_product_state":
        options.update(
            matrix_product_state_truncation_threshold=0.0,
            chop_threshold=0.0,
            mps_omp_threads=1,
        )

    backend_init_ms, backend = timed(lambda: AerSimulator(**options))

    def build_workload():
        circuit, symbols = parameterized_brickwork(qubits, 5, parameter_count)
        observables = pauli_observables(qubits, observable_count)
        points = parameter_points(point_count, parameter_count)
        return circuit, symbols, observables, points

    workload_build_ms, workload = timed(build_workload)
    circuit, symbols, observables, points = workload
    transpile_ms, compiled_pair = timed(lambda: compile_circuit(circuit, backend))
    transpile_mode, compiled_base = compiled_pair
    attach_ms, compiled = timed(lambda: attach_observables(compiled_base, observables))

    if int(qsa[f"{qsa_prefix}_operations"]) != len(circuit.data):
        raise RuntimeError(f"{qubits}q operation count differs between QSA and Aer")
    if int(qsa[f"{qsa_prefix}_queries"]) != observable_count:
        raise RuntimeError(f"{qubits}q observable count differs between QSA and Aer")
    if int(qsa[f"{qsa_prefix}_parameters"]) != parameter_count:
        raise RuntimeError(f"{qubits}q parameter count differs between QSA and Aer")
    if int(qsa[f"{qsa_prefix}_points"]) != point_count:
        raise RuntimeError(f"{qubits}q point count differs between QSA and Aer")

    wall_times: list[float] = []
    bind_times: list[float] = []
    backend_times: list[float] = []
    values: list[list[complex]] = []
    for _ in range(repetitions):
        candidate, wall_ms, bind_ms, run_ms = execute_sweep(
            backend, compiled, symbols, points, observable_count
        )
        wall_times.append(wall_ms)
        bind_times.append(bind_ms)
        backend_times.append(run_ms)
        if not values:
            values = candidate

    max_error = 0.0
    checksum = 0.0 + 0.0j
    for point in range(point_count):
        for observable in range(observable_count):
            value = values[point][observable]
            checksum += value
            max_error = max(
                max_error,
                abs(value - qsa_value(qsa, qsa_prefix, point, observable)),
            )
    if max_error > VALUE_TOLERANCE:
        raise RuntimeError(
            f"{qubits}q QSA/Aer sweep value error {max_error:.6g} exceeds "
            f"{VALUE_TOLERANCE:.6g}"
        )

    first_sweep_ms = wall_times[0]
    best_sweep_ms = min(wall_times)
    setup_ms = backend_init_ms + workload_build_ms + transpile_ms + attach_ms
    qsa_first_ms = qsa[f"{qsa_prefix}_first_sweep_ms"]
    qsa_best_ms = qsa[f"{qsa_prefix}_best_sweep_ms"]
    qsa_setup_first_ms = qsa[f"{qsa_prefix}_setup_plus_first_ms"]

    print(f"{prefix}_method={method}")
    print(f"{prefix}_qubits={qubits}")
    print(f"{prefix}_operations={len(circuit.data)}")
    print(f"{prefix}_queries={observable_count}")
    print(f"{prefix}_parameters={parameter_count}")
    print(f"{prefix}_points={point_count}")
    print(f"{prefix}_transpile_mode={transpile_mode}")
    print(f"{prefix}_backend_init_ms={backend_init_ms:.12g}")
    print(f"{prefix}_workload_build_ms={workload_build_ms:.12g}")
    print(f"{prefix}_transpile_ms={transpile_ms:.12g}")
    print(f"{prefix}_observable_attach_ms={attach_ms:.12g}")
    print(f"{prefix}_first_bind_ms={bind_times[0]:.12g}")
    print(f"{prefix}_best_bind_ms={min(bind_times):.12g}")
    print(f"{prefix}_first_backend_run_ms={backend_times[0]:.12g}")
    print(f"{prefix}_best_backend_run_ms={min(backend_times):.12g}")
    print(f"{prefix}_first_sweep_ms={first_sweep_ms:.12g}")
    print(f"{prefix}_best_sweep_ms={best_sweep_ms:.12g}")
    print(f"{prefix}_setup_plus_first_ms={setup_ms + first_sweep_ms:.12g}")
    print(f"{prefix}_checksum_real={checksum.real:.17g}")
    print(f"{prefix}_checksum_imag={checksum.imag:.17g}")
    print(f"{prefix}_qsa_max_value_error={max_error:.12g}")
    print(f"qsa_pest{qubits}_vs_aer_first={first_sweep_ms / qsa_first_ms:.12g}")
    print(f"qsa_pest{qubits}_vs_aer_best={best_sweep_ms / qsa_best_ms:.12g}")
    print(
        f"qsa_pest{qubits}_end_to_end_vs_aer="
        f"{(setup_ms + first_sweep_ms) / qsa_setup_first_ms:.12g}"
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
    print("aer_sweep_cpu_thread_ceiling=4")
    print("aer_sweep_parallel_axis=experiments")
    print("aer_sweep_transpile_optimization_level=0")
    print("aer_sweep_mps_truncation_threshold=0")
    print("aer_sweep_mps_omp_threads=1")

    run_case(18, 24, 6, 12, "statevector", qsa, args.repetitions)
    run_case(100, 8, 4, 8, "matrix_product_state", qsa, args.repetitions)


if __name__ == "__main__":
    main()
