from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

VALUE_TOLERANCE = 5.0e-9
GRADIENT_TOLERANCE = 5.0e-9


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


def parameter_point(parameter_count: int) -> list[float]:
    return [
        -0.47 + 0.059 + 0.031 * float(parameter + 1)
        for parameter in range(parameter_count)
    ]


def parameterized_brickwork(
    qubits: int,
    layers: int,
    parameter_count: int,
) -> tuple[QuantumCircuit, list[Parameter]]:
    parameters = [Parameter(f"theta_{index:02d}") for index in range(parameter_count)]
    circuit = QuantumCircuit(qubits)
    next_parameter = 0
    for layer in range(layers):
        for qubit in range(qubits):
            ry_angle = 0.007 * float((layer + 1) * (qubit + 3))
            if next_parameter < parameter_count:
                ry_angle = parameters[next_parameter]
                next_parameter += 1
            circuit.ry(ry_angle, qubit)

            rz_angle = -0.005 * float((layer + 2) * (qubit + 1))
            if next_parameter < parameter_count:
                rz_angle = parameters[next_parameter]
                next_parameter += 1
            circuit.rz(rz_angle, qubit)

        start = layer & 1
        for qubit in range(start, qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)

    if next_parameter != parameter_count:
        raise RuntimeError("not all benchmark parameters were assigned")
    return circuit, parameters


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


def result_values(result, count: int) -> list[complex]:
    data = result.data(0)
    return [complex(data[f"obs_{index}"]) for index in range(count)]


def evaluate_bound(
    backend: AerSimulator,
    compiled: QuantumCircuit,
    symbols: list[Parameter],
    parameter_values: list[float],
    query_count: int,
) -> tuple[list[complex], float, float]:
    bindings = dict(zip(symbols, parameter_values, strict=True))
    bind_ms, bound = timed(lambda: compiled.assign_parameters(bindings, inplace=False))
    run_ms, result = timed(lambda: backend.run(bound).result())
    return result_values(result, query_count), bind_ms, run_ms


def exact_parameter_shift(
    backend: AerSimulator,
    compiled: QuantumCircuit,
    symbols: list[Parameter],
    base_parameters: list[float],
    query_count: int,
) -> tuple[list[complex], list[complex], float, float, float]:
    wall_start = time.perf_counter_ns()
    values, bind_ms, backend_run_ms = evaluate_bound(
        backend, compiled, symbols, base_parameters, query_count
    )
    gradients = [0.0 + 0.0j] * (query_count * len(base_parameters))
    shift = math.pi / 2.0

    for parameter in range(len(base_parameters)):
        plus_parameters = list(base_parameters)
        minus_parameters = list(base_parameters)
        plus_parameters[parameter] += shift
        minus_parameters[parameter] -= shift

        plus, plus_bind_ms, plus_run_ms = evaluate_bound(
            backend, compiled, symbols, plus_parameters, query_count
        )
        minus, minus_bind_ms, minus_run_ms = evaluate_bound(
            backend, compiled, symbols, minus_parameters, query_count
        )
        bind_ms += plus_bind_ms + minus_bind_ms
        backend_run_ms += plus_run_ms + minus_run_ms

        for observable in range(query_count):
            gradients[observable * len(base_parameters) + parameter] = (
                plus[observable] - minus[observable]
            ) * 0.5

    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
    return values, gradients, wall_ms, bind_ms, backend_run_ms


def qsa_complex(qsa: dict[str, float], real_key: str, imag_key: str) -> complex:
    return complex(qsa[real_key], qsa[imag_key])


def run_case(
    qubits: int,
    query_count: int,
    parameter_count: int,
    method: str,
    qsa: dict[str, float],
    repetitions: int,
) -> None:
    prefix = f"aer_gradient{qubits}"
    qsa_prefix = f"qsa_gradient{qubits}"

    backend_options = {
        "method": method,
        "device": "CPU",
        "precision": "double",
        "max_parallel_threads": 4,
        "max_parallel_experiments": 1,
        "max_parallel_shots": 1,
        "zero_threshold": 0.0,
    }
    if method == "matrix_product_state":
        backend_options.update(
            matrix_product_state_truncation_threshold=0.0,
            chop_threshold=0.0,
            mps_omp_threads=4,
        )

    backend_init_ms, backend = timed(lambda: AerSimulator(**backend_options))

    def build_workload():
        circuit, symbols = parameterized_brickwork(qubits, 5, parameter_count)
        observables = pauli_observables(qubits, query_count)
        parameters = parameter_point(parameter_count)
        return circuit, symbols, observables, parameters

    workload_build_ms, workload = timed(build_workload)
    circuit, symbols, observables, parameters = workload
    transpile_ms, compiled_pair = timed(lambda: compile_circuit(circuit, backend))
    transpile_mode, compiled_base = compiled_pair
    attach_ms, compiled = timed(lambda: attach_observables(compiled_base, observables))

    if int(qsa[f"{qsa_prefix}_operations"]) != len(circuit.data):
        raise RuntimeError(f"{qubits}q operation count differs between QSA and Aer")
    if int(qsa[f"{qsa_prefix}_queries"]) != query_count:
        raise RuntimeError(f"{qubits}q observable count differs between QSA and Aer")
    if int(qsa[f"{qsa_prefix}_parameters"]) != parameter_count:
        raise RuntimeError(f"{qubits}q parameter count differs between QSA and Aer")

    run_times: list[float] = []
    bind_times: list[float] = []
    backend_times: list[float] = []
    values: list[complex] = []
    gradients: list[complex] = []
    for _ in range(repetitions):
        candidate_values, candidate_gradients, wall_ms, bind_ms, backend_run_ms = (
            exact_parameter_shift(
                backend,
                compiled,
                symbols,
                parameters,
                query_count,
            )
        )
        run_times.append(wall_ms)
        bind_times.append(bind_ms)
        backend_times.append(backend_run_ms)
        if not values:
            values = candidate_values
            gradients = candidate_gradients

    value_max_error = 0.0
    gradient_max_error = 0.0
    for observable in range(query_count):
        qsa_value = qsa_complex(
            qsa,
            f"{qsa_prefix}_value_{observable}_real",
            f"{qsa_prefix}_value_{observable}_imag",
        )
        value_max_error = max(value_max_error, abs(qsa_value - values[observable]))
        for parameter in range(parameter_count):
            index = observable * parameter_count + parameter
            qsa_gradient = qsa_complex(
                qsa,
                f"{qsa_prefix}_gradient_{observable}_{parameter}_real",
                f"{qsa_prefix}_gradient_{observable}_{parameter}_imag",
            )
            gradient_max_error = max(
                gradient_max_error,
                abs(qsa_gradient - gradients[index]),
            )

    if value_max_error > VALUE_TOLERANCE:
        raise RuntimeError(
            f"{qubits}q QSA/Aer value error {value_max_error:.6g} exceeds "
            f"{VALUE_TOLERANCE:.6g}"
        )
    if gradient_max_error > GRADIENT_TOLERANCE:
        raise RuntimeError(
            f"{qubits}q QSA/Aer gradient error {gradient_max_error:.6g} exceeds "
            f"{GRADIENT_TOLERANCE:.6g}"
        )

    value_checksum = sum(values, 0.0 + 0.0j)
    gradient_checksum = sum(gradients, 0.0 + 0.0j)
    qsa_value_checksum = qsa_complex(
        qsa,
        f"{qsa_prefix}_value_checksum_real",
        f"{qsa_prefix}_value_checksum_imag",
    )
    qsa_gradient_checksum = qsa_complex(
        qsa,
        f"{qsa_prefix}_gradient_checksum_real",
        f"{qsa_prefix}_gradient_checksum_imag",
    )
    value_checksum_scale = max(1.0, abs(value_checksum), abs(qsa_value_checksum))
    gradient_checksum_scale = max(
        1.0, abs(gradient_checksum), abs(qsa_gradient_checksum)
    )

    first_execution_ms = run_times[0]
    best_execution_ms = min(run_times)
    first_bind_ms = bind_times[0]
    best_bind_ms = min(bind_times)
    first_backend_ms = backend_times[0]
    best_backend_ms = min(backend_times)
    setup_ms = backend_init_ms + workload_build_ms + transpile_ms + attach_ms
    qsa_first_ms = qsa[f"{qsa_prefix}_first_execution_ms"]
    qsa_best_ms = qsa[f"{qsa_prefix}_best_execution_ms"]
    qsa_setup_plus_first_ms = qsa[f"{qsa_prefix}_setup_plus_first_ms"]

    print(f"{prefix}_method={method}")
    print(f"{prefix}_qubits={qubits}")
    print(f"{prefix}_operations={len(circuit.data)}")
    print(f"{prefix}_queries={query_count}")
    print(f"{prefix}_parameters={parameter_count}")
    print(f"{prefix}_parameter_shift_evaluations={1 + 2 * parameter_count}")
    print(f"{prefix}_repetitions={repetitions}")
    print(f"{prefix}_backend_target_qubits={backend.target.num_qubits}")
    print(f"{prefix}_transpile_mode={transpile_mode}")
    print(f"{prefix}_backend_init_ms={backend_init_ms:.12g}")
    print(f"{prefix}_workload_build_ms={workload_build_ms:.12g}")
    print(f"{prefix}_transpile_ms={transpile_ms:.12g}")
    print(f"{prefix}_observable_attach_ms={attach_ms:.12g}")
    print(f"{prefix}_setup_ms={setup_ms:.12g}")
    print(f"{prefix}_first_bind_ms={first_bind_ms:.12g}")
    print(f"{prefix}_best_bind_ms={best_bind_ms:.12g}")
    print(f"{prefix}_first_backend_run_ms={first_backend_ms:.12g}")
    print(f"{prefix}_best_backend_run_ms={best_backend_ms:.12g}")
    print(f"{prefix}_first_execution_ms={first_execution_ms:.12g}")
    print(f"{prefix}_best_execution_ms={best_execution_ms:.12g}")
    print(f"{prefix}_setup_plus_first_ms={setup_ms + first_execution_ms:.12g}")
    print(f"{prefix}_value_checksum_real={value_checksum.real:.17g}")
    print(f"{prefix}_value_checksum_imag={value_checksum.imag:.17g}")
    print(f"{prefix}_gradient_checksum_real={gradient_checksum.real:.17g}")
    print(f"{prefix}_gradient_checksum_imag={gradient_checksum.imag:.17g}")
    print(f"{prefix}_value_max_error={value_max_error:.12g}")
    print(f"{prefix}_gradient_max_error={gradient_max_error:.12g}")
    print(
        f"{prefix}_value_checksum_relative_error="
        f"{abs(qsa_value_checksum - value_checksum) / value_checksum_scale:.12g}"
    )
    print(
        f"{prefix}_gradient_checksum_relative_error="
        f"{abs(qsa_gradient_checksum - gradient_checksum) / gradient_checksum_scale:.12g}"
    )
    print(f"qsa_gradient{qubits}_vs_aer_first={first_execution_ms / qsa_first_ms:.12g}")
    print(f"qsa_gradient{qubits}_vs_aer_best={best_execution_ms / qsa_best_ms:.12g}")
    print(
        f"qsa_gradient{qubits}_vs_aer_backend_first="
        f"{first_backend_ms / qsa_first_ms:.12g}"
    )
    print(
        f"qsa_gradient{qubits}_end_to_end_vs_aer="
        f"{(setup_ms + first_execution_ms) / qsa_setup_plus_first_ms:.12g}"
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
    print("aer_gradient_cpu_thread_ceiling=4")
    print("aer_gradient_transpile_optimization_level=0")
    print("aer_gradient_transpile_count=1")
    print("aer_gradient_parameter_shift_radians=1.5707963267948966")
    print("aer_gradient_mps_truncation_threshold=0")
    print(f"aer_gradient_value_tolerance={VALUE_TOLERANCE:.12g}")
    print(f"aer_gradient_gradient_tolerance={GRADIENT_TOLERANCE:.12g}")

    run_case(18, 24, 6, "statevector", qsa, args.repetitions)
    run_case(100, 8, 4, "matrix_product_state", qsa, args.repetitions)


if __name__ == "__main__":
    main()
