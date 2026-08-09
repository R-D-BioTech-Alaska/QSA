from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def timed(function):
    start = time.perf_counter_ns()
    value = function()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return elapsed_ms, value


def parse_qsa_output(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = raw_line.partition("=")
        if separator:
            values[key.strip()] = raw_value.strip()
    return values


def qsa_float(values: dict[str, str], key: str) -> float:
    if key not in values:
        raise KeyError(f"missing QSA evidence field: {key}")
    return float(values[key])


def qsa_complex_vector(
    values: dict[str, str],
    prefix: str,
    kind: str,
    count: int,
) -> list[complex]:
    result: list[complex] = []
    for index in range(count):
        real = qsa_float(values, f"{prefix}_{kind}_{index}_real")
        imag = qsa_float(values, f"{prefix}_{kind}_{index}_imag")
        result.append(complex(real, imag))
    return result


def parameterized_brickwork(
    qubits: int,
    layers: int,
    parameter_count: int,
) -> tuple[QuantumCircuit, list[Parameter]]:
    circuit = QuantumCircuit(qubits)
    parameters = ParameterVector(f"theta_{qubits}", parameter_count)
    parameter_objects = list(parameters)
    next_parameter = 0

    for layer in range(layers):
        for qubit in range(qubits):
            ry_angle: float | Parameter = 0.007 * float((layer + 1) * (qubit + 3))
            if next_parameter < parameter_count:
                ry_angle = parameter_objects[next_parameter]
                next_parameter += 1
            circuit.ry(ry_angle, qubit)

            rz_angle: float | Parameter = -0.005 * float((layer + 2) * (qubit + 1))
            if next_parameter < parameter_count:
                rz_angle = parameter_objects[next_parameter]
                next_parameter += 1
            circuit.rz(rz_angle, qubit)

        start = layer & 1
        for qubit in range(start, qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)

    if next_parameter != parameter_count:
        raise RuntimeError("parameterized circuit did not bind the requested parameter count")
    return circuit, parameter_objects


def parameter_point(parameter_count: int) -> list[float]:
    return [
        -0.47 + 0.059 + 0.031 * float(parameter + 1)
        for parameter in range(parameter_count)
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


def evaluate_parameter_shift(
    backend: AerSimulator,
    circuit: QuantumCircuit,
    parameters: list[Parameter],
    parameter_values: list[float],
    query_count: int,
) -> tuple[list[complex], list[complex], int]:
    shift = math.pi / 2.0
    assignments: list[dict[Parameter, float]] = [
        dict(zip(parameters, parameter_values, strict=True))
    ]
    for parameter_index in range(len(parameters)):
        plus = list(parameter_values)
        minus = list(parameter_values)
        plus[parameter_index] += shift
        minus[parameter_index] -= shift
        assignments.append(dict(zip(parameters, plus, strict=True)))
        assignments.append(dict(zip(parameters, minus, strict=True)))

    bound = [circuit.assign_parameters(mapping, inplace=False) for mapping in assignments]
    result = backend.run(bound).result()
    values = result_values(result, 0, query_count)
    parameter_count = len(parameters)
    gradients = [0.0 + 0.0j] * (query_count * parameter_count)
    for parameter_index in range(parameter_count):
        plus = result_values(result, 1 + 2 * parameter_index, query_count)
        minus = result_values(result, 2 + 2 * parameter_index, query_count)
        for observable_index in range(query_count):
            gradients[observable_index * parameter_count + parameter_index] = (
                plus[observable_index] - minus[observable_index]
            ) * 0.5
    return values, gradients, len(bound)


def checksum(values: list[complex]) -> complex:
    return sum(values, 0.0 + 0.0j)


def relative_error(actual: complex, expected: complex) -> float:
    scale = max(1.0, abs(actual), abs(expected))
    return abs(actual - expected) / scale


def max_error(actual: list[complex], expected: list[complex]) -> float:
    if len(actual) != len(expected):
        raise RuntimeError("comparison vectors have different lengths")
    return max((abs(first - second) for first, second in zip(actual, expected, strict=True)), default=0.0)


def run_case(
    qubits: int,
    query_count: int,
    parameter_count: int,
    method: str,
    qsa: dict[str, str],
) -> None:
    qsa_prefix = f"scheduler{qubits}p1"
    prefix = f"aergrad{qubits}"

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
    circuit_build_ms, circuit_data = timed(
        lambda: parameterized_brickwork(qubits, 5, parameter_count)
    )
    circuit, parameters = circuit_data
    observable_build_ms, observables = timed(
        lambda: pauli_observables(qubits, query_count)
    )
    transpile_ms, compiled_pair = timed(lambda: compile_circuit(circuit, backend))
    transpile_mode, compiled_base = compiled_pair
    attach_ms, compiled = timed(lambda: attach_observables(compiled_base, observables))

    values_at_point = parameter_point(parameter_count)
    execution_ms, evaluation = timed(
        lambda: evaluate_parameter_shift(
            backend,
            compiled,
            parameters,
            values_at_point,
            query_count,
        )
    )
    values, gradients, evaluation_count = evaluation

    qsa_values = qsa_complex_vector(qsa, qsa_prefix, "value", query_count)
    qsa_gradients = qsa_complex_vector(
        qsa,
        qsa_prefix,
        "gradient",
        query_count * parameter_count,
    )
    value_error = max_error(values, qsa_values)
    gradient_error = max_error(gradients, qsa_gradients)
    value_checksum = checksum(values)
    gradient_checksum = checksum(gradients)
    qsa_value_checksum = complex(
        qsa_float(qsa, f"{qsa_prefix}_value_checksum_real"),
        qsa_float(qsa, f"{qsa_prefix}_value_checksum_imag"),
    )
    qsa_gradient_checksum = complex(
        qsa_float(qsa, f"{qsa_prefix}_gradient_checksum_real"),
        qsa_float(qsa, f"{qsa_prefix}_gradient_checksum_imag"),
    )

    qsa_execution_ms = qsa_float(qsa, f"{qsa_prefix}_scheduler_ms")
    qsa_compile_ms = qsa_float(qsa, f"{qsa_prefix}_scheduler_compile_ms")
    aer_compile_setup_ms = transpile_ms + attach_ms
    aer_total_setup_ms = (
        backend_init_ms + circuit_build_ms + observable_build_ms + aer_compile_setup_ms
    )
    qsa_end_to_end_ms = qsa_compile_ms + qsa_execution_ms
    aer_comparable_end_to_end_ms = aer_compile_setup_ms + execution_ms

    print(f"{prefix}_method={method}")
    print(f"{prefix}_qubits={qubits}")
    print(f"{prefix}_operations={len(circuit.data)}")
    print(f"{prefix}_queries={query_count}")
    print(f"{prefix}_parameters={parameter_count}")
    print(f"{prefix}_parameter_shift_evaluations={evaluation_count}")
    print(f"{prefix}_transpile_count=1")
    print(f"{prefix}_backend_target_qubits={backend.target.num_qubits}")
    print(f"{prefix}_transpile_mode={transpile_mode}")
    print(f"{prefix}_backend_init_ms={backend_init_ms:.12g}")
    print(f"{prefix}_circuit_build_ms={circuit_build_ms:.12g}")
    print(f"{prefix}_observable_build_ms={observable_build_ms:.12g}")
    print(f"{prefix}_transpile_ms={transpile_ms:.12g}")
    print(f"{prefix}_observable_attach_ms={attach_ms:.12g}")
    print(f"{prefix}_compile_setup_ms={aer_compile_setup_ms:.12g}")
    print(f"{prefix}_total_setup_ms={aer_total_setup_ms:.12g}")
    print(f"{prefix}_execution_ms={execution_ms:.12g}")
    print(f"{prefix}_comparable_setup_plus_execution_ms={aer_comparable_end_to_end_ms:.12g}")
    print(f"{prefix}_value_checksum_real={value_checksum.real:.17g}")
    print(f"{prefix}_value_checksum_imag={value_checksum.imag:.17g}")
    print(f"{prefix}_gradient_checksum_real={gradient_checksum.real:.17g}")
    print(f"{prefix}_gradient_checksum_imag={gradient_checksum.imag:.17g}")
    print(f"{prefix}_value_max_error_vs_qsa={value_error:.12g}")
    print(f"{prefix}_gradient_max_error_vs_qsa={gradient_error:.12g}")
    print(
        f"{prefix}_value_checksum_relative_error_vs_qsa="
        f"{relative_error(value_checksum, qsa_value_checksum):.12g}"
    )
    print(
        f"{prefix}_gradient_checksum_relative_error_vs_qsa="
        f"{relative_error(gradient_checksum, qsa_gradient_checksum):.12g}"
    )
    print(f"qsa_grad{qubits}_execution_vs_aer={execution_ms / qsa_execution_ms:.12g}")
    print(
        f"qsa_grad{qubits}_comparable_end_to_end_vs_aer="
        f"{aer_comparable_end_to_end_ms / qsa_end_to_end_ms:.12g}"
    )

    value_tolerance = 2e-10 if qubits == 18 else 2e-9
    gradient_tolerance = 2e-10 if qubits == 18 else 2e-9
    if value_error > value_tolerance:
        raise RuntimeError(
            f"{qubits}q Aer value error {value_error} exceeds {value_tolerance}"
        )
    if gradient_error > gradient_tolerance:
        raise RuntimeError(
            f"{qubits}q Aer gradient error {gradient_error} exceeds {gradient_tolerance}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsa-output", type=Path, required=True)
    args = parser.parse_args()

    qsa = parse_qsa_output(args.qsa_output)
    print(f"qiskit_version={qiskit.__version__}")
    print(f"qiskit_aer_version={qiskit_aer.__version__}")
    print("aer_gradient_cpu_thread_ceiling=4")
    print("aer_gradient_max_parallel_experiments=1")
    print("aer_gradient_transpile_optimization_level=0")
    print("aer_gradient_parameter_shift=pi_over_2")
    print("aer_gradient_mps_truncation_threshold=0")
    print("aer_gradient_chop_threshold=0")
    print("aer_gradient_execution_includes_parameter_binding=1")
    print("aer_gradient_comparable_setup_excludes_python_workload_construction=1")

    run_case(18, 24, 6, "statevector", qsa)
    run_case(100, 8, 4, "matrix_product_state", qsa)


if __name__ == "__main__":
    main()
