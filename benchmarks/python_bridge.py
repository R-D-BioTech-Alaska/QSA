"""Measure scalar-vs-batch bridge overhead and QSC caching."""

from __future__ import annotations

import statistics
import time

from qsa import OperationPlan, Parameter, ParameterizedPlan, QubitRegister


def timed(function, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def scalar_run(operation_count: int) -> None:
    with QubitRegister(1) as state:
        for _ in range(operation_count):
            state.h(0)


def batch_run(plan: OperationPlan) -> None:
    with QubitRegister(1) as state:
        state.apply_plan(plan)


def qsc_run(qubits: int = 200) -> tuple[float, float, float]:
    with QubitRegister(qubits) as state:
        operations = []
        for qubit in range(0, qubits, 2):
            operations.extend([("h", qubit), ("cnot", qubit, qubit + 1)])
        state.apply_batch(operations)
        start = time.perf_counter()
        first = state.encode_qsc()
        first_time = time.perf_counter() - start
        start = time.perf_counter()
        second = state.encode_qsc()
        cached_time = time.perf_counter() - start
        assert first == second
        state.z(0)
        start = time.perf_counter()
        changed = state.encode_qsc()
        invalidated_time = time.perf_counter() - start
        assert changed != first
        return first_time, cached_time, invalidated_time


def readout_run(qubits: int = 4096, repeats: int = 20) -> tuple[float, float]:
    with QubitRegister(qubits) as state:
        state.apply_plan(OperationPlan([("ry", qubit, 0.17) for qubit in range(qubits)]))
        start = time.perf_counter()
        for _ in range(repeats):
            [state.probability_one(qubit) for qubit in range(qubits)]
        scalar = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(repeats):
            state.probabilities_one()
        bulk = time.perf_counter() - start
        return scalar, bulk


def parameter_sweep(repeats: int = 300) -> tuple[float, float]:
    theta = Parameter("theta")
    phi = Parameter("phi")
    template_operations = []
    for qubit in range(8):
        template_operations.extend([("ry", qubit, theta), ("rz", qubit, phi)])
    for qubit in range(1, 8):
        template_operations.append(("cnot", qubit - 1, qubit))
    template = ParameterizedPlan(template_operations)

    start = time.perf_counter()
    for index in range(repeats):
        concrete = []
        theta_value = 0.001 + index * 0.000001
        phi_value = -0.002 + index * 0.000001
        for operation in template_operations:
            concrete.append(
                tuple(
                    theta_value if item is theta else phi_value if item is phi else item
                    for item in operation
                )
            )
        with QubitRegister(8) as state:
            state.apply_plan(OperationPlan(concrete))
    rebuilt = time.perf_counter() - start

    start = time.perf_counter()
    for index in range(repeats):
        with QubitRegister(8) as state:
            template.apply(
                state,
                (0.001 + index * 0.000001, -0.002 + index * 0.000001),
            )
    parameterized = time.perf_counter() - start
    template.close()
    return rebuilt, parameterized


def ensemble_run(registers: int = 256) -> tuple[float, float]:
    plan = OperationPlan(
        [("ry", qubit, 0.03) for qubit in range(10)]
        + [("cnot", qubit - 1, qubit) for qubit in range(1, 10)]
    )
    serial = [QubitRegister(10) for _ in range(registers)]
    parallel = [QubitRegister(10) for _ in range(registers)]
    try:
        start = time.perf_counter()
        plan.apply_many(serial, workers=1)
        serial_time = time.perf_counter() - start
        start = time.perf_counter()
        plan.apply_many(parallel, workers=0)
        parallel_time = time.perf_counter() - start
        return serial_time, parallel_time
    finally:
        for state in serial + parallel:
            state.close()
        plan.close()


def main() -> None:
    operation_count = 50_000
    plan = OperationPlan([("h", 0)] * operation_count)
    scalar = timed(lambda: scalar_run(operation_count), repeats=3)
    batch = timed(lambda: batch_run(plan), repeats=3)
    first, cached, invalidated = qsc_run()
    scalar_readout, bulk_readout = readout_run()
    rebuilt, parameterized = parameter_sweep()
    serial_ensemble, parallel_ensemble = ensemble_run()
    print(f"operations: {operation_count}")
    print(f"scalar bridge: {scalar * 1000:.3f} ms")
    print(f"batch bridge:  {batch * 1000:.3f} ms")
    print(f"batch speedup: {scalar / batch:.2f}x")
    print(f"QSC first encode:       {first * 1000:.3f} ms")
    print(f"QSC cached encode:      {cached * 1000:.3f} ms")
    print(f"QSC after invalidation: {invalidated * 1000:.3f} ms")
    print(f"scalar readout: {scalar_readout * 1000:.3f} ms")
    print(f"bulk readout:   {bulk_readout * 1000:.3f} ms")
    print(f"readout speedup: {scalar_readout / bulk_readout:.2f}x")
    print(f"rebuilt parameter sweep: {rebuilt * 1000:.3f} ms")
    print(f"native parameter sweep:  {parameterized * 1000:.3f} ms")
    print(f"parameter sweep speedup: {rebuilt / parameterized:.2f}x")
    print(f"serial ensemble:   {serial_ensemble * 1000:.3f} ms")
    print(f"parallel ensemble: {parallel_ensemble * 1000:.3f} ms")
    print(f"ensemble speedup: {serial_ensemble / parallel_ensemble:.2f}x")


if __name__ == "__main__":
    main()
