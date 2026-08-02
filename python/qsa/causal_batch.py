from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence

from .causal import (
    CausalParameterizedPlan,
    CausalPauliPlan,
    CausalRegister,
    CausalRuntimeError,
    _check,
    _configure,
)


_BATCH_CONFIGURED: set[int] = set()


def _configure_batch(state: CausalRegister) -> None:
    bindings = state._bindings
    _configure(bindings)
    key = id(bindings)
    if key in _BATCH_CONFIGURED:
        return
    lib = bindings.lib
    required = (
        "qcausal_fork_many",
        "qcausal_parameterized_plan_execute_many",
        "qcausal_pauli_plan_execute_many",
    )
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        raise CausalRuntimeError(
            "Loaded QSA library does not provide causal batches: "
            + ", ".join(missing)
        )

    handle = ctypes.c_void_p
    size = ctypes.c_size_t
    double_p = ctypes.POINTER(ctypes.c_double)
    lib.qcausal_fork_many.argtypes = [handle, ctypes.POINTER(handle), size]
    lib.qcausal_fork_many.restype = ctypes.c_int
    lib.qcausal_parameterized_plan_execute_many.argtypes = [
        handle,
        ctypes.POINTER(handle),
        size,
        double_p,
        size,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_parameterized_plan_execute_many.restype = ctypes.c_int
    lib.qcausal_pauli_plan_execute_many.argtypes = [
        handle,
        ctypes.POINTER(handle),
        size,
        double_p,
        size,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_pauli_plan_execute_many.restype = ctypes.c_int
    _BATCH_CONFIGURED.add(key)


def _state_handles(states: Sequence[CausalRegister]) -> tuple[object, object]:
    materialized = tuple(states)
    if not materialized:
        array_type = ctypes.c_void_p * 0
        return materialized, array_type()
    first = materialized[0]
    first._ensure_open()
    for state in materialized[1:]:
        state._ensure_open()
        if state._bindings is not first._bindings:
            raise ValueError("all causal states must use the same native library")
    array_type = ctypes.c_void_p * len(materialized)
    return materialized, array_type(*(state._handle for state in materialized))


def fork_many(state: CausalRegister, count: int) -> list[CausalRegister]:
    """Create all branches in one native call without QSC reconstruction."""
    state._ensure_open()
    branch_count = int(count)
    if branch_count < 0:
        raise ValueError("fork count cannot be negative")
    _configure_batch(state)
    array_type = ctypes.c_void_p * branch_count
    output = array_type()
    _check(
        state._bindings,
        state._bindings.lib.qcausal_fork_many(
            state._handle,
            output,
            branch_count,
        ),
    )
    branches: list[CausalRegister] = []
    try:
        for raw in output:
            branches.append(
                CausalRegister(
                    int(state.qubit_count),
                    _bindings=state._bindings,
                    _handle=raw,
                )
            )
        return branches
    except Exception:
        for branch in branches:
            branch.close()
        for raw in output[len(branches):]:
            if raw:
                state._bindings.lib.qcausal_destroy(raw)
        raise


def apply_many(
    plan: CausalParameterizedPlan,
    states: Sequence[CausalRegister],
    values: Sequence[Mapping[str, float] | Sequence[float]],
    *,
    workers: int = 0,
) -> list[CausalRegister]:
    """Apply one plan with a distinct parameter row for every causal branch."""
    if not isinstance(plan, CausalParameterizedPlan):
        raise TypeError("plan must be a CausalParameterizedPlan")
    materialized, handles = _state_handles(states)
    rows = tuple(values)
    if len(rows) != len(materialized):
        raise ValueError("one parameter row is required for every causal state")
    if not materialized:
        return []
    _configure_batch(materialized[0])
    bound_rows = tuple(plan._values(row) for row in rows)
    width = plan.parameter_count
    flattened = tuple(value for row in bound_rows for value in row)
    array_type = ctypes.c_double * len(flattened)
    parameter_buffer = array_type(*flattened)
    completed = ctypes.c_size_t()
    native = plan._native_handle(materialized[0]._bindings)
    _check(
        materialized[0]._bindings,
        materialized[0]._bindings.lib.qcausal_parameterized_plan_execute_many(
            native,
            handles,
            len(materialized),
            parameter_buffer,
            width,
            max(0, int(workers)),
            ctypes.byref(completed),
        ),
    )
    if completed.value != len(materialized):
        raise CausalRuntimeError(
            f"causal batch completed {completed.value} of {len(materialized)} states"
        )
    return list(materialized)


def observe_many(
    plan: CausalPauliPlan,
    states: Sequence[CausalRegister],
    *,
    workers: int = 0,
) -> tuple[tuple[float, ...], ...]:
    """Evaluate one Pauli closure over many branches in one native call."""
    if not isinstance(plan, CausalPauliPlan):
        raise TypeError("plan must be a CausalPauliPlan")
    materialized, handles = _state_handles(states)
    if not materialized:
        return ()
    _configure_batch(materialized[0])
    if any(int(state.qubit_count) != plan.qubits for state in materialized):
        raise ValueError("Pauli plan qubit count differs from a causal state")
    observable_count = len(plan.words)
    total = len(materialized) * observable_count
    array_type = ctypes.c_double * total
    output = array_type()
    completed = ctypes.c_size_t()
    native = plan._native_handle(materialized[0]._bindings)
    _check(
        materialized[0]._bindings,
        materialized[0]._bindings.lib.qcausal_pauli_plan_execute_many(
            native,
            handles,
            len(materialized),
            output,
            total,
            max(0, int(workers)),
            ctypes.byref(completed),
        ),
    )
    if completed.value != len(materialized):
        raise CausalRuntimeError(
            f"Pauli batch completed {completed.value} of {len(materialized)} states"
        )
    return tuple(
        tuple(
            float(output[row * observable_count + column])
            for column in range(observable_count)
        )
        for row in range(len(materialized))
    )


__all__ = ["apply_many", "fork_many", "observe_many"]
