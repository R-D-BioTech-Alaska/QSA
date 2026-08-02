from __future__ import annotations

import ctypes
from collections.abc import Iterable
from typing import Optional

from .causal import (
    CausalParameterizedPlan,
    CausalRegister,
    CausalRuntimeError,
    _Bindings,
    _configure,
    _error,
)
from .causal_support import CausalPauliSupportPlan


_CONFIGURED_BINDINGS = set()

_ONE_QUBIT = {"x", "y", "z", "h", "s", "sdg", "t", "tdg"}
_ROTATIONS = {"rx", "ry", "rz"}
_TWO_QUBIT = {"cnot", "cz", "swap"}
_TRAJECTORIES = {
    "bit_flip_trajectory",
    "phase_flip_trajectory",
    "depolarizing_trajectory",
    "amplitude_damping_trajectory",
}


def _configure_components(bindings: _Bindings) -> None:
    _configure(bindings)
    key = id(bindings)
    if key in _CONFIGURED_BINDINGS:
        return
    lib = bindings.lib
    if not hasattr(lib, "qcausal_extract_component_closure"):
        raise CausalRuntimeError(
            "Loaded QSA library does not provide component closure extraction"
        )
    handle = ctypes.c_void_p
    size = ctypes.c_size_t
    u32_p = ctypes.POINTER(ctypes.c_uint32)
    lib.qcausal_extract_component_closure.argtypes = [
        handle,
        u32_p,
        size,
        size,
        u32_p,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_extract_component_closure.restype = handle
    _CONFIGURED_BINDINGS.add(key)


def parameterized_plan_qubits(plan: CausalParameterizedPlan) -> tuple[int, ...]:
    if not isinstance(plan, CausalParameterizedPlan):
        raise TypeError("plan must be a CausalParameterizedPlan")
    qubits = set()
    for operation in plan.operations:
        name = str(operation[0]).lower()
        if name in _ONE_QUBIT or name in _ROTATIONS or name in _TRAJECTORIES:
            qubits.add(int(operation[1]))
        elif name in _TWO_QUBIT:
            qubits.add(int(operation[1]))
            qubits.add(int(operation[2]))
        else:
            raise ValueError(f"unsupported causal component operation: {name}")
    return tuple(sorted(qubits))


def support_plan_qubits(plan: CausalPauliSupportPlan) -> tuple[int, ...]:
    if not isinstance(plan, CausalPauliSupportPlan):
        raise TypeError("plan must be a CausalPauliSupportPlan")
    return tuple(
        sorted(
            {
                int(qubit)
                for observable in plan.observables
                for qubit, _axis in observable
            }
        )
    )


def remap_parameterized_plan(
    plan: CausalParameterizedPlan,
    local_by_global: dict[int, int],
) -> CausalParameterizedPlan:
    remapped = []
    for operation in plan.operations:
        name = str(operation[0]).lower()
        if name in _ONE_QUBIT:
            remapped.append((name, local_by_global[int(operation[1])]))
        elif name in _ROTATIONS:
            remapped.append(
                (name, local_by_global[int(operation[1])], operation[2])
            )
        elif name in _TWO_QUBIT:
            remapped.append(
                (
                    name,
                    local_by_global[int(operation[1])],
                    local_by_global[int(operation[2])],
                )
            )
        elif name in _TRAJECTORIES:
            remapped.append(
                (name, local_by_global[int(operation[1])], operation[2], operation[3])
            )
        else:
            raise ValueError(f"unsupported causal component operation: {name}")
    local = CausalParameterizedPlan(remapped, optimize=plan.optimize)
    if local.parameter_names != plan.parameter_names:
        local.close()
        raise CausalRuntimeError("local parameter order differs from the global plan")
    return local


def remap_support_plan(
    plan: CausalPauliSupportPlan,
    local_by_global: dict[int, int],
) -> CausalPauliSupportPlan:
    observables = tuple(
        tuple((local_by_global[int(qubit)], axis) for qubit, axis in observable)
        for observable in plan.observables
    )
    return CausalPauliSupportPlan(
        len(local_by_global),
        observables,
        imaginary_tolerance=plan.imaginary_tolerance,
    )


class ExtractedCausalComponents:
    def __init__(
        self,
        state: CausalRegister,
        global_qubits: tuple[int, ...],
    ) -> None:
        self.state = state
        self.global_qubits = global_qubits
        self.local_by_global = {
            global_qubit: local_qubit
            for local_qubit, global_qubit in enumerate(global_qubits)
        }
        self._closed = False

    @property
    def local_qubit_count(self) -> int:
        return len(self.global_qubits)

    def close(self) -> None:
        if self._closed:
            return
        self.state.close()
        self._closed = True

    def __enter__(self) -> "ExtractedCausalComponents":
        if self._closed:
            raise CausalRuntimeError("extracted causal components are closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def extract_component_closure(
    state: CausalRegister,
    requested_qubits: Iterable[int],
    *,
    max_local_qubits: int = 24,
) -> ExtractedCausalComponents:
    state._ensure_open()
    requested = tuple(sorted({int(qubit) for qubit in requested_qubits}))
    if not requested:
        raise ValueError("component closure request is empty")
    limit = int(max_local_qubits)
    if limit <= 0 or limit > 24:
        raise ValueError("max_local_qubits must be between 1 and 24")

    _configure_components(state._bindings)
    requested_type = ctypes.c_uint32 * len(requested)
    requested_buffer = requested_type(*requested)
    mapping_type = ctypes.c_uint32 * limit
    mapping_buffer = mapping_type()
    mapping_count = ctypes.c_size_t()
    raw = state._bindings.lib.qcausal_extract_component_closure(
        state._handle,
        requested_buffer,
        len(requested),
        limit,
        mapping_buffer,
        limit,
        ctypes.byref(mapping_count),
    )
    if not raw:
        raise CausalRuntimeError(_error(state._bindings))
    try:
        count = int(mapping_count.value)
        if count <= 0 or count > limit:
            raise CausalRuntimeError("component closure returned an invalid mapping")
        global_qubits = tuple(int(mapping_buffer[index]) for index in range(count))
        if len(set(global_qubits)) != len(global_qubits):
            raise CausalRuntimeError("component closure mapping contains duplicates")
        local_state = CausalRegister(
            count,
            _bindings=state._bindings,
            _handle=raw,
        )
        return ExtractedCausalComponents(local_state, global_qubits)
    except Exception:
        state._bindings.lib.qcausal_destroy(raw)
        raise


__all__ = [
    "ExtractedCausalComponents",
    "extract_component_closure",
    "parameterized_plan_qubits",
    "remap_parameterized_plan",
    "remap_support_plan",
    "support_plan_qubits",
]
