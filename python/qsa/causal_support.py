from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Tuple

from .causal import (
    CausalRegister,
    CausalRuntimeError,
    _Bindings,
    _check,
    _configure,
    _error,
)
from .causal_batch import _state_handles


class _NativeSupportTerm(ctypes.Structure):
    _fields_ = [
        ("qubit", ctypes.c_uint32),
        ("axis", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


_CONFIGURED_BINDINGS = set()


def _configure_support(bindings: _Bindings) -> None:
    _configure(bindings)
    key = id(bindings)
    if key in _CONFIGURED_BINDINGS:
        return

    lib = bindings.lib
    required = (
        "qcausal_pauli_support_plan_create",
        "qcausal_pauli_support_plan_destroy",
        "qcausal_pauli_support_plan_observable_count",
        "qcausal_pauli_support_plan_term_count",
        "qcausal_pauli_support_plan_execute",
        "qcausal_pauli_support_plan_execute_many",
    )
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        raise CausalRuntimeError(
            "Loaded QSA library does not provide compact Pauli support: "
            + ", ".join(missing)
        )

    handle = ctypes.c_void_p
    size = ctypes.c_size_t
    double_p = ctypes.POINTER(ctypes.c_double)
    support_p = ctypes.POINTER(_NativeSupportTerm)

    lib.qcausal_pauli_support_plan_create.argtypes = [
        size,
        support_p,
        size,
        ctypes.POINTER(size),
        size,
        ctypes.c_double,
    ]
    lib.qcausal_pauli_support_plan_create.restype = handle
    lib.qcausal_pauli_support_plan_destroy.argtypes = [handle]
    lib.qcausal_pauli_support_plan_destroy.restype = None
    lib.qcausal_pauli_support_plan_observable_count.argtypes = [handle]
    lib.qcausal_pauli_support_plan_observable_count.restype = size
    lib.qcausal_pauli_support_plan_term_count.argtypes = [handle]
    lib.qcausal_pauli_support_plan_term_count.restype = size
    lib.qcausal_pauli_support_plan_execute.argtypes = [
        handle,
        handle,
        double_p,
        size,
    ]
    lib.qcausal_pauli_support_plan_execute.restype = ctypes.c_int
    lib.qcausal_pauli_support_plan_execute_many.argtypes = [
        handle,
        ctypes.POINTER(handle),
        size,
        double_p,
        size,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_pauli_support_plan_execute_many.restype = ctypes.c_int
    _CONFIGURED_BINDINGS.add(key)


def _normalize_observables(
    qubits: int,
    observables: Sequence[Sequence[Tuple[int, str]]],
) -> Tuple[Tuple[Tuple[int, str], ...], ...]:
    normalized = []
    for observable in observables:
        terms = []
        seen = set()
        for term in observable:
            if len(term) != 2:
                raise ValueError("each Pauli support term must contain qubit and axis")
            qubit = int(term[0])
            axis = str(term[1]).upper()
            if qubit < 0 or qubit >= qubits:
                raise ValueError("Pauli support qubit is outside the register")
            if axis == "I":
                continue
            if axis not in {"X", "Y", "Z"}:
                raise ValueError("Pauli support axes may contain only I, X, Y, and Z")
            if qubit in seen:
                raise ValueError("Pauli support contains a duplicate qubit")
            seen.add(qubit)
            terms.append((qubit, axis))
        terms.sort(key=lambda item: item[0])
        normalized.append(tuple(terms))
    if not normalized:
        raise ValueError("at least one Pauli support observable is required")
    return tuple(normalized)


class CausalPauliSupportPlan:
    """Exact Pauli observables stored by non-identity support."""

    def __init__(
        self,
        qubits: int,
        observables: Sequence[Sequence[Tuple[int, str]]],
        *,
        imaginary_tolerance: float = 1.0e-10,
    ) -> None:
        if int(qubits) <= 0:
            raise ValueError("qubits must be positive")
        self.qubits = int(qubits)
        self.observables = _normalize_observables(self.qubits, observables)
        self.imaginary_tolerance = float(imaginary_tolerance)

        native_terms = []
        offsets = [0]
        for observable in self.observables:
            for qubit, axis in observable:
                term = _NativeSupportTerm()
                term.qubit = qubit
                term.axis = ord(axis)
                native_terms.append(term)
            offsets.append(len(native_terms))

        term_array = _NativeSupportTerm * len(native_terms)
        offset_array = ctypes.c_size_t * len(offsets)
        self._term_buffer = term_array(*native_terms)
        self._offset_buffer = offset_array(*offsets)
        self._native_handles = {}

    @property
    def observable_count(self) -> int:
        return len(self.observables)

    @property
    def term_count(self) -> int:
        return len(self._term_buffer)

    def _native_handle(self, bindings: _Bindings) -> ctypes.c_void_p:
        _configure_support(bindings)
        key = id(bindings)
        cached = self._native_handles.get(key)
        if cached is not None:
            return cached[1]

        raw = bindings.lib.qcausal_pauli_support_plan_create(
            self.qubits,
            self._term_buffer,
            self.term_count,
            self._offset_buffer,
            self.observable_count,
            self.imaginary_tolerance,
        )
        if not raw:
            raise CausalRuntimeError(_error(bindings))
        handle = ctypes.c_void_p(raw)
        native_observables = int(
            bindings.lib.qcausal_pauli_support_plan_observable_count(handle)
        )
        native_terms = int(
            bindings.lib.qcausal_pauli_support_plan_term_count(handle)
        )
        if native_observables != self.observable_count or native_terms != self.term_count:
            bindings.lib.qcausal_pauli_support_plan_destroy(handle)
            raise CausalRuntimeError("compact Pauli support plan metadata differs")
        self._native_handles[key] = (bindings, handle)
        return handle

    def execute(self, state: CausalRegister) -> Tuple[float, ...]:
        state._ensure_open()
        if int(state.qubit_count) != self.qubits:
            raise ValueError("Pauli support qubit count differs from causal state")
        native = self._native_handle(state._bindings)
        output_type = ctypes.c_double * self.observable_count
        output = output_type()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_pauli_support_plan_execute(
                native,
                state._handle,
                output,
                self.observable_count,
            ),
        )
        return tuple(float(value) for value in output)

    def execute_many(
        self,
        states: Sequence[CausalRegister],
        *,
        workers: int = 0,
    ) -> Tuple[Tuple[float, ...], ...]:
        materialized, handles = _state_handles(states)
        if not materialized:
            return ()
        if any(int(state.qubit_count) != self.qubits for state in materialized):
            raise ValueError("Pauli support qubit count differs from a causal state")

        bindings = materialized[0]._bindings
        native = self._native_handle(bindings)
        total = len(materialized) * self.observable_count
        output_type = ctypes.c_double * total
        output = output_type()
        completed = ctypes.c_size_t()
        _check(
            bindings,
            bindings.lib.qcausal_pauli_support_plan_execute_many(
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
                f"Pauli support batch completed {completed.value} of "
                f"{len(materialized)} states"
            )
        return tuple(
            tuple(
                float(output[row * self.observable_count + column])
                for column in range(self.observable_count)
            )
            for row in range(len(materialized))
        )

    def close(self) -> None:
        handles = tuple(self._native_handles.values())
        self._native_handles.clear()
        for bindings, handle in handles:
            try:
                bindings.lib.qcausal_pauli_support_plan_destroy(handle)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["CausalPauliSupportPlan"]
