from __future__ import annotations

import ctypes
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .causal import (
    CausalParameterizedPlan,
    CausalRegister,
    CausalRuntimeError,
    _Bindings,
    _check,
)
from .causal_adjoint import _configure_adjoint
from .causal_support import CausalPauliSupportPlan


_CONFIGURED_DENSE_BINDINGS: set[int] = set()


def _configure_dense_batch(bindings: _Bindings) -> None:
    _configure_adjoint(bindings)
    key = id(bindings)
    if key in _CONFIGURED_DENSE_BINDINGS:
        return
    lib = bindings.lib
    if not hasattr(lib, "qcausal_observables_many_dense"):
        raise CausalRuntimeError(
            "Loaded QSA library does not provide fused dense batch primals"
        )
    handle = ctypes.c_void_p
    size = ctypes.c_size_t
    double_p = ctypes.POINTER(ctypes.c_double)
    lib.qcausal_observables_many_dense.argtypes = [
        handle,
        handle,
        handle,
        double_p,
        size,
        size,
        size,
        double_p,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_observables_many_dense.restype = ctypes.c_int
    _CONFIGURED_DENSE_BINDINGS.add(key)


@dataclass(frozen=True)
class DenseBatchValuesResult:
    values: tuple[tuple[float, ...], ...]
    parameter_names: tuple[str, ...]
    observable_supports: tuple[tuple[tuple[int, str], ...], ...]
    qubit_count: int
    forward_sweeps: int
    native_calls: int = 1
    state_materializations: int = 1
    structural_branches: int = 0

    @property
    def row_count(self) -> int:
        return len(self.values)


class CausalDenseBatchPrimal:
    """Exact fused batch primal over one bounded local causal register.

    The state is materialized once inside the native call. Every parameter row
    reuses that immutable root amplitude vector, applies the shared gate
    schedule directly, and emits compact Pauli expectations row-major. No
    causal forks, persistent candidates, or per-row QRegister reconstruction
    are used.
    """

    def __init__(
        self,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
        *,
        max_qubits: int = 16,
    ) -> None:
        limit = int(max_qubits)
        if limit <= 0 or limit > 24:
            raise ValueError("max_qubits must be between 1 and 24")
        self.plan = plan
        self.observables = observables
        self.parameter_names = tuple(plan.parameter_names)
        self.max_qubits = limit

    def evaluate(
        self,
        state: CausalRegister,
        values: Sequence[Mapping[str, float] | Sequence[float]],
    ) -> DenseBatchValuesResult:
        state._ensure_open()
        if int(state.qubit_count) != self.observables.qubits:
            raise ValueError("observable qubit count differs from causal state")
        if int(state.qubit_count) > self.max_qubits:
            raise ValueError("causal state exceeds the dense batch qubit limit")

        rows = tuple(values)
        if not rows:
            return DenseBatchValuesResult(
                values=(),
                parameter_names=self.parameter_names,
                observable_supports=tuple(self.observables.observables),
                qubit_count=int(state.qubit_count),
                forward_sweeps=0,
                native_calls=0,
                state_materializations=0,
            )
        parameter_rows = tuple(
            tuple(float(value) for value in self.plan._values(row))
            for row in rows
        )
        parameter_width = len(self.parameter_names)
        observable_width = self.observables.observable_count
        flattened = tuple(value for row in parameter_rows for value in row)

        _configure_dense_batch(state._bindings)
        native_plan = self.plan._native_handle(state._bindings)
        native_observables = self.observables._native_handle(state._bindings)
        parameter_type = ctypes.c_double * len(flattened)
        output_type = ctypes.c_double * (len(rows) * observable_width)
        parameter_buffer = parameter_type(*flattened)
        output = output_type()
        completed = ctypes.c_size_t()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_observables_many_dense(
                state._handle,
                native_plan,
                native_observables,
                parameter_buffer,
                len(rows),
                parameter_width,
                self.max_qubits,
                output,
                len(output),
                ctypes.byref(completed),
            ),
        )
        if completed.value != len(rows):
            raise CausalRuntimeError(
                f"dense batch primal completed {completed.value} of {len(rows)} rows"
            )
        result_rows = tuple(
            tuple(
                float(output[row * observable_width + column])
                for column in range(observable_width)
            )
            for row in range(len(rows))
        )
        return DenseBatchValuesResult(
            values=result_rows,
            parameter_names=self.parameter_names,
            observable_supports=tuple(self.observables.observables),
            qubit_count=int(state.qubit_count),
            forward_sweeps=len(rows),
        )


__all__ = ["CausalDenseBatchPrimal", "DenseBatchValuesResult"]
