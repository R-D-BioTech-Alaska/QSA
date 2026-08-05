from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence

from .causal import (
    CausalParameterizedPlan,
    CausalRegister,
    CausalRuntimeError,
    _Bindings,
    _check,
)
from .causal_adjoint import WeightedAdjointBatchResult, _configure_adjoint
from .causal_adjoint_dense import DenseBatchValuesResult
from .causal_support import CausalPauliSupportPlan


_CONFIGURED_VECTORIZED_BINDINGS: set[int] = set()


def _configure_vectorized(bindings: _Bindings) -> None:
    _configure_adjoint(bindings)
    key = id(bindings)
    if key in _CONFIGURED_VECTORIZED_BINDINGS:
        return
    lib = bindings.lib
    required = (
        "qcausal_observables_many_vectorized",
        "qcausal_weighted_adjoint_many_vectorized",
    )
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        raise CausalRuntimeError(
            "Loaded QSA library does not provide vectorized batch adjoints: "
            + ", ".join(missing)
        )
    handle = ctypes.c_void_p
    size = ctypes.c_size_t
    double_p = ctypes.POINTER(ctypes.c_double)
    lib.qcausal_observables_many_vectorized.argtypes = [
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
    lib.qcausal_observables_many_vectorized.restype = ctypes.c_int
    lib.qcausal_weighted_adjoint_many_vectorized.argtypes = [
        handle,
        handle,
        handle,
        double_p,
        size,
        size,
        double_p,
        size,
        size,
        double_p,
        size,
        double_p,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_weighted_adjoint_many_vectorized.restype = ctypes.c_int
    _CONFIGURED_VECTORIZED_BINDINGS.add(key)


class CausalVectorizedBatchAdjoint:
    """Operation-major exact batch primals and VJPs on one local root."""

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

    def _validate_state(self, state: CausalRegister) -> None:
        state._ensure_open()
        if int(state.qubit_count) != self.observables.qubits:
            raise ValueError("observable qubit count differs from causal state")
        if int(state.qubit_count) > self.max_qubits:
            raise ValueError("causal state exceeds the vectorized qubit limit")

    def _parameter_rows(self, values):
        return tuple(
            tuple(float(value) for value in self.plan._values(row))
            for row in tuple(values)
        )

    def values_many(
        self,
        state: CausalRegister,
        values: Sequence[Mapping[str, float] | Sequence[float]],
    ) -> DenseBatchValuesResult:
        self._validate_state(state)
        rows = self._parameter_rows(values)
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
        parameter_width = len(self.parameter_names)
        observable_width = self.observables.observable_count
        flattened = tuple(value for row in rows for value in row)
        _configure_vectorized(state._bindings)
        native_plan = self.plan._native_handle(state._bindings)
        native_observables = self.observables._native_handle(state._bindings)
        parameter_type = ctypes.c_double * len(flattened)
        output_type = ctypes.c_double * (len(rows) * observable_width)
        parameter_buffer = parameter_type(*flattened)
        output = output_type()
        completed = ctypes.c_size_t()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_observables_many_vectorized(
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
                f"vectorized primal completed {completed.value} of {len(rows)} rows"
            )
        value_rows = tuple(
            tuple(
                float(output[row * observable_width + column])
                for column in range(observable_width)
            )
            for row in range(len(rows))
        )
        return DenseBatchValuesResult(
            values=value_rows,
            parameter_names=self.parameter_names,
            observable_supports=tuple(self.observables.observables),
            qubit_count=int(state.qubit_count),
            forward_sweeps=len(rows),
        )

    def evaluate_many(
        self,
        state: CausalRegister,
        values: Sequence[Mapping[str, float] | Sequence[float]],
        cotangents: Sequence[Sequence[float]],
    ) -> WeightedAdjointBatchResult:
        self._validate_state(state)
        rows = self._parameter_rows(values)
        weight_rows = tuple(
            tuple(float(value) for value in row) for row in tuple(cotangents)
        )
        if len(rows) != len(weight_rows):
            raise ValueError("one cotangent row is required for every parameter row")
        observable_width = self.observables.observable_count
        for row in weight_rows:
            if len(row) != observable_width:
                raise ValueError("cotangent width must match the observable count")
        if not rows:
            return WeightedAdjointBatchResult(
                values=(),
                gradients=(),
                parameter_names=self.parameter_names,
                observable_supports=tuple(self.observables.observables),
                global_qubits=tuple(range(int(state.qubit_count))),
                forward_sweeps=0,
                reverse_sweeps=0,
                native_calls=0,
            )

        parameter_width = len(self.parameter_names)
        flattened_parameters = tuple(value for row in rows for value in row)
        flattened_cotangents = tuple(
            value for row in weight_rows for value in row
        )
        _configure_vectorized(state._bindings)
        native_plan = self.plan._native_handle(state._bindings)
        native_observables = self.observables._native_handle(state._bindings)
        parameter_type = ctypes.c_double * len(flattened_parameters)
        cotangent_type = ctypes.c_double * len(flattened_cotangents)
        value_type = ctypes.c_double * (len(rows) * observable_width)
        gradient_type = ctypes.c_double * (len(rows) * parameter_width)
        parameter_buffer = parameter_type(*flattened_parameters)
        cotangent_buffer = cotangent_type(*flattened_cotangents)
        value_output = value_type()
        gradient_output = gradient_type()
        completed = ctypes.c_size_t()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_weighted_adjoint_many_vectorized(
                state._handle,
                native_plan,
                native_observables,
                parameter_buffer,
                len(rows),
                parameter_width,
                cotangent_buffer,
                observable_width,
                self.max_qubits,
                value_output,
                len(value_output),
                gradient_output,
                len(gradient_output),
                ctypes.byref(completed),
            ),
        )
        if completed.value != len(rows):
            raise CausalRuntimeError(
                f"vectorized adjoint completed {completed.value} of {len(rows)} rows"
            )
        value_rows = tuple(
            tuple(
                float(value_output[row * observable_width + column])
                for column in range(observable_width)
            )
            for row in range(len(rows))
        )
        gradient_rows = tuple(
            tuple(
                float(gradient_output[row * parameter_width + column])
                for column in range(parameter_width)
            )
            for row in range(len(rows))
        )
        return WeightedAdjointBatchResult(
            values=value_rows,
            gradients=gradient_rows,
            parameter_names=self.parameter_names,
            observable_supports=tuple(self.observables.observables),
            global_qubits=tuple(range(int(state.qubit_count))),
            forward_sweeps=len(rows),
            reverse_sweeps=len(rows),
        )


__all__ = ["CausalVectorizedBatchAdjoint"]
