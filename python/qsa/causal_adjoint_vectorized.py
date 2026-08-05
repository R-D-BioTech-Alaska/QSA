from __future__ import annotations

import ctypes
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

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
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


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
    lib.qcausal_observables_many_vectorized.argtypes = [
        handle,
        handle,
        handle,
        _DOUBLE_POINTER,
        size,
        size,
        size,
        _DOUBLE_POINTER,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_observables_many_vectorized.restype = ctypes.c_int
    lib.qcausal_weighted_adjoint_many_vectorized.argtypes = [
        handle,
        handle,
        handle,
        _DOUBLE_POINTER,
        size,
        size,
        _DOUBLE_POINTER,
        size,
        size,
        _DOUBLE_POINTER,
        size,
        _DOUBLE_POINTER,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_weighted_adjoint_many_vectorized.restype = ctypes.c_int
    _CONFIGURED_VECTORIZED_BINDINGS.add(key)


@dataclass(frozen=True)
class NumpyWeightedAdjointResult:
    values: np.ndarray
    gradients: np.ndarray
    native_calls: int = 1
    forward_sweeps: int = 0
    reverse_sweeps: int = 0
    shifted_evaluations: int = 0
    state_materializations: int = 1

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])


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

    def _numpy_parameters(self, values) -> np.ndarray:
        parameters = np.asarray(values)
        if parameters.ndim != 2:
            raise ValueError("parameter rows must be a two-dimensional matrix")
        if parameters.shape[1] != len(self.parameter_names):
            raise ValueError("parameter matrix width must match the plan")
        return np.ascontiguousarray(parameters, dtype=np.float64)

    def values_numpy(
        self,
        state: CausalRegister,
        parameters,
    ) -> np.ndarray:
        """Return a contiguous [rows, observables] float64 array.

        A contiguous float64 input is passed directly to the native runtime;
        no Python scalar expansion or ctypes-array copy is performed.
        """
        self._validate_state(state)
        parameter_rows = self._numpy_parameters(parameters)
        row_count = int(parameter_rows.shape[0])
        observable_width = self.observables.observable_count
        output = np.empty((row_count, observable_width), dtype=np.float64)
        if row_count == 0:
            return output

        _configure_vectorized(state._bindings)
        native_plan = self.plan._native_handle(state._bindings)
        native_observables = self.observables._native_handle(state._bindings)
        completed = ctypes.c_size_t()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_observables_many_vectorized(
                state._handle,
                native_plan,
                native_observables,
                parameter_rows.ctypes.data_as(_DOUBLE_POINTER),
                row_count,
                int(parameter_rows.shape[1]),
                self.max_qubits,
                output.ctypes.data_as(_DOUBLE_POINTER),
                int(output.size),
                ctypes.byref(completed),
            ),
        )
        if completed.value != row_count:
            raise CausalRuntimeError(
                f"vectorized primal completed {completed.value} of {row_count} rows"
            )
        return output

    def evaluate_numpy(
        self,
        state: CausalRegister,
        parameters,
        cotangents,
    ) -> NumpyWeightedAdjointResult:
        """Return contiguous value and gradient arrays without scalar copies."""
        self._validate_state(state)
        parameter_rows = self._numpy_parameters(parameters)
        weight_rows = np.asarray(cotangents)
        if weight_rows.ndim != 2:
            raise ValueError("cotangent rows must be a two-dimensional matrix")
        if weight_rows.shape[0] != parameter_rows.shape[0]:
            raise ValueError("one cotangent row is required for every parameter row")
        if weight_rows.shape[1] != self.observables.observable_count:
            raise ValueError("cotangent matrix width must match the observables")
        weight_rows = np.ascontiguousarray(weight_rows, dtype=np.float64)
        row_count = int(parameter_rows.shape[0])
        observable_width = int(weight_rows.shape[1])
        parameter_width = int(parameter_rows.shape[1])
        value_output = np.empty((row_count, observable_width), dtype=np.float64)
        gradient_output = np.empty((row_count, parameter_width), dtype=np.float64)
        if row_count == 0:
            return NumpyWeightedAdjointResult(
                values=value_output,
                gradients=gradient_output,
                native_calls=0,
                forward_sweeps=0,
                reverse_sweeps=0,
                state_materializations=0,
            )

        _configure_vectorized(state._bindings)
        native_plan = self.plan._native_handle(state._bindings)
        native_observables = self.observables._native_handle(state._bindings)
        completed = ctypes.c_size_t()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_weighted_adjoint_many_vectorized(
                state._handle,
                native_plan,
                native_observables,
                parameter_rows.ctypes.data_as(_DOUBLE_POINTER),
                row_count,
                parameter_width,
                weight_rows.ctypes.data_as(_DOUBLE_POINTER),
                observable_width,
                self.max_qubits,
                value_output.ctypes.data_as(_DOUBLE_POINTER),
                int(value_output.size),
                gradient_output.ctypes.data_as(_DOUBLE_POINTER),
                int(gradient_output.size),
                ctypes.byref(completed),
            ),
        )
        if completed.value != row_count:
            raise CausalRuntimeError(
                f"vectorized adjoint completed {completed.value} of {row_count} rows"
            )
        return NumpyWeightedAdjointResult(
            values=value_output,
            gradients=gradient_output,
            forward_sweeps=row_count,
            reverse_sweeps=row_count,
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
        output = self.values_numpy(state, rows)
        return DenseBatchValuesResult(
            values=tuple(tuple(float(value) for value in row) for row in output),
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
        for row in weight_rows:
            if len(row) != self.observables.observable_count:
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
        result = self.evaluate_numpy(state, rows, weight_rows)
        return WeightedAdjointBatchResult(
            values=tuple(
                tuple(float(value) for value in row) for row in result.values
            ),
            gradients=tuple(
                tuple(float(value) for value in row) for row in result.gradients
            ),
            parameter_names=self.parameter_names,
            observable_supports=tuple(self.observables.observables),
            global_qubits=tuple(range(int(state.qubit_count))),
            forward_sweeps=result.forward_sweeps,
            reverse_sweeps=result.reverse_sweeps,
        )


__all__ = [
    "CausalVectorizedBatchAdjoint",
    "NumpyWeightedAdjointResult",
]
