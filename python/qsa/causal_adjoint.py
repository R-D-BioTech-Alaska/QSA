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
from .causal_components import (
    extract_component_closure,
    parameterized_plan_qubits,
    remap_parameterized_plan,
    remap_support_plan,
    support_plan_qubits,
)
from .causal_support import (
    CausalPauliSupportPlan,
    _configure_support,
)


_CONFIGURED_BINDINGS: set[int] = set()


def _configure_adjoint(bindings: _Bindings) -> None:
    _configure_support(bindings)
    key = id(bindings)
    if key in _CONFIGURED_BINDINGS:
        return
    lib = bindings.lib
    if not hasattr(lib, "qcausal_weighted_adjoint"):
        raise CausalRuntimeError(
            "Loaded QSA library does not provide the weighted adjoint runtime"
        )
    handle = ctypes.c_void_p
    size = ctypes.c_size_t
    double_p = ctypes.POINTER(ctypes.c_double)
    lib.qcausal_weighted_adjoint.argtypes = [
        handle,
        handle,
        handle,
        double_p,
        size,
        double_p,
        size,
        size,
        double_p,
        size,
        double_p,
        size,
    ]
    lib.qcausal_weighted_adjoint.restype = ctypes.c_int
    if hasattr(lib, "qcausal_weighted_adjoint_many"):
        lib.qcausal_weighted_adjoint_many.argtypes = [
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
        lib.qcausal_weighted_adjoint_many.restype = ctypes.c_int
    _CONFIGURED_BINDINGS.add(key)


@dataclass(frozen=True)
class WeightedAdjointResult:
    values: tuple[float, ...]
    gradient: tuple[float, ...]
    parameter_names: tuple[str, ...]
    observable_supports: tuple[tuple[tuple[int, str], ...], ...]
    global_qubits: tuple[int, ...]
    forward_sweeps: int = 1
    reverse_sweeps: int = 1
    shifted_evaluations: int = 0
    stored_forward_states: int = 0

    @property
    def local_qubit_count(self) -> int:
        return len(self.global_qubits)


@dataclass(frozen=True)
class WeightedAdjointBatchResult:
    values: tuple[tuple[float, ...], ...]
    gradients: tuple[tuple[float, ...], ...]
    parameter_names: tuple[str, ...]
    observable_supports: tuple[tuple[tuple[int, str], ...], ...]
    global_qubits: tuple[int, ...]
    forward_sweeps: int
    reverse_sweeps: int
    shifted_evaluations: int = 0
    stored_forward_states: int = 0
    native_calls: int = 1

    @property
    def row_count(self) -> int:
        return len(self.values)

    @property
    def local_qubit_count(self) -> int:
        return len(self.global_qubits)


class CausalWeightedAdjoint:
    """Exact one-cotangent reverse sweep over a bounded local QSA state."""

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
            raise ValueError("causal state exceeds the adjoint qubit limit")

    def evaluate(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        cotangent: Sequence[float],
    ) -> WeightedAdjointResult:
        self._validate_state(state)

        parameters = tuple(float(value) for value in self.plan._values(values))
        weights = tuple(float(value) for value in cotangent)
        if len(weights) != self.observables.observable_count:
            raise ValueError("cotangent width must match the observable count")

        _configure_adjoint(state._bindings)
        native_plan = self.plan._native_handle(state._bindings)
        native_observables = self.observables._native_handle(state._bindings)
        parameter_type = ctypes.c_double * len(parameters)
        cotangent_type = ctypes.c_double * len(weights)
        value_type = ctypes.c_double * self.observables.observable_count
        gradient_type = ctypes.c_double * len(self.parameter_names)
        parameter_buffer = parameter_type(*parameters)
        cotangent_buffer = cotangent_type(*weights)
        value_output = value_type()
        gradient_output = gradient_type()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_weighted_adjoint(
                state._handle,
                native_plan,
                native_observables,
                parameter_buffer,
                len(parameters),
                cotangent_buffer,
                len(weights),
                self.max_qubits,
                value_output,
                self.observables.observable_count,
                gradient_output,
                len(self.parameter_names),
            ),
        )
        return WeightedAdjointResult(
            values=tuple(float(value) for value in value_output),
            gradient=tuple(float(value) for value in gradient_output),
            parameter_names=self.parameter_names,
            observable_supports=tuple(self.observables.observables),
            global_qubits=tuple(range(int(state.qubit_count))),
        )

    def evaluate_many(
        self,
        state: CausalRegister,
        values: Sequence[Mapping[str, float] | Sequence[float]],
        cotangents: Sequence[Sequence[float]],
    ) -> WeightedAdjointBatchResult:
        """Evaluate exact row-wise VJPs through one deterministic native call."""
        self._validate_state(state)
        rows = tuple(values)
        weight_rows = tuple(tuple(float(value) for value in row) for row in cotangents)
        if len(rows) != len(weight_rows):
            raise ValueError("one cotangent row is required for every parameter row")
        for weights in weight_rows:
            if len(weights) != self.observables.observable_count:
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

        parameter_rows = tuple(
            tuple(float(value) for value in self.plan._values(row))
            for row in rows
        )
        parameter_width = len(self.parameter_names)
        observable_width = self.observables.observable_count
        flattened_parameters = tuple(
            value for row in parameter_rows for value in row
        )
        flattened_cotangents = tuple(
            value for row in weight_rows for value in row
        )

        _configure_adjoint(state._bindings)
        if not hasattr(state._bindings.lib, "qcausal_weighted_adjoint_many"):
            raise CausalRuntimeError(
                "Loaded QSA library does not provide batch weighted adjoints"
            )
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
            state._bindings.lib.qcausal_weighted_adjoint_many(
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
                f"batch adjoint completed {completed.value} of {len(rows)} rows"
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

    def vjp(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        cotangent: Sequence[float],
    ) -> tuple[float, ...]:
        return self.evaluate(state, values, cotangent).gradient

    def vjp_many(
        self,
        state: CausalRegister,
        values: Sequence[Mapping[str, float] | Sequence[float]],
        cotangents: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        return self.evaluate_many(state, values, cotangents).gradients


class CausalComponentWeightedAdjoint:
    """Extract one complete connected closure and run the exact adjoint."""

    def __init__(
        self,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
        *,
        max_local_qubits: int = 16,
    ) -> None:
        limit = int(max_local_qubits)
        if limit <= 0 or limit > 24:
            raise ValueError("max_local_qubits must be between 1 and 24")
        self.plan = plan
        self.observables = observables
        self.parameter_names = tuple(plan.parameter_names)
        self.max_local_qubits = limit
        self._requested_qubits = tuple(
            sorted(
                set(parameterized_plan_qubits(plan))
                | set(support_plan_qubits(observables))
            )
        )
        if not self._requested_qubits:
            raise ValueError("component adjoint has no requested qubits")
        self._local_adjoint: dict[
            tuple[int, ...],
            tuple[
                CausalParameterizedPlan,
                CausalPauliSupportPlan,
                CausalWeightedAdjoint,
            ],
        ] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise CausalRuntimeError("component adjoint is closed")

    def _local_runtime(
        self,
        global_qubits: Sequence[int],
    ) -> CausalWeightedAdjoint:
        key = tuple(int(value) for value in global_qubits)
        cached = self._local_adjoint.get(key)
        if cached is not None:
            return cached[2]
        local_by_global = {
            global_qubit: local_qubit
            for local_qubit, global_qubit in enumerate(key)
        }
        local_plan = remap_parameterized_plan(self.plan, local_by_global)
        try:
            local_observables = remap_support_plan(
                self.observables,
                local_by_global,
            )
        except Exception:
            local_plan.close()
            raise
        try:
            runtime = CausalWeightedAdjoint(
                local_plan,
                local_observables,
                max_qubits=self.max_local_qubits,
            )
        except Exception:
            local_observables.close()
            local_plan.close()
            raise
        self._local_adjoint[key] = (
            local_plan,
            local_observables,
            runtime,
        )
        return runtime

    def evaluate(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        cotangent: Sequence[float],
    ) -> WeightedAdjointResult:
        self._ensure_open()
        state._ensure_open()
        if int(state.qubit_count) != self.observables.qubits:
            raise ValueError("observable qubit count differs from causal state")
        extracted = extract_component_closure(
            state,
            self._requested_qubits,
            max_local_qubits=self.max_local_qubits,
        )
        try:
            local = self._local_runtime(extracted.global_qubits).evaluate(
                extracted.state,
                values,
                cotangent,
            )
            return WeightedAdjointResult(
                values=local.values,
                gradient=local.gradient,
                parameter_names=self.parameter_names,
                observable_supports=tuple(self.observables.observables),
                global_qubits=tuple(extracted.global_qubits),
            )
        finally:
            extracted.close()

    def evaluate_many(
        self,
        state: CausalRegister,
        values: Sequence[Mapping[str, float] | Sequence[float]],
        cotangents: Sequence[Sequence[float]],
    ) -> WeightedAdjointBatchResult:
        """Extract one closure once, then execute all row VJPs natively."""
        self._ensure_open()
        state._ensure_open()
        if int(state.qubit_count) != self.observables.qubits:
            raise ValueError("observable qubit count differs from causal state")
        extracted = extract_component_closure(
            state,
            self._requested_qubits,
            max_local_qubits=self.max_local_qubits,
        )
        try:
            local = self._local_runtime(extracted.global_qubits).evaluate_many(
                extracted.state,
                values,
                cotangents,
            )
            return WeightedAdjointBatchResult(
                values=local.values,
                gradients=local.gradients,
                parameter_names=self.parameter_names,
                observable_supports=tuple(self.observables.observables),
                global_qubits=tuple(extracted.global_qubits),
                forward_sweeps=local.forward_sweeps,
                reverse_sweeps=local.reverse_sweeps,
                shifted_evaluations=local.shifted_evaluations,
                stored_forward_states=local.stored_forward_states,
                native_calls=local.native_calls,
            )
        finally:
            extracted.close()

    def vjp(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        cotangent: Sequence[float],
    ) -> tuple[float, ...]:
        return self.evaluate(state, values, cotangent).gradient

    def vjp_many(
        self,
        state: CausalRegister,
        values: Sequence[Mapping[str, float] | Sequence[float]],
        cotangents: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        return self.evaluate_many(state, values, cotangents).gradients

    def close(self) -> None:
        if self._closed:
            return
        for local_plan, local_observables, _runtime in self._local_adjoint.values():
            local_observables.close()
            local_plan.close()
        self._local_adjoint.clear()
        self._closed = True

    def __enter__(self) -> "CausalComponentWeightedAdjoint":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CausalComponentWeightedAdjoint",
    "CausalWeightedAdjoint",
    "WeightedAdjointBatchResult",
    "WeightedAdjointResult",
]
