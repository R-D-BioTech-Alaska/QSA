from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .causal import CausalParameterizedPlan, CausalRegister
from .causal_components import (
    extract_component_closure,
    parameterized_plan_qubits,
    remap_parameterized_plan,
    remap_support_plan,
    support_plan_qubits,
)
from .causal_support import CausalPauliSupportPlan
from .causal_support_grad import CausalSupportParameterShift


@dataclass(frozen=True)
class ComponentParameterShiftResult:
    values: tuple[float, ...]
    jacobian: tuple[tuple[float, ...], ...]
    parameter_names: tuple[str, ...]
    observable_supports: tuple[tuple[tuple[int, str], ...], ...]
    global_qubits: tuple[int, ...]

    @property
    def local_qubit_count(self) -> int:
        return len(self.global_qubits)

    def vjp(self, cotangent: Sequence[float]) -> tuple[float, ...]:
        weights = tuple(float(value) for value in cotangent)
        if len(weights) != len(self.values):
            raise ValueError("cotangent width must match the observable count")
        return tuple(
            sum(
                weights[row] * self.jacobian[row][column]
                for row in range(len(weights))
            )
            for column in range(len(self.parameter_names))
        )

    def jvp(self, tangent: Sequence[float]) -> tuple[float, ...]:
        direction = tuple(float(value) for value in tangent)
        if len(direction) != len(self.parameter_names):
            raise ValueError("tangent width must match the parameter count")
        return tuple(
            sum(
                self.jacobian[row][column] * direction[column]
                for column in range(len(direction))
            )
            for row in range(len(self.values))
        )


class CausalComponentParameterShift:
    """Exact parameter-shift gradients over a complete component closure."""

    def __init__(
        self,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
        *,
        max_local_qubits: int = 24,
    ) -> None:
        if not isinstance(plan, CausalParameterizedPlan):
            raise TypeError("plan must be a CausalParameterizedPlan")
        if not isinstance(observables, CausalPauliSupportPlan):
            raise TypeError("observables must be a CausalPauliSupportPlan")
        limit = int(max_local_qubits)
        if limit <= 0 or limit > 24:
            raise ValueError("max_local_qubits must be between 1 and 24")

        self.plan = plan
        self.observables = observables
        self.max_local_qubits = limit
        self.parameter_names = tuple(plan.parameter_names)
        self._requested_qubits = tuple(
            sorted(
                set(parameterized_plan_qubits(plan))
                | set(support_plan_qubits(observables))
            )
        )
        if not self._requested_qubits:
            raise ValueError("component gradient has no requested qubits")

        CausalSupportParameterShift(plan, observables)
        self._local_gradients = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("component gradient is closed")

    def _local_gradient(self, global_qubits):
        key = tuple(global_qubits)
        cached = self._local_gradients.get(key)
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
            gradient = CausalSupportParameterShift(
                local_plan,
                local_observables,
            )
        except Exception:
            local_observables.close()
            local_plan.close()
            raise
        self._local_gradients[key] = (
            local_plan,
            local_observables,
            gradient,
        )
        return gradient

    def evaluate_and_jacobian(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        *,
        workers: int = 0,
    ) -> ComponentParameterShiftResult:
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
            local_gradient = self._local_gradient(extracted.global_qubits)
            local = local_gradient.evaluate_and_jacobian(
                extracted.state,
                values,
                workers=workers,
            )
            return ComponentParameterShiftResult(
                values=local.values,
                jacobian=local.jacobian,
                parameter_names=self.parameter_names,
                observable_supports=tuple(self.observables.observables),
                global_qubits=extracted.global_qubits,
            )
        finally:
            extracted.close()

    def jacobian(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        *,
        workers: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        return self.evaluate_and_jacobian(
            state,
            values,
            workers=workers,
        ).jacobian

    def vjp(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        cotangent: Sequence[float],
        *,
        workers: int = 0,
    ) -> tuple[float, ...]:
        return self.evaluate_and_jacobian(
            state,
            values,
            workers=workers,
        ).vjp(cotangent)

    def jvp(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        tangent: Sequence[float],
        *,
        workers: int = 0,
    ) -> tuple[float, ...]:
        return self.evaluate_and_jacobian(
            state,
            values,
            workers=workers,
        ).jvp(tangent)

    def close(self) -> None:
        if self._closed:
            return
        for local_plan, local_observables, _gradient in self._local_gradients.values():
            local_observables.close()
            local_plan.close()
        self._local_gradients.clear()
        self._closed = True

    def __enter__(self) -> "CausalComponentParameterShift":
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
    "CausalComponentParameterShift",
    "ComponentParameterShiftResult",
]
