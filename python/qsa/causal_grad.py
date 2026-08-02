from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from . import Parameter
from .causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from .causal_batch import apply_many, fork_many, observe_many


@dataclass(frozen=True)
class ParameterShiftResult:
    """Exact observable values and Jacobian for one causal circuit step."""

    values: tuple[float, ...]
    jacobian: tuple[tuple[float, ...], ...]
    parameter_names: tuple[str, ...]
    observable_words: tuple[str, ...]

    def vjp(self, cotangent: Sequence[float]) -> tuple[float, ...]:
        weights = tuple(float(value) for value in cotangent)
        if len(weights) != len(self.values):
            raise ValueError("cotangent width must match the observable count")
        return tuple(
            sum(weights[row] * self.jacobian[row][column]
                for row in range(len(weights)))
            for column in range(len(self.parameter_names))
        )

    def jvp(self, tangent: Sequence[float]) -> tuple[float, ...]:
        direction = tuple(float(value) for value in tangent)
        if len(direction) != len(self.parameter_names):
            raise ValueError("tangent width must match the parameter count")
        return tuple(
            sum(self.jacobian[row][column] * direction[column]
                for column in range(len(direction)))
            for row in range(len(self.values))
        )


class CausalParameterShift:
    """Exact batched parameter-shift differentiation for QSA rotations.

    Every trainable parameter must occur exactly once and only as the angle of
    one Rx, Ry, or Rz gate. Under that checked contract, the standard
    plus/minus pi/2 shift with coefficient 1/2 is exact.
    """

    shift = math.pi / 2.0
    coefficient = 0.5

    def __init__(
        self,
        plan: CausalParameterizedPlan,
        observables: CausalPauliPlan,
    ) -> None:
        if not isinstance(plan, CausalParameterizedPlan):
            raise TypeError("plan must be a CausalParameterizedPlan")
        if not isinstance(observables, CausalPauliPlan):
            raise TypeError("observables must be a CausalPauliPlan")
        self.plan = plan
        self.observables = observables
        self.parameter_names = tuple(plan.parameter_names)
        self._validate_shift_contract()

    def _validate_shift_contract(self) -> None:
        occurrences = {name: 0 for name in self.parameter_names}
        for operation in self.plan.operations:
            name = str(operation[0]).lower()
            for position, item in enumerate(operation[1:], start=1):
                if not isinstance(item, Parameter):
                    continue
                if item.name not in occurrences:
                    raise ValueError(f"undeclared parameter in causal plan: {item.name}")
                if name not in {"rx", "ry", "rz"} or position != 2:
                    raise ValueError(
                        f"parameter {item.name} is not a single rotation angle"
                    )
                occurrences[item.name] += 1
        invalid = {
            name: count for name, count in occurrences.items() if count != 1
        }
        if invalid:
            raise ValueError(
                "exact parameter shift requires one rotation occurrence per "
                f"parameter: {invalid}"
            )

    def _rows(
        self,
        values: Mapping[str, float] | Sequence[float],
    ) -> tuple[tuple[float, ...], ...]:
        base = self.plan._values(values)
        if any(not math.isfinite(value) for value in base):
            raise ValueError("parameter values must be finite")
        rows: list[tuple[float, ...]] = [base]
        for index in range(len(base)):
            positive = list(base)
            negative = list(base)
            positive[index] += self.shift
            negative[index] -= self.shift
            rows.append(tuple(positive))
            rows.append(tuple(negative))
        return tuple(rows)

    def evaluate_and_jacobian(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        *,
        workers: int = 0,
    ) -> ParameterShiftResult:
        state._ensure_open()
        if int(state.qubit_count) != self.observables.qubits:
            raise ValueError("observable plan qubit count differs from causal state")
        rows = self._rows(values)
        branches = fork_many(state, len(rows))
        try:
            apply_many(self.plan, branches, rows, workers=workers)
            measured = observe_many(self.observables, branches, workers=workers)
        finally:
            for branch in branches:
                branch.close()

        base_values = tuple(measured[0])
        parameter_count = len(self.parameter_names)
        observable_count = len(base_values)
        columns: list[tuple[float, ...]] = []
        for parameter in range(parameter_count):
            positive = measured[1 + 2 * parameter]
            negative = measured[2 + 2 * parameter]
            columns.append(tuple(
                self.coefficient * (positive[row] - negative[row])
                for row in range(observable_count)
            ))
        jacobian = tuple(
            tuple(columns[column][row] for column in range(parameter_count))
            for row in range(observable_count)
        )
        return ParameterShiftResult(
            values=base_values,
            jacobian=jacobian,
            parameter_names=self.parameter_names,
            observable_words=tuple(self.observables.words),
        )

    def jacobian(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        *,
        workers: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        return self.evaluate_and_jacobian(
            state, values, workers=workers
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
            state, values, workers=workers
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
            state, values, workers=workers
        ).jvp(tangent)


__all__ = ["CausalParameterShift", "ParameterShiftResult"]
