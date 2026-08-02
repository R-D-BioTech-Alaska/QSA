from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .causal import CausalParameterizedPlan, CausalRegister
from .causal_batch import apply_many, fork_many
from .causal_grad import CausalParameterShift
from .causal_support import CausalPauliSupportPlan


@dataclass(frozen=True)
class SupportParameterShiftResult:
    values: tuple[float, ...]
    jacobian: tuple[tuple[float, ...], ...]
    parameter_names: tuple[str, ...]
    observable_supports: tuple[tuple[tuple[int, str], ...], ...]

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


class CausalSupportParameterShift(CausalParameterShift):
    """Exact parameter-shift gradients with compact Pauli support."""

    def __init__(
        self,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
    ) -> None:
        if not isinstance(plan, CausalParameterizedPlan):
            raise TypeError("plan must be a CausalParameterizedPlan")
        if not isinstance(observables, CausalPauliSupportPlan):
            raise TypeError("observables must be a CausalPauliSupportPlan")
        self.plan = plan
        self.observables = observables
        self.parameter_names = tuple(plan.parameter_names)
        self._validate_shift_contract()

    def evaluate_and_jacobian(
        self,
        state: CausalRegister,
        values: Mapping[str, float] | Sequence[float],
        *,
        workers: int = 0,
    ) -> SupportParameterShiftResult:
        state._ensure_open()
        if int(state.qubit_count) != self.observables.qubits:
            raise ValueError("observable plan qubit count differs from causal state")

        rows = self._rows(values)
        branches = fork_many(state, len(rows))
        try:
            apply_many(self.plan, branches, rows, workers=workers)
            measured = self.observables.execute_many(
                branches,
                workers=workers,
            )
        finally:
            for branch in branches:
                branch.close()

        base_values = tuple(measured[0])
        parameter_count = len(self.parameter_names)
        observable_count = len(base_values)
        columns = []
        for parameter in range(parameter_count):
            positive = measured[1 + 2 * parameter]
            negative = measured[2 + 2 * parameter]
            columns.append(
                tuple(
                    self.coefficient * (positive[row] - negative[row])
                    for row in range(observable_count)
                )
            )
        jacobian = tuple(
            tuple(
                columns[column][row]
                for column in range(parameter_count)
            )
            for row in range(observable_count)
        )
        return SupportParameterShiftResult(
            values=base_values,
            jacobian=jacobian,
            parameter_names=self.parameter_names,
            observable_supports=tuple(self.observables.observables),
        )


__all__ = ["CausalSupportParameterShift", "SupportParameterShiftResult"]
