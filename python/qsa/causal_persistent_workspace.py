from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from .causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from .causal_adjoint import CausalWeightedAdjoint, WeightedAdjointResult
from .causal_batch import apply_many, fork_many
from .causal_component_grad import ComponentParameterShiftResult
from .causal_components import (
    ExtractedCausalComponents,
    extract_component_closure,
    parameterized_plan_qubits,
    remap_parameterized_plan,
    remap_support_plan,
    support_plan_qubits,
)
from .causal_support import CausalPauliSupportPlan
from .causal_support_grad import CausalSupportParameterShift


class PersistentComponentBatch:
    def __init__(
        self,
        runtime: "CausalPersistentComponentRuntime",
        rows,
        branches,
        observations,
        global_qubits,
        generation: int,
    ) -> None:
        self._runtime = runtime
        self._rows = tuple(rows)
        self._branches = list(branches)
        self._generation = int(generation)
        self.observations = tuple(observations)
        self.global_qubits = tuple(global_qubits)
        self.local_qubit_count = len(self.global_qubits)
        self._closed = False

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_owner(self, runtime: "CausalPersistentComponentRuntime") -> None:
        if self._closed:
            raise CausalRuntimeError("persistent component batch is closed")
        if self._runtime is not runtime:
            raise ValueError("persistent component batch belongs to another runtime")
        if self._generation != runtime.generation:
            raise CausalRuntimeError(
                "persistent component batch belongs to an older state generation"
            )

    def parameter_row(self, index: int):
        if self._closed:
            raise CausalRuntimeError("persistent component batch is closed")
        return self._rows[index]

    def _take_branch(self, index: int) -> CausalRegister:
        if self._closed:
            raise CausalRuntimeError("persistent component batch is closed")
        branch = self._branches[index]
        if branch is None:
            raise CausalRuntimeError("persistent component branch was already consumed")
        self._branches[index] = None
        return branch

    def close(self) -> None:
        if self._closed:
            return
        for branch in self._branches:
            if branch is not None:
                branch.close()
        self._branches = []
        self._closed = True
        self._runtime._release_batch(self)

    def __enter__(self) -> "PersistentComponentBatch":
        if self._closed:
            raise CausalRuntimeError("persistent component batch is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CausalPersistentComponentRuntime:
    """Persistent global state with one retained exact component workspace."""

    def __init__(
        self,
        state: CausalRegister,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
        *,
        workers: int = 0,
        max_local_qubits: int = 24,
    ) -> None:
        if not isinstance(state, CausalRegister):
            raise TypeError("state must be a CausalRegister")
        if not isinstance(plan, CausalParameterizedPlan):
            raise TypeError("plan must be a CausalParameterizedPlan")
        if not isinstance(observables, CausalPauliSupportPlan):
            raise TypeError("observables must be a CausalPauliSupportPlan")
        state._ensure_open()
        if int(state.qubit_count) != observables.qubits:
            raise ValueError("observable qubit count differs from causal state")
        limit = int(max_local_qubits)
        if limit <= 0 or limit > 24:
            raise ValueError("max_local_qubits must be between 1 and 24")

        self._state = state
        self.plan = plan
        self.observables = observables
        self.workers = max(0, int(workers))
        self.max_local_qubits = limit
        self._requested_qubits = tuple(
            sorted(
                set(parameterized_plan_qubits(plan))
                | set(support_plan_qubits(observables))
            )
        )
        if not self._requested_qubits:
            raise ValueError("persistent component runtime has no requested qubits")

        self._generation = 0
        self._active_batch: Optional[PersistentComponentBatch] = None
        self._workspace: Optional[ExtractedCausalComponents] = None
        self._workspace_extractions = 0
        self._local_plans = {}
        self._local_gradients = {}
        self._local_adjoints = {}
        self._closed = False

    @classmethod
    def create(
        cls,
        qubits: int,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
        *,
        library_path: Optional[str] = None,
        workers: int = 0,
        max_local_qubits: int = 24,
    ) -> "CausalPersistentComponentRuntime":
        return cls(
            CausalRegister(int(qubits), library_path=library_path),
            plan,
            observables,
            workers=workers,
            max_local_qubits=max_local_qubits,
        )

    @classmethod
    def decode_qsc(
        cls,
        packet: bytes,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
        *,
        library_path: Optional[str] = None,
        workers: int = 0,
        max_local_qubits: int = 24,
    ) -> "CausalPersistentComponentRuntime":
        return cls(
            CausalRegister.decode_qsc(packet, library_path=library_path),
            plan,
            observables,
            workers=workers,
            max_local_qubits=max_local_qubits,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise CausalRuntimeError("persistent component runtime is closed")
        self._state._ensure_open()

    @property
    def state(self) -> CausalRegister:
        self._ensure_open()
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def has_active_batch(self) -> bool:
        return self._active_batch is not None and not self._active_batch.closed

    @property
    def workspace_ready(self) -> bool:
        return self._workspace is not None

    @property
    def workspace_extractions(self) -> int:
        return self._workspace_extractions

    @property
    def workspace_qubit_count(self) -> int:
        return 0 if self._workspace is None else self._workspace.local_qubit_count

    def _ensure_workspace(self) -> ExtractedCausalComponents:
        if self._workspace is None:
            self._workspace = extract_component_closure(
                self._state,
                self._requested_qubits,
                max_local_qubits=self.max_local_qubits,
            )
            self._workspace_extractions += 1
        return self._workspace

    def refresh_workspace(self) -> None:
        self._ensure_open()
        if self.has_active_batch:
            raise CausalRuntimeError("commit or discard the active batch before refresh")
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None

    def _local_runtime(self, global_qubits):
        key = tuple(global_qubits)
        cached = self._local_plans.get(key)
        if cached is not None:
            return cached

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
        self._local_plans[key] = (local_plan, local_observables)
        return local_plan, local_observables

    def _local_gradient(self, global_qubits):
        key = tuple(global_qubits)
        cached = self._local_gradients.get(key)
        if cached is not None:
            return cached
        local_plan, local_observables = self._local_runtime(key)
        gradient = CausalSupportParameterShift(
            local_plan,
            local_observables,
        )
        self._local_gradients[key] = gradient
        return gradient

    def _local_adjoint(self, global_qubits):
        key = tuple(global_qubits)
        cached = self._local_adjoints.get(key)
        if cached is not None:
            return cached
        local_plan, local_observables = self._local_runtime(key)
        adjoint = CausalWeightedAdjoint(
            local_plan,
            local_observables,
            max_qubits=self.max_local_qubits,
        )
        self._local_adjoints[key] = adjoint
        return adjoint

    def evaluate(
        self,
        parameter_rows: Sequence[Mapping[str, float] | Sequence[float]],
    ) -> PersistentComponentBatch:
        self._ensure_open()
        if self.has_active_batch:
            raise CausalRuntimeError("commit or discard the active batch first")
        rows = tuple(parameter_rows)
        if not rows:
            raise ValueError("at least one candidate parameter row is required")

        workspace = self._ensure_workspace()
        branches = fork_many(workspace.state, len(rows))
        try:
            local_plan, local_observables = self._local_runtime(
                workspace.global_qubits
            )
            apply_many(local_plan, branches, rows, workers=self.workers)
            observations = local_observables.execute_many(
                branches,
                workers=self.workers,
            )
            batch = PersistentComponentBatch(
                self,
                rows,
                branches,
                observations,
                workspace.global_qubits,
                self._generation,
            )
            self._active_batch = batch
            return batch
        except Exception:
            for branch in branches:
                branch.close()
            raise

    def selected_gradient(
        self,
        batch: PersistentComponentBatch,
        selected_index: int,
        *,
        workers: Optional[int] = None,
    ) -> ComponentParameterShiftResult:
        self._ensure_open()
        if not isinstance(batch, PersistentComponentBatch):
            raise TypeError("batch must be a PersistentComponentBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("persistent component batch is not active")
        index = int(selected_index)
        if index < 0 or index >= len(batch):
            raise IndexError("selected candidate index is out of range")

        workspace = self._ensure_workspace()
        gradient = self._local_gradient(batch.global_qubits)
        rows = gradient._rows(batch.parameter_row(index))
        shifted_rows = rows[1:]
        shifted_branches = fork_many(workspace.state, len(shifted_rows))
        selected_workers = self.workers if workers is None else max(0, int(workers))
        try:
            apply_many(
                gradient.plan,
                shifted_branches,
                shifted_rows,
                workers=selected_workers,
            )
            shifted_values = gradient.observables.execute_many(
                shifted_branches,
                workers=selected_workers,
            )
        finally:
            for branch in shifted_branches:
                branch.close()

        base_values = tuple(batch.observations[index])
        parameter_count = len(self.plan.parameter_names)
        observable_count = len(base_values)
        columns = []
        for parameter in range(parameter_count):
            positive = shifted_values[2 * parameter]
            negative = shifted_values[2 * parameter + 1]
            columns.append(
                tuple(
                    gradient.coefficient * (positive[row] - negative[row])
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
        return ComponentParameterShiftResult(
            values=base_values,
            jacobian=jacobian,
            parameter_names=tuple(self.plan.parameter_names),
            observable_supports=tuple(self.observables.observables),
            global_qubits=batch.global_qubits,
        )

    def selected_pullback(
        self,
        batch: PersistentComponentBatch,
        selected_index: int,
        cotangent: Sequence[float],
    ) -> WeightedAdjointResult:
        self._ensure_open()
        if not isinstance(batch, PersistentComponentBatch):
            raise TypeError("batch must be a PersistentComponentBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("persistent component batch is not active")
        index = int(selected_index)
        if index < 0 or index >= len(batch):
            raise IndexError("selected candidate index is out of range")

        workspace = self._ensure_workspace()
        adjoint = self._local_adjoint(batch.global_qubits)
        result = adjoint.evaluate(
            workspace.state,
            batch.parameter_row(index),
            cotangent,
        )
        base_values = tuple(float(value) for value in batch.observations[index])
        maximum_error = max(
            abs(left - right)
            for left, right in zip(base_values, result.values)
        )
        if maximum_error > 2.0e-12:
            raise CausalRuntimeError(
                "selected adjoint values differ from the evaluated candidate"
            )
        return WeightedAdjointResult(
            values=base_values,
            gradient=result.gradient,
            parameter_names=tuple(self.plan.parameter_names),
            observable_supports=tuple(self.observables.observables),
            global_qubits=batch.global_qubits,
            forward_sweeps=result.forward_sweeps,
            reverse_sweeps=result.reverse_sweeps,
            shifted_evaluations=result.shifted_evaluations,
            stored_forward_states=result.stored_forward_states,
        )

    def selected_vjp(
        self,
        batch: PersistentComponentBatch,
        selected_index: int,
        cotangent: Sequence[float],
        *,
        workers: Optional[int] = None,
    ) -> tuple[float, ...]:
        del workers
        return self.selected_pullback(
            batch,
            selected_index,
            cotangent,
        ).gradient

    def commit(
        self,
        batch: PersistentComponentBatch,
        selected_index: int,
    ) -> None:
        self._ensure_open()
        if not isinstance(batch, PersistentComponentBatch):
            raise TypeError("batch must be a PersistentComponentBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("persistent component batch is not active")
        index = int(selected_index)
        if index < 0 or index >= len(batch):
            raise IndexError("selected candidate index is out of range")

        selected_row = batch.parameter_row(index)
        self._state.apply(self.plan, selected_row)
        selected_local = batch._take_branch(index)
        previous_workspace = self._workspace
        self._workspace = ExtractedCausalComponents(
            selected_local,
            batch.global_qubits,
        )
        batch.close()
        if previous_workspace is not None:
            previous_workspace.close()
        self._generation += 1

    def discard(self, batch: PersistentComponentBatch) -> None:
        self._ensure_open()
        if not isinstance(batch, PersistentComponentBatch):
            raise TypeError("batch must be a PersistentComponentBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("persistent component batch is not active")
        batch.close()

    def checkpoint(self) -> bytes:
        self._ensure_open()
        if self.has_active_batch:
            raise CausalRuntimeError("commit or discard the active batch first")
        return self._state.encode_qsc()

    def _release_batch(self, batch: PersistentComponentBatch) -> None:
        if self._active_batch is batch:
            self._active_batch = None

    def close(self) -> None:
        if self._closed:
            return
        if self._active_batch is not None:
            self._active_batch.close()
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None
        self._local_gradients.clear()
        self._local_adjoints.clear()
        for local_plan, local_observables in self._local_plans.values():
            local_observables.close()
            local_plan.close()
        self._local_plans.clear()
        self._state.close()
        self._closed = True

    def __enter__(self) -> "CausalPersistentComponentRuntime":
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
    "CausalPersistentComponentRuntime",
    "PersistentComponentBatch",
]
