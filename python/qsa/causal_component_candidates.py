from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from .causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from .causal_batch import apply_many, fork_many
from .causal_components import (
    extract_component_closure,
    parameterized_plan_qubits,
    remap_parameterized_plan,
    remap_support_plan,
    support_plan_qubits,
)
from .causal_support import CausalPauliSupportPlan


class CausalComponentCandidateBatch:
    def __init__(
        self,
        runtime: "CausalComponentCandidateRuntime",
        rows,
        branches,
        extracted,
        observations,
        generation: int,
    ) -> None:
        self._runtime = runtime
        self._rows = tuple(rows)
        self._branches = list(branches)
        self._extracted = extracted
        self._generation = int(generation)
        self.observations = tuple(observations)
        self.local_qubit_count = extracted.local_qubit_count
        self.global_qubits = extracted.global_qubits
        self._closed = False

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_owner(self, runtime: "CausalComponentCandidateRuntime") -> None:
        if self._closed:
            raise CausalRuntimeError("component candidate batch is closed")
        if self._runtime is not runtime:
            raise ValueError("component candidate batch belongs to another runtime")
        if self._generation != runtime.generation:
            raise CausalRuntimeError(
                "component candidate batch belongs to an older state generation"
            )

    def parameter_row(self, index: int):
        if self._closed:
            raise CausalRuntimeError("component candidate batch is closed")
        return self._rows[index]

    def close(self) -> None:
        if self._closed:
            return
        for branch in self._branches:
            branch.close()
        self._branches = []
        self._extracted.close()
        self._closed = True
        self._runtime._release_batch(self)

    def __enter__(self) -> "CausalComponentCandidateBatch":
        if self._closed:
            raise CausalRuntimeError("component candidate batch is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CausalComponentCandidateRuntime:
    """Candidate transactions over only the complete components they touch."""

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
            raise ValueError("component candidate runtime has no requested qubits")

        self._generation = 0
        self._active_batch: Optional[CausalComponentCandidateBatch] = None
        self._local_plans = {}
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
    ) -> "CausalComponentCandidateRuntime":
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
    ) -> "CausalComponentCandidateRuntime":
        return cls(
            CausalRegister.decode_qsc(packet, library_path=library_path),
            plan,
            observables,
            workers=workers,
            max_local_qubits=max_local_qubits,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise CausalRuntimeError("component candidate runtime is closed")
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

    def _local_runtime(self, global_qubits):
        key = tuple(global_qubits)
        cached = self._local_plans.get(key)
        if cached is not None:
            return cached
        local_by_global = {
            global_qubit: local_qubit
            for local_qubit, global_qubit in enumerate(key)
        }
        try:
            local_plan = remap_parameterized_plan(self.plan, local_by_global)
            local_observables = remap_support_plan(
                self.observables,
                local_by_global,
            )
        except Exception:
            if "local_plan" in locals():
                local_plan.close()
            raise
        self._local_plans[key] = (local_plan, local_observables)
        return local_plan, local_observables

    def evaluate(
        self,
        parameter_rows: Sequence[Mapping[str, float] | Sequence[float]],
    ) -> CausalComponentCandidateBatch:
        self._ensure_open()
        if self.has_active_batch:
            raise CausalRuntimeError("commit or discard the active candidate batch first")
        rows = tuple(parameter_rows)
        if not rows:
            raise ValueError("at least one candidate parameter row is required")

        extracted = extract_component_closure(
            self._state,
            self._requested_qubits,
            max_local_qubits=self.max_local_qubits,
        )
        branches = []
        try:
            local_plan, local_observables = self._local_runtime(
                extracted.global_qubits
            )
            branches = fork_many(extracted.state, len(rows))
            apply_many(local_plan, branches, rows, workers=self.workers)
            observations = local_observables.execute_many(
                branches,
                workers=self.workers,
            )
            batch = CausalComponentCandidateBatch(
                self,
                rows,
                branches,
                extracted,
                observations,
                self._generation,
            )
            self._active_batch = batch
            return batch
        except Exception:
            for branch in branches:
                branch.close()
            extracted.close()
            raise

    def commit(
        self,
        batch: CausalComponentCandidateBatch,
        selected_index: int,
    ) -> None:
        self._ensure_open()
        if not isinstance(batch, CausalComponentCandidateBatch):
            raise TypeError("batch must be a CausalComponentCandidateBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("component candidate batch is not active")
        index = int(selected_index)
        if index < 0 or index >= len(batch):
            raise IndexError("selected candidate index is out of range")

        selected_row = batch.parameter_row(index)
        self._state.apply(self.plan, selected_row)
        batch.close()
        self._generation += 1

    def discard(self, batch: CausalComponentCandidateBatch) -> None:
        self._ensure_open()
        if not isinstance(batch, CausalComponentCandidateBatch):
            raise TypeError("batch must be a CausalComponentCandidateBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("component candidate batch is not active")
        batch.close()

    def checkpoint(self) -> bytes:
        self._ensure_open()
        if self.has_active_batch:
            raise CausalRuntimeError("commit or discard the active candidate batch first")
        return self._state.encode_qsc()

    def _release_batch(self, batch: CausalComponentCandidateBatch) -> None:
        if self._active_batch is batch:
            self._active_batch = None

    def close(self) -> None:
        if self._closed:
            return
        if self._active_batch is not None:
            self._active_batch.close()
        for local_plan, local_observables in self._local_plans.values():
            local_observables.close()
            local_plan.close()
        self._local_plans.clear()
        self._state.close()
        self._closed = True

    def __enter__(self) -> "CausalComponentCandidateRuntime":
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
    "CausalComponentCandidateBatch",
    "CausalComponentCandidateRuntime",
]
