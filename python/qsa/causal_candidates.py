from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, Tuple

from .causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from .causal_batch import apply_many, fork_many
from .causal_support import CausalPauliSupportPlan


class CausalCandidateBatch:
    """One uncommitted set of candidate branches and observable values."""

    def __init__(
        self,
        runtime: "CausalCandidateRuntime",
        branches,
        observations: Tuple[Tuple[float, ...], ...],
        generation: int,
    ) -> None:
        self._runtime = runtime
        self._branches = list(branches)
        self._generation = int(generation)
        self.observations = observations
        self._closed = False

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise CausalRuntimeError("causal candidate batch is closed")

    def _ensure_owner(self, runtime: "CausalCandidateRuntime") -> None:
        self._ensure_open()
        if self._runtime is not runtime:
            raise ValueError("candidate batch belongs to a different runtime")
        if self._generation != runtime.generation:
            raise CausalRuntimeError("candidate batch belongs to an older state generation")

    def _take(self, index: int):
        self._ensure_open()
        selected = self._branches[index]
        self._branches[index] = None
        return selected

    def close(self) -> None:
        if self._closed:
            return
        for branch in self._branches:
            if branch is not None:
                branch.close()
        self._branches = []
        self._closed = True
        self._runtime._release_batch(self)

    def __enter__(self) -> "CausalCandidateBatch":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CausalCandidateRuntime:
    """Persistent causal state with explicit candidate commit authority."""

    def __init__(
        self,
        state: CausalRegister,
        plan: CausalParameterizedPlan,
        observables: CausalPauliSupportPlan,
        *,
        workers: int = 0,
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

        self._state = state
        self.plan = plan
        self.observables = observables
        self.workers = max(0, int(workers))
        self._generation = 0
        self._active_batch: Optional[CausalCandidateBatch] = None
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
    ) -> "CausalCandidateRuntime":
        return cls(
            CausalRegister(int(qubits), library_path=library_path),
            plan,
            observables,
            workers=workers,
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
    ) -> "CausalCandidateRuntime":
        return cls(
            CausalRegister.decode_qsc(packet, library_path=library_path),
            plan,
            observables,
            workers=workers,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise CausalRuntimeError("causal candidate runtime is closed")
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

    def evaluate(
        self,
        parameter_rows: Sequence[Mapping[str, float]],
    ) -> CausalCandidateBatch:
        self._ensure_open()
        if self.has_active_batch:
            raise CausalRuntimeError("commit or discard the active candidate batch first")
        rows = tuple(parameter_rows)
        if not rows:
            raise ValueError("at least one candidate parameter row is required")

        branches = fork_many(self._state, len(rows))
        try:
            apply_many(self.plan, branches, rows, workers=self.workers)
            values = self.observables.execute_many(
                branches,
                workers=self.workers,
            )
            batch = CausalCandidateBatch(
                self,
                branches,
                values,
                self._generation,
            )
            self._active_batch = batch
            return batch
        except Exception:
            for branch in branches:
                branch.close()
            raise

    def commit(self, batch: CausalCandidateBatch, selected_index: int) -> None:
        self._ensure_open()
        if not isinstance(batch, CausalCandidateBatch):
            raise TypeError("batch must be a CausalCandidateBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("candidate batch is not active")
        index = int(selected_index)
        if index < 0 or index >= len(batch):
            raise IndexError("selected candidate index is out of range")

        selected = batch._take(index)
        try:
            self._state.adopt(selected)
        finally:
            batch.close()
        self._generation += 1

    def discard(self, batch: CausalCandidateBatch) -> None:
        self._ensure_open()
        if not isinstance(batch, CausalCandidateBatch):
            raise TypeError("batch must be a CausalCandidateBatch")
        batch._ensure_owner(self)
        if self._active_batch is not batch:
            raise CausalRuntimeError("candidate batch is not active")
        batch.close()

    def checkpoint(self) -> bytes:
        self._ensure_open()
        if self.has_active_batch:
            raise CausalRuntimeError("commit or discard the active candidate batch first")
        return self._state.encode_qsc()

    def _release_batch(self, batch: CausalCandidateBatch) -> None:
        if self._active_batch is batch:
            self._active_batch = None

    def close(self) -> None:
        if self._closed:
            return
        if self._active_batch is not None:
            self._active_batch.close()
        self._state.close()
        self._closed = True

    def __enter__(self) -> "CausalCandidateRuntime":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["CausalCandidateBatch", "CausalCandidateRuntime"]
