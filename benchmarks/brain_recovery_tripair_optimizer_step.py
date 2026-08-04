from __future__ import annotations

import argparse
import copy
import io
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan
from qsa.causal_persistent_workspace import CausalPersistentComponentRuntime
from qsa.causal_support import CausalPauliSupportPlan

QSA_SOURCE = "78cc67d6c52469486976bb75e9471b87da1fc7fd"
BRAIN_PARENT = "f9a183e7bdbffc3d838e8774da548a053f883152"
BRAIN_QUANTUM_SOURCE = (
    "external/nueralnet/dc25ec28cf577bd3fd950afc381edd381d3a8137/"
    "brain_neural_net/quantum.py"
)
BRAIN_QUANTUM_BLOB = "caed1ccb866314c9011523f4799dd95a30ebaa3b"
FEATURE_WIDTH = 256
QUBITS = 3
DEPTH = 2
OBSERVABLE_NAMES = (
    "X0",
    "Y0",
    "Z0",
    "X1",
    "Y1",
    "Z1",
    "X2",
    "Y2",
    "Z2",
    "X0 X1",
    "Z0 Z1",
    "X1 X2",
    "Z1 Z2",
    "X0 X1 X2",
)
OBSERVABLES = (
    ((0, "X"),),
    ((0, "Y"),),
    ((0, "Z"),),
    ((1, "X"),),
    ((1, "Y"),),
    ((1, "Z"),),
    ((2, "X"),),
    ((2, "Y"),),
    ((2, "Z"),),
    ((0, "X"), (1, "X")),
    ((0, "Z"), (1, "Z")),
    ((1, "X"), (2, "X")),
    ((1, "Z"), (2, "Z")),
    ((0, "X"), (1, "X"), (2, "X")),
)
BRAIN_INPUT_NAMES = (
    "theta_0",
    "theta_1",
    "theta_2",
    "phi_0",
    "phi_1",
    "phi_2",
)
QSA_INPUT_NAMES = (
    "theta_0",
    "phi_0",
    "theta_1",
    "phi_1",
    "theta_2",
    "phi_2",
)
BRAIN_TO_QSA = (0, 3, 1, 4, 2, 5)
QSA_TO_BRAIN = (0, 2, 4, 1, 3, 5)
EXPECTED_PARAMETER_COUNT = 5_395


@lru_cache(maxsize=64)
def basis_indices(qubit: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    zero = tuple(index for index in range(1 << QUBITS) if ((index >> qubit) & 1) == 0)
    return zero, tuple(index | (1 << qubit) for index in zero)


@lru_cache(maxsize=64)
def cnot_permutation(control: int, target: int) -> tuple[int, ...]:
    return tuple(
        index ^ (1 << target) if ((index >> control) & 1) else index
        for index in range(1 << QUBITS)
    )


def apply_ry(state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
    zero_raw, one_raw = basis_indices(qubit)
    zero = torch.tensor(zero_raw, device=state.device)
    one = torch.tensor(one_raw, device=state.device)
    a0 = state.index_select(1, zero)
    a1 = state.index_select(1, one)
    half = angle.reshape(-1, 1) * 0.5
    cosine = torch.cos(half).to(state.dtype)
    sine = torch.sin(half).to(state.dtype)
    result = state.clone()
    result[:, zero] = cosine * a0 - sine * a1
    result[:, one] = sine * a0 + cosine * a1
    return result


def apply_rz(state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
    zero_raw, one_raw = basis_indices(qubit)
    zero = torch.tensor(zero_raw, device=state.device)
    one = torch.tensor(one_raw, device=state.device)
    half = angle.reshape(-1, 1) * 0.5
    cosine = torch.cos(half)
    sine = torch.sin(half)
    negative = torch.complex(cosine, -sine).to(state.dtype)
    positive = torch.complex(cosine, sine).to(state.dtype)
    result = state.clone()
    result[:, zero] = state.index_select(1, zero) * negative
    result[:, one] = state.index_select(1, one) * positive
    return result


def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    permutation = torch.tensor(cnot_permutation(control, target), device=state.device)
    return state.index_select(1, permutation)


def apply_pauli(
    state: torch.Tensor,
    paulis: Iterable[tuple[str, int]],
) -> torch.Tensor:
    result = state
    for name, qubit in paulis:
        bit = 1 << qubit
        indices = torch.arange(1 << QUBITS, device=state.device)
        if name == "Z":
            phase = torch.where((indices & bit) == 0, 1.0, -1.0).to(result.dtype)
            result = result * phase.unsqueeze(0)
        elif name == "X":
            result = result.index_select(1, indices ^ bit)
        elif name == "Y":
            source = indices ^ bit
            source_bit = (source & bit) != 0
            phase = torch.where(
                source_bit,
                torch.full_like(indices, -1j, dtype=result.dtype),
                torch.full_like(indices, 1j, dtype=result.dtype),
            )
            result = result.index_select(1, source) * phase.unsqueeze(0)
        else:
            raise ValueError(name)
    return result


def torch_observables(brain_angles: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    batch = brain_angles.shape[0]
    state = torch.zeros(batch, 1 << QUBITS, dtype=torch.complex128)
    state[:, 0] = 1.0 + 0.0j
    theta = brain_angles[:, :QUBITS]
    phi = brain_angles[:, QUBITS:]
    for qubit in range(QUBITS):
        state = apply_ry(state, theta[:, qubit], qubit)
        state = apply_rz(state, phi[:, qubit], qubit)
    state = apply_cnot(state, 0, 1)
    state = apply_cnot(state, 1, 2)
    state = apply_cnot(state, 2, 0)
    angles = shared.reshape(DEPTH, QUBITS, 2)
    for layer in range(DEPTH):
        for qubit in range(QUBITS):
            state = apply_ry(state, angles[layer, qubit, 0].expand(batch), qubit)
            state = apply_rz(state, angles[layer, qubit, 1].expand(batch), qubit)
        state = apply_cnot(state, 0, 1)
        state = apply_cnot(state, 1, 2)
    values = []
    for observable in OBSERVABLES:
        operated = apply_pauli(state, observable)
        values.append(torch.sum(torch.conj(state) * operated, dim=-1).real)
    return torch.stack(values, dim=-1)


def qsa_plan() -> CausalParameterizedPlan:
    parameters = {name: Parameter(name) for name in QSA_INPUT_NAMES}
    operations = []
    for qubit in range(QUBITS):
        operations.append(("ry", qubit, parameters[f"theta_{qubit}"]))
        operations.append(("rz", qubit, parameters[f"phi_{qubit}"]))
    operations.extend((("cnot", 0, 1), ("cnot", 1, 2), ("cnot", 2, 0)))
    for layer in range(DEPTH):
        for qubit in range(QUBITS):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_{qubit}_ry"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_{qubit}_rz"))
            )
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    return CausalParameterizedPlan(tuple(operations))


class QSATripairBridge:
    def __init__(self) -> None:
        self.plan = qsa_plan()
        self.support = CausalPauliSupportPlan(QUBITS, OBSERVABLES)
        self.runtime = CausalPersistentComponentRuntime.create(
            QUBITS,
            self.plan,
            self.support,
            workers=1,
            max_local_qubits=QUBITS,
        )
        self.forward_seconds = 0.0
        self.backward_seconds = 0.0
        self.forward_calls = 0
        self.backward_calls = 0
        self.forward_sweeps = 0
        self.reverse_sweeps = 0
        self.shifted_evaluations = 0
        self._root_before = bytes(self.runtime.state.encode_qsc())
        self._workspace_before = bytes(self.runtime._workspace.state.encode_qsc())

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.plan.parameter_names

    def parameter_rows(
        self,
        brain_angles: torch.Tensor,
        shared: torch.Tensor,
    ) -> tuple[dict[str, float], ...]:
        brain_cpu = brain_angles.detach().cpu()
        shared_cpu = shared.detach().reshape(-1).cpu()
        rows = []
        for sample in range(brain_cpu.shape[0]):
            row = {
                "theta_0": float(brain_cpu[sample, 0]),
                "phi_0": float(brain_cpu[sample, 3]),
                "theta_1": float(brain_cpu[sample, 1]),
                "phi_1": float(brain_cpu[sample, 4]),
                "theta_2": float(brain_cpu[sample, 2]),
                "phi_2": float(brain_cpu[sample, 5]),
            }
            offset = 0
            for layer in range(DEPTH):
                for qubit in range(QUBITS):
                    row[f"layer_{layer}_{qubit}_ry"] = float(shared_cpu[offset])
                    offset += 1
                    row[f"layer_{layer}_{qubit}_rz"] = float(shared_cpu[offset])
                    offset += 1
            rows.append(row)
        return tuple(rows)

    def forward(
        self,
        brain_angles: torch.Tensor,
        shared: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        started = time.perf_counter()
        batch = self.runtime.evaluate(self.parameter_rows(brain_angles, shared))
        observations = torch.tensor(
            batch.observations,
            dtype=brain_angles.dtype,
            device=brain_angles.device,
        )
        self.forward_seconds += time.perf_counter() - started
        self.forward_calls += 1
        return observations, batch

    def backward(
        self,
        batch: object,
        cotangents: torch.Tensor,
        input_dtype: torch.dtype,
        input_device: torch.device,
        shared_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        cotangent_rows = cotangents.detach().cpu().tolist()
        input_gradients = torch.zeros(
            len(cotangent_rows),
            len(BRAIN_INPUT_NAMES),
            dtype=input_dtype,
            device=input_device,
        )
        shared_gradient = torch.zeros(
            12,
            dtype=input_dtype,
            device=input_device,
        )
        name_to_position = {
            name: position for position, name in enumerate(self.parameter_names)
        }
        for sample, cotangent in enumerate(cotangent_rows):
            result = self.runtime.selected_pullback(batch, sample, cotangent)
            gradient = result.gradient
            for brain_position, name in enumerate(BRAIN_INPUT_NAMES):
                input_gradients[sample, brain_position] = gradient[
                    name_to_position[name]
                ]
            offset = 0
            for layer in range(DEPTH):
                for qubit in range(QUBITS):
                    for axis in ("ry", "rz"):
                        name = f"layer_{layer}_{qubit}_{axis}"
                        shared_gradient[offset] += gradient[name_to_position[name]]
                        offset += 1
            self.forward_sweeps += int(result.forward_sweeps)
            self.reverse_sweeps += int(result.reverse_sweeps)
            self.shifted_evaluations += int(result.shifted_evaluations)
        self.runtime.discard(batch)
        self.backward_seconds += time.perf_counter() - started
        self.backward_calls += 1
        return input_gradients, shared_gradient.reshape(shared_shape)

    def state_receipt(self) -> dict[str, object]:
        return {
            "root_immutable": bytes(self.runtime.state.encode_qsc()) == self._root_before,
            "workspace_immutable": bytes(self.runtime._workspace.state.encode_qsc())
            == self._workspace_before,
            "workspace_extractions": int(self.runtime.workspace_extractions),
            "generation": int(self.runtime.generation),
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "forward_sweeps": self.forward_sweeps,
            "reverse_sweeps": self.reverse_sweeps,
            "shifted_evaluations": self.shifted_evaluations,
            "forward_seconds": self.forward_seconds,
            "backward_seconds": self.backward_seconds,
        }

    def close(self) -> None:
        self.runtime.close()
        self.support.close()
        self.plan.close()


class QSAObservables(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        brain_angles: torch.Tensor,
        shared: torch.Tensor,
        bridge: QSATripairBridge,
    ) -> torch.Tensor:
        observations, batch = bridge.forward(brain_angles, shared)
        ctx.bridge = bridge
        ctx.batch = batch
        ctx.input_dtype = brain_angles.dtype
        ctx.input_device = brain_angles.device
        ctx.shared_shape = shared.shape
        return observations

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        input_gradient, shared_gradient = ctx.bridge.backward(
            ctx.batch,
            grad_output,
            ctx.input_dtype,
            ctx.input_device,
            ctx.shared_shape,
        )
        return input_gradient, shared_gradient, None


class TripairCircuitModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_angles = nn.Linear(FEATURE_WIDTH, QUBITS * 2, dtype=torch.float64)
        self.trainable_angles = nn.Parameter(
            torch.empty(DEPTH, QUBITS, 2, dtype=torch.float64)
        )
        nn.init.normal_(self.trainable_angles, mean=0.0, std=0.08)
        self.observable_projection = nn.Linear(
            len(OBSERVABLES), FEATURE_WIDTH, dtype=torch.float64
        )
        self.residual_gate = nn.Parameter(torch.tensor([-2.0], dtype=torch.float64))

    def forward(
        self,
        features: torch.Tensor,
        backend: str,
        bridge: QSATripairBridge | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = F.linear(features, self.input_angles.weight, self.input_angles.bias)
        theta, phi = encoded.chunk(2, dim=-1)
        brain_angles = torch.cat(
            (torch.pi * torch.tanh(theta), torch.pi * torch.tanh(phi)), dim=-1
        )
        if backend == "torch":
            observables = torch_observables(brain_angles, self.trainable_angles)
        elif backend == "qsa":
            if bridge is None:
                raise ValueError("QSA backend requires a bridge")
            observables = QSAObservables.apply(
                brain_angles, self.trainable_angles, bridge
            )
        else:
            raise ValueError(backend)
        residual = torch.sigmoid(self.residual_gate) * F.linear(
            observables,
            self.observable_projection.weight,
            self.observable_projection.bias,
        )
        return residual, observables


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def maximum_state_error(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> float:
    return max(
        float(torch.max(torch.abs(first[name] - second[name])))
        for name in first
    )


def maximum_gradient_error(first: nn.Module, second: nn.Module) -> float:
    errors = []
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    for name in first_parameters:
        left = first_parameters[name].grad
        right = second_parameters[name].grad
        if left is None or right is None:
            raise RuntimeError(f"missing gradient for {name}")
        errors.append(float(torch.max(torch.abs(left - right))))
    return max(errors)


def all_finite_gradients(model: nn.Module) -> bool:
    return all(
        parameter.grad is not None
        and bool(torch.all(torch.isfinite(parameter.grad)))
        for parameter in model.parameters()
    )


def optimizer_state_equal(first: dict, second: dict) -> bool:
    if first.keys() != second.keys():
        return False
    if first["param_groups"] != second["param_groups"]:
        return False
    if first["state"].keys() != second["state"].keys():
        return False
    for key in first["state"]:
        left = first["state"][key]
        right = second["state"][key]
        if left.keys() != right.keys():
            return False
        for name in left:
            if isinstance(left[name], torch.Tensor):
                if not torch.equal(left[name], right[name]):
                    return False
            elif left[name] != right[name]:
                return False
    return True


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


@dataclass
class StepTiming:
    forward: float
    backward: float
    clip: float
    optimizer: float
    total: float


def run_step(
    model: TripairCircuitModule,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    target: torch.Tensor,
    backend: str,
    bridge: QSATripairBridge | None,
    clip_norm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, StepTiming, float]:
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    residual, observables = model(features, backend, bridge)
    loss = F.mse_loss(residual, target)
    forward_done = time.perf_counter()
    loss.backward()
    backward_done = time.perf_counter()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm))
    clip_done = time.perf_counter()
    optimizer.step()
    optimizer_done = time.perf_counter()
    return (
        residual.detach(),
        observables.detach(),
        loss.detach(),
        StepTiming(
            forward=forward_done - started,
            backward=backward_done - forward_done,
            clip=clip_done - backward_done,
            optimizer=optimizer_done - clip_done,
            total=optimizer_done - started,
        ),
        gradient_norm,
    )


def training_batches(batch_size: int, steps: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator().manual_seed(20260804 + batch_size)
    batches = []
    for step in range(steps):
        features = torch.randn(
            batch_size,
            FEATURE_WIDTH,
            generator=generator,
            dtype=torch.float64,
        )
        target = 0.18 * torch.tanh(
            torch.roll(features, shifts=step % 11 + 1, dims=1)
            + 0.37 * torch.roll(features, shifts=7, dims=1)
        )
        batches.append((features, target))
    return tuple(batches)


def timing_receipt(timings: Sequence[StepTiming]) -> dict[str, object]:
    def values(name: str) -> list[float]:
        return [1000.0 * float(getattr(timing, name)) for timing in timings]

    return {
        name: {
            "median_ms": statistics.median(values(name)),
            "p95_ms": percentile(values(name), 0.95),
            "first_ms": values(name)[0],
        }
        for name in ("forward", "backward", "clip", "optimizer", "total")
    }


def run_case(batch_size: int, steps: int) -> dict[str, object]:
    torch.manual_seed(370038 + batch_size)
    baseline = TripairCircuitModule()
    candidate = copy.deepcopy(baseline)
    if count_parameters(baseline) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Tripair circuit parameter contract changed")
    initial_baseline = clone_state(baseline.state_dict())
    initial_candidate = clone_state(candidate.state_dict())

    baseline_optimizer = torch.optim.AdamW(
        baseline.parameters(), lr=3.0e-4, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.01
    )
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=3.0e-4, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.01
    )
    initial_baseline_optimizer = copy.deepcopy(baseline_optimizer.state_dict())
    initial_candidate_optimizer = copy.deepcopy(candidate_optimizer.state_dict())
    baseline_buffer = io.BytesIO()
    candidate_buffer = io.BytesIO()
    torch.save(
        {"model": initial_baseline, "optimizer": initial_baseline_optimizer},
        baseline_buffer,
    )
    torch.save(
        {"model": initial_candidate, "optimizer": initial_candidate_optimizer},
        candidate_buffer,
    )

    bridge = QSATripairBridge()
    baseline_timings = []
    candidate_timings = []
    losses = []
    maximum_output_error = 0.0
    maximum_observable_error = 0.0
    maximum_loss_error = 0.0
    maximum_grad_error = 0.0
    maximum_parameter_error = 0.0
    maximum_clip_norm_error = 0.0
    all_finite = True
    try:
        for features, target in training_batches(batch_size, steps):
            (
                baseline_output,
                baseline_observables,
                baseline_loss,
                baseline_timing,
                baseline_norm,
            ) = run_step(
                baseline,
                baseline_optimizer,
                features,
                target,
                "torch",
                None,
                1.0,
            )
            (
                candidate_output,
                candidate_observables,
                candidate_loss,
                candidate_timing,
                candidate_norm,
            ) = run_step(
                candidate,
                candidate_optimizer,
                features,
                target,
                "qsa",
                bridge,
                1.0,
            )
            baseline_timings.append(baseline_timing)
            candidate_timings.append(candidate_timing)
            maximum_output_error = max(
                maximum_output_error,
                float(torch.max(torch.abs(baseline_output - candidate_output))),
            )
            maximum_observable_error = max(
                maximum_observable_error,
                float(
                    torch.max(
                        torch.abs(baseline_observables - candidate_observables)
                    )
                ),
            )
            maximum_loss_error = max(
                maximum_loss_error,
                abs(float(baseline_loss) - float(candidate_loss)),
            )
            maximum_grad_error = max(
                maximum_grad_error, maximum_gradient_error(baseline, candidate)
            )
            maximum_clip_norm_error = max(
                maximum_clip_norm_error, abs(baseline_norm - candidate_norm)
            )
            maximum_parameter_error = max(
                maximum_parameter_error,
                maximum_state_error(baseline.state_dict(), candidate.state_dict()),
            )
            all_finite = (
                all_finite
                and all_finite_gradients(baseline)
                and all_finite_gradients(candidate)
            )
            losses.append(
                {
                    "baseline": float(baseline_loss),
                    "candidate": float(candidate_loss),
                }
            )

        bridge_state = bridge.state_receipt()
        final_baseline = clone_state(baseline.state_dict())
        final_candidate = clone_state(candidate.state_dict())

        baseline.load_state_dict(initial_baseline)
        candidate.load_state_dict(initial_candidate)
        baseline_optimizer.load_state_dict(initial_baseline_optimizer)
        candidate_optimizer.load_state_dict(initial_candidate_optimizer)
        rollback = {
            "baseline_model_max_error": maximum_state_error(
                baseline.state_dict(), initial_baseline
            ),
            "candidate_model_max_error": maximum_state_error(
                candidate.state_dict(), initial_candidate
            ),
            "baseline_optimizer_exact": optimizer_state_equal(
                baseline_optimizer.state_dict(), initial_baseline_optimizer
            ),
            "candidate_optimizer_exact": optimizer_state_equal(
                candidate_optimizer.state_dict(), initial_candidate_optimizer
            ),
            "baseline_checkpoint_bytes": len(baseline_buffer.getvalue()),
            "candidate_checkpoint_bytes": len(candidate_buffer.getvalue()),
        }
    finally:
        bridge.close()

    baseline_timing_receipt = timing_receipt(baseline_timings)
    candidate_timing_receipt = timing_receipt(candidate_timings)
    baseline_total = baseline_timing_receipt["total"]["median_ms"]
    candidate_total = candidate_timing_receipt["total"]["median_ms"]
    result: dict[str, object] = {
        "batch_size": batch_size,
        "steps": steps,
        "active_trainable_parameters": count_parameters(baseline),
        "parameter_names": list(dict(baseline.named_parameters()).keys()),
        "parameter_shapes": {
            name: list(parameter.shape)
            for name, parameter in baseline.named_parameters()
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3.0e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 0.01,
        },
        "gradient_clip_norm": 1.0,
        "baseline": {
            "backend": "PyTorch exact complex128 reverse-mode autograd",
            "timing": baseline_timing_receipt,
            "initial_loss": losses[0]["baseline"],
            "final_loss": losses[-1]["baseline"],
        },
        "candidate": {
            "backend": "QSA persistent component-local weighted adjoint",
            "timing": candidate_timing_receipt,
            "initial_loss": losses[0]["candidate"],
            "final_loss": losses[-1]["candidate"],
            "qsa": bridge_state,
        },
        "baseline_over_candidate_total_speed": baseline_total / candidate_total,
        "candidate_faster_than_pytorch": candidate_total < baseline_total,
        "maximum_output_error": maximum_output_error,
        "maximum_observable_error": maximum_observable_error,
        "maximum_loss_error": maximum_loss_error,
        "maximum_gradient_error": maximum_grad_error,
        "maximum_clip_norm_error": maximum_clip_norm_error,
        "maximum_parameter_error_after_step": maximum_parameter_error,
        "maximum_final_parameter_error": maximum_state_error(
            final_baseline, final_candidate
        ),
        "all_gradients_finite_and_present": all_finite,
        "rollback": rollback,
    }
    result["passed"] = (
        result["active_trainable_parameters"] == EXPECTED_PARAMETER_COUNT
        and all_finite
        and maximum_output_error <= 2.0e-12
        and maximum_observable_error <= 2.0e-12
        and maximum_loss_error <= 2.0e-13
        and maximum_grad_error <= 2.0e-11
        and maximum_parameter_error <= 2.0e-12
        and result["maximum_final_parameter_error"] <= 2.0e-12
        and bridge_state["root_immutable"]
        and bridge_state["workspace_immutable"]
        and bridge_state["workspace_extractions"] == 1
        and bridge_state["generation"] == 0
        and bridge_state["shifted_evaluations"] == 0
        and rollback["baseline_model_max_error"] == 0.0
        and rollback["candidate_model_max_error"] == 0.0
        and rollback["baseline_optimizer_exact"]
        and rollback["candidate_optimizer_exact"]
    )
    return result


def build_receipt(steps: int) -> dict[str, object]:
    cases = [run_case(batch_size, steps) for batch_size in (1, 4, 16, 64)]
    receipt: dict[str, object] = {
        "schema_version": "qsa.brain-recovery-tripair-optimizer-step.v1",
        "source": {
            "accepted_brain_parent": BRAIN_PARENT,
            "brain_quantum_source": BRAIN_QUANTUM_SOURCE,
            "brain_quantum_blob": BRAIN_QUANTUM_BLOB,
            "qsa_source": QSA_SOURCE,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64/complex128",
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "feature_width": FEATURE_WIDTH,
            "logical_qubits": QUBITS,
            "depth": DEPTH,
            "observables": list(OBSERVABLE_NAMES),
            "active_trainable_parameters": EXPECTED_PARAMETER_COUNT,
            "brain_input_order": list(BRAIN_INPUT_NAMES),
            "qsa_input_order": list(QSA_INPUT_NAMES),
            "brain_to_qsa_indices": list(BRAIN_TO_QSA),
            "qsa_gradient_to_brain_indices": list(QSA_TO_BRAIN),
            "same_model_parameter_names_shapes_and_initial_values": True,
            "same_batches_targets_optimizer_clipping_and_step_count": True,
            "speed_is_measured_not_required_for_correctness_pass": True,
        },
        "cases": cases,
        "classification_policy": {
            "correctness_pass": (
                "Requires forward, loss, gradient, clipped-update, parameter, "
                "persistent-state, and rollback parity."
            ),
            "engineering_speed_advantage": (
                "Established only for batch sizes whose complete AdamW median "
                "step is lower than the PyTorch control."
            ),
            "nonunique_boundary": (
                "PyTorch exact reverse mode is the strongest conventional "
                "control and may remain faster because it is batched and avoids "
                "Python/native per-sample pullback calls."
            ),
        },
        "not_established": [
            "complete 203539-parameter Tripair attachment advantage",
            "Brain language/canonical/safety capability contribution",
            "GPU advantage",
            "quantum uniqueness",
            "physical-QPU advantage",
            "production readiness",
        ],
    }
    receipt["passed"] = all(bool(case["passed"]) for case in cases)
    receipt["candidate_faster_batch_sizes"] = [
        int(case["batch_size"])
        for case in cases
        if bool(case["candidate_faster_than_pytorch"])
    ]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-optimizer-step.json"),
    )
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 4:
        raise SystemExit("--steps must be at least 4")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = build_receipt(args.steps)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("Brain Tripair complete optimizer-step contract failed")


if __name__ == "__main__":
    main()
