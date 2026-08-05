from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import brain_recovery_tripair_optimizer_step_v9_vectorized as v9


v8 = v9.v8
v7 = v9.v7
v1 = v9.v1
v4 = v9.v4

EXPECTED_PARAMETER_NAMES = (
    "theta_0",
    "phi_0",
    "theta_1",
    "phi_1",
    "theta_2",
    "phi_2",
    "layer_0_0_ry",
    "layer_0_0_rz",
    "layer_0_1_ry",
    "layer_0_1_rz",
    "layer_0_2_ry",
    "layer_0_2_rz",
    "layer_1_0_ry",
    "layer_1_0_rz",
    "layer_1_1_ry",
    "layer_1_1_rz",
    "layer_1_2_ry",
    "layer_1_2_rz",
)
BRAIN_GRADIENT_COLUMNS = (0, 2, 4, 1, 3, 5)


@dataclass(frozen=True)
class ContiguousTripairBatch:
    parameter_tensor: torch.Tensor
    parameter_array: np.ndarray
    observations: np.ndarray


class ContiguousVectorizedTripairBridge(v9.VectorizedBatchTripairBridge):
    """Exact CPU bridge using contiguous Torch/NumPy/native buffers.

    The bridge deliberately rejects non-CPU execution. Brain angles and shared
    circuit angles are assembled into one [batch, 18] float64 tensor in exact
    QSA parameter-occurrence order. NumPy and ctypes then view the same memory.
    Observable and gradient arrays return through torch.from_numpy without
    scalar expansion, dictionaries, per-row tuples, or list conversion.
    """

    def __init__(self) -> None:
        super().__init__()
        if tuple(self.parameter_names) != EXPECTED_PARAMETER_NAMES:
            raise RuntimeError(
                "Tripair parameter occurrence order changed; contiguous binding "
                "must fail closed"
            )
        self.parameter_matrix_build_seconds = 0.0
        self.gradient_projection_seconds = 0.0
        self.parameter_zero_copy_calls = 0
        self.observation_zero_copy_calls = 0
        self.cotangent_zero_copy_calls = 0
        self.input_gradient_zero_copy_calls = 0
        self.shared_gradient_zero_copy_calls = 0
        self.python_scalar_parameter_bindings = 0
        self.python_scalar_gradient_assignments = 0
        self.parameter_matrix_bytes = 0
        self.observable_matrix_bytes = 0
        self.gradient_matrix_bytes = 0

    @staticmethod
    def _require_cpu_float64(tensor: torch.Tensor, label: str) -> None:
        if tensor.device.type != "cpu":
            raise ValueError(f"{label} must remain on CPU for the safe pilot")
        if tensor.dtype != torch.float64:
            raise ValueError(f"{label} must use float64 for exact comparison")

    def parameter_matrix(
        self,
        brain_angles: torch.Tensor,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        self._require_cpu_float64(brain_angles, "brain angles")
        self._require_cpu_float64(shared, "shared angles")
        if brain_angles.ndim != 2 or brain_angles.shape[1] != 6:
            raise ValueError("brain angle matrix must have shape [batch, 6]")
        if shared.numel() != 12:
            raise ValueError("shared Tripair angle tensor must contain 12 values")
        ordered_brain = brain_angles.detach()[:, (0, 3, 1, 4, 2, 5)]
        shared_rows = shared.detach().reshape(1, 12).expand(
            brain_angles.shape[0], 12
        )
        return torch.cat((ordered_brain, shared_rows), dim=1).contiguous()

    @staticmethod
    def _shares_numpy_storage(tensor: torch.Tensor, array: np.ndarray) -> bool:
        return bool(
            tensor.device.type == "cpu"
            and tensor.is_contiguous()
            and int(tensor.data_ptr()) == int(array.__array_interface__["data"][0])
        )

    def forward(self, brain_angles, shared):
        started = time.perf_counter()
        parameter_tensor = self.parameter_matrix(brain_angles, shared)
        parameter_array = parameter_tensor.numpy()
        matrix_done = time.perf_counter()
        observations = self.vectorized.values_numpy(self.root, parameter_array)
        output = torch.from_numpy(observations)
        finished = time.perf_counter()

        self.parameter_matrix_build_seconds += matrix_done - started
        self.forward_seconds += finished - started
        self.forward_calls += 1
        self.forward_batch_calls += 1
        self.forward_core_native_calls += 1
        self.forward_executions += int(parameter_tensor.shape[0])
        self.parameter_rows_bound += int(parameter_tensor.shape[0])
        self.observable_rows += int(parameter_tensor.shape[0])
        self.vectorized_primal_materializations += 1
        self.parameter_matrix_bytes = max(
            self.parameter_matrix_bytes,
            int(parameter_array.nbytes),
        )
        self.observable_matrix_bytes = max(
            self.observable_matrix_bytes,
            int(observations.nbytes),
        )
        if self._shares_numpy_storage(parameter_tensor, parameter_array):
            self.parameter_zero_copy_calls += 1
        if self._shares_numpy_storage(output, observations):
            self.observation_zero_copy_calls += 1
        return output, ContiguousTripairBatch(
            parameter_tensor=parameter_tensor,
            parameter_array=parameter_array,
            observations=observations,
        )

    def backward(
        self,
        batch,
        cotangents,
        input_dtype,
        input_device,
        shared_shape,
    ):
        started = time.perf_counter()
        self._require_cpu_float64(cotangents, "observable cotangents")
        cotangent_tensor = cotangents.detach().contiguous()
        cotangent_array = cotangent_tensor.numpy()
        result = self.vectorized.evaluate_numpy(
            self.root,
            batch.parameter_array,
            cotangent_array,
        )
        native_done = time.perf_counter()

        input_array = np.ascontiguousarray(
            result.gradients[:, BRAIN_GRADIENT_COLUMNS],
            dtype=np.float64,
        )
        shared_array = np.ascontiguousarray(
            np.sum(result.gradients[:, 6:], axis=0, dtype=np.float64).reshape(
                tuple(shared_shape)
            ),
            dtype=np.float64,
        )
        input_gradient = torch.from_numpy(input_array)
        shared_gradient = torch.from_numpy(shared_array)
        if input_dtype != torch.float64 or input_device.type != "cpu":
            input_gradient = input_gradient.to(
                dtype=input_dtype,
                device=input_device,
            )
            shared_gradient = shared_gradient.to(
                dtype=input_dtype,
                device=input_device,
            )
        finished = time.perf_counter()

        self.backward_native_calls += int(result.native_calls)
        self.backward_batch_calls += 1
        self.vectorized_reverse_materializations += int(result.state_materializations)
        self.adjoint_forward_sweeps += int(result.forward_sweeps)
        self.adjoint_reverse_sweeps += int(result.reverse_sweeps)
        self.shifted_evaluations += int(result.shifted_evaluations)
        self.local_widths.extend(
            [v1.QUBITS] * int(result.row_count)
        )
        self.maximum_backward_value_replay_error = max(
            self.maximum_backward_value_replay_error,
            float(np.max(np.abs(batch.observations - result.values))),
        )
        self.gradient_projection_seconds += finished - native_done
        self.backward_seconds += finished - started
        self.backward_calls += 1
        self.gradient_matrix_bytes = max(
            self.gradient_matrix_bytes,
            int(result.gradients.nbytes),
        )
        if self._shares_numpy_storage(cotangent_tensor, cotangent_array):
            self.cotangent_zero_copy_calls += 1
        if self._shares_numpy_storage(input_gradient, input_array):
            self.input_gradient_zero_copy_calls += 1
        if self._shares_numpy_storage(shared_gradient, shared_array):
            self.shared_gradient_zero_copy_calls += 1
        return input_gradient, shared_gradient

    def state_receipt(self) -> dict[str, object]:
        receipt = super().state_receipt()
        receipt.update(
            {
                "mode": "safe_contiguous_zero_copy_vectorized_batch",
                "parameter_binding": (
                    "one contiguous [batch,18] Torch tensor in occurrence order"
                ),
                "python_scalar_parameter_bindings": (
                    self.python_scalar_parameter_bindings
                ),
                "python_scalar_gradient_assignments": (
                    self.python_scalar_gradient_assignments
                ),
                "parameter_matrix_build_seconds": (
                    self.parameter_matrix_build_seconds
                ),
                "gradient_projection_seconds": self.gradient_projection_seconds,
                "parameter_zero_copy_calls": self.parameter_zero_copy_calls,
                "observation_zero_copy_calls": self.observation_zero_copy_calls,
                "cotangent_zero_copy_calls": self.cotangent_zero_copy_calls,
                "input_gradient_zero_copy_calls": (
                    self.input_gradient_zero_copy_calls
                ),
                "shared_gradient_zero_copy_calls": (
                    self.shared_gradient_zero_copy_calls
                ),
                "parameter_matrix_peak_bytes": self.parameter_matrix_bytes,
                "observable_matrix_peak_bytes": self.observable_matrix_bytes,
                "gradient_matrix_peak_bytes": self.gradient_matrix_bytes,
                "torch_numpy_parameter_copy": False,
                "torch_numpy_observable_copy": False,
                "torch_numpy_cotangent_copy": False,
                "shared_gradient_reduction": (
                    "NumPy float64 sample-major sum over columns 6:18"
                ),
                "brain_gradient_reorder": "columns [0,2,4,1,3,5]",
            }
        )
        return receipt


def contiguous_case_passed(case: dict[str, object]) -> bool:
    qsa = case["candidate"]["qsa"]
    rollback = case["rollback"]
    expected_steps = int(case["steps"])
    expected_rows = int(case["batch_size"]) * expected_steps
    return bool(
        case["active_trainable_parameters"] == v1.EXPECTED_PARAMETER_COUNT
        and case["all_gradients_finite_and_present"]
        and case["maximum_output_error"] <= 3.0e-12
        and case["maximum_observable_error"] <= 3.0e-12
        and case["maximum_loss_error"] <= 3.0e-13
        and case["maximum_gradient_error"] <= 3.0e-11
        and case["maximum_clip_norm_error"] <= 3.0e-11
        and case["maximum_parameter_error_after_step"] <= 3.0e-12
        and case["maximum_final_parameter_error"] <= 3.0e-12
        and qsa["root_immutable"]
        and not qsa["persistent_workspace_used"]
        and not qsa["retained_candidate_used"]
        and qsa["forward_batch_calls"] == expected_steps
        and qsa["backward_batch_calls"] == expected_steps
        and qsa["forward_core_native_calls"] == expected_steps
        and qsa["backward_native_calls"] == expected_steps
        and qsa["forward_executions"] == expected_rows
        and qsa["forward_sweeps"] == expected_rows
        and qsa["reverse_sweeps"] == expected_rows
        and qsa["shifted_evaluations"] == 0
        and qsa["local_widths"] == [v1.QUBITS] * expected_rows
        and qsa["component_closure_extractions"] == 0
        and qsa["branch_state_constructions"] == 0
        and qsa["branch_state_destructions"] == 0
        and qsa["vectorized_primal_materializations"] == expected_steps
        and qsa["vectorized_reverse_materializations"] == expected_steps
        and qsa["parameter_rows_bound"] == expected_rows
        and qsa["observable_rows"] == expected_rows
        and qsa["parameter_zero_copy_calls"] == expected_steps
        and qsa["observation_zero_copy_calls"] == expected_steps
        and qsa["cotangent_zero_copy_calls"] == expected_steps
        and qsa["input_gradient_zero_copy_calls"] == expected_steps
        and qsa["shared_gradient_zero_copy_calls"] == expected_steps
        and qsa["python_scalar_parameter_bindings"] == 0
        and qsa["python_scalar_gradient_assignments"] == 0
        and qsa["maximum_backward_value_replay_error"] <= 3.0e-12
        and rollback["baseline_model_max_error"] == 0.0
        and rollback["candidate_model_max_error"] == 0.0
        and rollback["baseline_optimizer_exact"]
        and rollback["candidate_optimizer_exact"]
    )


def build_receipt(steps: int) -> dict[str, object]:
    batch_sizes = (1, 4, 16, 64, 128)
    safe_cases = v7.run_cases(v4.SafeComponentTripairBridge, batch_sizes, steps)
    vectorized_cases = v7.run_cases(v9.VectorizedBatchTripairBridge, batch_sizes, steps)
    contiguous_cases = v7.run_cases(
        ContiguousVectorizedTripairBridge,
        batch_sizes,
        steps,
    )
    safe_by_batch = {int(case["batch_size"]): case for case in safe_cases}
    vectorized_by_batch = {
        int(case["batch_size"]): case for case in vectorized_cases
    }

    for case in contiguous_cases:
        batch_size = int(case["batch_size"])
        safe = safe_by_batch[batch_size]
        vectorized = vectorized_by_batch[batch_size]
        case["candidate"]["backend"] = (
            "QSA contiguous Torch/NumPy operation-major exact batch adjoint"
        )
        case["safe_per_sample_reference"] = {
            "passed": bool(v4.case_passed(safe)),
            "timing": safe["candidate"]["timing"],
            "qsa": safe["candidate"]["qsa"],
        }
        case["vectorized_sequence_reference"] = {
            "passed": bool(v9.vectorized_case_passed(vectorized)),
            "timing": vectorized["candidate"]["timing"],
            "qsa": vectorized["candidate"]["qsa"],
        }
        contiguous_total = float(
            case["candidate"]["timing"]["total"]["median_ms"]
        )
        safe_total = float(safe["candidate"]["timing"]["total"]["median_ms"])
        vectorized_total = float(
            vectorized["candidate"]["timing"]["total"]["median_ms"]
        )
        case["safe_over_contiguous_total_speed"] = safe_total / contiguous_total
        case["sequence_over_contiguous_total_speed"] = (
            vectorized_total / contiguous_total
        )
        case["contiguous_faster_than_safe"] = contiguous_total < safe_total
        case["contiguous_faster_than_sequence"] = (
            contiguous_total < vectorized_total
        )
        case["passed"] = bool(
            contiguous_case_passed(case)
            and case["safe_per_sample_reference"]["passed"]
            and case["vectorized_sequence_reference"]["passed"]
        )

    rerun = v7.run_cases(ContiguousVectorizedTripairBridge, (64,), steps)[0]
    deterministic = {
        "batch_size": 64,
        "first": v8.deterministic_signature(
            next(
                case
                for case in contiguous_cases
                if int(case["batch_size"]) == 64
            )
        ),
        "second": v8.deterministic_signature(rerun),
    }
    deterministic["equal"] = deterministic["first"] == deterministic["second"]

    repository_root = Path(__file__).resolve().parents[1]
    receipt: dict[str, object] = {
        "schema_version": (
            "qsa.brain-recovery-tripair-optimizer-step.v10-contiguous"
        ),
        "source": {
            "runtime_head": v7.git_head(repository_root),
            "runtime_parent_exactness_repair": (
                "748e4a40483de72ea5e9092d1db45a62ec6b4a16"
            ),
            "frozen_evidence_head": v7.git_head(v7.EVIDENCE_BENCHMARKS.parent),
            "accepted_brain_parent": v1.BRAIN_PARENT,
            "brain_quantum_source": v1.BRAIN_QUANTUM_SOURCE,
            "brain_quantum_blob": v1.BRAIN_QUANTUM_BLOB,
        },
        "environment": {
            "python": v7.sys.version,
            "platform": v7.platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64/complex128",
            "device": "cpu",
            "gpu_used": False,
        },
        "contract": {
            "same_5395_trainable_parameters": True,
            "same_names_shapes_initialization_batches_targets_optimizer_clipping": True,
            "nonzero_learning_rate": 3.0e-4,
            "exact_rollback": True,
            "shifted_evaluations": 0,
            "persistent_candidate_reuse": False,
            "batch_sizes": list(batch_sizes),
            "steps_per_case": steps,
            "parameter_matrix": "contiguous float64 [batch,18]",
            "parameter_occurrence_order": list(EXPECTED_PARAMETER_NAMES),
            "python_scalar_parameter_bindings": 0,
            "python_scalar_gradient_assignments": 0,
            "torch_numpy_zero_copy": True,
            "native_calls_per_optimizer_step": 2,
            "stored_depth_trajectory": False,
            "deterministic_shared_gradient_order": (
                "NumPy sample-major float64 sum"
            ),
        },
        "cases": contiguous_cases,
        "deterministic_rerun": deterministic,
        "classification_policy": {
            "correctness": (
                "Requires exact output, observable, loss, gradient, clipping, "
                "parameter trajectory, root immutability, deterministic rerun, "
                "zero-copy bridge receipts, and exact model/optimizer rollback."
            ),
            "complete_step_advantage": (
                "Established only where contiguous QSA complete AdamW median "
                "step is lower than PyTorch, V9 sequence binding, and the safe "
                "per-sample QSA bridge."
            ),
            "claim_boundary": (
                "A positive result is a QSA engineering and quantum-native "
                "structural execution advantage, not quantum uniqueness."
            ),
        },
        "not_established": [
            "held-out Brain-data learning preservation",
            "Brain language canonical or safety capability contribution",
            "persistent selected-pullback safety",
            "quantum uniqueness",
            "physical-QPU advantage",
            "production readiness",
        ],
    }
    receipt["passed"] = bool(
        all(bool(case["passed"]) for case in contiguous_cases)
        and deterministic["equal"]
    )
    receipt["contiguous_faster_than_safe_batch_sizes"] = [
        int(case["batch_size"])
        for case in contiguous_cases
        if bool(case["contiguous_faster_than_safe"])
    ]
    receipt["contiguous_faster_than_sequence_batch_sizes"] = [
        int(case["batch_size"])
        for case in contiguous_cases
        if bool(case["contiguous_faster_than_sequence"])
    ]
    receipt["contiguous_faster_than_pytorch_batch_sizes"] = [
        int(case["batch_size"])
        for case in contiguous_cases
        if bool(case["candidate_faster_than_pytorch"])
    ]
    receipt["complete_step_advantage_over_all_controls_batch_sizes"] = [
        int(case["batch_size"])
        for case in contiguous_cases
        if bool(case["contiguous_faster_than_safe"])
        and bool(case["contiguous_faster_than_sequence"])
        and bool(case["candidate_faster_than_pytorch"])
    ]
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-tripair-optimizer-step-v10-contiguous.json"
        ),
    )
    parser.add_argument("--steps", type=int, default=20)
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
        raise SystemExit("contiguous batch optimizer-step contract failed")


if __name__ == "__main__":
    main()
