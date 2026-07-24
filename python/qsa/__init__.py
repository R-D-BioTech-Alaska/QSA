from __future__ import annotations

import ctypes
import os
import random
import sys
import threading

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

__version__ = "0.1.6"

class QubitNativeError(RuntimeError):

class _CallableInt(int):

    def __call__(self) -> int:
        return int(self)

def _library_names() -> tuple[str, ...]:
    if sys.platform.startswith("win"):
        return ("qstate.dll", "libqstate.dll")
    if sys.platform == "darwin":
        return ("libqstate.dylib",)
    return ("libqstate.so",)

def _explicit_library_path(path: Optional[str]) -> Optional[str]:
    return path or os.environ.get("QSA_NATIVE_LIB") or os.environ.get("QUBIT_NATIVE_LIB")

def _load_library(path: Optional[str] = None) -> ctypes.CDLL:
    candidates: list[Path] = []
    explicit = _explicit_library_path(path)
    if explicit:
        candidates.append(Path(explicit).expanduser())

    package_dir = Path(__file__).resolve().parent
    source_root = package_dir.parents[1] if len(package_dir.parents) > 1 else package_dir
    for name in _library_names():
        candidates.extend(
            [
                package_dir / name,
                source_root / "build" / name,
                source_root / "build" / "Release" / name,
                source_root / "lib" / name,
                Path(name),
            ]
        )

    failures: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        try:
            return ctypes.CDLL(candidate_text)
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    raise QubitNativeError(
        "Unable to load the QSA native library. Build it with CMake or set "
        "QSA_NATIVE_LIB (preferred) or QUBIT_NATIVE_LIB (legacy). Tried:\n"
        + "\n".join(failures)
    )

class _NativeOperation(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint32),
        ("first", ctypes.c_uint32),
        ("second", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("parameter", ctypes.c_double),
        ("sample", ctypes.c_double),
    ]

class _NativeParameterizedOperation(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint32),
        ("first", ctypes.c_uint32),
        ("second", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("parameter", ctypes.c_double),
        ("sample", ctypes.c_double),
        ("parameter_slot", ctypes.c_int32),
        ("sample_slot", ctypes.c_int32),
    ]

class Parameter:

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        text = str(name).strip()
        if not text:
            raise ValueError("parameter name cannot be empty")
        self.name = text

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Parameter) and self.name == other.name

_OPERATION_CODES = {
    "x": 1,
    "y": 2,
    "z": 3,
    "h": 4,
    "s": 5,
    "sdg": 6,
    "t": 7,
    "tdg": 8,
    "rx": 9,
    "ry": 10,
    "rz": 11,
    "cnot": 12,
    "cz": 13,
    "swap": 14,
    "bit_flip_trajectory": 15,
    "phase_flip_trajectory": 16,
    "depolarizing_trajectory": 17,
    "amplitude_damping_trajectory": 18,
}

class OperationPlan:

    def __init__(self, operations: object, *, optimize: bool = True) -> None:
        normalized: list[tuple[object, ...]] = []
        encoded: list[_NativeOperation] = []
        one_qubit = {"x", "y", "z", "h", "s", "sdg", "t", "tdg"}
        rotations = {"rx", "ry", "rz"}
        two_qubit = {"cnot", "cz", "swap"}
        trajectories = {
            "bit_flip_trajectory",
            "phase_flip_trajectory",
            "depolarizing_trajectory",
            "amplitude_damping_trajectory",
        }
        for index, raw_operation in enumerate(operations):
            operation = tuple(raw_operation)
            if not operation:
                raise ValueError(f"batch operation {index} is empty")
            name = str(operation[0]).lower()
            opcode = _OPERATION_CODES.get(name)
            if opcode is None:
                raise ValueError(f"batch operation {index} has unknown name: {name}")
            if name in one_qubit and len(operation) == 2:
                native = _NativeOperation(opcode, int(operation[1]), 0, 0, 0.0, 0.0)
            elif name in rotations and len(operation) == 3:
                native = _NativeOperation(
                    opcode, int(operation[1]), 0, 0, float(operation[2]), 0.0
                )
            elif name in two_qubit and len(operation) == 3:
                native = _NativeOperation(
                    opcode, int(operation[1]), int(operation[2]), 0, 0.0, 0.0
                )
            elif name in trajectories and len(operation) == 4:
                native = _NativeOperation(
                    opcode,
                    int(operation[1]),
                    0,
                    0,
                    float(operation[2]),
                    float(operation[3]),
                )
            else:
                raise ValueError(f"batch operation {index} has invalid arguments: {operation!r}")
            normalized.append((name, *operation[1:]))
            encoded.append(native)

        self.operations = tuple(normalized)
        self.count = len(encoded)
        self.optimize = bool(optimize)
        array_type = _NativeOperation * self.count
        self._buffer = array_type(*encoded)
        self._native_handles: dict[int, tuple["_Bindings", ctypes.c_void_p]] = {}
        self._native_lock = threading.Lock()

    def __len__(self) -> int:
        return self.count

    def _native_handle(self, bindings: "_Bindings") -> ctypes.c_void_p | None:
        if not bindings.has_plan_api:
            return None
        key = id(bindings)
        with self._native_lock:
            cached = self._native_handles.get(key)
            if cached is not None:
                return cached[1]
            flags = 1 if self.optimize else 0
            handle = bindings.lib.qstate_plan_create(self._buffer, len(self), flags)
            if not handle:
                raise QubitNativeError(bindings.error())
            native_handle = ctypes.c_void_p(handle)
            self._native_handles[key] = (bindings, native_handle)
            return native_handle

    def compiled_step_count(self, register: "QubitRegister") -> int:
        register._ensure_open()
        native = self._native_handle(register._bindings)
        if native is None:
            return len(self)
        return int(register._bindings.lib.qstate_plan_compiled_step_count(native))

    def apply_many(
        self,
        registers: object,
        *,
        workers: int = 0,
    ) -> list["QubitRegister"]:
        states = list(registers)
        if not states:
            return states
        for state in states:
            if not isinstance(state, QubitRegister):
                raise TypeError("all ensemble entries must be QubitRegister instances")
            state._ensure_open()
        bindings = states[0]._bindings
        if any(state._bindings is not bindings for state in states[1:]):
            raise ValueError("all registers in an ensemble must use the same native library")

        native = self._native_handle(bindings)
        if native is None or not bindings.has_plan_ensemble_api:
            for state in states:
                state.apply_plan(self)
            return states

        array_type = ctypes.c_void_p * len(states)
        handles = array_type(*(state._handle for state in states))
        completed = ctypes.c_size_t()
        code = bindings.lib.qstate_plan_execute_many(
            native,
            handles,
            len(states),
            max(0, int(workers)),
            ctypes.byref(completed),
        )
        if code != 0:
            raise QubitNativeError(
                f"{bindings.error()} (completed {completed.value} of {len(states)} registers)"
            )
        return states

    def close(self) -> None:
        with self._native_lock:
            handles = tuple(self._native_handles.values())
            self._native_handles.clear()
        for bindings, handle in handles:
            try:
                bindings.lib.qstate_plan_destroy(handle)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

class ParameterizedPlan:

    def __init__(self, operations: object, *, optimize: bool = True) -> None:
        names: list[str] = []
        slots: dict[str, int] = {}
        normalized: list[tuple[object, ...]] = []
        encoded: list[_NativeParameterizedOperation] = []
        one_qubit = {"x", "y", "z", "h", "s", "sdg", "t", "tdg"}
        rotations = {"rx", "ry", "rz"}
        two_qubit = {"cnot", "cz", "swap"}
        trajectories = {
            "bit_flip_trajectory",
            "phase_flip_trajectory",
            "depolarizing_trajectory",
            "amplitude_damping_trajectory",
        }

        def numeric(value: object) -> tuple[float, int]:
            if isinstance(value, Parameter):
                slot = slots.get(value.name)
                if slot is None:
                    slot = len(names)
                    slots[value.name] = slot
                    names.append(value.name)
                return 0.0, slot
            return float(value), -1

        for index, raw_operation in enumerate(operations):
            operation = tuple(raw_operation)
            if not operation:
                raise ValueError(f"parameterized operation {index} is empty")
            name = str(operation[0]).lower()
            opcode = _OPERATION_CODES.get(name)
            if opcode is None:
                raise ValueError(f"parameterized operation {index} has unknown name: {name}")
            parameter = 0.0
            sample = 0.0
            parameter_slot = -1
            sample_slot = -1
            first = 0
            second = 0
            if name in one_qubit and len(operation) == 2:
                first = int(operation[1])
            elif name in rotations and len(operation) == 3:
                first = int(operation[1])
                parameter, parameter_slot = numeric(operation[2])
            elif name in two_qubit and len(operation) == 3:
                first = int(operation[1])
                second = int(operation[2])
            elif name in trajectories and len(operation) == 4:
                first = int(operation[1])
                parameter, parameter_slot = numeric(operation[2])
                sample, sample_slot = numeric(operation[3])
            else:
                raise ValueError(
                    f"parameterized operation {index} has invalid arguments: {operation!r}"
                )
            normalized.append((name, *operation[1:]))
            encoded.append(
                _NativeParameterizedOperation(
                    opcode,
                    first,
                    second,
                    0,
                    parameter,
                    sample,
                    parameter_slot,
                    sample_slot,
                )
            )

        self.operations = tuple(normalized)
        self.parameter_names = tuple(names)
        self.optimize = bool(optimize)
        self.count = len(encoded)
        array_type = _NativeParameterizedOperation * self.count
        self._buffer = array_type(*encoded)
        self._native_handles: dict[int, tuple["_Bindings", ctypes.c_void_p]] = {}
        self._native_lock = threading.Lock()

    def __len__(self) -> int:
        return self.count

    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def _values(self, values: object) -> tuple[float, ...]:
        if isinstance(values, Mapping):
            missing = [name for name in self.parameter_names if name not in values]
            if missing:
                raise ValueError("missing parameter values: " + ", ".join(missing))
            return tuple(float(values[name]) for name in self.parameter_names)
        result = tuple(float(value) for value in values)
        if len(result) != self.parameter_count:
            raise ValueError(
                f"expected {self.parameter_count} parameter values, received {len(result)}"
            )
        return result

    def bind(self, values: object) -> OperationPlan:
        bound_values = self._values(values)
        by_name = dict(zip(self.parameter_names, bound_values))
        concrete: list[tuple[object, ...]] = []
        for operation in self.operations:
            concrete.append(
                tuple(by_name[item.name] if isinstance(item, Parameter) else item for item in operation)
            )
        return OperationPlan(concrete, optimize=self.optimize)

    def _native_handle(self, bindings: "_Bindings") -> ctypes.c_void_p | None:
        if not bindings.has_parameterized_plan_api:
            return None
        key = id(bindings)
        with self._native_lock:
            cached = self._native_handles.get(key)
            if cached is not None:
                return cached[1]
            flags = 1 if self.optimize else 0
            handle = bindings.lib.qstate_parameterized_plan_create(
                self._buffer, len(self), flags
            )
            if not handle:
                raise QubitNativeError(bindings.error())
            native_handle = ctypes.c_void_p(handle)
            self._native_handles[key] = (bindings, native_handle)
            return native_handle

    def _value_buffer(values: tuple[float, ...]) -> object:
        array_type = ctypes.c_double * len(values)
        return array_type(*values)

    def apply(self, register: "QubitRegister", values: object) -> "QubitRegister":
        register._ensure_open()
        bound_values = self._values(values)
        native = self._native_handle(register._bindings)
        if native is None:
            register.apply_plan(self.bind(bound_values))
            return register
        value_buffer = self._value_buffer(bound_values)
        completed = ctypes.c_size_t()
        code = register._bindings.lib.qstate_parameterized_plan_execute(
            register._handle,
            native,
            value_buffer,
            len(bound_values),
            ctypes.byref(completed),
        )
        if code != 0:
            raise QubitNativeError(
                f"{register._bindings.error()} (completed {completed.value} of {len(self)})"
            )
        return register

    def apply_many(
        self,
        registers: object,
        values: object,
        *,
        workers: int = 0,
    ) -> list["QubitRegister"]:
        states = list(registers)
        if not states:
            return states
        for state in states:
            if not isinstance(state, QubitRegister):
                raise TypeError("all ensemble entries must be QubitRegister instances")
            state._ensure_open()
        bindings = states[0]._bindings
        if any(state._bindings is not bindings for state in states[1:]):
            raise ValueError("all registers in an ensemble must use the same native library")
        bound_values = self._values(values)
        native = self._native_handle(bindings)
        if native is None or not bindings.has_parameterized_plan_ensemble_api:
            concrete = self.bind(bound_values)
            return concrete.apply_many(states, workers=workers)
        value_buffer = self._value_buffer(bound_values)
        array_type = ctypes.c_void_p * len(states)
        handles = array_type(*(state._handle for state in states))
        completed = ctypes.c_size_t()
        code = bindings.lib.qstate_parameterized_plan_execute_many(
            native,
            handles,
            len(states),
            value_buffer,
            len(bound_values),
            max(0, int(workers)),
            ctypes.byref(completed),
        )
        if code != 0:
            raise QubitNativeError(
                f"{bindings.error()} (completed {completed.value} of {len(states)} registers)"
            )
        return states

    def close(self) -> None:
        with self._native_lock:
            handles = tuple(self._native_handles.values())
            self._native_handles.clear()
        for bindings, handle in handles:
            try:
                bindings.lib.qstate_parameterized_plan_destroy(handle)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

class _Bindings:
    def __init__(self, library_path: Optional[str] = None) -> None:
        self.lib = _load_library(library_path)
        handle = ctypes.c_void_p
        u8 = ctypes.c_uint8
        u32 = ctypes.c_uint32
        u64 = ctypes.c_uint64
        size = ctypes.c_size_t
        int_p = ctypes.POINTER(ctypes.c_int)
        u8_p = ctypes.POINTER(u8)
        u64_p = ctypes.POINTER(u64)
        double_p = ctypes.POINTER(ctypes.c_double)

        self.lib.qstate_create.argtypes = [size]
        self.lib.qstate_create.restype = handle
        self.lib.qstate_destroy.argtypes = [handle]
        self.lib.qstate_destroy.restype = None
        self.lib.qstate_last_error.argtypes = []
        self.lib.qstate_last_error.restype = ctypes.c_char_p

        for name in ("x", "y", "z", "h", "s", "t"):
            function = getattr(self.lib, f"qstate_apply_{name}")
            function.argtypes = [handle, u32]
            function.restype = ctypes.c_int
        for name in ("rx", "ry", "rz"):
            function = getattr(self.lib, f"qstate_apply_{name}")
            function.argtypes = [handle, u32, ctypes.c_double]
            function.restype = ctypes.c_int
        for name in ("cnot", "cz", "swap"):
            function = getattr(self.lib, f"qstate_apply_{name}")
            function.argtypes = [handle, u32, u32]
            function.restype = ctypes.c_int

        self.lib.qstate_probability_one.argtypes = [handle, u32, double_p]
        self.lib.qstate_probability_one.restype = ctypes.c_int
        self.lib.qstate_measure.argtypes = [handle, u32, ctypes.c_double, int_p]
        self.lib.qstate_measure.restype = ctypes.c_int
        self.lib.qstate_amplitude.argtypes = [handle, u64, double_p, double_p]
        self.lib.qstate_amplitude.restype = ctypes.c_int

        for name in (
            "qubit_count",
            "component_count",
            "estimated_bytes",
            "description_size",
            "qsc_size",
        ):
            function = getattr(self.lib, f"qstate_{name}")
            function.argtypes = [handle]
            function.restype = size
        for name in ("component_size", "component_nonzero_count"):
            function = getattr(self.lib, f"qstate_{name}")
            function.argtypes = [handle, u32]
            function.restype = size

        self.lib.qstate_description_write.argtypes = [handle, ctypes.c_char_p, size]
        self.lib.qstate_description_write.restype = ctypes.c_int
        self.lib.qstate_qsc_write.argtypes = [handle, u8_p, size]
        self.lib.qstate_qsc_write.restype = ctypes.c_int
        self.lib.qstate_qsc_read.argtypes = [u8_p, size]
        self.lib.qstate_qsc_read.restype = handle

        self.has_extended_api = hasattr(self.lib, "qstate_abi_version_major")
        if self.has_extended_api:
            for name in ("major", "minor", "patch"):
                function = getattr(self.lib, f"qstate_abi_version_{name}")
                function.argtypes = []
                function.restype = u32
            self.lib.qstate_version_string.argtypes = []
            self.lib.qstate_version_string.restype = ctypes.c_char_p

            for name in ("sdg", "tdg"):
                function = getattr(self.lib, f"qstate_apply_{name}")
                function.argtypes = [handle, u32]
                function.restype = ctypes.c_int
            for name in (
                "bit_flip_trajectory",
                "phase_flip_trajectory",
                "depolarizing_trajectory",
                "amplitude_damping_trajectory",
            ):
                function = getattr(self.lib, f"qstate_apply_{name}")
                function.argtypes = [handle, u32, ctypes.c_double, ctypes.c_double]
                function.restype = ctypes.c_int

            self.lib.qstate_measure_all.argtypes = [handle, u64, u8_p, size]
            self.lib.qstate_measure_all.restype = ctypes.c_int
            self.lib.qstate_amplitude_bits.argtypes = [handle, u8_p, size, double_p, double_p]
            self.lib.qstate_amplitude_bits.restype = ctypes.c_int
            self.lib.qstate_component_kind.argtypes = [handle, u32, int_p]
            self.lib.qstate_component_kind.restype = ctypes.c_int
            self.lib.qstate_validate.argtypes = [handle]
            self.lib.qstate_validate.restype = ctypes.c_int
            self.has_batch_api = hasattr(self.lib, "qstate_apply_operations")
            if self.has_batch_api:
                self.lib.qstate_apply_operations.argtypes = [
                    handle,
                    ctypes.POINTER(_NativeOperation),
                    size,
                    ctypes.POINTER(size),
                ]
                self.lib.qstate_apply_operations.restype = ctypes.c_int

            self.has_plan_api = hasattr(self.lib, "qstate_plan_create")
            self.has_plan_ensemble_api = hasattr(self.lib, "qstate_plan_execute_many")
            if self.has_plan_api:
                self.lib.qstate_plan_create.argtypes = [
                    ctypes.POINTER(_NativeOperation),
                    size,
                    u32,
                ]
                self.lib.qstate_plan_create.restype = handle
                self.lib.qstate_plan_destroy.argtypes = [handle]
                self.lib.qstate_plan_destroy.restype = None
                self.lib.qstate_plan_source_operation_count.argtypes = [handle]
                self.lib.qstate_plan_source_operation_count.restype = size
                self.lib.qstate_plan_compiled_step_count.argtypes = [handle]
                self.lib.qstate_plan_compiled_step_count.restype = size
                self.lib.qstate_plan_execute.argtypes = [
                    handle,
                    handle,
                    ctypes.POINTER(size),
                ]
                self.lib.qstate_plan_execute.restype = ctypes.c_int
            if self.has_plan_ensemble_api:
                self.lib.qstate_plan_execute_many.argtypes = [
                    handle,
                    ctypes.POINTER(handle),
                    size,
                    size,
                    ctypes.POINTER(size),
                ]
                self.lib.qstate_plan_execute_many.restype = ctypes.c_int

            self.has_bulk_probability_api = hasattr(self.lib, "qstate_probabilities_one")
            if self.has_bulk_probability_api:
                self.lib.qstate_probabilities_one.argtypes = [handle, double_p, size]
                self.lib.qstate_probabilities_one.restype = ctypes.c_int

            self.has_parameterized_plan_api = hasattr(
                self.lib, "qstate_parameterized_plan_create"
            )
            self.has_parameterized_plan_ensemble_api = hasattr(
                self.lib, "qstate_parameterized_plan_execute_many"
            )
            if self.has_parameterized_plan_api:
                self.lib.qstate_parameterized_plan_create.argtypes = [
                    ctypes.POINTER(_NativeParameterizedOperation),
                    size,
                    u32,
                ]
                self.lib.qstate_parameterized_plan_create.restype = handle
                self.lib.qstate_parameterized_plan_destroy.argtypes = [handle]
                self.lib.qstate_parameterized_plan_destroy.restype = None
                self.lib.qstate_parameterized_plan_source_operation_count.argtypes = [handle]
                self.lib.qstate_parameterized_plan_source_operation_count.restype = size
                self.lib.qstate_parameterized_plan_parameter_count.argtypes = [handle]
                self.lib.qstate_parameterized_plan_parameter_count.restype = size
                self.lib.qstate_parameterized_plan_execute.argtypes = [
                    handle,
                    handle,
                    double_p,
                    size,
                    ctypes.POINTER(size),
                ]
                self.lib.qstate_parameterized_plan_execute.restype = ctypes.c_int
            if self.has_parameterized_plan_ensemble_api:
                self.lib.qstate_parameterized_plan_execute_many.argtypes = [
                    handle,
                    ctypes.POINTER(handle),
                    size,
                    double_p,
                    size,
                    size,
                    ctypes.POINTER(size),
                ]
                self.lib.qstate_parameterized_plan_execute_many.restype = ctypes.c_int

            self.has_grover_api = hasattr(self.lib, "qstate_grover_create")
            if self.has_grover_api:
                self.lib.qstate_apply_grover_oracle.argtypes = [handle, u64_p, size]
                self.lib.qstate_apply_grover_oracle.restype = ctypes.c_int
                self.lib.qstate_apply_grover_diffusion.argtypes = [handle]
                self.lib.qstate_apply_grover_diffusion.restype = ctypes.c_int
                self.lib.qstate_apply_grover_iterations.argtypes = [handle, u64_p, size, u64]
                self.lib.qstate_apply_grover_iterations.restype = ctypes.c_int

                self.lib.qstate_grover_create.argtypes = [size, u64_p, size]
                self.lib.qstate_grover_create.restype = handle
                self.lib.qstate_grover_create_count.argtypes = [size, u64]
                self.lib.qstate_grover_create_count.restype = handle
                self.lib.qstate_grover_destroy.argtypes = [handle]
                self.lib.qstate_grover_destroy.restype = None
                for name in ("reset", "apply_oracle", "apply_diffusion", "run_optimal", "validate"):
                    function = getattr(self.lib, f"qstate_grover_{name}")
                    function.argtypes = [handle]
                    function.restype = ctypes.c_int
                self.lib.qstate_grover_iterate.argtypes = [handle, u64]
                self.lib.qstate_grover_iterate.restype = ctypes.c_int
                self.lib.qstate_grover_qubit_count.argtypes = [handle]
                self.lib.qstate_grover_qubit_count.restype = size
                for name in (
                    "space_size",
                    "marked_count",
                    "iteration_count",
                    "optimal_iterations",
                ):
                    function = getattr(self.lib, f"qstate_grover_{name}")
                    function.argtypes = [handle]
                    function.restype = u64
                self.lib.qstate_grover_estimated_bytes.argtypes = [handle]
                self.lib.qstate_grover_estimated_bytes.restype = size
                self.lib.qstate_grover_has_explicit_marked_indices.argtypes = [handle]
                self.lib.qstate_grover_has_explicit_marked_indices.restype = ctypes.c_int
                self.lib.qstate_grover_success_probability.argtypes = [handle, double_p]
                self.lib.qstate_grover_success_probability.restype = ctypes.c_int
                for name in ("marked_amplitude", "unmarked_amplitude"):
                    function = getattr(self.lib, f"qstate_grover_{name}")
                    function.argtypes = [handle, double_p, double_p]
                    function.restype = ctypes.c_int
                self.lib.qstate_grover_amplitude.argtypes = [handle, u64, double_p, double_p]
                self.lib.qstate_grover_amplitude.restype = ctypes.c_int
                self.lib.qstate_grover_sample_basis.argtypes = [
                    handle,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(u64),
                ]
                self.lib.qstate_grover_sample_basis.restype = ctypes.c_int
                self.lib.qstate_grover_description_size.argtypes = [handle]
                self.lib.qstate_grover_description_size.restype = size
                self.lib.qstate_grover_description_write.argtypes = [handle, ctypes.c_char_p, size]
                self.lib.qstate_grover_description_write.restype = ctypes.c_int

            self.has_symmetry_api = hasattr(self.lib, "qstate_symmetry_create_ordered")
            if self.has_symmetry_api:
                self.lib.qstate_symmetry_create_ordered.argtypes = [size, u64_p, size]
                self.lib.qstate_symmetry_create_ordered.restype = handle
                self.lib.qstate_symmetry_create_count_only.argtypes = [size, u64_p, size]
                self.lib.qstate_symmetry_create_count_only.restype = handle
                self.lib.qstate_symmetry_create_labels.argtypes = [size, ctypes.POINTER(u32), size]
                self.lib.qstate_symmetry_create_labels.restype = handle
                self.lib.qstate_symmetry_create_hamming_weight.argtypes = [size]
                self.lib.qstate_symmetry_create_hamming_weight.restype = handle
                self.lib.qstate_symmetry_discover.argtypes = [
                    handle, size, ctypes.c_double, size
                ]
                self.lib.qstate_symmetry_discover.restype = handle
                self.lib.qstate_symmetry_destroy.argtypes = [handle]
                self.lib.qstate_symmetry_destroy.restype = None
                self.lib.qstate_symmetry_reset_uniform.argtypes = [handle]
                self.lib.qstate_symmetry_reset_uniform.restype = ctypes.c_int
                self.lib.qstate_symmetry_set_amplitudes.argtypes = [
                    handle, double_p, double_p, size, ctypes.c_int
                ]
                self.lib.qstate_symmetry_set_amplitudes.restype = ctypes.c_int
                self.lib.qstate_symmetry_apply_class_phase.argtypes = [
                    handle, size, ctypes.c_double
                ]
                self.lib.qstate_symmetry_apply_class_phase.restype = ctypes.c_int
                self.lib.qstate_symmetry_apply_class_phases.argtypes = [handle, double_p, size]
                self.lib.qstate_symmetry_apply_class_phases.restype = ctypes.c_int
                self.lib.qstate_symmetry_apply_reflection.argtypes = [handle]
                self.lib.qstate_symmetry_apply_reflection.restype = ctypes.c_int
                self.lib.qstate_symmetry_split_class.argtypes = [
                    handle, size, u64, ctypes.POINTER(size)
                ]
                self.lib.qstate_symmetry_split_class.restype = ctypes.c_int
                self.lib.qstate_symmetry_merge_equivalent.argtypes = [
                    handle, ctypes.c_double, ctypes.POINTER(size)
                ]
                self.lib.qstate_symmetry_merge_equivalent.restype = ctypes.c_int
                self.lib.qstate_symmetry_apply_unitary.argtypes = [
                    handle, double_p, double_p, size, ctypes.c_double
                ]
                self.lib.qstate_symmetry_apply_unitary.restype = ctypes.c_int
                self.lib.qstate_symmetry_iterate_unitary.argtypes = [
                    handle, double_p, double_p, size, u64, ctypes.c_double
                ]
                self.lib.qstate_symmetry_iterate_unitary.restype = ctypes.c_int
                self.lib.qstate_symmetry_qubit_count.argtypes = [handle]
                self.lib.qstate_symmetry_qubit_count.restype = size
                self.lib.qstate_symmetry_space_size.argtypes = [handle]
                self.lib.qstate_symmetry_space_size.restype = u64
                self.lib.qstate_symmetry_class_count.argtypes = [handle]
                self.lib.qstate_symmetry_class_count.restype = size
                self.lib.qstate_symmetry_membership_mode.argtypes = [handle]
                self.lib.qstate_symmetry_membership_mode.restype = ctypes.c_int
                self.lib.qstate_symmetry_class_size.argtypes = [handle, size]
                self.lib.qstate_symmetry_class_size.restype = u64
                self.lib.qstate_symmetry_class_amplitude.argtypes = [
                    handle, size, double_p, double_p
                ]
                self.lib.qstate_symmetry_class_amplitude.restype = ctypes.c_int
                self.lib.qstate_symmetry_class_probability.argtypes = [
                    handle, size, double_p
                ]
                self.lib.qstate_symmetry_class_probability.restype = ctypes.c_int
                self.lib.qstate_symmetry_amplitude.argtypes = [
                    handle, u64, double_p, double_p
                ]
                self.lib.qstate_symmetry_amplitude.restype = ctypes.c_int
                self.lib.qstate_symmetry_sample_class.argtypes = [
                    handle, ctypes.c_double, ctypes.POINTER(size)
                ]
                self.lib.qstate_symmetry_sample_class.restype = ctypes.c_int
                self.lib.qstate_symmetry_sample_basis.argtypes = [
                    handle, ctypes.c_double, ctypes.c_double, ctypes.POINTER(u64)
                ]
                self.lib.qstate_symmetry_sample_basis.restype = ctypes.c_int
                self.lib.qstate_symmetry_to_register.argtypes = [handle, size]
                self.lib.qstate_symmetry_to_register.restype = handle
                self.lib.qstate_symmetry_estimated_bytes.argtypes = [handle]
                self.lib.qstate_symmetry_estimated_bytes.restype = size
                self.lib.qstate_symmetry_discovery_error.argtypes = [handle, double_p]
                self.lib.qstate_symmetry_discovery_error.restype = ctypes.c_int
                self.lib.qstate_symmetry_validate.argtypes = [handle]
                self.lib.qstate_symmetry_validate.restype = ctypes.c_int
                self.lib.qstate_symmetry_description_size.argtypes = [handle]
                self.lib.qstate_symmetry_description_size.restype = size
                self.lib.qstate_symmetry_description_write.argtypes = [
                    handle, ctypes.c_char_p, size
                ]
                self.lib.qstate_symmetry_description_write.restype = ctypes.c_int
        else:
            self.has_batch_api = False
            self.has_plan_api = False
            self.has_plan_ensemble_api = False
            self.has_bulk_probability_api = False
            self.has_parameterized_plan_api = False
            self.has_parameterized_plan_ensemble_api = False
            self.has_grover_api = False
            self.has_symmetry_api = False

    def error(self) -> str:
        raw = self.lib.qstate_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "Unknown QSA native error"

    def check(self, code: int) -> None:
        if code != 0:
            raise QubitNativeError(self.error())

    def require_extended(self, operation: str) -> None:
        if not self.has_extended_api:
            raise QubitNativeError(
                f"{operation} requires QSA native ABI 1.1 or newer; an older library was loaded"
            )

    def require_grover(self, operation: str) -> None:
        if not self.has_grover_api:
            raise QubitNativeError(
                f"{operation} requires QSA native ABI 1.4 or newer; an older library was loaded"
            )

    def require_symmetry(self, operation: str) -> None:
        if not self.has_symmetry_api:
            raise QubitNativeError(
                f"{operation} requires QSA native ABI 1.5 or newer; an older library was loaded"
            )

_BINDINGS: dict[str, _Bindings] = {}
_BINDINGS_LOCK = threading.Lock()

def _bindings_key(path: Optional[str]) -> str:
    explicit = _explicit_library_path(path)
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    return "<automatic>"

def _get_bindings(path: Optional[str] = None) -> _Bindings:
    key = _bindings_key(path)
    with _BINDINGS_LOCK:
        bindings = _BINDINGS.get(key)
        if bindings is None:
            bindings = _Bindings(path)
            _BINDINGS[key] = bindings
        return bindings

class GroverSearch:

    def __init__(
        self,
        qubits: int,
        marked_indices: object,
        *,
        library_path: Optional[str] = None,
        _bindings: Optional[_Bindings] = None,
        _handle: Optional[int] = None,
        _explicit_marked: Optional[tuple[int, ...]] = None,
    ) -> None:
        self._bindings = _bindings or _get_bindings(library_path)
        self._bindings.require_grover("GroverSearch")
        if _handle is not None:
            self._handle = ctypes.c_void_p(_handle)
            self._explicit_marked = _explicit_marked
        else:
            if isinstance(marked_indices, int):
                marked = (int(marked_indices),)
            else:
                marked = tuple(int(value) for value in marked_indices)
            if not marked:
                raise ValueError("marked_indices cannot be empty")
            array_type = ctypes.c_uint64 * len(marked)
            buffer = array_type(*marked)
            handle = self._bindings.lib.qstate_grover_create(int(qubits), buffer, len(marked))
            if not handle:
                raise QubitNativeError(self._bindings.error())
            self._handle = ctypes.c_void_p(handle)
            self._explicit_marked = tuple(sorted(marked))
        self._closed = False

    def from_marked_count(
        cls,
        qubits: int,
        marked_count: int,
        *,
        library_path: Optional[str] = None,
    ) -> "GroverSearch":
        bindings = _get_bindings(library_path)
        bindings.require_grover("GroverSearch.from_marked_count")
        handle = bindings.lib.qstate_grover_create_count(int(qubits), int(marked_count))
        if not handle:
            raise QubitNativeError(bindings.error())
        return cls(
            int(qubits),
            (),
            _bindings=bindings,
            _handle=handle,
            _explicit_marked=None,
        )

    def _ensure_open(self) -> None:
        if self._closed or not self._handle:
            raise QubitNativeError("GroverSearch is closed")

    def close(self) -> None:
        if not self._closed and self._handle:
            self._bindings.lib.qstate_grover_destroy(self._handle)
            self._handle = ctypes.c_void_p()
            self._closed = True

    def __enter__(self) -> "GroverSearch":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def reset(self) -> "GroverSearch":
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_grover_reset(self._handle))
        return self

    def oracle(self) -> "GroverSearch":
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_grover_apply_oracle(self._handle))
        return self

    def diffuse(self) -> "GroverSearch":
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_grover_apply_diffusion(self._handle))
        return self

    def iterate(self, count: int = 1) -> "GroverSearch":
        self._ensure_open()
        if count < 0:
            raise ValueError("Grover iteration count cannot be negative")
        self._bindings.check(self._bindings.lib.qstate_grover_iterate(self._handle, int(count)))
        return self

    def run_optimal(self) -> "GroverSearch":
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_grover_run_optimal(self._handle))
        return self

    def qubit_count(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_grover_qubit_count(self._handle))

    def space_size(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_grover_space_size(self._handle))

    def marked_count(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_grover_marked_count(self._handle))

    def iteration_count(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_grover_iteration_count(self._handle))

    def optimal_iterations(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_grover_optimal_iterations(self._handle))

    def estimated_bytes(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_grover_estimated_bytes(self._handle))

    def has_explicit_marked_indices(self) -> bool:
        self._ensure_open()
        result = int(
            self._bindings.lib.qstate_grover_has_explicit_marked_indices(self._handle)
        )
        if result < 0:
            raise QubitNativeError(self._bindings.error())
        return bool(result)

    def success_probability(self) -> float:
        self._ensure_open()
        result = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_grover_success_probability(
                self._handle, ctypes.byref(result)
            )
        )
        return float(result.value)

    def _amplitude_call(self, name: str, *arguments: object) -> complex:
        self._ensure_open()
        real = ctypes.c_double()
        imag = ctypes.c_double()
        function = getattr(self._bindings.lib, name)
        self._bindings.check(
            function(self._handle, *arguments, ctypes.byref(real), ctypes.byref(imag))
        )
        return complex(real.value, imag.value)

    def marked_amplitude(self) -> complex:
        return self._amplitude_call("qstate_grover_marked_amplitude")

    def unmarked_amplitude(self) -> complex:
        return self._amplitude_call("qstate_grover_unmarked_amplitude")

    def amplitude(self, basis_index: int) -> complex:
        return self._amplitude_call("qstate_grover_amplitude", int(basis_index))

    def sample(
        self,
        *,
        branch_sample: Optional[float] = None,
        index_sample: Optional[float] = None,
    ) -> int:
        self._ensure_open()
        branch = random.random() if branch_sample is None else float(branch_sample)
        index = random.random() if index_sample is None else float(index_sample)
        result = ctypes.c_uint64()
        self._bindings.check(
            self._bindings.lib.qstate_grover_sample_basis(
                self._handle, branch, index, ctypes.byref(result)
            )
        )
        return int(result.value)

    def validate(self) -> bool:
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_grover_validate(self._handle))
        return True

    def describe(self) -> str:
        self._ensure_open()
        size = int(self._bindings.lib.qstate_grover_description_size(self._handle))
        if size <= 0:
            raise QubitNativeError(self._bindings.error())
        output = ctypes.create_string_buffer(size)
        self._bindings.check(
            self._bindings.lib.qstate_grover_description_write(self._handle, output, size)
        )
        return output.value.decode("utf-8")

class SymmetryState:

    _MEMBERSHIP_NAMES = {
        0: "count_only",
        1: "ordered_ranges",
        2: "explicit_labels",
        3: "hamming_weight",
    }

    def __init__(
        self,
        qubits: int,
        class_counts: object,
        *,
        count_only: bool = False,
        library_path: Optional[str] = None,
        _bindings: Optional[_Bindings] = None,
        _handle: Optional[int] = None,
    ) -> None:
        self._bindings = _bindings or _get_bindings(library_path)
        self._bindings.require_symmetry("SymmetryState")
        if _handle is not None:
            self._handle = ctypes.c_void_p(_handle)
        else:
            counts = tuple(int(value) for value in class_counts)
            if not counts:
                raise ValueError("class_counts cannot be empty")
            if any(value <= 0 for value in counts):
                raise ValueError("every symmetry class must contain at least one basis state")
            array_type = ctypes.c_uint64 * len(counts)
            buffer = array_type(*counts)
            function = (
                self._bindings.lib.qstate_symmetry_create_count_only
                if count_only
                else self._bindings.lib.qstate_symmetry_create_ordered
            )
            handle = function(int(qubits), buffer, len(counts))
            if not handle:
                raise QubitNativeError(self._bindings.error())
            self._handle = ctypes.c_void_p(handle)
        self._closed = False

    def from_counts(
        cls,
        qubits: int,
        class_counts: object,
        *,
        library_path: Optional[str] = None,
    ) -> "SymmetryState":
        return cls(qubits, class_counts, count_only=True, library_path=library_path)

    def from_labels(
        cls,
        qubits: int,
        labels: object,
        *,
        library_path: Optional[str] = None,
    ) -> "SymmetryState":
        bindings = _get_bindings(library_path)
        bindings.require_symmetry("SymmetryState.from_labels")
        values = tuple(int(value) for value in labels)
        if not values:
            raise ValueError("labels cannot be empty")
        if any(value < 0 for value in values):
            raise ValueError("symmetry labels cannot be negative")
        array_type = ctypes.c_uint32 * len(values)
        buffer = array_type(*values)
        handle = bindings.lib.qstate_symmetry_create_labels(int(qubits), buffer, len(values))
        if not handle:
            raise QubitNativeError(bindings.error())
        return cls(qubits, (), _bindings=bindings, _handle=handle)

    def hamming_weight(
        cls,
        qubits: int,
        *,
        library_path: Optional[str] = None,
    ) -> "SymmetryState":
        bindings = _get_bindings(library_path)
        bindings.require_symmetry("SymmetryState.hamming_weight")
        handle = bindings.lib.qstate_symmetry_create_hamming_weight(int(qubits))
        if not handle:
            raise QubitNativeError(bindings.error())
        return cls(int(qubits), (), _bindings=bindings, _handle=handle)

    def from_register(
        cls,
        register: "QubitRegister",
        *,
        max_qubits: int = 24,
        tolerance: float = 0.0,
        max_classes: int = 1_000_000,
    ) -> "SymmetryState":
        register._ensure_open()
        bindings = register._bindings
        bindings.require_symmetry("SymmetryState.from_register")
        handle = bindings.lib.qstate_symmetry_discover(
            register._handle,
            int(max_qubits),
            float(tolerance),
            int(max_classes),
        )
        if not handle:
            raise QubitNativeError(bindings.error())
        return cls(int(register.qubit_count), (), _bindings=bindings, _handle=handle)

    def _ensure_open(self) -> None:
        if self._closed or not self._handle:
            raise QubitNativeError("SymmetryState is closed")

    def close(self) -> None:
        if not self._closed and self._handle:
            self._bindings.lib.qstate_symmetry_destroy(self._handle)
            self._handle = ctypes.c_void_p()
            self._closed = True

    def __enter__(self) -> "SymmetryState":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _complex_arrays(values: object) -> tuple[object, object, int]:
        entries = tuple(complex(value) for value in values)
        array_type = ctypes.c_double * len(entries)
        return (
            array_type(*(value.real for value in entries)),
            array_type(*(value.imag for value in entries)),
            len(entries),
        )

    def _matrix_arrays(self, matrix: object) -> tuple[object, object, int]:
        rows = tuple(matrix)
        if not rows:
            raise ValueError("unitary matrix cannot be empty")
        if isinstance(rows[0], (list, tuple)):
            if len(rows) != int(self.class_count):
                raise ValueError("unitary matrix row count must match class_count")
            flattened: list[complex] = []
            for row in rows:
                values = tuple(row)
                if len(values) != int(self.class_count):
                    raise ValueError("unitary matrix must be square")
                flattened.extend(complex(value) for value in values)
            return self._complex_arrays(flattened)
        if len(rows) != int(self.class_count) ** 2:
            raise ValueError("flat unitary matrix must contain class_count squared values")
        return self._complex_arrays(rows)

    def reset(self) -> "SymmetryState":
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_symmetry_reset_uniform(self._handle))
        return self

    def set_amplitudes(
        self,
        amplitudes: object,
        *,
        normalize: bool = True,
    ) -> "SymmetryState":
        self._ensure_open()
        real, imag, count = self._complex_arrays(amplitudes)
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_set_amplitudes(
                self._handle, real, imag, count, int(normalize)
            )
        )
        return self

    def phase(self, class_index: int, angle: float) -> "SymmetryState":
        self._ensure_open()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_apply_class_phase(
                self._handle, int(class_index), float(angle)
            )
        )
        return self

    def phases(self, angles: object) -> "SymmetryState":
        self._ensure_open()
        values = tuple(float(value) for value in angles)
        array_type = ctypes.c_double * len(values)
        buffer = array_type(*values)
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_apply_class_phases(
                self._handle, buffer, len(values)
            )
        )
        return self

    def reflect(self) -> "SymmetryState":
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_symmetry_apply_reflection(self._handle))
        return self

    def split_class(self, class_index: int, first_count: int) -> int:
        self._ensure_open()
        result = ctypes.c_size_t()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_split_class(
                self._handle, int(class_index), int(first_count), ctypes.byref(result)
            )
        )
        return int(result.value)

    def merge_equivalent(self, *, tolerance: float = 1e-12) -> int:
        self._ensure_open()
        result = ctypes.c_size_t()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_merge_equivalent(
                self._handle, float(tolerance), ctypes.byref(result)
            )
        )
        return int(result.value)

    def unitary(self, matrix: object, *, tolerance: float = 1e-10) -> "SymmetryState":
        self._ensure_open()
        real, imag, count = self._matrix_arrays(matrix)
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_apply_unitary(
                self._handle, real, imag, count, float(tolerance)
            )
        )
        return self

    def iterate_unitary(
        self,
        matrix: object,
        count: int,
        *,
        tolerance: float = 1e-10,
    ) -> "SymmetryState":
        self._ensure_open()
        if count < 0:
            raise ValueError("unitary iteration count cannot be negative")
        real, imag, elements = self._matrix_arrays(matrix)
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_iterate_unitary(
                self._handle, real, imag, elements, int(count), float(tolerance)
            )
        )
        return self

    def qubit_count(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_symmetry_qubit_count(self._handle))

    def space_size(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_symmetry_space_size(self._handle))

    def class_count(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_symmetry_class_count(self._handle))

    def membership(self) -> str:
        self._ensure_open()
        value = int(self._bindings.lib.qstate_symmetry_membership_mode(self._handle))
        if value < 0:
            raise QubitNativeError(self._bindings.error())
        return self._MEMBERSHIP_NAMES.get(value, f"unknown_{value}")

    def estimated_bytes(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_symmetry_estimated_bytes(self._handle))

    def discovery_error(self) -> float:
        self._ensure_open()
        result = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_discovery_error(
                self._handle, ctypes.byref(result)
            )
        )
        return float(result.value)

    def class_size(self, class_index: int) -> int:
        self._ensure_open()
        value = int(
            self._bindings.lib.qstate_symmetry_class_size(self._handle, int(class_index))
        )
        if value == 0:
            error = self._bindings.error()
            if error:
                raise QubitNativeError(error)
        return value

    def class_amplitude(self, class_index: int) -> complex:
        self._ensure_open()
        real = ctypes.c_double()
        imag = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_class_amplitude(
                self._handle, int(class_index), ctypes.byref(real), ctypes.byref(imag)
            )
        )
        return complex(real.value, imag.value)

    def class_probability(self, class_index: int) -> float:
        self._ensure_open()
        result = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_class_probability(
                self._handle, int(class_index), ctypes.byref(result)
            )
        )
        return float(result.value)

    def amplitude(self, basis_index: int) -> complex:
        self._ensure_open()
        real = ctypes.c_double()
        imag = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_amplitude(
                self._handle, int(basis_index), ctypes.byref(real), ctypes.byref(imag)
            )
        )
        return complex(real.value, imag.value)

    def sample_class(self, sample: Optional[float] = None) -> int:
        self._ensure_open()
        value = random.random() if sample is None else float(sample)
        result = ctypes.c_size_t()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_sample_class(
                self._handle, value, ctypes.byref(result)
            )
        )
        return int(result.value)

    def sample(
        self,
        *,
        class_sample: Optional[float] = None,
        index_sample: Optional[float] = None,
    ) -> int:
        self._ensure_open()
        branch = random.random() if class_sample is None else float(class_sample)
        index = random.random() if index_sample is None else float(index_sample)
        result = ctypes.c_uint64()
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_sample_basis(
                self._handle, branch, index, ctypes.byref(result)
            )
        )
        return int(result.value)

    def to_register(self, *, max_qubits: int = 24) -> "QubitRegister":
        self._ensure_open()
        handle = self._bindings.lib.qstate_symmetry_to_register(
            self._handle, int(max_qubits)
        )
        if not handle:
            raise QubitNativeError(self._bindings.error())
        qubits = int(self._bindings.lib.qstate_qubit_count(handle))
        return QubitRegister(qubits, _bindings=self._bindings, _handle=handle)

    def validate(self) -> bool:
        self._ensure_open()
        self._bindings.check(self._bindings.lib.qstate_symmetry_validate(self._handle))
        return True

    def describe(self) -> str:
        self._ensure_open()
        size = int(self._bindings.lib.qstate_symmetry_description_size(self._handle))
        if size <= 0:
            raise QubitNativeError(self._bindings.error())
        output = ctypes.create_string_buffer(size)
        self._bindings.check(
            self._bindings.lib.qstate_symmetry_description_write(
                self._handle, output, size
            )
        )
        return output.value.decode("utf-8")

class QubitRegister:

    def __init__(
        self,
        qubits: int,
        *,
        library_path: Optional[str] = None,
        _bindings: Optional[_Bindings] = None,
        _handle: Optional[int] = None,
    ) -> None:
        if qubits <= 0:
            raise ValueError("qubits must be positive")
        self._bindings = _bindings or _get_bindings(library_path)
        if _handle is None:
            self._handle = self._bindings.lib.qstate_create(qubits)
        elif isinstance(_handle, ctypes.c_void_p):
            self._handle = _handle
        else:
            self._handle = ctypes.c_void_p(_handle)
        if not self._handle:
            raise QubitNativeError(self._bindings.error())
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed or not self._handle:
            raise QubitNativeError("QubitRegister is closed")

    def close(self) -> None:
        if not self._closed and self._handle:
            self._bindings.lib.qstate_destroy(self._handle)
            self._closed = True
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "QubitRegister":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _call(self, name: str, *arguments: object) -> "QubitRegister":
        self._ensure_open()
        function = getattr(self._bindings.lib, f"qstate_apply_{name}")
        self._bindings.check(function(self._handle, *arguments))
        return self

    def x(self, qubit: int) -> "QubitRegister":
        return self._call("x", qubit)

    def y(self, qubit: int) -> "QubitRegister":
        return self._call("y", qubit)

    def z(self, qubit: int) -> "QubitRegister":
        return self._call("z", qubit)

    def h(self, qubit: int) -> "QubitRegister":
        return self._call("h", qubit)

    def s(self, qubit: int) -> "QubitRegister":
        return self._call("s", qubit)

    def sdg(self, qubit: int) -> "QubitRegister":
        self._bindings.require_extended("sdg")
        return self._call("sdg", qubit)

    def t(self, qubit: int) -> "QubitRegister":
        return self._call("t", qubit)

    def tdg(self, qubit: int) -> "QubitRegister":
        self._bindings.require_extended("tdg")
        return self._call("tdg", qubit)

    def rx(self, qubit: int, theta: float) -> "QubitRegister":
        return self._call("rx", qubit, theta)

    def ry(self, qubit: int, theta: float) -> "QubitRegister":
        return self._call("ry", qubit, theta)

    def rz(self, qubit: int, theta: float) -> "QubitRegister":
        return self._call("rz", qubit, theta)

    def cnot(self, control: int, target: int) -> "QubitRegister":
        return self._call("cnot", control, target)

    def cz(self, first: int, second: int) -> "QubitRegister":
        return self._call("cz", first, second)

    def swap(self, first: int, second: int) -> "QubitRegister":
        return self._call("swap", first, second)

    def grover_oracle(self, marked_indices: object) -> "QubitRegister":
        self._ensure_open()
        self._bindings.require_grover("grover_oracle")
        marked = (int(marked_indices),) if isinstance(marked_indices, int) else tuple(
            int(value) for value in marked_indices
        )
        if not marked:
            raise ValueError("marked_indices cannot be empty")
        array_type = ctypes.c_uint64 * len(marked)
        buffer = array_type(*marked)
        self._bindings.check(
            self._bindings.lib.qstate_apply_grover_oracle(
                self._handle, buffer, len(marked)
            )
        )
        return self

    def grover_diffusion(self) -> "QubitRegister":
        self._ensure_open()
        self._bindings.require_grover("grover_diffusion")
        self._bindings.check(
            self._bindings.lib.qstate_apply_grover_diffusion(self._handle)
        )
        return self

    def grover_iterations(
        self,
        marked_indices: object,
        count: int = 1,
    ) -> "QubitRegister":
        self._ensure_open()
        self._bindings.require_grover("grover_iterations")
        if count < 0:
            raise ValueError("Grover iteration count cannot be negative")
        marked = (int(marked_indices),) if isinstance(marked_indices, int) else tuple(
            int(value) for value in marked_indices
        )
        if not marked:
            raise ValueError("marked_indices cannot be empty")
        array_type = ctypes.c_uint64 * len(marked)
        buffer = array_type(*marked)
        self._bindings.check(
            self._bindings.lib.qstate_apply_grover_iterations(
                self._handle, buffer, len(marked), int(count)
            )
        )
        return self

    def bit_flip_trajectory(self, qubit: int, probability: float, sample: float) -> "QubitRegister":
        self._bindings.require_extended("bit_flip_trajectory")
        return self._call("bit_flip_trajectory", qubit, probability, sample)

    def phase_flip_trajectory(self, qubit: int, probability: float, sample: float) -> "QubitRegister":
        self._bindings.require_extended("phase_flip_trajectory")
        return self._call("phase_flip_trajectory", qubit, probability, sample)

    def depolarizing_trajectory(self, qubit: int, probability: float, sample: float) -> "QubitRegister":
        self._bindings.require_extended("depolarizing_trajectory")
        return self._call("depolarizing_trajectory", qubit, probability, sample)

    def amplitude_damping_trajectory(self, qubit: int, gamma: float, sample: float) -> "QubitRegister":
        self._bindings.require_extended("amplitude_damping_trajectory")
        return self._call("amplitude_damping_trajectory", qubit, gamma, sample)

    def compile_operations(operations: object, *, optimize: bool = True) -> OperationPlan:
        return OperationPlan(operations, optimize=optimize)

    def apply_batch(self, operations: object) -> "QubitRegister":
        self._ensure_open()
        plan = operations if isinstance(operations, OperationPlan) else OperationPlan(operations)
        if len(plan) == 0:
            return self
        native_plan = plan._native_handle(self._bindings)
        if native_plan is not None:
            completed = ctypes.c_size_t()
            code = self._bindings.lib.qstate_plan_execute(
                self._handle, native_plan, ctypes.byref(completed)
            )
            if code != 0:
                raise QubitNativeError(
                    f"{self._bindings.error()} (completed {completed.value} of {len(plan)})"
                )
            return self
        if not self._bindings.has_batch_api:
            for operation in plan.operations:
                name = str(operation[0])
                getattr(self, name)(*operation[1:])
            return self

        completed = ctypes.c_size_t()
        code = self._bindings.lib.qstate_apply_operations(
            self._handle, plan._buffer, len(plan), ctypes.byref(completed)
        )
        if code != 0:
            raise QubitNativeError(
                f"{self._bindings.error()} (completed {completed.value} of {len(plan)})"
            )
        return self

    def apply_plan(self, plan: OperationPlan) -> "QubitRegister":
        return self.apply_batch(plan)

    def apply_plan_many(
        registers: object,
        plan: OperationPlan,
        *,
        workers: int = 0,
    ) -> list["QubitRegister"]:
        return plan.apply_many(registers, workers=workers)

    def apply_parameterized_plan(
        self,
        plan: ParameterizedPlan,
        values: object,
    ) -> "QubitRegister":
        return plan.apply(self, values)

    def apply_parameterized_plan_many(
        registers: object,
        plan: ParameterizedPlan,
        values: object,
        *,
        workers: int = 0,
    ) -> list["QubitRegister"]:
        return plan.apply_many(registers, values, workers=workers)

    def probability_one(self, qubit: int) -> float:
        self._ensure_open()
        result = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_probability_one(self._handle, qubit, ctypes.byref(result))
        )
        return float(result.value)

    def probabilities_one(self) -> list[float]:
        self._ensure_open()
        count = int(self.qubit_count)
        if not self._bindings.has_bulk_probability_api:
            return [self.probability_one(qubit) for qubit in range(count)]
        array_type = ctypes.c_double * count
        output = array_type()
        self._bindings.check(
            self._bindings.lib.qstate_probabilities_one(self._handle, output, count)
        )
        return [float(value) for value in output]

    def measure(self, qubit: int, sample: Optional[float] = None) -> int:
        self._ensure_open()
        value = random.random() if sample is None else sample
        result = ctypes.c_int()
        self._bindings.check(
            self._bindings.lib.qstate_measure(self._handle, qubit, value, ctypes.byref(result))
        )
        return int(result.value)

    def measure_all(self, seed: Optional[int] = None) -> list[int]:
        self._ensure_open()
        self._bindings.require_extended("measure_all")
        actual_seed = random.getrandbits(64) if seed is None else seed
        count = int(self.qubit_count)
        array_type = ctypes.c_uint8 * count
        output = array_type()
        self._bindings.check(
            self._bindings.lib.qstate_measure_all(self._handle, actual_seed, output, count)
        )
        return [int(value) for value in output]

    def amplitude(self, basis_index: int) -> complex:
        self._ensure_open()
        real = ctypes.c_double()
        imag = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_amplitude(
                self._handle, basis_index, ctypes.byref(real), ctypes.byref(imag)
            )
        )
        return complex(real.value, imag.value)

    def amplitude_bits(self, bits: list[int] | tuple[int, ...] | bytes | bytearray) -> complex:
        self._ensure_open()
        self._bindings.require_extended("amplitude_bits")
        values = bytes(bits)
        array_type = ctypes.c_uint8 * len(values)
        input_buffer = array_type.from_buffer_copy(values)
        real = ctypes.c_double()
        imag = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_amplitude_bits(
                self._handle,
                input_buffer,
                len(values),
                ctypes.byref(real),
                ctypes.byref(imag),
            )
        )
        return complex(real.value, imag.value)

    def qubit_count(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_qubit_count(self._handle))

    def component_count(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_component_count(self._handle))

    def estimated_bytes(self) -> _CallableInt:
        self._ensure_open()
        return _CallableInt(self._bindings.lib.qstate_estimated_bytes(self._handle))

    def component_size(self, qubit: int) -> int:
        self._ensure_open()
        return int(self._bindings.lib.qstate_component_size(self._handle, qubit))

    def component_nonzero_count(self, qubit: int) -> int:
        self._ensure_open()
        return int(self._bindings.lib.qstate_component_nonzero_count(self._handle, qubit))

    def component_kind(self, qubit: int) -> str:
        self._ensure_open()
        self._bindings.require_extended("component_kind")
        result = ctypes.c_int()
        self._bindings.check(
            self._bindings.lib.qstate_component_kind(self._handle, qubit, ctypes.byref(result))
        )
        return {0: "cell", 1: "sparse", 2: "dense"}.get(result.value, "unknown")

    def validate(self) -> bool:
        self._ensure_open()
        if not self._bindings.has_extended_api:
            return True
        self._bindings.check(self._bindings.lib.qstate_validate(self._handle))
        return True

    def native_version(self) -> str:
        if not self._bindings.has_extended_api:
            return "0.1.0-compatible"
        raw = self._bindings.lib.qstate_version_string()
        return raw.decode("ascii", errors="replace") if raw else "unknown"

    def abi_version(self) -> tuple[int, int, int]:
        if not self._bindings.has_extended_api:
            return (1, 0, 0)
        return (
            int(self._bindings.lib.qstate_abi_version_major()),
            int(self._bindings.lib.qstate_abi_version_minor()),
            int(self._bindings.lib.qstate_abi_version_patch()),
        )

    def describe(self) -> str:
        self._ensure_open()
        size = int(self._bindings.lib.qstate_description_size(self._handle))
        if size <= 0:
            raise QubitNativeError(self._bindings.error())
        output = ctypes.create_string_buffer(size)
        self._bindings.check(
            self._bindings.lib.qstate_description_write(self._handle, output, size)
        )
        return output.value.decode("utf-8")

    def encode_qsc(self) -> bytes:
        self._ensure_open()
        size = int(self._bindings.lib.qstate_qsc_size(self._handle))
        if size <= 0:
            raise QubitNativeError(self._bindings.error())
        array_type = ctypes.c_uint8 * size
        output = array_type()
        self._bindings.check(self._bindings.lib.qstate_qsc_write(self._handle, output, size))
        return bytes(output)

    def save_qsc(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_bytes(self.encode_qsc())

    def decode_qsc(
        cls,
        data: bytes,
        *,
        library_path: Optional[str] = None,
    ) -> "QubitRegister":
        if not data:
            raise ValueError("QSC data is empty")
        bindings = _get_bindings(library_path)
        array_type = ctypes.c_uint8 * len(data)
        input_buffer = array_type.from_buffer_copy(data)
        handle = bindings.lib.qstate_qsc_read(input_buffer, len(data))
        if not handle:
            raise QubitNativeError(bindings.error())
        qubits = int(bindings.lib.qstate_qubit_count(handle))
        return cls(qubits, _bindings=bindings, _handle=handle)

    def load_qsc(
        cls,
        path: str | os.PathLike[str],
        *,
        library_path: Optional[str] = None,
    ) -> "QubitRegister":
        return cls.decode_qsc(Path(path).read_bytes(), library_path=library_path)

QRegister = QubitRegister

__all__ = [
    "GroverSearch",
    "OperationPlan",
    "Parameter",
    "ParameterizedPlan",
    "QRegister",
    "SymmetryState",
    "QubitNativeError",
    "QubitRegister",
    "__version__",
]
