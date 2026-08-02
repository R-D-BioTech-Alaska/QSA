from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence
from typing import Optional

from . import (
    ParameterizedPlan,
    QubitNativeError,
    _Bindings,
    _NativeParameterizedOperation,
    _get_bindings,
)


class CausalRuntimeError(QubitNativeError):
    """Raised when the experimental native causal runtime rejects an operation."""


_CONFIGURED_BINDINGS: set[int] = set()


def _configure(bindings: _Bindings) -> None:
    key = id(bindings)
    if key in _CONFIGURED_BINDINGS:
        return

    lib = bindings.lib
    required = (
        "qcausal_api_version_major",
        "qcausal_api_version_minor",
        "qcausal_api_version_patch",
        "qcausal_last_error",
        "qcausal_create",
        "qcausal_from_qsc",
        "qcausal_fork",
        "qcausal_adopt",
        "qcausal_destroy",
        "qcausal_qubit_count",
        "qcausal_component_count",
        "qcausal_estimated_bytes",
        "qcausal_shared_owner_count",
        "qcausal_validate",
        "qcausal_amplitude",
        "qcausal_probabilities_one",
        "qcausal_qsc_size",
        "qcausal_qsc_write",
        "qcausal_parameterized_plan_create",
        "qcausal_parameterized_plan_destroy",
        "qcausal_parameterized_plan_parameter_count",
        "qcausal_parameterized_plan_execute",
        "qcausal_pauli_plan_create",
        "qcausal_pauli_plan_destroy",
        "qcausal_pauli_plan_observable_count",
        "qcausal_pauli_plan_execute",
    )
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        raise CausalRuntimeError(
            "Loaded QSA library does not provide the causal runtime: "
            + ", ".join(missing)
        )

    handle = ctypes.c_void_p
    size = ctypes.c_size_t
    u8_p = ctypes.POINTER(ctypes.c_uint8)
    double_p = ctypes.POINTER(ctypes.c_double)

    for name in (
        "qcausal_api_version_major",
        "qcausal_api_version_minor",
        "qcausal_api_version_patch",
    ):
        function = getattr(lib, name)
        function.argtypes = []
        function.restype = ctypes.c_uint32
    lib.qcausal_last_error.argtypes = []
    lib.qcausal_last_error.restype = ctypes.c_char_p

    lib.qcausal_create.argtypes = [size]
    lib.qcausal_create.restype = handle
    lib.qcausal_from_qsc.argtypes = [u8_p, size]
    lib.qcausal_from_qsc.restype = handle
    lib.qcausal_fork.argtypes = [handle]
    lib.qcausal_fork.restype = handle
    lib.qcausal_adopt.argtypes = [handle, handle]
    lib.qcausal_adopt.restype = ctypes.c_int
    lib.qcausal_destroy.argtypes = [handle]
    lib.qcausal_destroy.restype = None

    for name in (
        "qcausal_qubit_count",
        "qcausal_component_count",
        "qcausal_estimated_bytes",
        "qcausal_shared_owner_count",
    ):
        function = getattr(lib, name)
        function.argtypes = [handle]
        function.restype = size
    lib.qcausal_validate.argtypes = [handle]
    lib.qcausal_validate.restype = ctypes.c_int
    lib.qcausal_amplitude.argtypes = [
        handle,
        ctypes.c_uint64,
        double_p,
        double_p,
    ]
    lib.qcausal_amplitude.restype = ctypes.c_int
    lib.qcausal_probabilities_one.argtypes = [handle, double_p, size]
    lib.qcausal_probabilities_one.restype = ctypes.c_int
    lib.qcausal_qsc_size.argtypes = [handle]
    lib.qcausal_qsc_size.restype = size
    lib.qcausal_qsc_write.argtypes = [handle, u8_p, size]
    lib.qcausal_qsc_write.restype = ctypes.c_int

    lib.qcausal_parameterized_plan_create.argtypes = [
        ctypes.POINTER(_NativeParameterizedOperation),
        size,
        ctypes.c_uint32,
    ]
    lib.qcausal_parameterized_plan_create.restype = handle
    lib.qcausal_parameterized_plan_destroy.argtypes = [handle]
    lib.qcausal_parameterized_plan_destroy.restype = None
    lib.qcausal_parameterized_plan_parameter_count.argtypes = [handle]
    lib.qcausal_parameterized_plan_parameter_count.restype = size
    lib.qcausal_parameterized_plan_execute.argtypes = [
        handle,
        handle,
        double_p,
        size,
        ctypes.POINTER(size),
    ]
    lib.qcausal_parameterized_plan_execute.restype = ctypes.c_int

    lib.qcausal_pauli_plan_create.argtypes = [
        size,
        ctypes.c_char_p,
        size,
        ctypes.c_double,
    ]
    lib.qcausal_pauli_plan_create.restype = handle
    lib.qcausal_pauli_plan_destroy.argtypes = [handle]
    lib.qcausal_pauli_plan_destroy.restype = None
    lib.qcausal_pauli_plan_observable_count.argtypes = [handle]
    lib.qcausal_pauli_plan_observable_count.restype = size
    lib.qcausal_pauli_plan_execute.argtypes = [handle, handle, double_p, size]
    lib.qcausal_pauli_plan_execute.restype = ctypes.c_int

    version = (
        int(lib.qcausal_api_version_major()),
        int(lib.qcausal_api_version_minor()),
        int(lib.qcausal_api_version_patch()),
    )
    if version < (0, 1, 0):
        raise CausalRuntimeError(
            f"QSA causal runtime 0.1.0 or newer is required, loaded {version}"
        )
    _CONFIGURED_BINDINGS.add(key)


def _error(bindings: _Bindings) -> str:
    raw = bindings.lib.qcausal_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "Unknown causal runtime error"


def _check(bindings: _Bindings, code: int) -> None:
    if code != 0:
        raise CausalRuntimeError(_error(bindings))


class CausalParameterizedPlan:
    """Reusable native plan for mutation of copy-on-write causal branches."""

    def __init__(self, operations: object, *, optimize: bool = True) -> None:
        self.template = ParameterizedPlan(operations, optimize=optimize)
        self.parameter_names = self.template.parameter_names
        self.operations = self.template.operations
        self.optimize = bool(optimize)
        self._native_handles: dict[int, tuple[_Bindings, ctypes.c_void_p]] = {}

    @property
    def parameter_count(self) -> int:
        return self.template.parameter_count

    def _values(self, values: object) -> tuple[float, ...]:
        return self.template._values(values)

    def _native_handle(self, bindings: _Bindings) -> ctypes.c_void_p:
        _configure(bindings)
        key = id(bindings)
        cached = self._native_handles.get(key)
        if cached is not None:
            return cached[1]
        flags = 1 if self.optimize else 0
        raw = bindings.lib.qcausal_parameterized_plan_create(
            self.template._buffer,
            len(self.template),
            flags,
        )
        if not raw:
            raise CausalRuntimeError(_error(bindings))
        handle = ctypes.c_void_p(raw)
        native_count = int(
            bindings.lib.qcausal_parameterized_plan_parameter_count(handle)
        )
        if native_count != self.parameter_count:
            bindings.lib.qcausal_parameterized_plan_destroy(handle)
            raise CausalRuntimeError(
                "Causal plan parameter count differs from Python template"
            )
        self._native_handles[key] = (bindings, handle)
        return handle

    def close(self) -> None:
        handles = tuple(self._native_handles.values())
        self._native_handles.clear()
        for bindings, handle in handles:
            try:
                bindings.lib.qcausal_parameterized_plan_destroy(handle)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CausalPauliPlan:
    """Exact component-wise Pauli observable closure for causal states."""

    def __init__(
        self,
        qubits: int,
        words: Sequence[str],
        *,
        imaginary_tolerance: float = 1.0e-10,
    ) -> None:
        if int(qubits) <= 0:
            raise ValueError("qubits must be positive")
        normalized = tuple(str(word).upper() for word in words)
        if not normalized:
            raise ValueError("at least one Pauli word is required")
        if any(len(word) != int(qubits) for word in normalized):
            raise ValueError("every Pauli word must match the qubit count")
        if any(set(word) - {"I", "X", "Y", "Z"} for word in normalized):
            raise ValueError("Pauli words may contain only I, X, Y, and Z")
        self.qubits = int(qubits)
        self.words = normalized
        self.imaginary_tolerance = float(imaginary_tolerance)
        self._encoded = "".join(normalized).encode("ascii")
        self._native_handles: dict[int, tuple[_Bindings, ctypes.c_void_p]] = {}

    def _native_handle(self, bindings: _Bindings) -> ctypes.c_void_p:
        _configure(bindings)
        key = id(bindings)
        cached = self._native_handles.get(key)
        if cached is not None:
            return cached[1]
        raw = bindings.lib.qcausal_pauli_plan_create(
            self.qubits,
            self._encoded,
            len(self.words),
            self.imaginary_tolerance,
        )
        if not raw:
            raise CausalRuntimeError(_error(bindings))
        handle = ctypes.c_void_p(raw)
        native_count = int(bindings.lib.qcausal_pauli_plan_observable_count(handle))
        if native_count != len(self.words):
            bindings.lib.qcausal_pauli_plan_destroy(handle)
            raise CausalRuntimeError(
                "Causal Pauli observable count differs from Python plan"
            )
        self._native_handles[key] = (bindings, handle)
        return handle

    def execute(self, state: "CausalRegister") -> tuple[float, ...]:
        state._ensure_open()
        if int(state.qubit_count) != self.qubits:
            raise ValueError("Pauli plan qubit count differs from causal state")
        native = self._native_handle(state._bindings)
        array_type = ctypes.c_double * len(self.words)
        output = array_type()
        _check(
            state._bindings,
            state._bindings.lib.qcausal_pauli_plan_execute(
                native,
                state._handle,
                output,
                len(self.words),
            ),
        )
        return tuple(float(value) for value in output)

    def close(self) -> None:
        handles = tuple(self._native_handles.values())
        self._native_handles.clear()
        for bindings, handle in handles:
            try:
                bindings.lib.qcausal_pauli_plan_destroy(handle)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CausalRegister:
    """Persistent exact QSA state with constant-time native forks."""

    def __init__(
        self,
        qubits: int,
        *,
        library_path: Optional[str] = None,
        _bindings: Optional[_Bindings] = None,
        _handle: Optional[int | ctypes.c_void_p] = None,
    ) -> None:
        self._bindings = _bindings or _get_bindings(library_path)
        _configure(self._bindings)
        if _handle is None:
            raw = self._bindings.lib.qcausal_create(int(qubits))
        else:
            raw = _handle
        if isinstance(raw, ctypes.c_void_p):
            self._handle = raw
        else:
            self._handle = ctypes.c_void_p(raw)
        if not self._handle:
            raise CausalRuntimeError(_error(self._bindings))
        self._closed = False

    @classmethod
    def decode_qsc(
        cls,
        data: bytes,
        *,
        library_path: Optional[str] = None,
    ) -> "CausalRegister":
        if not data:
            raise ValueError("QSC data is empty")
        bindings = _get_bindings(library_path)
        _configure(bindings)
        array_type = ctypes.c_uint8 * len(data)
        buffer = array_type.from_buffer_copy(data)
        raw = bindings.lib.qcausal_from_qsc(buffer, len(data))
        if not raw:
            raise CausalRuntimeError(_error(bindings))
        qubits = int(bindings.lib.qcausal_qubit_count(raw))
        return cls(qubits, _bindings=bindings, _handle=raw)

    def _ensure_open(self) -> None:
        if self._closed or not self._handle:
            raise CausalRuntimeError("CausalRegister is closed")

    def close(self) -> None:
        if not self._closed and self._handle:
            self._bindings.lib.qcausal_destroy(self._handle)
            self._handle = ctypes.c_void_p()
            self._closed = True

    def __enter__(self) -> "CausalRegister":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def fork(self) -> "CausalRegister":
        self._ensure_open()
        raw = self._bindings.lib.qcausal_fork(self._handle)
        if not raw:
            raise CausalRuntimeError(_error(self._bindings))
        return CausalRegister(
            int(self.qubit_count),
            _bindings=self._bindings,
            _handle=raw,
        )

    def fork_many(self, count: int) -> list["CausalRegister"]:
        if int(count) < 0:
            raise ValueError("fork count cannot be negative")
        branches: list[CausalRegister] = []
        try:
            for _ in range(int(count)):
                branches.append(self.fork())
            return branches
        except Exception:
            for branch in branches:
                branch.close()
            raise

    def adopt(self, selected: "CausalRegister") -> "CausalRegister":
        self._ensure_open()
        selected._ensure_open()
        if selected._bindings is not self._bindings:
            raise ValueError("causal states use different native libraries")
        _check(
            self._bindings,
            self._bindings.lib.qcausal_adopt(self._handle, selected._handle),
        )
        selected.close()
        return self

    def apply(
        self,
        plan: CausalParameterizedPlan,
        values: Mapping[str, float] | Sequence[float],
    ) -> "CausalRegister":
        self._ensure_open()
        if not isinstance(plan, CausalParameterizedPlan):
            raise TypeError("plan must be a CausalParameterizedPlan")
        bound = plan._values(values)
        array_type = ctypes.c_double * len(bound)
        parameter_buffer = array_type(*bound)
        completed = ctypes.c_size_t()
        native = plan._native_handle(self._bindings)
        _check(
            self._bindings,
            self._bindings.lib.qcausal_parameterized_plan_execute(
                self._handle,
                native,
                parameter_buffer,
                len(bound),
                ctypes.byref(completed),
            ),
        )
        if completed.value != len(plan.template):
            raise CausalRuntimeError(
                f"causal plan completed {completed.value} of {len(plan.template)} operations"
            )
        return self

    @property
    def qubit_count(self) -> int:
        self._ensure_open()
        value = int(self._bindings.lib.qcausal_qubit_count(self._handle))
        if value <= 0:
            raise CausalRuntimeError(_error(self._bindings))
        return value

    @property
    def component_count(self) -> int:
        self._ensure_open()
        return int(self._bindings.lib.qcausal_component_count(self._handle))

    @property
    def estimated_bytes(self) -> int:
        self._ensure_open()
        return int(self._bindings.lib.qcausal_estimated_bytes(self._handle))

    @property
    def shared_owner_count(self) -> int:
        self._ensure_open()
        return int(self._bindings.lib.qcausal_shared_owner_count(self._handle))

    def amplitude(self, basis_index: int) -> complex:
        self._ensure_open()
        real = ctypes.c_double()
        imag = ctypes.c_double()
        _check(
            self._bindings,
            self._bindings.lib.qcausal_amplitude(
                self._handle,
                int(basis_index),
                ctypes.byref(real),
                ctypes.byref(imag),
            ),
        )
        return complex(real.value, imag.value)

    def probabilities_one(self) -> tuple[float, ...]:
        self._ensure_open()
        count = int(self.qubit_count)
        array_type = ctypes.c_double * count
        output = array_type()
        _check(
            self._bindings,
            self._bindings.lib.qcausal_probabilities_one(
                self._handle,
                output,
                count,
            ),
        )
        return tuple(float(value) for value in output)

    def encode_qsc(self) -> bytes:
        self._ensure_open()
        size = int(self._bindings.lib.qcausal_qsc_size(self._handle))
        if size <= 0:
            raise CausalRuntimeError(_error(self._bindings))
        array_type = ctypes.c_uint8 * size
        output = array_type()
        _check(
            self._bindings,
            self._bindings.lib.qcausal_qsc_write(self._handle, output, size),
        )
        return bytes(output)

    def validate(self) -> bool:
        self._ensure_open()
        _check(self._bindings, self._bindings.lib.qcausal_validate(self._handle))
        return True


__all__ = [
    "CausalParameterizedPlan",
    "CausalPauliPlan",
    "CausalRegister",
    "CausalRuntimeError",
]
