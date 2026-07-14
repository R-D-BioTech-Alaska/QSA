"""Python bridge for the Qubit Native State Engine.

This module intentionally does not import NumPy.  Numeric state evolution lives
in the compiled C++ engine; Python is only the control/API layer.
"""

from __future__ import annotations

import ctypes
import os
import random
import sys
from pathlib import Path
from typing import Optional

class QubitNativeError(RuntimeError):
    """Raised when the native Qubit engine rejects an operation."""

def _library_names() -> tuple[str, ...]:
    if sys.platform.startswith("win"):
        return ("qstate.dll", "libqstate.dll")
    if sys.platform == "darwin":
        return ("libqstate.dylib",)
    return ("libqstate.so",)

def _load_library(path: Optional[str] = None) -> ctypes.CDLL:
    candidates: list[Path] = []
    explicit = path or os.environ.get("QUBIT_NATIVE_LIB")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    root = Path(__file__).resolve().parents[1]
    for name in _library_names():
        candidates.extend(
            [
                root / "build" / name,
                root / "build" / "Release" / name,
                root / "lib" / name,
                Path(name),
            ]
        )

    failures: list[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    raise QubitNativeError(
        "Unable to load the Qubit native library. Build it with CMake or set "
        "QUBIT_NATIVE_LIB. Tried:\n" + "\n".join(failures)
    )

class _Bindings:
    def __init__(self, library_path: Optional[str] = None) -> None:
        self.lib = _load_library(library_path)
        handle = ctypes.c_void_p
        u32 = ctypes.c_uint32
        u64 = ctypes.c_uint64
        size = ctypes.c_size_t
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
        self.lib.qstate_measure.argtypes = [handle, u32, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
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
        self.lib.qstate_qsc_write.argtypes = [handle, ctypes.POINTER(ctypes.c_uint8), size]
        self.lib.qstate_qsc_write.restype = ctypes.c_int
        self.lib.qstate_qsc_read.argtypes = [ctypes.POINTER(ctypes.c_uint8), size]
        self.lib.qstate_qsc_read.restype = handle

    def error(self) -> str:
        raw = self.lib.qstate_last_error()
        return raw.decode("utf-8", errors="replace") if raw else "Unknown Qubit native error"

    def check(self, code: int) -> None:
        if code != 0:
            raise QubitNativeError(self.error())

class QubitRegister:
    """Adaptive Qubit-native register backed by the compiled state engine."""
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
        self._bindings = _bindings or _Bindings(library_path)
        self._handle = ctypes.c_void_p(_handle) if _handle is not None else self._bindings.lib.qstate_create(qubits)
        if not self._handle:
            raise QubitNativeError(self._bindings.error())
        self._closed = False

    def close(self) -> None:
        if not self._closed and self._handle:
            self._bindings.lib.qstate_destroy(self._handle)
            self._closed = True
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "QubitRegister":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _call(self, name: str, *arguments: object) -> "QubitRegister":
        if self._closed:
            raise QubitNativeError("QubitRegister is closed")
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

    def t(self, qubit: int) -> "QubitRegister":
        return self._call("t", qubit)

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

    def probability_one(self, qubit: int) -> float:
        result = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_probability_one(self._handle, qubit, ctypes.byref(result))
        )
        return float(result.value)

    def measure(self, qubit: int, sample: Optional[float] = None) -> int:
        value = random.random() if sample is None else sample
        result = ctypes.c_int()
        self._bindings.check(
            self._bindings.lib.qstate_measure(self._handle, qubit, value, ctypes.byref(result))
        )
        return int(result.value)

    def amplitude(self, basis_index: int) -> complex:
        real = ctypes.c_double()
        imag = ctypes.c_double()
        self._bindings.check(
            self._bindings.lib.qstate_amplitude(
                self._handle, basis_index, ctypes.byref(real), ctypes.byref(imag)
            )
        )
        return complex(real.value, imag.value)

    def qubit_count(self) -> int:
        return int(self._bindings.lib.qstate_qubit_count(self._handle))

    def component_count(self) -> int:
        return int(self._bindings.lib.qstate_component_count(self._handle))

    def estimated_bytes(self) -> int:
        return int(self._bindings.lib.qstate_estimated_bytes(self._handle))

    def component_size(self, qubit: int) -> int:
        return int(self._bindings.lib.qstate_component_size(self._handle, qubit))

    def component_nonzero_count(self, qubit: int) -> int:
        return int(self._bindings.lib.qstate_component_nonzero_count(self._handle, qubit))

    def describe(self) -> str:
        size = int(self._bindings.lib.qstate_description_size(self._handle))
        if size <= 0:
            raise QubitNativeError(self._bindings.error())
        output = ctypes.create_string_buffer(size)
        self._bindings.check(
            self._bindings.lib.qstate_description_write(self._handle, output, size)
        )
        return output.value.decode("utf-8")

    def encode_qsc(self) -> bytes:
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
        bindings = _Bindings(library_path)
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
