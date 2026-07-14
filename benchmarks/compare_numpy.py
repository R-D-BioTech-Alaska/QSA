"""Structured-state memory comparison with a dense NumPy statevector.

This does not claim universal superiority. It demonstrates the states the QSA
representation is specifically designed not to inflate into one global array.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qubit_native import QubitRegister

try:
    import numpy as np
except ImportError:
    np = None

def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
    number = float(value)
    unit = 0
    while number >= 1024.0 and unit < len(units) - 1:
        number /= 1024.0
        unit += 1
    return f"{number:,.2f} {units[unit]}"

def build_ghz(qubits: int) -> tuple[QubitRegister, float]:
    start = time.perf_counter()
    state = QubitRegister(qubits)
    state.h(0)
    for target in range(1, qubits):
        state.cnot(0, target)
    return state, (time.perf_counter() - start) * 1_000.0

print("Qubit State Algebra vs. mandatory dense complex128 allocation")
print("-" * 79)
print(f"{'qubits':>6}  {'QSA engine bytes':>18}  {'dense complex128':>18}  {'QSA support':>12}")
for width in (20, 30, 40, 50):
    state, elapsed = build_ghz(width)
    dense_bytes = (1 << width) * 16
    print(
        f"{width:>6}  {human_bytes(state.estimated_bytes):>18}  "
        f"{human_bytes(dense_bytes):>18}  {state.component_nonzero_count(0):>12}"
        f"   ({elapsed:.3f} ms)"
    )
    state.close()

if np is not None:
    width = 20
    start = time.perf_counter()
    dense = np.zeros(1 << width, dtype=np.complex128)
    dense[0] = 1 / math.sqrt(2)
    dense[-1] = 1 / math.sqrt(2)
    elapsed = (time.perf_counter() - start) * 1_000.0
    print("\nActual NumPy allocation check")
    print(f"20-qubit dense GHZ array: {human_bytes(dense.nbytes)} in {elapsed:.3f} ms")
else:
    print("\nNumPy is not installed; theoretical dense sizes are still exact.")

print("\nBoundary: arbitrary highly entangled states may still require exponential storage.")
