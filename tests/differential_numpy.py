"""Random-circuit differential validation against an independent NumPy reference."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from qubit_native import QubitRegister


I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)


def rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(np.complex128)


def apply_one(state: np.ndarray, qubit: int, gate: np.ndarray) -> None:
    mask = 1 << qubit
    for base in range(len(state)):
        if base & mask:
            continue
        one = base | mask
        a0, a1 = state[base], state[one]
        state[base] = gate[0, 0] * a0 + gate[0, 1] * a1
        state[one] = gate[1, 0] * a0 + gate[1, 1] * a1


def apply_two(state: np.ndarray, first: int, second: int, gate: np.ndarray) -> None:
    first_mask = 1 << first
    second_mask = 1 << second
    for base in range(len(state)):
        if base & first_mask or base & second_mask:
            continue
        indices = [base, base | second_mask, base | first_mask, base | first_mask | second_mask]
        state[indices] = gate @ state[indices]


def compare(reference: np.ndarray, engine: QubitRegister, label: str) -> None:
    candidate = np.array([engine.amplitude(i) for i in range(len(reference))], dtype=np.complex128)
    overlap = np.vdot(reference, candidate)
    fidelity = abs(overlap) ** 2
    if not math.isclose(float(fidelity), 1.0, rel_tol=0.0, abs_tol=2e-10):
        difference = np.max(np.abs(reference - candidate))
        raise AssertionError(f"{label}: fidelity={fidelity}, raw max difference={difference}")


def run(seed: int = 6142026, circuits: int = 120, depth: int = 120) -> None:
    rng = random.Random(seed)
    one_names = ("x", "y", "z", "h", "s", "t", "rx", "ry", "rz")
    two_names = ("cnot", "cz", "swap")

    for circuit in range(circuits):
        qubits = rng.randint(1, 6)
        reference = np.zeros(1 << qubits, dtype=np.complex128)
        reference[0] = 1.0
        engine = QubitRegister(qubits)

        for step in range(depth):
            if qubits == 1 or rng.random() < 0.68:
                name = rng.choice(one_names)
                q = rng.randrange(qubits)
                if name == "x":
                    engine.x(q); apply_one(reference, q, X)
                elif name == "y":
                    engine.y(q); apply_one(reference, q, Y)
                elif name == "z":
                    engine.z(q); apply_one(reference, q, Z)
                elif name == "h":
                    engine.h(q); apply_one(reference, q, H)
                elif name == "s":
                    engine.s(q); apply_one(reference, q, S)
                elif name == "t":
                    engine.t(q); apply_one(reference, q, T)
                else:
                    theta = rng.uniform(-2 * math.pi, 2 * math.pi)
                    getattr(engine, name)(q, theta)
                    apply_one(reference, q, {"rx": rx, "ry": ry, "rz": rz}[name](theta))
            else:
                first, second = rng.sample(range(qubits), 2)
                name = rng.choice(two_names)
                getattr(engine, name)(first, second)
                apply_two(reference, first, second, {"cnot": CNOT, "cz": CZ, "swap": SWAP}[name])

            if step % 23 == 0:
                compare(reference, engine, f"circuit {circuit}, step {step}")

        compare(reference, engine, f"circuit {circuit}, final")

        q = rng.randrange(qubits)
        p1 = float(sum(abs(reference[i]) ** 2 for i in range(len(reference)) if i & (1 << q)))
        sample = rng.random()
        expected = 1 if sample < p1 else 0
        actual = engine.measure(q, sample)
        if actual != expected:
            raise AssertionError(f"measurement mismatch: expected {expected}, got {actual}")
        for index in range(len(reference)):
            if ((index >> q) & 1) != expected:
                reference[index] = 0
        reference /= np.linalg.norm(reference)
        compare(reference, engine, f"circuit {circuit}, post-measurement")
        engine.close()

    print(f"Differential validation passed: {circuits} circuits x {depth} random gates.")


if __name__ == "__main__":
    run()
