from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import sympy


ROUNDS = 7
SAMPLES = 8


def evaluate(index: int) -> sympy.Rational:
    i = index % 997 + 2
    a = sympy.Rational(i + 1, i + 2)
    b = sympy.Rational(2 * i + 3, i + 5)
    c = sympy.Rational(3 * i + 7, 2 * i + 9)
    d = sympy.Rational(i + 11, 4 * i + 13)
    return (a + b) * (c - d)


def canonical(value: sympy.Rational) -> str:
    numerator, denominator = value.as_numer_denom()
    return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"


def run_round(iterations: int) -> float:
    sink = 0.0
    started = time.perf_counter()
    for index in range(iterations):
        sink += float(evaluate(index))
    elapsed = time.perf_counter() - started
    if not math.isfinite(sink):
        raise RuntimeError("SymPy exact benchmark sink became nonfinite")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qsa = json.loads(args.qsa.read_text(encoding="utf-8"))
    if qsa.get("schema") != "qsa.qmath-exact-platform.v1":
        raise RuntimeError("QSA exact benchmark schema changed")
    iterations = int(qsa["iterations"])
    if int(qsa.get("rounds", -1)) != ROUNDS or iterations <= 0:
        raise RuntimeError("QSA exact benchmark workload changed")

    for _ in range(2):
        run_round(iterations)
    rounds = [run_round(iterations) for _ in range(ROUNDS)]
    median = statistics.median(rounds)

    samples = [canonical(evaluate(index)) for index in range(SAMPLES)]
    exact_match = samples == list(qsa.get("samples") or [])
    qsa_seconds = float(qsa["median_seconds"])
    if not math.isfinite(qsa_seconds) or qsa_seconds <= 0.0:
        raise RuntimeError("QSA exact benchmark timing is invalid")
    ratio = median / qsa_seconds

    result = {
        "schema": "qsa.qmath-sympy-control.v1",
        "workload": "construct and evaluate independent exact Rational expressions (a+b)*(c-d), including float observation sink",
        "iterations_per_round": iterations,
        "rounds": ROUNDS,
        "qsa": {
            "implementation": str(qsa.get("implementation")),
            "median_seconds": qsa_seconds,
            "samples": list(qsa.get("samples") or []),
        },
        "control": {
            "implementation": "sympy.Rational",
            "version": sympy.__version__,
            "median_seconds": median,
            "samples": samples,
        },
        "exact_outputs_match": exact_match,
        "sympy_over_qsa_median_time_ratio": ratio,
        "speed_claim_accepted": bool(exact_match and ratio > 1.0),
        "claim_boundary": "Narrow in-process exact-rational platform workload only. This does not establish general CAS superiority or a universal QSA speed advantage.",
    }
    if not exact_match:
        raise RuntimeError("QSA and SymPy exact-rational outputs differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
