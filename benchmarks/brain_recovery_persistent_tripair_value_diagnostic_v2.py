from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from qsa.causal import CausalRegister

import brain_recovery_persistent_tripair_value_diagnostic as v1


def corrected_direct_global_values(
    bridge: v1.v2.LazyWorkspaceTripairBridge,
    row: Mapping[str, float] | Sequence[float],
) -> tuple[float, ...]:
    with CausalRegister(v1.v1.QUBITS) as state:
        state.apply(bridge.plan, row)
        return tuple(float(value) for value in bridge.support.execute(state))


def build_receipt() -> dict[str, object]:
    original = v1.direct_global_values
    v1.direct_global_values = corrected_direct_global_values
    try:
        receipt = v1.build_receipt()
    finally:
        v1.direct_global_values = original

    receipt["schema_version"] = (
        "qsa.brain-recovery-persistent-tripair-value-diagnostic.v2"
    )
    receipt["iteration"] = {
        "predecessor": (
            "qsa.brain-recovery-persistent-tripair-value-diagnostic.v1"
        ),
        "predecessor_workflow_run": 30941512237,
        "predecessor_job": 92100764337,
        "predecessor_artifact_id": 8905327264,
        "predecessor_artifact_zip_sha256": (
            "21b1f53f4da421e7158e48b240c8b76fc4a747b1b918b2de9a929c926f1da1ca"
        ),
        "predecessor_reached_comparisons": False,
        "causal_failure": "missing_causal_register_import_in_diagnostic_control",
        "successor_change": (
            "Import CausalRegister directly from qsa.causal and replace only "
            "the direct-global diagnostic control constructor."
        ),
        "persistent_runtime_changed": False,
        "diagnostic_rows_changed": False,
        "scale_sweep_changed": False,
        "parity_threshold_changed": False,
    }
    receipt["gate_policy"]["predecessor_failure_preserved"] = True
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "brain-recovery-persistent-tripair-value-diagnostic-v2.json"
        ),
    )
    args = parser.parse_args()
    v1.v1.torch.set_num_threads(1)
    v1.v1.torch.set_num_interop_threads(1)
    receipt = build_receipt()
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit(
            "persistent Tripair value diagnostic v2 did not complete"
        )


if __name__ == "__main__":
    main()
