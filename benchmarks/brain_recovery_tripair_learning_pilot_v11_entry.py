from __future__ import annotations

import argparse
import json
from pathlib import Path

import brain_recovery_tripair_learning_pilot_v11 as pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("brain-recovery-tripair-learning-pilot-v11.json"),
    )
    args = parser.parse_args()
    pilot.torch.set_num_threads(1)
    pilot.torch.set_num_interop_threads(1)
    # All initial, training, and final evaluations must use the same corrected
    # Pauli convention. run_arm preserves and restores this exact reference.
    pilot.v1.apply_pauli = pilot.v3.corrected_apply_pauli
    receipt = pilot.build_receipt()
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("bounded Tripair learning pilot failed")


if __name__ == "__main__":
    main()
