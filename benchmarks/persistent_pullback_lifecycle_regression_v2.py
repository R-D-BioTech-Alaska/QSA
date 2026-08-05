from __future__ import annotations

import argparse
import json
from pathlib import Path

import persistent_pullback_lifecycle_diagnostic_v1 as diagnostic


ERROR_TOLERANCE = 2.0e-12


def result_passed(result: dict[str, object]) -> bool:
    return bool(
        result.get("succeeded")
        and float(result.get("maximum_value_error", 0.0)) <= ERROR_TOLERANCE
        and float(result.get("maximum_gradient_error", 0.0)) <= ERROR_TOLERANCE
    )


def lifecycle_case_passed(case: dict[str, object]) -> bool:
    return bool(
        all(result_passed(result) for result in case["cached"])
        and all(result_passed(result) for result in case["fresh_adjoint"])
        and all(result_passed(result) for result in case["decoded_workspace"])
        and all(result_passed(result) for result in case["cotangent_variants"])
        and result_passed(case["selected_variant"])
        and case["root_qsc_equal"]
        and case["workspace_qsc_equal"]
    )


def build_receipt() -> dict[str, object]:
    receipt = diagnostic.build_receipt()
    named_cases = (
        "same_batch_generation_0",
        "new_batch_same_generation",
        "same_batch_generation_1",
        "after_workspace_refresh",
        "decoded_rollback",
    )
    case_results = {
        name: lifecycle_case_passed(receipt["lifecycle"][name])
        for name in named_cases
    }
    current_passed = bool(
        all(case_results.values())
        and receipt["lifecycle"]["discard_rejection"]["rejected"]
        and receipt["lifecycle"]["generation_after_commit"] == 1
        and receipt["lifecycle"]["workspace_ready_after_refresh"] is False
        and receipt["lifecycle"]["rollback_root_exact"]
    )
    receipt.update(
        {
            "schema_version": "qsa.persistent-pullback-lifecycle-regression.v2",
            "historical_status": {
                "known_red": True,
                "historical_failure": (
                    "A retained persistent candidate previously passed the first "
                    "pullback and failed a later pullback while root and workspace "
                    "QSC bytes remained identical."
                ),
                "historical_reported_mismatch": 0.2868323271,
                "historical_red_must_reproduce_now": False,
            },
            "current_case_results": case_results,
            "current_lifecycle_passed": current_passed,
            "interpretation": (
                "The historical persistent failure is not reproduced by the "
                "current exact-repair baseline. This receipt is a current-state "
                "regression, not proof that PR #36 itself caused the correction."
            ),
            "passed": current_passed,
        }
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("persistent-pullback-lifecycle-regression-v2.json"),
    )
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("current persistent pullback lifecycle regression failed")


if __name__ == "__main__":
    main()
