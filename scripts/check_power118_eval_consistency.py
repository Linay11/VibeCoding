from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_eval_consistency(records_payload: dict[str, Any], summary_payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    records = records_payload.get("records", [])
    summary = summary_payload
    evaluation = summary.get("evaluation", {})
    mode_rows = summary.get("modes", [])
    mode_map = {row.get("requestedMode"): row for row in mode_rows}

    exact_records = [row for row in records if row.get("requestedMode") == "exact"]
    exact_available_from_records = any(
        row.get("adapterMode") == "real"
        and (bool(row.get("optimal")) or (bool(row.get("feasible")) and bool(row.get("hasIncumbent"))))
        for row in exact_records
    )
    if bool(evaluation.get("exactRealBaselineAvailable")) != exact_available_from_records:
        issues.append(
            "exact baseline availability mismatch between summary and records"
        )

    for mode_name, row in mode_map.items():
        group = [record for record in records if record.get("requestedMode") == mode_name]
        if int(row.get("runCount", 0)) != len(group):
            issues.append(f"runCount mismatch for mode={mode_name}")
        success_count = sum(1 for record in group if bool(record.get("feasible")))
        failure_count = len(group) - success_count
        fallback_count = sum(1 for record in group if bool(record.get("isFallback")))
        if int(row.get("successCount", 0)) != success_count:
            issues.append(f"successCount mismatch for mode={mode_name}")
        if int(row.get("failureCount", 0)) != failure_count:
            issues.append(f"failureCount mismatch for mode={mode_name}")
        if int(row.get("fallbackCount", 0)) != fallback_count:
            issues.append(f"fallbackCount mismatch for mode={mode_name}")

    if bool(evaluation.get("exactRealBaselineAvailable")):
        non_exact_records = [row for row in records if row.get("requestedMode") in {"hybrid", "ml"}]
        if non_exact_records and all(row.get("objectiveGapVsExact") is None for row in non_exact_records):
            issues.append("exact baseline is available but all objectiveGapVsExact values are null")

    for record in exact_records:
        if record.get("adapterMode") == "real" and record.get("status") == "COMPAT":
            issues.append("real exact record was incorrectly classified as COMPAT")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check consistency between power118 eval records and summary outputs.")
    parser.add_argument(
        "--records-path",
        type=Path,
        required=True,
        help="Path to power118_eval_records.json",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path to summary.json",
    )
    args = parser.parse_args()

    records_payload = _load_json(args.records_path.resolve())
    summary_payload = _load_json(args.summary_path.resolve())
    issues = check_eval_consistency(records_payload, summary_payload)

    if issues:
        print("Power118 eval consistency check FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Power118 eval consistency check PASSED")
    print(f"- Records: {args.records_path.resolve()}")
    print(f"- Summary: {args.summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
