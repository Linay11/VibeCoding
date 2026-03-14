from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend_adapter.services.power118_data_augment import generate_power118_override_set
from backend_adapter.services.power118_dataset import load_power118_data
from backend_adapter.services.power118_ml_model import load_power118_model_artifacts, resolve_power118_model_paths
from backend_adapter.services.power118_service import run_power118_once


DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend_adapter" / "data" / "power118_eval"
RECORDS_JSON_NAME = "power118_eval_records.json"
RECORDS_CSV_NAME = "power118_eval_records.csv"
SUMMARY_JSON_NAME = "summary.json"
REPORT_MD_NAME = "report.md"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_status(run: dict[str, Any]) -> str:
    if run.get("adapterMode") == "compat":
        return "COMPAT"
    if run.get("feasible"):
        return "FEASIBLE"
    return "INFEASIBLE"


def _gap_vs_exact(objective_value: float | None, exact_objective_value: float | None) -> float | None:
    if objective_value is None or exact_objective_value is None:
        return None
    denominator = max(abs(exact_objective_value), 1.0)
    return float((objective_value - exact_objective_value) / denominator)


def _cost_gap(objective_value: float | None, exact_objective_value: float | None) -> float | None:
    if objective_value is None or exact_objective_value is None:
        return None
    return float(objective_value - exact_objective_value)


def _dispatch_mae(run: dict[str, Any], exact_run: dict[str, Any] | None) -> tuple[float | None, str | None]:
    if not isinstance(exact_run, dict):
        return None, "exact baseline payload unavailable"
    exact_dispatch = exact_run.get("generatorDispatchByHour")
    run_dispatch = run.get("generatorDispatchByHour")
    if exact_dispatch is None or run_dispatch is None:
        return None, "dispatch outputs unavailable in one or both runs"
    try:
        exact_frame = pd.DataFrame(exact_dispatch)
        run_frame = pd.DataFrame(run_dispatch)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"dispatch outputs could not be converted to tabular form: {exc}"
    if exact_frame.shape != run_frame.shape:
        return None, f"dispatch shape mismatch exact={exact_frame.shape} run={run_frame.shape}"
    return float((exact_frame - run_frame).abs().to_numpy().mean()), None


def build_eval_record(case_id: str, requested_mode: str, run: dict[str, Any], exact_run: dict[str, Any] | None) -> dict[str, Any]:
    exact_objective_value = exact_run.get("objectiveValue") if isinstance(exact_run, dict) else None
    objective_value = run.get("objectiveValue")
    fallback_reason = run.get("fallbackReason")
    solver_mode_used = run.get("solverModeUsed")
    adapter_mode = run.get("adapterMode")
    is_fallback = bool(fallback_reason) or adapter_mode == "compat" or (solver_mode_used not in {"", requested_mode})
    dispatch_mae, dispatch_mae_unavailable_reason = _dispatch_mae(run, exact_run)

    return {
        "caseId": case_id,
        "requestedMode": requested_mode,
        "solverModeUsed": solver_mode_used,
        "status": _derive_status(run),
        "adapterMode": adapter_mode,
        "feasible": bool(run.get("feasible", False)),
        "fallbackReason": fallback_reason,
        "isFallback": is_fallback,
        "repairApplied": run.get("repairApplied"),
        "mlConfidence": run.get("mlConfidence"),
        "runtimeMs": run.get("runtimeMs", run.get("metrics", {}).get("solveTimeMs")),
        "objectiveValue": objective_value,
        "objectiveGapVsExact": _gap_vs_exact(objective_value, exact_objective_value),
        "costGap": _cost_gap(objective_value, exact_objective_value),
        "dispatchMAE": dispatch_mae,
        "dispatchMAEUnavailableReason": dispatch_mae_unavailable_reason,
        "usedModelVersion": run.get("modelVersion"),
        "featureSchemaVersion": run.get("featureSchemaVersion"),
        "modelLoadStatus": run.get("modelLoadStatus"),
    }


def _status_counts(group: pd.DataFrame) -> dict[str, int]:
    counts = group["status"].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def summarize_eval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    summary_rows = []
    frame = pd.DataFrame(records)
    for requested_mode, group in frame.groupby("requestedMode", dropna=False):
        group = group.copy()
        runtime_values = pd.to_numeric(group["runtimeMs"], errors="coerce")
        objective_gap_values = pd.to_numeric(group["objectiveGapVsExact"], errors="coerce")
        cost_gap_values = pd.to_numeric(group["costGap"], errors="coerce")
        dispatch_mae_values = pd.to_numeric(group["dispatchMAE"], errors="coerce")
        fallback_rate = float(group["isFallback"].astype(bool).mean())
        feasible_rate = float(group["feasible"].astype(bool).mean())
        summary_rows.append(
            {
                "requestedMode": str(requested_mode),
                "runCount": int(len(group)),
                "successCount": int(group["status"].eq("FEASIBLE").sum()),
                "failureCount": int(group["status"].ne("FEASIBLE").sum()),
                "fallbackCount": int(group["isFallback"].astype(bool).sum()),
                "compatCount": int(group["adapterMode"].eq("compat").sum()),
                "statusCounts": _status_counts(group),
                "fallbackRate": fallback_rate,
                "feasibilityRate": feasible_rate,
                "averageRuntimeMs": float(runtime_values.mean()) if runtime_values.notna().any() else None,
                "objectiveGapVsExact": float(objective_gap_values.mean()) if objective_gap_values.notna().any() else None,
                "costGap": float(cost_gap_values.mean()) if cost_gap_values.notna().any() else None,
                "dispatchMAE": float(dispatch_mae_values.mean()) if dispatch_mae_values.notna().any() else None,
                "dispatchMAEUnavailableReason": None
                if dispatch_mae_values.notna().any()
                else "dispatch outputs unavailable for at least one compared mode or exact baseline",
            }
        )
    return summary_rows


def _summary_payload(
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    exact_real_available: bool,
    model_artifacts: dict[str, Any],
    requested_modes: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "evaluation": {
            "generatedAt": _utc_now_iso(),
            "caseCount": int(len({record["caseId"] for record in records})),
            "requestedModes": requested_modes,
            "exactRealBaselineAvailable": bool(exact_real_available),
            "modelLoaded": bool(model_artifacts.get("loadSuccess")),
            "modelPath": model_artifacts.get("modelPath"),
            "metadataPath": model_artifacts.get("metadataPath"),
            "modelVersion": model_artifacts.get("modelVersion"),
            "featureSchemaVersion": model_artifacts.get("featureSchemaVersion"),
            "modelLoadStatus": model_artifacts.get("loadStatus"),
            "modelLoadFailureReason": model_artifacts.get("loadFailureReason"),
            "outputDir": str(output_dir),
        },
        "modes": summary_rows,
    }


def _markdown_table(summary_rows: list[dict[str, Any]]) -> str:
    header = "| Mode | Runs | Success | Failure | Fallback | Feasible Rate | Avg Runtime (ms) | Avg Gap vs Exact | Avg Cost Gap | Avg Dispatch MAE |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = []
    for row in summary_rows:
        rows.append(
            "| "
            f"{row['requestedMode']} | "
            f"{row['runCount']} | "
            f"{row['successCount']} | "
            f"{row['failureCount']} | "
            f"{row['fallbackCount']} | "
            f"{row['feasibilityRate']:.3f} | "
            f"{row['averageRuntimeMs'] if row['averageRuntimeMs'] is not None else 'NA'} | "
            f"{row['objectiveGapVsExact'] if row['objectiveGapVsExact'] is not None else 'NA'} | "
            f"{row['costGap'] if row['costGap'] is not None else 'NA'} | "
            f"{row['dispatchMAE'] if row['dispatchMAE'] is not None else 'NA'} |"
        )
    return "\n".join([header] + rows)


def _write_report(
    output_dir: Path,
    summary_payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> Path:
    evaluation = summary_payload["evaluation"]
    report_lines = [
        "# Power-118 Evaluation Report",
        "",
        "## Run Info",
        f"- Generated at: `{evaluation['generatedAt']}`",
        f"- Case count: `{evaluation['caseCount']}`",
        f"- Requested modes: `{', '.join(evaluation['requestedModes'])}`",
        f"- Model path: `{evaluation['modelPath']}`",
        f"- Metadata path: `{evaluation['metadataPath']}`",
        f"- Model version: `{evaluation['modelVersion']}`",
        f"- Feature schema version: `{evaluation['featureSchemaVersion']}`",
        f"- Model load status: `{evaluation['modelLoadStatus']}`",
        f"- Exact real baseline available: `{evaluation['exactRealBaselineAvailable']}`",
        "",
        "## Mode Summary",
        _markdown_table(summary_rows),
        "",
        "## Limits",
    ]
    if evaluation["exactRealBaselineAvailable"]:
        report_lines.append("- Objective and cost gap metrics are based on a real feasible exact baseline.")
    else:
        report_lines.append("- Exact baseline was not a real feasible solve in this environment, so gap metrics may be null.")
    if evaluation.get("modelLoadFailureReason"):
        report_lines.append(f"- Model artifacts did not fully load: `{evaluation['modelLoadFailureReason']}`")
    else:
        report_lines.append("- Model artifacts loaded successfully for the evaluation process.")
    report_lines.extend(
        [
            "- `compat` rows indicate that the backend did not complete a real exact or hybrid solve for that case.",
            "- `dispatchMAE` stays unavailable when dispatch outputs are missing or when no exact baseline dispatch is available.",
            "",
            "## Artifact Index",
            f"- JSON records: `{RECORDS_JSON_NAME}`",
            f"- CSV records: `{RECORDS_CSV_NAME}`",
            f"- Summary JSON: `{SUMMARY_JSON_NAME}`",
            f"- Markdown report: `{REPORT_MD_NAME}`",
        ]
    )
    report_path = output_dir / REPORT_MD_NAME
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def print_summary(summary_payload: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    evaluation = summary_payload["evaluation"]
    print("Power118 evaluation summary")
    print(f"- Output dir: {evaluation['outputDir']}")
    print(f"- Model loaded: {'YES' if evaluation['modelLoaded'] else 'NO'}")
    print(f"- Model path: {evaluation['modelPath']}")
    print(f"- Exact real baseline available: {'YES' if evaluation['exactRealBaselineAvailable'] else 'NO'}")
    for row in summary_rows:
        print(
            "- "
            f"mode={row['requestedMode']} "
            f"runs={row['runCount']} "
            f"success={row['successCount']} "
            f"failure={row['failureCount']} "
            f"fallback={row['fallbackCount']} "
            f"avgRuntimeMs={row['averageRuntimeMs'] if row['averageRuntimeMs'] is not None else 'NA'} "
            f"gapVsExact={row['objectiveGapVsExact'] if row['objectiveGapVsExact'] is not None else 'NA'}"
        )
    if not evaluation["exactRealBaselineAvailable"]:
        print("- Exact baseline was not a real feasible solve in this environment, so objective-gap metrics remain unavailable.")
    if evaluation.get("modelLoadFailureReason"):
        print(f"- Model load failure reason: {evaluation['modelLoadFailureReason']}")


def evaluate_modes(
    num_cases: int,
    seed: int,
    output_dir: Path,
    time_limit_ms: int | None,
    modes: list[str],
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    require_exact_baseline: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    base_data = load_power118_data()
    overrides_list = generate_power118_override_set(base_data=base_data, n_samples=max(num_cases - 1, 0), seed=seed)
    cases = [{"caseId": "case-00000", "overrides": None}]
    for index, overrides in enumerate(overrides_list, start=1):
        cases.append({"caseId": f"case-{index:05d}", "overrides": overrides})

    model_artifacts = load_power118_model_artifacts(model_path=model_path, metadata_path=metadata_path)
    records: list[dict[str, Any]] = []
    exact_real_available = False

    for case in cases[:num_cases]:
        exact_run = run_power118_once(
            run_mode="exact",
            time_limit_ms=time_limit_ms,
            fallback_to_exact=True,
            overrides=case["overrides"],
            model_path=model_path,
            metadata_path=metadata_path,
        )
        exact_real_available = exact_real_available or (
            exact_run.get("adapterMode") == "real" and bool(exact_run.get("feasible"))
        )

        for requested_mode in modes:
            run = exact_run if requested_mode == "exact" else run_power118_once(
                run_mode=requested_mode,
                time_limit_ms=time_limit_ms,
                fallback_to_exact=True,
                overrides=case["overrides"],
                model_path=model_path,
                metadata_path=metadata_path,
            )
            records.append(build_eval_record(case["caseId"], requested_mode, run, exact_run))

    if require_exact_baseline and not exact_real_available:
        raise RuntimeError("Exact baseline was requested but no real feasible exact baseline was available.")

    summary_rows = summarize_eval_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_json_path = output_dir / RECORDS_JSON_NAME
    records_csv_path = output_dir / RECORDS_CSV_NAME
    summary_json_path = output_dir / SUMMARY_JSON_NAME

    summary_payload = _summary_payload(
        records=records,
        summary_rows=summary_rows,
        exact_real_available=exact_real_available,
        model_artifacts=model_artifacts,
        requested_modes=modes,
        output_dir=output_dir,
    )
    records_json_path.write_text(
        json.dumps(
            {
                "records": records,
                "summary": summary_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(records).to_csv(records_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    report_path = _write_report(output_dir, summary_payload, summary_rows)

    print_summary(summary_payload, summary_rows)
    print(f"Saved JSON records to {records_json_path}")
    print(f"Saved CSV records to {records_csv_path}")
    print(f"Saved JSON summary to {summary_json_path}")
    print(f"Saved Markdown report to {report_path}")
    return records, summary_payload, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate exact, hybrid, and ml modes for power-118.")
    parser.add_argument("--num-cases", type=int, default=8, help="Number of evaluation cases including the unperturbed base case.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for perturbation generation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON, CSV, summary JSON, and Markdown evaluation outputs.",
    )
    parser.add_argument("--time-limit-ms", type=int, default=None, help="Optional exact or hybrid solver time limit in milliseconds.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["exact", "hybrid", "ml"],
        help="Requested modes to evaluate.",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Optional explicit model artifact path for evaluation.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional explicit metadata path for evaluation.")
    parser.add_argument(
        "--require-exact-baseline",
        action="store_true",
        help="Fail the script if no real feasible exact baseline is available.",
    )
    args = parser.parse_args()

    modes = []
    for mode in args.modes:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"exact", "hybrid", "ml"}:
            raise ValueError(f"Unsupported mode for evaluation: {mode}")
        modes.append(normalized_mode)

    evaluate_modes(
        num_cases=max(1, args.num_cases),
        seed=args.seed,
        output_dir=args.output_dir.resolve(),
        time_limit_ms=args.time_limit_ms,
        modes=modes,
        model_path=args.model_path.resolve() if args.model_path is not None else None,
        metadata_path=args.metadata_path.resolve() if args.metadata_path is not None else None,
        require_exact_baseline=args.require_exact_baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
