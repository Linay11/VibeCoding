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


def _exact_baseline_available(run: dict[str, Any] | None) -> bool:
    if not isinstance(run, dict):
        return False
    if run.get("adapterMode") != "real":
        return False
    if bool(run.get("optimal", False)):
        return True
    return bool(run.get("feasible", False) and run.get("hasIncumbent", False))


def _derive_status(run: dict[str, Any]) -> str:
    if run.get("adapterMode") == "compat":
        return "COMPAT"
    if bool(run.get("optimal", False)):
        return "OPTIMAL"
    if bool(run.get("terminatedByTimeLimit", False)) and bool(run.get("hasIncumbent", False)):
        return "TIME_LIMIT_FEASIBLE"
    if bool(run.get("terminatedByTimeLimit", False)) and not bool(run.get("hasIncumbent", False)):
        return "TIME_LIMIT_NO_INCUMBENT"
    if bool(run.get("feasible", False)):
        return "FEASIBLE"
    return "FAILED"


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
    if not _exact_baseline_available(exact_run):
        return None, "exact baseline unavailable"
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
    exact_available = _exact_baseline_available(exact_run)
    exact_objective_value = exact_run.get("objectiveValue") if exact_available and isinstance(exact_run, dict) else None
    exact_runtime_ms = exact_run.get("runtimeMs") if exact_available and isinstance(exact_run, dict) else None
    objective_value = run.get("objectiveValue")
    fallback_reason = run.get("fallbackReason")
    solver_mode_used = run.get("solverModeUsed")
    adapter_mode = run.get("adapterMode")
    base_mode = "hybrid" if requested_mode.startswith("hybrid_") else requested_mode
    hybrid_strategy = requested_mode.split("_", 1)[1] if requested_mode.startswith("hybrid_") else None
    is_fallback = bool(fallback_reason) or adapter_mode == "compat" or (solver_mode_used not in {"", base_mode})
    fallback_to_mode = solver_mode_used if is_fallback and solver_mode_used not in {"", base_mode} else None
    dispatch_mae, dispatch_mae_unavailable_reason = _dispatch_mae(run, exact_run)

    return {
        "caseId": case_id,
        "requestedMode": run.get("requestedMode") or requested_mode,
        "baseMode": base_mode,
        "hybridStrategy": hybrid_strategy,
        "solverModeUsed": solver_mode_used,
        "status": _derive_status(run),
        "adapterMode": adapter_mode,
        "isRealSolve": adapter_mode == "real",
        "feasible": bool(run.get("feasible", False)),
        "fallbackReason": fallback_reason,
        "isFallback": is_fallback,
        "fallbackToMode": fallback_to_mode,
        "repairApplied": run.get("repairApplied"),
        "mlConfidence": run.get("mlConfidence"),
        "runtimeMs": run.get("runtimeMs", run.get("metrics", {}).get("solveTimeMs")),
        "exactRuntimeMs": exact_runtime_ms,
        "objectiveValue": objective_value,
        "objectiveGapVsExact": _gap_vs_exact(objective_value, exact_objective_value),
        "costGap": _cost_gap(objective_value, exact_objective_value),
        "dispatchMAE": dispatch_mae,
        "dispatchMAEUnavailableReason": dispatch_mae_unavailable_reason,
        "usedModelVersion": run.get("modelVersion"),
        "featureSchemaVersion": run.get("featureSchemaVersion"),
        "modelLoadStatus": run.get("modelLoadStatus"),
        "constraintAwareHybridUsed": run.get("constraintAwareHybridUsed"),
        "reducedSolveApplied": run.get("reducedSolveApplied"),
        "fixedCommitmentCount": run.get("fixedCommitmentCount"),
        "predictedActiveConstraintCount": run.get("predictedActiveConstraintCount"),
        "constraintConfidence": run.get("constraintConfidence"),
        "reducedSolveFallbackReason": run.get("reducedSolveFallbackReason"),
        "constraintScoringUsed": run.get("constraintScoringUsed"),
        "criticalConstraintCount": run.get("criticalConstraintCount"),
        "deferredConstraintCount": run.get("deferredConstraintCount"),
        "constraintReactivationCount": run.get("constraintReactivationCount"),
        "stagedSolveRounds": run.get("stagedSolveRounds"),
        "constraintAwareReductionMode": run.get("constraintAwareReductionMode"),
        "reducedModelValidated": run.get("reducedModelValidated"),
        "reductionRejectedReason": run.get("reductionRejectedReason"),
        "statusName": run.get("statusName"),
        "statusCode": run.get("statusCode"),
        "solutionCount": run.get("solutionCount"),
        "terminatedByTimeLimit": run.get("terminatedByTimeLimit"),
        "hasIncumbent": run.get("hasIncumbent"),
        "optimal": run.get("optimal"),
        "exactBaselineAvailable": exact_available,
        "noSpeedupAgainstExact": bool(
            exact_available
            and requested_mode.startswith("hybrid_")
            and run.get("runtimeMs") is not None
            and exact_runtime_ms is not None
            and float(run.get("runtimeMs")) >= float(exact_runtime_ms)
        ),
    }


def _status_counts(group: pd.DataFrame) -> dict[str, int]:
    counts = group["status"].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _reason_counts(group: pd.DataFrame, column_name: str) -> dict[str, int]:
    normalized = group[column_name].fillna("").astype(str).str.strip()
    normalized = normalized[normalized != ""]
    counts = normalized.value_counts(dropna=False).to_dict()
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
        fixed_commitment_values = pd.to_numeric(group["fixedCommitmentCount"], errors="coerce")
        predicted_active_values = pd.to_numeric(group["predictedActiveConstraintCount"], errors="coerce")
        critical_constraint_values = pd.to_numeric(group["criticalConstraintCount"], errors="coerce")
        deferred_constraint_values = pd.to_numeric(group["deferredConstraintCount"], errors="coerce")
        reactivation_values = pd.to_numeric(group["constraintReactivationCount"], errors="coerce")
        staged_round_values = pd.to_numeric(group["stagedSolveRounds"], errors="coerce")
        fallback_rate = float(group["isFallback"].astype(bool).mean())
        feasible_rate = float(group["feasible"].astype(bool).mean())
        success_count = int(group["feasible"].astype(bool).sum())
        failure_count = int((~group["feasible"].astype(bool)).sum())
        fallback_reason_counts = _reason_counts(group, "fallbackReason")
        solver_mode_counts = _reason_counts(group, "solverModeUsed")
        fallback_to_mode_counts = _reason_counts(group, "fallbackToMode")
        no_speedup_feasible_count = 0
        if str(requested_mode).startswith("hybrid_"):
            comparable = group.loc[group["feasible"].astype(bool) & group["noSpeedupAgainstExact"].astype(bool)]
            no_speedup_feasible_count = int(len(comparable))
        fallback_case_ids = group.loc[group["isFallback"].astype(bool), "caseId"].astype(str).tolist()
        summary_rows.append(
            {
                "requestedMode": str(requested_mode),
                "runCount": int(len(group)),
                "successCount": success_count,
                "failureCount": failure_count,
                "fallbackCount": int(group["isFallback"].astype(bool).sum()),
                "compatCount": int(group["adapterMode"].eq("compat").sum()),
                "statusCounts": _status_counts(group),
                "solverModeUsedCounts": solver_mode_counts,
                "fallbackReasonCounts": fallback_reason_counts,
                "fallbackToModeCounts": fallback_to_mode_counts,
                "fallbackCaseIds": fallback_case_ids,
                "fallbackRate": fallback_rate,
                "feasibilityRate": feasible_rate,
                "averageRuntimeMs": float(runtime_values.mean()) if runtime_values.notna().any() else None,
                "objectiveGapVsExact": float(objective_gap_values.mean()) if objective_gap_values.notna().any() else None,
                "costGap": float(cost_gap_values.mean()) if cost_gap_values.notna().any() else None,
                "dispatchMAE": float(dispatch_mae_values.mean()) if dispatch_mae_values.notna().any() else None,
                "dispatchMAEUnavailableReason": None
                if dispatch_mae_values.notna().any()
                else "dispatch outputs unavailable for at least one compared mode or exact baseline",
                "exactFallbackCount": int(group["solverModeUsed"].fillna("").eq("exact").sum()) if str(requested_mode).startswith("hybrid_") else 0,
                "noSpeedupFeasibleCount": no_speedup_feasible_count if str(requested_mode).startswith("hybrid_") else 0,
                "constraintAwareHybridUsedCount": int(group["constraintAwareHybridUsed"].fillna(False).astype(bool).sum()),
                "reducedSolveAppliedCount": int(group["reducedSolveApplied"].fillna(False).astype(bool).sum()),
                "reductionRejectedCount": int(group["reductionRejectedReason"].fillna("").astype(str).str.strip().ne("").sum()),
                "averageFixedCommitmentCount": float(fixed_commitment_values.mean()) if fixed_commitment_values.notna().any() else None,
                "averagePredictedActiveConstraintCount": float(predicted_active_values.mean()) if predicted_active_values.notna().any() else None,
                "averageCriticalConstraintCount": float(critical_constraint_values.mean()) if critical_constraint_values.notna().any() else None,
                "averageDeferredConstraintCount": float(deferred_constraint_values.mean()) if deferred_constraint_values.notna().any() else None,
                "averageConstraintReactivationCount": float(reactivation_values.mean()) if reactivation_values.notna().any() else None,
                "averageStagedSolveRounds": float(staged_round_values.mean()) if staged_round_values.notna().any() else None,
            }
        )
    return summary_rows


def _summary_payload(
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    exact_real_available: bool,
    exact_baseline_status: str | None,
    exact_baseline_time_limited: bool,
    exact_baseline_has_incumbent: bool,
    model_artifacts: dict[str, Any],
    requested_modes: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    hybrid_rows = [row for row in summary_rows if str(row["requestedMode"]).startswith("hybrid_")]
    hybrid_fallback_reason_counts: dict[str, int] = {}
    hybrid_fallback_to_mode_counts: dict[str, int] = {}
    for row in hybrid_rows:
        for key, value in row.get("fallbackReasonCounts", {}).items():
            hybrid_fallback_reason_counts[key] = hybrid_fallback_reason_counts.get(key, 0) + int(value)
        for key, value in row.get("fallbackToModeCounts", {}).items():
            hybrid_fallback_to_mode_counts[key] = hybrid_fallback_to_mode_counts.get(key, 0) + int(value)
    return {
        "evaluation": {
            "generatedAt": _utc_now_iso(),
            "caseCount": int(len({record["caseId"] for record in records})),
            "requestedModes": requested_modes,
            "exactRealBaselineAvailable": bool(exact_real_available),
            "exactBaselineStatus": exact_baseline_status,
            "exactBaselineTimeLimited": bool(exact_baseline_time_limited),
            "exactBaselineHasIncumbent": bool(exact_baseline_has_incumbent),
            "modelLoaded": bool(model_artifacts.get("loadSuccess")),
            "modelPath": model_artifacts.get("modelPath"),
            "metadataPath": model_artifacts.get("metadataPath"),
            "modelVersion": model_artifacts.get("modelVersion"),
            "featureSchemaVersion": model_artifacts.get("featureSchemaVersion"),
            "modelLoadStatus": model_artifacts.get("loadStatus"),
            "modelLoadFailureReason": model_artifacts.get("loadFailureReason"),
            "outputDir": str(output_dir),
            "hybridFallbackReasonCounts": hybrid_fallback_reason_counts,
            "hybridFallbackToModeCounts": hybrid_fallback_to_mode_counts,
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
        f"- Exact baseline status: `{evaluation['exactBaselineStatus']}`",
        f"- Exact baseline time-limited: `{evaluation['exactBaselineTimeLimited']}`",
        f"- Exact baseline has incumbent: `{evaluation['exactBaselineHasIncumbent']}`",
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
            "## Hybrid Fallback Detail",
        ]
    )
    hybrid_rows = [row for row in summary_rows if str(row["requestedMode"]).startswith("hybrid_")]
    if hybrid_rows:
        report_lines.extend(
            [
                f"- Hybrid fallback reason distribution: `{evaluation['hybridFallbackReasonCounts']}`",
                f"- Hybrid fallback-to-mode distribution: `{evaluation['hybridFallbackToModeCounts']}`",
                "",
            ]
        )
        for hybrid_row in hybrid_rows:
            report_lines.extend(
                [
                    f"### {hybrid_row['requestedMode']}",
                    f"- fallback count: `{hybrid_row['fallbackCount']}`",
                    f"- fallback reason distribution: `{hybrid_row['fallbackReasonCounts']}`",
                    f"- fallback-to-mode distribution: `{hybrid_row['fallbackToModeCounts']}`",
                    f"- solver mode distribution: `{hybrid_row['solverModeUsedCounts']}`",
                    f"- fallback caseIds: `{hybrid_row['fallbackCaseIds']}`",
                    f"- cases that ended up using exact: `{hybrid_row['exactFallbackCount']}`",
                    f"- feasible cases with no demonstrated speedup signal: `{hybrid_row['noSpeedupFeasibleCount']}`",
                    "",
                ]
            )
    else:
        report_lines.extend(
            [
                "- Hybrid mode was not requested in this evaluation run.",
                "",
            ]
        )
    report_lines.extend(
        [
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
    print(f"- Exact baseline status: {evaluation['exactBaselineStatus']}")
    print(f"- Exact baseline time-limited: {'YES' if evaluation['exactBaselineTimeLimited'] else 'NO'}")
    print(f"- Exact baseline has incumbent: {'YES' if evaluation['exactBaselineHasIncumbent'] else 'NO'}")
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
        if str(row["requestedMode"]).startswith("hybrid_"):
            print(f"  fallbackReasonCounts={row['fallbackReasonCounts']}")
            print(f"  fallbackToModeCounts={row['fallbackToModeCounts']}")
            print(f"  solverModeUsedCounts={row['solverModeUsedCounts']}")
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
    exact_baseline_status = None
    exact_baseline_time_limited = False
    exact_baseline_has_incumbent = False

    for case in cases[:num_cases]:
        exact_run = run_power118_once(
            run_mode="exact",
            time_limit_ms=time_limit_ms,
            fallback_to_exact=True,
            overrides=case["overrides"],
            model_path=model_path,
            metadata_path=metadata_path,
        )
        baseline_available = _exact_baseline_available(exact_run)
        exact_real_available = exact_real_available or baseline_available
        if exact_baseline_status is None:
            exact_baseline_status = _derive_status(exact_run)
            exact_baseline_time_limited = bool(exact_run.get("terminatedByTimeLimit", False))
            exact_baseline_has_incumbent = bool(exact_run.get("hasIncumbent", False))

        requested_runs: list[tuple[str, str, str | None]] = []
        for requested_mode in modes:
            if requested_mode == "hybrid":
                requested_runs.append(("hybrid_warm_start", "hybrid_warm_start", None))
            elif requested_mode in {"exact", "hybrid_warm_start", "hybrid_constraint_aware_v2", "hybrid_constraint_aware_v3", "ml"}:
                requested_runs.append((requested_mode, requested_mode, None))

        for requested_label, requested_mode, hybrid_strategy in requested_runs:
            run = exact_run if requested_mode == "exact" else run_power118_once(
                run_mode=requested_mode,
                time_limit_ms=time_limit_ms,
                fallback_to_exact=True,
                overrides=case["overrides"],
                model_path=model_path,
                metadata_path=metadata_path,
                hybrid_strategy=hybrid_strategy or "warm_start",
            )
            records.append(build_eval_record(case["caseId"], requested_label, run, exact_run))

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
        exact_baseline_status=exact_baseline_status,
        exact_baseline_time_limited=exact_baseline_time_limited,
        exact_baseline_has_incumbent=exact_baseline_has_incumbent,
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
        default=["exact", "hybrid_warm_start", "hybrid_constraint_aware_v2", "hybrid_constraint_aware_v3", "ml"],
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
