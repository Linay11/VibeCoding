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
V3_REDUCTION_STRENGTHS = (0.1, 0.2, 0.4)
DEFAULT_ABLATION_PRIMARY_MODE_ORDER = (
    "hybrid_constraint_aware_v3",
    "hybrid_constraint_aware_v2",
    "hybrid_warm_start",
    "ml",
    "exact",
)
DEFAULT_OBJECTIVE_ABLATION_ORDER = ("proxy-only", "mixed", "exact-priority")
DEFAULT_REPRESENTATION_ABLATION_ORDER = ("inst-only", "abs-only", "inst+abs")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or "variant"))
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "variant"


def _load_variant_specs(
    variant_config_path: Path | None,
    default_model_path: str | Path | None,
    default_metadata_path: str | Path | None,
) -> list[dict[str, Any]]:
    if variant_config_path is None:
        return []
    payload = json.loads(variant_config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        variant_items = payload.get("variants", [])
    else:
        variant_items = payload
    if not isinstance(variant_items, list):
        raise ValueError("variant config payload must be a list or {\"variants\": [...]} object")

    normalized_specs: list[dict[str, Any]] = []
    config_base_dir = variant_config_path.parent if variant_config_path is not None else Path.cwd()

    def _resolve_optional_path(value: Any, fallback: str | Path | None) -> str | Path | None:
        candidate = value if value is not None else fallback
        if candidate is None:
            return None
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (config_base_dir / candidate_path).resolve()
        else:
            candidate_path = candidate_path.resolve()
        return candidate_path

    for index, item in enumerate(variant_items, start=1):
        if not isinstance(item, dict):
            continue
        model_variant = str(item.get("modelVariant") or item.get("name") or f"variant-{index:02d}")
        normalized_specs.append(
            {
                "modelVariant": model_variant,
                "modelPath": _resolve_optional_path(item.get("modelPath"), default_model_path),
                "metadataPath": _resolve_optional_path(item.get("metadataPath"), default_metadata_path),
                "constraintTrainingObjective": item.get("constraintTrainingObjective"),
                "featureAblationMode": item.get("featureAblationMode"),
                "criticalClassificationThreshold": item.get("criticalClassificationThreshold"),
                "constraintAuxRankingModelEnabled": item.get("constraintAuxRankingModelEnabled"),
                "runEnabled": bool(item.get("runEnabled", True)),
                "note": item.get("note"),
            }
        )
    return normalized_specs


def _resolve_variant_tags(
    model_artifacts: dict[str, Any],
    variant_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_overrides = variant_overrides or {}
    metadata = model_artifacts.get("metadata") if isinstance(model_artifacts.get("metadata"), dict) else {}
    model_variant = (
        variant_overrides.get("modelVariant")
        or metadata.get("modelVariant")
        or metadata.get("modelVersion")
        or model_artifacts.get("modelVersion")
        or "default"
    )
    feature_mode = (
        variant_overrides.get("featureAblationMode")
        or metadata.get("featureAblationMode")
        or metadata.get("featureAblationModeEffective")
        or "unknown"
    )
    objective = (
        variant_overrides.get("constraintTrainingObjective")
        or metadata.get("constraintTrainingObjective")
        or "unknown"
    )
    threshold = variant_overrides.get("criticalClassificationThreshold")
    if threshold is None:
        threshold = metadata.get("criticalClassificationThreshold")
    try:
        threshold_value = float(threshold) if threshold is not None else None
    except Exception:
        threshold_value = None
    aux_ranking = variant_overrides.get("constraintAuxRankingModelEnabled")
    if aux_ranking is None:
        aux_ranking = metadata.get("constraintAuxRankingModelEnabled")
    return {
        "modelVariant": str(model_variant),
        "constraintTrainingObjective": str(objective),
        "featureAblationMode": str(feature_mode),
        "exactLabelCoverage": _as_float(metadata.get("exactLabelCoverage")),
        "proxyLabelCoverage": _as_float(metadata.get("proxyLabelCoverage")),
        "criticalClassificationThreshold": threshold_value,
        "constraintAuxRankingModelEnabled": bool(aux_ranking) if aux_ranking is not None else None,
        "modelVersion": model_artifacts.get("modelVersion"),
        "featureSchemaVersion": model_artifacts.get("featureSchemaVersion"),
        "modelPath": model_artifacts.get("modelPath"),
        "metadataPath": model_artifacts.get("metadataPath"),
    }


def _format_metric(value: Any) -> str:
    return str(value) if value is not None else "NA"


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


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _mode_family(requested_mode: str) -> str:
    if requested_mode == "exact":
        return "exact"
    if requested_mode == "ml":
        return "ml"
    if requested_mode == "hybrid_warm_start":
        return "hybrid_warm_start"
    if requested_mode.startswith("hybrid_constraint_aware") or requested_mode == "hybrid_constraint_aware":
        return "hybrid_constraint_aware"
    if requested_mode.startswith("hybrid_"):
        return "hybrid"
    return requested_mode


def _constraint_eval_fields(run: dict[str, Any]) -> dict[str, Any]:
    constraint_family = run.get("constraintReductionFamily") or run.get("constraintAwareReductionMode")
    reduction_rate = _as_float(run.get("constraintReductionRate"))
    if reduction_rate is None:
        reduction_rate = _as_float(run.get("constraintReductionRatio"))

    critical_count = _as_int(run.get("criticalConstraintCount"))
    deferred_count = _as_int(run.get("deferredConstraintCount"))
    fixed_count = _as_int(run.get("fixedCommitmentCount"))
    fixed_ratio = _as_float(run.get("fixedBinaryRatio"))
    used_reduction = run.get("usedConstraintReduction")
    if used_reduction is None:
        used_reduction = bool(run.get("reducedSolveApplied", False))

    candidate_count = _as_int(run.get("constraintCandidateCount"))
    if candidate_count is None and critical_count is not None and deferred_count is not None:
        candidate_count = critical_count + deferred_count
    if candidate_count is None and fixed_count is not None and fixed_ratio is not None and fixed_ratio > 0:
        candidate_count = int(round(float(fixed_count) / float(fixed_ratio)))
    if candidate_count is None and fixed_count is not None:
        commitment = run.get("unitCommitmentByHour")
        if isinstance(commitment, list) and commitment and isinstance(commitment[0], list):
            candidate_count = int(len(commitment) * len(commitment[0]))

    selected_count = _as_int(run.get("constraintSelectedCount"))
    if selected_count is None and critical_count is not None and bool(run.get("constraintScoringUsed")):
        selected_count = critical_count
    if selected_count is None and fixed_count is not None and str(constraint_family or "") == "fixed_commitment_mask":
        selected_count = fixed_count
    if selected_count is None and fixed_count is not None and bool(used_reduction):
        selected_count = fixed_count

    if reduction_rate is None and candidate_count is not None and candidate_count > 0 and selected_count is not None:
        if str(constraint_family or "") == "critical_constraint_subset":
            reduction_rate = float((candidate_count - selected_count) / candidate_count)
        elif str(constraint_family or "") == "fixed_commitment_mask":
            reduction_rate = float(selected_count / candidate_count)

    critical_prediction_available = run.get("criticalPredictionAvailable")
    if critical_prediction_available is None:
        critical_prediction_available = bool(run.get("constraintScoringUsed", False))

    # Be conservative: only mark label availability true when it is explicitly provided.
    critical_label_available = run.get("criticalLabelAvailable")
    if critical_label_available is None:
        critical_label_available = False
    else:
        critical_label_available = bool(critical_label_available)

    critical_exact_label_available = run.get("criticalExactLabelAvailable")
    if critical_exact_label_available is None:
        critical_exact_label_available = False
    else:
        critical_exact_label_available = bool(critical_exact_label_available)

    return {
        "usedConstraintReduction": bool(used_reduction),
        "constraintReductionFamily": constraint_family,
        "constraintCandidateCount": candidate_count,
        "constraintSelectedCount": selected_count,
        "constraintReductionRate": reduction_rate,
        "criticalLabelAvailable": critical_label_available,
        "criticalExactLabelAvailable": critical_exact_label_available,
        "criticalPredictionAvailable": bool(critical_prediction_available),
        "criticalSelectionPrecision": _as_float(run.get("criticalSelectionPrecision")),
        "criticalSelectionRecall": _as_float(run.get("criticalSelectionRecall")),
    }


def build_eval_record(case_id: str, requested_mode: str, run: dict[str, Any], exact_run: dict[str, Any] | None) -> dict[str, Any]:
    exact_available = _exact_baseline_available(exact_run)
    exact_objective_value = exact_run.get("objectiveValue") if exact_available and isinstance(exact_run, dict) else None
    exact_runtime_ms = exact_run.get("runtimeMs") if exact_available and isinstance(exact_run, dict) else None
    objective_value = run.get("objectiveValue")
    fallback_reason = run.get("fallbackReason")
    solver_mode_used = run.get("solverModeUsed")
    solver_mode_used_normalized = str(solver_mode_used or "").strip()
    adapter_mode = run.get("adapterMode")
    base_mode = requested_mode
    hybrid_strategy = requested_mode.split("_", 1)[1] if requested_mode.startswith("hybrid_") else None
    comparison_mode = "hybrid" if requested_mode.startswith("hybrid_") else requested_mode
    is_fallback = bool(fallback_reason) or adapter_mode == "compat" or (
        bool(solver_mode_used_normalized) and solver_mode_used_normalized != comparison_mode
    )
    fallback_to_mode = (
        solver_mode_used_normalized
        if is_fallback and solver_mode_used_normalized and solver_mode_used_normalized != comparison_mode
        else None
    )
    dispatch_mae, dispatch_mae_unavailable_reason = _dispatch_mae(run, exact_run)
    constraint_eval = _constraint_eval_fields(run)
    is_real_solver_result = bool(adapter_mode == "real" and solver_mode_used_normalized in {"exact", "hybrid"})

    return {
        "caseId": case_id,
        "requestedMode": run.get("requestedMode") or requested_mode,
        "modeFamily": _mode_family(str(run.get("requestedMode") or requested_mode)),
        "baseMode": base_mode,
        "hybridStrategy": hybrid_strategy,
        "solverModeUsed": solver_mode_used,
        "status": _derive_status(run),
        "adapterMode": adapter_mode,
        "isRealSolve": adapter_mode == "real",
        "isRealSolverResult": is_real_solver_result,
        "feasible": bool(run.get("feasible", False)),
        "fallbackReason": fallback_reason,
        "isFallback": is_fallback,
        "fallbackOccurred": is_fallback,
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
        "reductionStrength": run.get("reductionStrength"),
        "criticalTopKRatio": run.get("criticalTopKRatio"),
        "statusName": run.get("statusName"),
        "statusCode": run.get("statusCode"),
        "solutionCount": run.get("solutionCount"),
        "terminatedByTimeLimit": run.get("terminatedByTimeLimit"),
        "hasIncumbent": run.get("hasIncumbent"),
        "optimal": run.get("optimal"),
        "exactBaselineAvailable": exact_available,
        **constraint_eval,
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
        reduction_strength_values = pd.to_numeric(group["reductionStrength"], errors="coerce")
        reduction_rate_values = pd.to_numeric(group["constraintReductionRate"], errors="coerce")
        critical_precision_values = pd.to_numeric(group["criticalSelectionPrecision"], errors="coerce")
        critical_recall_values = pd.to_numeric(group["criticalSelectionRecall"], errors="coerce")
        fallback_rate = float(group["isFallback"].astype(bool).mean())
        feasible_rate = float(group["feasible"].astype(bool).mean())
        success_count = int(group["feasible"].astype(bool).sum())
        failure_count = int((~group["feasible"].astype(bool)).sum())
        fallback_occurred = group["fallbackOccurred"] if "fallbackOccurred" in group.columns else group["isFallback"]
        no_fallback_group = group.loc[~fallback_occurred.astype(bool)].copy()
        no_fallback_feasible_rate = (
            float(no_fallback_group["feasible"].astype(bool).mean())
            if not no_fallback_group.empty
            else None
        )
        no_fallback_runtime_gain_vs_exact = None
        if not no_fallback_group.empty:
            no_fallback_runtime = pd.to_numeric(no_fallback_group["runtimeMs"], errors="coerce")
            no_fallback_exact_runtime = pd.to_numeric(no_fallback_group["exactRuntimeMs"], errors="coerce")
            valid_runtime = no_fallback_runtime.notna() & no_fallback_exact_runtime.notna() & (no_fallback_exact_runtime.abs() > 1e-9)
            if valid_runtime.any():
                runtime_gain_series = (
                    (no_fallback_exact_runtime[valid_runtime] - no_fallback_runtime[valid_runtime])
                    / no_fallback_exact_runtime[valid_runtime]
                )
                if runtime_gain_series.notna().any():
                    no_fallback_runtime_gain_vs_exact = float(runtime_gain_series.mean())
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
                "reductionStrength": float(reduction_strength_values.mean()) if reduction_strength_values.notna().any() else None,
                "averageConstraintReductionRate": float(reduction_rate_values.mean()) if reduction_rate_values.notna().any() else None,
                "criticalSelectionPrecision": float(critical_precision_values.mean()) if critical_precision_values.notna().any() else None,
                "criticalSelectionRecall": float(critical_recall_values.mean()) if critical_recall_values.notna().any() else None,
                "criticalPrecisionSampleCount": int(critical_precision_values.notna().sum()),
                "criticalRecallSampleCount": int(critical_recall_values.notna().sum()),
                "criticalMetricsAvailableCount": int((critical_precision_values.notna() & critical_recall_values.notna()).sum()),
                "noFallbackFeasibilityRate": no_fallback_feasible_rate,
                "noFallbackRuntimeGainVsExact": no_fallback_runtime_gain_vs_exact,
                "criticalPredictionAvailableCount": int(group["criticalPredictionAvailable"].fillna(False).astype(bool).sum())
                if "criticalPredictionAvailable" in group.columns
                else 0,
                "criticalLabelAvailableCount": int(group["criticalLabelAvailable"].fillna(False).astype(bool).sum())
                if "criticalLabelAvailable" in group.columns
                else 0,
                "criticalExactLabelAvailableCount": int(group["criticalExactLabelAvailable"].fillna(False).astype(bool).sum())
                if "criticalExactLabelAvailable" in group.columns
                else 0,
            }
        )
    return summary_rows


def _v3_tradeoff_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame([row for row in records if row.get("requestedMode") == "hybrid_constraint_aware_v3"])
    if frame.empty:
        return []
    tradeoff_rows: list[dict[str, Any]] = []
    for reduction_strength, group in frame.groupby("reductionStrength", dropna=False):
        runtime_values = pd.to_numeric(group["runtimeMs"], errors="coerce")
        gap_values = pd.to_numeric(group["objectiveGapVsExact"], errors="coerce")
        critical_values = pd.to_numeric(group["criticalConstraintCount"], errors="coerce")
        deferred_values = pd.to_numeric(group["deferredConstraintCount"], errors="coerce")
        tradeoff_rows.append(
            {
                "reductionStrength": float(reduction_strength) if reduction_strength == reduction_strength else None,
                "runCount": int(len(group)),
                "averageRuntimeMs": float(runtime_values.mean()) if runtime_values.notna().any() else None,
                "objectiveGapVsExact": float(gap_values.mean()) if gap_values.notna().any() else None,
                "fallbackRate": float(group["isFallback"].astype(bool).mean()),
                "averageCriticalConstraintCount": float(critical_values.mean()) if critical_values.notna().any() else None,
                "averageDeferredConstraintCount": float(deferred_values.mean()) if deferred_values.notna().any() else None,
            }
        )
    return sorted(tradeoff_rows, key=lambda item: (item["reductionStrength"] is None, item["reductionStrength"]))


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
    variant_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_tags = variant_tags or {}
    hybrid_rows = [row for row in summary_rows if str(row["requestedMode"]).startswith("hybrid_")]
    hybrid_fallback_reason_counts: dict[str, int] = {}
    hybrid_fallback_to_mode_counts: dict[str, int] = {}
    for row in hybrid_rows:
        for key, value in row.get("fallbackReasonCounts", {}).items():
            hybrid_fallback_reason_counts[key] = hybrid_fallback_reason_counts.get(key, 0) + int(value)
        for key, value in row.get("fallbackToModeCounts", {}).items():
            hybrid_fallback_to_mode_counts[key] = hybrid_fallback_to_mode_counts.get(key, 0) + int(value)
    mode_map = {row["requestedMode"]: row for row in summary_rows}
    warm_runtime = mode_map.get("hybrid_warm_start", {}).get("averageRuntimeMs")
    v2_runtime = mode_map.get("hybrid_constraint_aware_v2", {}).get("averageRuntimeMs")
    v3_runtime = mode_map.get("hybrid_constraint_aware_v3", {}).get("averageRuntimeMs")
    if v3_runtime is not None and warm_runtime is not None:
        mode_map["hybrid_constraint_aware_v3"]["hybridVsWarmStartRuntimeDelta"] = float(v3_runtime - warm_runtime)
    if v3_runtime is not None and v2_runtime is not None:
        mode_map["hybrid_constraint_aware_v3"]["hybridVsConstraintAwareV2RuntimeDelta"] = float(v3_runtime - v2_runtime)
    v3_tradeoff = _v3_tradeoff_records(records)
    family_records: list[dict[str, Any]] = []
    for record in records:
        family_record = dict(record)
        family_record["requestedMode"] = str(record.get("modeFamily") or _mode_family(str(record.get("requestedMode") or "")))
        family_records.append(family_record)
    mode_family_summary = summarize_eval_records(family_records)
    return {
        "evaluation": {
            "generatedAt": _utc_now_iso(),
            "caseCount": int(len({record["caseId"] for record in records})),
            "requestedModes": requested_modes,
            "modelVariant": variant_tags.get("modelVariant"),
            "constraintTrainingObjective": variant_tags.get("constraintTrainingObjective"),
            "featureAblationMode": variant_tags.get("featureAblationMode"),
            "exactLabelCoverage": variant_tags.get("exactLabelCoverage"),
            "proxyLabelCoverage": variant_tags.get("proxyLabelCoverage"),
            "criticalClassificationThreshold": variant_tags.get("criticalClassificationThreshold"),
            "constraintAuxRankingModelEnabled": variant_tags.get("constraintAuxRankingModelEnabled"),
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
        "modeFamilies": mode_family_summary,
        "v3ReductionTradeoff": v3_tradeoff,
    }


def _markdown_table(summary_rows: list[dict[str, Any]]) -> str:
    header = "| Mode | Runs | Success | Failure | Fallback | Feasible Rate | Avg Runtime (ms) | Avg Gap vs Exact | Avg Constraint Reduction | Critical P | Critical R | No-Fallback Feasible | No-Fallback Runtime Gain |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
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
            f"{row['averageConstraintReductionRate'] if row.get('averageConstraintReductionRate') is not None else 'NA'} | "
            f"{row['criticalSelectionPrecision'] if row.get('criticalSelectionPrecision') is not None else 'NA'} | "
            f"{row['criticalSelectionRecall'] if row.get('criticalSelectionRecall') is not None else 'NA'} | "
            f"{row['noFallbackFeasibilityRate'] if row.get('noFallbackFeasibilityRate') is not None else 'NA'} | "
            f"{row['noFallbackRuntimeGainVsExact'] if row.get('noFallbackRuntimeGainVsExact') is not None else 'NA'} |"
        )
    return "\n".join([header] + rows)


def _markdown_ablation_table(summary_rows: list[dict[str, Any]]) -> str:
    rows = [row for row in summary_rows if row["requestedMode"] in {"hybrid_warm_start", "hybrid_constraint_aware_v2", "hybrid_constraint_aware_v3"}]
    header = "| Method | Avg Runtime (ms) | Gap vs Exact | Fallback Count | Avg Reduction | Avg Critical | Avg Deferred | Avg Rounds | No-Fallback Gain |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = []
    for row in rows:
        body.append(
            "| "
            f"{row['requestedMode']} | "
            f"{row['averageRuntimeMs'] if row['averageRuntimeMs'] is not None else 'NA'} | "
            f"{row['objectiveGapVsExact'] if row['objectiveGapVsExact'] is not None else 'NA'} | "
            f"{row['fallbackCount']} | "
            f"{row['averageConstraintReductionRate'] if row.get('averageConstraintReductionRate') is not None else 'NA'} | "
            f"{row['averageCriticalConstraintCount'] if row.get('averageCriticalConstraintCount') is not None else 'NA'} | "
            f"{row['averageDeferredConstraintCount'] if row.get('averageDeferredConstraintCount') is not None else 'NA'} | "
            f"{row['averageStagedSolveRounds'] if row.get('averageStagedSolveRounds') is not None else 'NA'} | "
            f"{row['noFallbackRuntimeGainVsExact'] if row.get('noFallbackRuntimeGainVsExact') is not None else 'NA'} |"
        )
    return "\n".join([header] + body)


def _markdown_v3_tradeoff_table(v3_tradeoff: list[dict[str, Any]]) -> str:
    header = "| reductionStrength | Avg Runtime (ms) | Gap vs Exact | Fallback Rate | Avg Critical | Avg Deferred |\n|---|---:|---:|---:|---:|---:|"
    body = []
    for row in v3_tradeoff:
        body.append(
            "| "
            f"{row['reductionStrength']} | "
            f"{row['averageRuntimeMs'] if row['averageRuntimeMs'] is not None else 'NA'} | "
            f"{row['objectiveGapVsExact'] if row['objectiveGapVsExact'] is not None else 'NA'} | "
            f"{row['fallbackRate']:.3f} | "
            f"{row['averageCriticalConstraintCount'] if row['averageCriticalConstraintCount'] is not None else 'NA'} | "
            f"{row['averageDeferredConstraintCount'] if row['averageDeferredConstraintCount'] is not None else 'NA'} |"
        )
    return "\n".join([header] + body)


def _variant_row_from_mode_row(
    mode_row: dict[str, Any] | None,
    variant_tags: dict[str, Any],
    status: str,
    source_mode: str | None = None,
) -> dict[str, Any]:
    mode_row = mode_row or {}
    return {
        "modelVariant": variant_tags.get("modelVariant"),
        "constraintTrainingObjective": variant_tags.get("constraintTrainingObjective"),
        "featureAblationMode": variant_tags.get("featureAblationMode"),
        "exactLabelCoverage": variant_tags.get("exactLabelCoverage"),
        "proxyLabelCoverage": variant_tags.get("proxyLabelCoverage"),
        "criticalClassificationThreshold": variant_tags.get("criticalClassificationThreshold"),
        "constraintAuxRankingModelEnabled": variant_tags.get("constraintAuxRankingModelEnabled"),
        "status": status,
        "sourceMode": source_mode or mode_row.get("requestedMode"),
        "runCount": mode_row.get("runCount"),
        "feasibilityRate": mode_row.get("feasibilityRate"),
        "fallbackRate": mode_row.get("fallbackRate"),
        "averageRuntimeMs": mode_row.get("averageRuntimeMs"),
        "objectiveGapVsExact": mode_row.get("objectiveGapVsExact"),
        "averageConstraintReductionRate": mode_row.get("averageConstraintReductionRate"),
        "criticalSelectionPrecision": mode_row.get("criticalSelectionPrecision"),
        "criticalSelectionRecall": mode_row.get("criticalSelectionRecall"),
        "noFallbackFeasibilityRate": mode_row.get("noFallbackFeasibilityRate"),
        "noFallbackRuntimeGainVsExact": mode_row.get("noFallbackRuntimeGainVsExact"),
    }


def _select_primary_mode_row(mode_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not mode_rows:
        return None
    mode_map = {str(row.get("requestedMode")): row for row in mode_rows}
    for preferred_mode in DEFAULT_ABLATION_PRIMARY_MODE_ORDER:
        if preferred_mode in mode_map:
            return mode_map[preferred_mode]
    return mode_rows[0]


def _pick_best_variant_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    status_priority = {
        "ran": 0,
        "configured_not_run": 1,
        "not_configured": 2,
    }

    def _row_rank(row: dict[str, Any]) -> tuple[int, float]:
        status = str(row.get("status") or "not_configured")
        runtime = _as_float(row.get("averageRuntimeMs"))
        runtime_rank = runtime if runtime is not None else float("inf")
        return status_priority.get(status, 9), runtime_rank

    return sorted(rows, key=_row_rank)[0]


def _objective_ablation_rows(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for objective in DEFAULT_OBJECTIVE_ABLATION_ORDER:
        objective_rows = [
            row for row in variant_rows
            if str(row.get("constraintTrainingObjective")) == objective
        ]
        best_row = _pick_best_variant_row(objective_rows)
        if best_row is not None:
            rows.append(best_row)
            continue
        rows.append(
            {
                "modelVariant": f"{objective}-unconfigured",
                "constraintTrainingObjective": objective,
                "featureAblationMode": "unknown",
                "exactLabelCoverage": None,
                "proxyLabelCoverage": None,
                "criticalClassificationThreshold": None,
                "constraintAuxRankingModelEnabled": None,
                "status": "not_configured",
                "sourceMode": None,
                "runCount": 0,
                "feasibilityRate": None,
                "fallbackRate": None,
                "averageRuntimeMs": None,
                "objectiveGapVsExact": None,
                "averageConstraintReductionRate": None,
                "criticalSelectionPrecision": None,
                "criticalSelectionRecall": None,
                "noFallbackFeasibilityRate": None,
                "noFallbackRuntimeGainVsExact": None,
            }
        )
    return rows


def _representation_ablation_rows(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_mode in DEFAULT_REPRESENTATION_ABLATION_ORDER:
        mode_rows = [
            row for row in variant_rows
            if str(row.get("featureAblationMode")) == feature_mode
        ]
        best_row = _pick_best_variant_row(mode_rows)
        if best_row is not None:
            rows.append(best_row)
            continue
        rows.append(
            {
                "modelVariant": f"{feature_mode}-unconfigured",
                "constraintTrainingObjective": "unknown",
                "featureAblationMode": feature_mode,
                "exactLabelCoverage": None,
                "proxyLabelCoverage": None,
                "criticalClassificationThreshold": None,
                "constraintAuxRankingModelEnabled": None,
                "status": "not_configured",
                "sourceMode": None,
                "runCount": 0,
                "feasibilityRate": None,
                "fallbackRate": None,
                "averageRuntimeMs": None,
                "objectiveGapVsExact": None,
                "averageConstraintReductionRate": None,
                "criticalSelectionPrecision": None,
                "criticalSelectionRecall": None,
                "noFallbackFeasibilityRate": None,
                "noFallbackRuntimeGainVsExact": None,
            }
        )
    return rows


def _markdown_variant_ablation_table(rows: list[dict[str, Any]], table_title: str) -> str:
    header = (
        f"### {table_title}\n\n"
        "| Variant | Objective | Feature Mode | Status | Source Mode | Feasibility | Fallback | Avg Runtime (ms) | Gap vs Exact | Avg Reduction | Critical P | Critical R | No-Fallback Feasibility | No-Fallback Runtime Gain |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    body = []
    for row in rows:
        body.append(
            "| "
            f"{row.get('modelVariant')} | "
            f"{row.get('constraintTrainingObjective')} | "
            f"{row.get('featureAblationMode')} | "
            f"{row.get('status')} | "
            f"{row.get('sourceMode') or 'NA'} | "
            f"{_format_metric(row.get('feasibilityRate'))} | "
            f"{_format_metric(row.get('fallbackRate'))} | "
            f"{_format_metric(row.get('averageRuntimeMs'))} | "
            f"{_format_metric(row.get('objectiveGapVsExact'))} | "
            f"{_format_metric(row.get('averageConstraintReductionRate'))} | "
            f"{_format_metric(row.get('criticalSelectionPrecision'))} | "
            f"{_format_metric(row.get('criticalSelectionRecall'))} | "
            f"{_format_metric(row.get('noFallbackFeasibilityRate'))} | "
            f"{_format_metric(row.get('noFallbackRuntimeGainVsExact'))} |"
        )
    return "\n".join([header] + body)


def _write_report(
    output_dir: Path,
    summary_payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> Path:
    evaluation = summary_payload["evaluation"]
    requested_modes = [str(mode) for mode in (evaluation.get("requestedModes") or [])]
    report_lines = [
        "# Power-118 Evaluation Report",
        "",
        "## Run Info",
        f"- Generated at: `{evaluation.get('generatedAt')}`",
        f"- Case count: `{evaluation.get('caseCount')}`",
        f"- Requested modes: `{', '.join(requested_modes)}`",
        f"- Model path: `{evaluation.get('modelPath', 'NA')}`",
        f"- Metadata path: `{evaluation.get('metadataPath', 'NA')}`",
        f"- Model version: `{evaluation.get('modelVersion', 'NA')}`",
        f"- Feature schema version: `{evaluation.get('featureSchemaVersion', 'NA')}`",
        f"- Model load status: `{evaluation.get('modelLoadStatus', 'NA')}`",
        f"- Exact real baseline available: `{evaluation.get('exactRealBaselineAvailable')}`",
        f"- Exact baseline status: `{evaluation.get('exactBaselineStatus')}`",
        f"- Exact baseline time-limited: `{evaluation.get('exactBaselineTimeLimited')}`",
        f"- Exact baseline has incumbent: `{evaluation.get('exactBaselineHasIncumbent')}`",
        "",
        "## Mode Summary",
        _markdown_table(summary_rows),
        "",
        "## Mode Family Summary",
        _markdown_table(summary_payload.get("modeFamilies", [])),
        "",
        "## Variant Ablation",
    ]
    objective_ablation_rows = summary_payload.get("objectiveAblation", [])
    if objective_ablation_rows:
        report_lines.extend(
            [
                _markdown_variant_ablation_table(objective_ablation_rows, "Objective Ablation"),
                "",
            ]
        )
    representation_ablation_rows = summary_payload.get("representationAblation", [])
    if representation_ablation_rows:
        report_lines.extend(
            [
                _markdown_variant_ablation_table(representation_ablation_rows, "Representation Ablation"),
                "",
            ]
        )
    if not objective_ablation_rows and not representation_ablation_rows:
        report_lines.extend(
            [
                "- Variant-level ablation summary is unavailable for this run.",
                "",
            ]
        )
    report_lines.extend(
        [
        "## Ablation",
        _markdown_ablation_table(summary_rows),
        "",
        "## Reduction Strength Tradeoff",
        _markdown_v3_tradeoff_table(summary_payload.get("v3ReductionTradeoff", [])),
        "",
        "## Limits",
        ]
    )
    if bool(evaluation.get("exactRealBaselineAvailable")):
        report_lines.append("- Objective and cost gap metrics are based on a real feasible exact baseline.")
    else:
        report_lines.append("- Exact baseline was not a real feasible solve in this environment, so gap metrics may be null.")
    critical_metric_rows = [
        row for row in summary_rows
        if int(row.get("criticalMetricsAvailableCount", 0) or 0) > 0
    ]
    if critical_metric_rows:
        report_lines.append("- Critical precision/recall are computed only on samples with exact critical labels.")
    else:
        report_lines.append("- Critical precision/recall are unavailable because exact critical labels were not present in evaluated runs.")
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
                f"- Hybrid fallback reason distribution: `{evaluation.get('hybridFallbackReasonCounts', {})}`",
                f"- Hybrid fallback-to-mode distribution: `{evaluation.get('hybridFallbackToModeCounts', {})}`",
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
            f"gapVsExact={row['objectiveGapVsExact'] if row['objectiveGapVsExact'] is not None else 'NA'} "
            f"avgReduction={row['averageConstraintReductionRate'] if row.get('averageConstraintReductionRate') is not None else 'NA'} "
            f"criticalP={row['criticalSelectionPrecision'] if row.get('criticalSelectionPrecision') is not None else 'NA'} "
            f"criticalR={row['criticalSelectionRecall'] if row.get('criticalSelectionRecall') is not None else 'NA'}"
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
    variant_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    base_data = load_power118_data()
    overrides_list = generate_power118_override_set(base_data=base_data, n_samples=max(num_cases - 1, 0), seed=seed)
    cases = [{"caseId": "case-00000", "overrides": None}]
    for index, overrides in enumerate(overrides_list, start=1):
        cases.append({"caseId": f"case-{index:05d}", "overrides": overrides})

    model_artifacts = load_power118_model_artifacts(model_path=model_path, metadata_path=metadata_path)
    variant_tags = _resolve_variant_tags(model_artifacts=model_artifacts, variant_overrides=variant_overrides)
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
            elif requested_mode == "hybrid_constraint_aware":
                requested_runs.append(("hybrid_constraint_aware", "hybrid_constraint_aware_v2", "constraint_aware_v2"))
            elif requested_mode in {"exact", "hybrid_warm_start", "hybrid_constraint_aware_v2", "hybrid_constraint_aware_v3", "ml"}:
                requested_runs.append((requested_mode, requested_mode, None))

        for requested_label, requested_mode, hybrid_strategy in requested_runs:
            reduction_strengths = [None]
            if requested_mode == "hybrid_constraint_aware_v3":
                reduction_strengths = list(V3_REDUCTION_STRENGTHS)

            for reduction_strength in reduction_strengths:
                run = exact_run if requested_mode == "exact" else run_power118_once(
                    run_mode=requested_mode,
                    time_limit_ms=time_limit_ms,
                    fallback_to_exact=True,
                    overrides=case["overrides"],
                    model_path=model_path,
                    metadata_path=metadata_path,
                    hybrid_strategy=hybrid_strategy or "warm_start",
                    critical_top_k_ratio=reduction_strength,
                )
                record = build_eval_record(case["caseId"], requested_label, run, exact_run)
                record.update(variant_tags)
                records.append(record)

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
        variant_tags=variant_tags,
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


def evaluate_ablation_variants(
    num_cases: int,
    seed: int,
    output_dir: Path,
    time_limit_ms: int | None,
    modes: list[str],
    variant_specs: list[dict[str, Any]],
    require_exact_baseline: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    if not variant_specs:
        raise ValueError("evaluate_ablation_variants requires at least one variant spec")

    combined_records: list[dict[str, Any]] = []
    variant_mode_summary: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    configured_variants: list[dict[str, Any]] = []
    ran_variant_count = 0

    variants_root = output_dir / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)

    for index, variant_spec in enumerate(variant_specs, start=1):
        model_path = variant_spec.get("modelPath")
        metadata_path = variant_spec.get("metadataPath")
        model_artifacts = load_power118_model_artifacts(model_path=model_path, metadata_path=metadata_path)
        variant_tags = _resolve_variant_tags(model_artifacts=model_artifacts, variant_overrides=variant_spec)
        run_enabled = bool(variant_spec.get("runEnabled", True))
        configured_entry = {
            **variant_tags,
            "runEnabled": run_enabled,
            "note": variant_spec.get("note"),
            "modelLoadSuccess": bool(model_artifacts.get("loadSuccess")),
            "modelLoadStatus": model_artifacts.get("loadStatus"),
            "modelLoadFailureReason": model_artifacts.get("loadFailureReason"),
        }
        if not run_enabled:
            configured_entry["status"] = "configured_not_run"
            configured_variants.append(configured_entry)
            variant_rows.append(
                _variant_row_from_mode_row(
                    mode_row=None,
                    variant_tags=variant_tags,
                    status="configured_not_run",
                    source_mode=None,
                )
            )
            continue

        ran_variant_count += 1
        variant_output_dir = variants_root / f"{index:02d}-{_safe_slug(str(variant_tags.get('modelVariant')))}"
        records, summary_payload, _ = evaluate_modes(
            num_cases=num_cases,
            seed=seed,
            output_dir=variant_output_dir,
            time_limit_ms=time_limit_ms,
            modes=modes,
            model_path=model_path,
            metadata_path=metadata_path,
            require_exact_baseline=require_exact_baseline,
            variant_overrides=variant_spec,
        )
        combined_records.extend(records)
        mode_rows = list(summary_payload.get("modes", []))
        for row in mode_rows:
            annotated_row = dict(row)
            annotated_row.update(variant_tags)
            annotated_row["status"] = "ran"
            variant_mode_summary.append(annotated_row)
        primary_mode_row = _select_primary_mode_row(mode_rows)
        variant_rows.append(
            _variant_row_from_mode_row(
                mode_row=primary_mode_row,
                variant_tags=variant_tags,
                status="ran",
                source_mode=primary_mode_row.get("requestedMode") if primary_mode_row else None,
            )
        )
        configured_entry["status"] = "ran"
        configured_variants.append(configured_entry)

    summary_rows = summarize_eval_records(combined_records) if combined_records else []
    family_records: list[dict[str, Any]] = []
    for record in combined_records:
        family_record = dict(record)
        family_record["requestedMode"] = str(
            record.get("modeFamily") or _mode_family(str(record.get("requestedMode") or ""))
        )
        family_records.append(family_record)
    mode_family_summary = summarize_eval_records(family_records) if family_records else []
    v3_tradeoff = _v3_tradeoff_records(combined_records)
    objective_ablation_rows = _objective_ablation_rows(variant_rows)
    representation_ablation_rows = _representation_ablation_rows(variant_rows)
    hybrid_rows = [row for row in summary_rows if str(row.get("requestedMode", "")).startswith("hybrid_")]
    hybrid_fallback_reason_counts: dict[str, int] = {}
    hybrid_fallback_to_mode_counts: dict[str, int] = {}
    for row in hybrid_rows:
        for key, value in row.get("fallbackReasonCounts", {}).items():
            hybrid_fallback_reason_counts[str(key)] = hybrid_fallback_reason_counts.get(str(key), 0) + int(value)
        for key, value in row.get("fallbackToModeCounts", {}).items():
            hybrid_fallback_to_mode_counts[str(key)] = hybrid_fallback_to_mode_counts.get(str(key), 0) + int(value)

    exact_records = [record for record in combined_records if record.get("requestedMode") == "exact"]
    exact_real_available = any(
        bool(record.get("adapterMode") == "real")
        and (bool(record.get("optimal")) or (bool(record.get("feasible")) and bool(record.get("hasIncumbent"))))
        for record in exact_records
    )
    exact_baseline_status = exact_records[0].get("status") if exact_records else None
    exact_baseline_time_limited = bool(exact_records[0].get("terminatedByTimeLimit")) if exact_records else False
    exact_baseline_has_incumbent = bool(exact_records[0].get("hasIncumbent")) if exact_records else False
    run_enabled_variants = [entry for entry in configured_variants if bool(entry.get("runEnabled"))]
    all_run_enabled_loaded = bool(run_enabled_variants) and all(
        bool(entry.get("modelLoadSuccess")) for entry in run_enabled_variants
    )
    run_variant_failures = {
        str(entry.get("modelVariant")): str(entry.get("modelLoadFailureReason"))
        for entry in run_enabled_variants
        if entry.get("modelLoadFailureReason")
    }

    summary_payload = {
        "evaluation": {
            "generatedAt": _utc_now_iso(),
            "caseCount": int(len({record["caseId"] for record in combined_records})) if combined_records else 0,
            "requestedModes": modes,
            "variantMode": "multi",
            "configuredVariantCount": int(len(variant_specs)),
            "ranVariantCount": int(ran_variant_count),
            "modelPath": "multi-variant",
            "metadataPath": "multi-variant",
            "modelVersion": "multi-variant",
            "featureSchemaVersion": "multi-variant",
            "modelLoaded": all_run_enabled_loaded,
            "modelLoadStatus": "multi-variant",
            "modelLoadFailureReason": run_variant_failures or None,
            "exactRealBaselineAvailable": bool(exact_real_available),
            "exactBaselineStatus": exact_baseline_status,
            "exactBaselineTimeLimited": exact_baseline_time_limited,
            "exactBaselineHasIncumbent": exact_baseline_has_incumbent,
            "outputDir": str(output_dir),
            "hybridFallbackReasonCounts": hybrid_fallback_reason_counts,
            "hybridFallbackToModeCounts": hybrid_fallback_to_mode_counts,
        },
        "modes": summary_rows,
        "modeFamilies": mode_family_summary,
        "v3ReductionTradeoff": v3_tradeoff,
        "variantModeSummary": variant_mode_summary,
        "variants": variant_rows,
        "objectiveAblation": objective_ablation_rows,
        "representationAblation": representation_ablation_rows,
        "configuredVariants": configured_variants,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    records_json_path = output_dir / RECORDS_JSON_NAME
    records_csv_path = output_dir / RECORDS_CSV_NAME
    summary_json_path = output_dir / SUMMARY_JSON_NAME

    records_json_path.write_text(
        json.dumps(
            {
                "records": combined_records,
                "summary": summary_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(combined_records).to_csv(records_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    report_path = _write_report(output_dir, summary_payload, summary_rows)

    print("Power118 ablation evaluation summary")
    print(f"- Output dir: {summary_payload['evaluation']['outputDir']}")
    print(f"- Configured variants: {summary_payload['evaluation']['configuredVariantCount']}")
    print(f"- Ran variants: {summary_payload['evaluation']['ranVariantCount']}")
    for row in variant_rows:
        print(
            "- "
            f"variant={row.get('modelVariant')} "
            f"objective={row.get('constraintTrainingObjective')} "
            f"feature={row.get('featureAblationMode')} "
            f"status={row.get('status')} "
            f"mode={row.get('sourceMode') or 'NA'} "
            f"feasibility={_format_metric(row.get('feasibilityRate'))} "
            f"runtimeMs={_format_metric(row.get('averageRuntimeMs'))}"
        )
    print(f"Saved JSON records to {records_json_path}")
    print(f"Saved CSV records to {records_csv_path}")
    print(f"Saved JSON summary to {summary_json_path}")
    print(f"Saved Markdown report to {report_path}")
    return combined_records, summary_payload, report_path


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
        "--variant-config-path",
        type=Path,
        default=None,
        help="Optional JSON config for multi-variant ablation runs.",
    )
    parser.add_argument(
        "--require-exact-baseline",
        action="store_true",
        help="Fail the script if no real feasible exact baseline is available.",
    )
    args = parser.parse_args()

    modes = []
    for mode in args.modes:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == "hybrid":
            normalized_mode = "hybrid_warm_start"
        if normalized_mode not in {
            "exact",
            "hybrid_warm_start",
            "hybrid_constraint_aware",
            "hybrid_constraint_aware_v2",
            "hybrid_constraint_aware_v3",
            "ml",
        }:
            raise ValueError(f"Unsupported mode for evaluation: {mode}")
        modes.append(normalized_mode)

    resolved_model_path = args.model_path.resolve() if args.model_path is not None else None
    resolved_metadata_path = args.metadata_path.resolve() if args.metadata_path is not None else None
    if args.variant_config_path is not None:
        variant_config_path = args.variant_config_path.resolve()
        variant_specs = _load_variant_specs(
            variant_config_path=variant_config_path,
            default_model_path=resolved_model_path,
            default_metadata_path=resolved_metadata_path,
        )
        if not variant_specs:
            raise ValueError(f"Variant config has no valid entries: {variant_config_path}")
        evaluate_ablation_variants(
            num_cases=max(1, args.num_cases),
            seed=args.seed,
            output_dir=args.output_dir.resolve(),
            time_limit_ms=args.time_limit_ms,
            modes=modes,
            variant_specs=variant_specs,
            require_exact_baseline=args.require_exact_baseline,
        )
    else:
        evaluate_modes(
            num_cases=max(1, args.num_cases),
            seed=args.seed,
            output_dir=args.output_dir.resolve(),
            time_limit_ms=args.time_limit_ms,
            modes=modes,
            model_path=resolved_model_path,
            metadata_path=resolved_metadata_path,
            require_exact_baseline=args.require_exact_baseline,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
