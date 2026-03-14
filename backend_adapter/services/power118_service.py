from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

from backend_adapter.services.power118_dataset import load_power118_data
from backend_adapter.services.power118_ml_model import (
    load_power118_model_artifacts,
    predict_power118_constraints,
    predict_power118_constraint_scores,
    predict_power118_schedule,
    resolve_power118_model_paths,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _power118_dir() -> Path:
    return _repo_root() / "external" / "power118"


def _power118_script() -> Path:
    return _power118_dir() / "SCUC_118_new.py"


def _power118_data() -> Path:
    return _power118_dir() / "118_data.xls"


def _power118_script_label() -> str:
    return "external/power118/SCUC_118_new.py"


def _power118_data_label() -> str:
    return "external/power118/118_data.xls"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"run-power-118-{ts}"


def _normalize_power118_modes(run_mode: str | None, hybrid_strategy: str | None) -> tuple[str, str | None]:
    requested_mode = (run_mode or "exact").strip().lower()
    requested_hybrid_strategy = (hybrid_strategy or "").strip().lower()

    if requested_mode == "hybrid":
        requested_mode = "hybrid_warm_start"
    elif requested_mode == "hybrid_constraint_aware":
        requested_mode = "hybrid_constraint_aware_v2"

    hybrid_strategy_aliases = {
        "warm_start": "warm_start",
        "constraint_aware": "constraint_aware_v2",
        "constraint_aware_v2": "constraint_aware_v2",
        "constraint_aware_v3": "constraint_aware_v3",
        "fixing": "constraint_aware_v2",
        "scoring": "constraint_aware_v3",
    }
    requested_hybrid_strategy = hybrid_strategy_aliases.get(requested_hybrid_strategy, requested_hybrid_strategy)

    if requested_mode == "hybrid_warm_start":
        return "hybrid_warm_start", "warm_start"
    if requested_mode == "hybrid_constraint_aware_v2":
        return "hybrid_constraint_aware_v2", "constraint_aware_v2"
    if requested_mode == "hybrid_constraint_aware_v3":
        return "hybrid_constraint_aware_v3", "constraint_aware_v3"
    if requested_mode == "ml":
        return "ml", None
    return "exact", None


def _base_model_artifacts(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    model_path, metadata_path = resolve_power118_model_paths(model_path=model_path, metadata_path=metadata_path)
    return {
        "modelPath": str(model_path),
        "metadataPath": str(metadata_path),
        "loadSuccess": False,
        "loadFailureReason": None,
        "loadStatus": "not_requested",
        "modelBundle": None,
        "metadata": None,
        "modelVersion": None,
        "featureSchemaVersion": None,
    }


def _load_module():
    script_path = _power118_script()
    if not script_path.exists():
        raise FileNotFoundError(f"power118 script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("external.power118.scuc_118_new", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detect_runtime(module: Any) -> Dict[str, Any]:
    data_exists = _power118_data().exists()
    runtime = {
        "available": False,
        "stage": "precheck",
        "reason": "",
        "scriptPath": _power118_script_label(),
        "dataPath": _power118_data_label(),
        "scriptExists": _power118_script().exists(),
        "dataExists": data_exists,
    }

    if not runtime["scriptExists"]:
        runtime["reason"] = f"missing script {_power118_script_label()}"
        return runtime

    if not data_exists:
        runtime["reason"] = f"missing workbook {_power118_data_label()}"
        return runtime

    if hasattr(module, "check_gurobi_runtime"):
        try:
            external_runtime = module.check_gurobi_runtime()
        except Exception as exc:  # pragma: no cover - defensive
            runtime["stage"] = "runtime_check"
            runtime["reason"] = f"runtime check failed: {exc}"
            return runtime

        if isinstance(external_runtime, dict):
            runtime.update(external_runtime)
            runtime["scriptPath"] = _power118_script_label()
            runtime["dataPath"] = _power118_data_label()
            runtime["scriptExists"] = True
            runtime["dataExists"] = True
            return runtime

    try:
        import gurobipy  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        runtime["stage"] = "import"
        runtime["reason"] = f"gurobipy import failed: {exc}"
        return runtime

    runtime["available"] = True
    runtime["stage"] = "ready"
    runtime["reason"] = "gurobi runtime assumed ready"
    return runtime


def _load_preview(
    module: Any,
    data_path: Path,
    overrides: dict[str, Any] | None = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    if not hasattr(module, "load_power118_data"):
        return None, "load_power118_data is missing"

    try:
        preview = module.load_power118_data(data_path=data_path, overrides=overrides)
    except Exception as exc:  # pragma: no cover - depends on optional deps/runtime
        return None, str(exc)

    return preview if isinstance(preview, dict) else None, None


def _translate_runtime_reason(runtime: Dict[str, Any]) -> str:
    stage = runtime.get("stage") or "unknown"
    reason = str(runtime.get("reason") or "unknown runtime failure").strip()
    return f"runtime blocked at {stage}: {reason}"


def _decorate_payload(
    payload: Dict[str, Any],
    requested_run_mode: str,
    solver_mode_used: str,
    runtime_ms: float,
    objective_value: float | None,
    feasible: bool,
    model_artifacts: dict[str, Any] | None = None,
    ml_confidence: float | None = None,
    repair_applied: bool | None = None,
    fallback_reason: str | None = None,
    status_name: str | None = None,
    status_code: int | None = None,
    solution_count: int | None = None,
    terminated_by_time_limit: bool | None = None,
    optimal: bool | None = None,
    has_incumbent: bool | None = None,
    hybrid_strategy_requested: str | None = None,
    hybrid_strategy_used: str | None = None,
    constraint_aware_hybrid_used: bool | None = None,
    reduced_solve_applied: bool | None = None,
    fixed_commitment_count: int | None = None,
    predicted_active_constraint_count: int | None = None,
    constraint_confidence: float | None = None,
    repair_after_reduced_solve: bool | None = None,
    reduced_solve_fallback_reason: str | None = None,
    constraint_scoring_used: bool | None = None,
    critical_constraint_count: int | None = None,
    deferred_constraint_count: int | None = None,
    constraint_reactivation_count: int | None = None,
    staged_solve_rounds: int | None = None,
    constraint_aware_reduction_mode: str | None = None,
    reduced_model_validated: bool | None = None,
    reduction_rejected_reason: str | None = None,
) -> Dict[str, Any]:
    artifacts = model_artifacts or _base_model_artifacts()
    payload["requestedRunMode"] = requested_run_mode
    payload["requestedMode"] = requested_run_mode
    payload["solverModeUsed"] = solver_mode_used
    payload["mlConfidence"] = float(ml_confidence) if ml_confidence is not None else None
    payload["repairApplied"] = bool(repair_applied) if repair_applied is not None else None
    payload["fallbackReason"] = fallback_reason or None
    payload["modelVersion"] = artifacts.get("modelVersion")
    payload["featureSchemaVersion"] = artifacts.get("featureSchemaVersion")
    payload["runtimeMs"] = float(max(runtime_ms, 0.0))
    payload["objectiveValue"] = float(objective_value) if objective_value is not None else None
    payload["feasible"] = bool(feasible)
    payload["modelPath"] = artifacts.get("modelPath")
    payload["modelLoadStatus"] = artifacts.get("loadStatus")
    payload["statusName"] = status_name
    payload["statusCode"] = status_code
    payload["solutionCount"] = solution_count
    payload["terminatedByTimeLimit"] = terminated_by_time_limit
    payload["optimal"] = optimal
    payload["hasIncumbent"] = has_incumbent
    payload["hybridStrategyRequested"] = hybrid_strategy_requested
    payload["hybridStrategyUsed"] = hybrid_strategy_used
    payload["constraintAwareHybridUsed"] = constraint_aware_hybrid_used
    payload["reducedSolveApplied"] = reduced_solve_applied
    payload["fixedCommitmentCount"] = fixed_commitment_count
    payload["predictedActiveConstraintCount"] = predicted_active_constraint_count
    payload["constraintConfidence"] = constraint_confidence
    payload["repairAfterReducedSolve"] = repair_after_reduced_solve
    payload["reducedSolveFallbackReason"] = reduced_solve_fallback_reason
    payload["constraintScoringUsed"] = constraint_scoring_used
    payload["criticalConstraintCount"] = critical_constraint_count
    payload["deferredConstraintCount"] = deferred_constraint_count
    payload["constraintReactivationCount"] = constraint_reactivation_count
    payload["stagedSolveRounds"] = staged_solve_rounds
    payload["constraintAwareReductionMode"] = constraint_aware_reduction_mode
    payload["reducedModelValidated"] = reduced_model_validated
    payload["reductionRejectedReason"] = reduction_rejected_reason
    total_binary_count = None
    if fixed_commitment_count is not None:
        dispatch = payload.get("unitCommitmentByHour")
        if isinstance(dispatch, list) and dispatch and isinstance(dispatch[0], list):
            total_binary_count = len(dispatch) * len(dispatch[0])
    payload["fixedBinaryRatio"] = (
        float(fixed_commitment_count) / float(total_binary_count)
        if fixed_commitment_count is not None and total_binary_count
        else None
    )
    payload["constraintReductionRatio"] = payload["fixedBinaryRatio"]
    return payload


def _compat_run_payload(
    elapsed_ms: float,
    reason: str,
    requested_run_mode: str,
    solver_mode_used: str,
    preview: Dict[str, Any] | None = None,
    preview_error: str | None = None,
    fallback_reason: str | None = None,
    model_artifacts: dict[str, Any] | None = None,
    hybrid_strategy_requested: str | None = None,
    hybrid_strategy_used: str | None = None,
    constraint_aware_hybrid_used: bool | None = None,
    reduced_solve_applied: bool | None = None,
    fixed_commitment_count: int | None = None,
    predicted_active_constraint_count: int | None = None,
    constraint_confidence: float | None = None,
    repair_after_reduced_solve: bool | None = None,
    reduced_solve_fallback_reason: str | None = None,
    constraint_scoring_used: bool | None = None,
    critical_constraint_count: int | None = None,
    deferred_constraint_count: int | None = None,
    constraint_reactivation_count: int | None = None,
    staged_solve_rounds: int | None = None,
    constraint_aware_reduction_mode: str | None = None,
    reduced_model_validated: bool | None = None,
    reduction_rejected_reason: str | None = None,
) -> Dict[str, Any]:
    preview = preview or {}
    total_load_by_hour = preview.get("totalLoadByHour") if isinstance(preview.get("totalLoadByHour"), list) else []
    generator_capacity_preview = (
        preview.get("generatorCapacityPreview")
        if isinstance(preview.get("generatorCapacityPreview"), list)
        else []
    )
    summary = preview.get("summary") if isinstance(preview.get("summary"), dict) else {}
    artifacts = model_artifacts or _base_model_artifacts()

    note_parts = [f"power-118 compat mode: {reason}."]
    if preview_error:
        note_parts.append(f"Preview load issue: {preview_error}.")
    if total_load_by_hour:
        note_parts.append("trend is mapped from real hourly load totals in 118_data.xls.")
    if generator_capacity_preview:
        note_parts.append("comparison is mapped from generator Pmax preview because solved dispatch is unavailable.")
    note_parts.append("metrics other than solveTimeMs are temporary compatibility placeholders.")
    note_parts.append(f"Workbook={_power118_data_label()}.")
    note_parts.append(f"Script={_power118_script_label()}.")
    note_parts.append(f"modelPath={artifacts.get('modelPath')}.")
    note_parts.append(f"modelLoadStatus={artifacts.get('loadStatus')}.")
    if artifacts.get("loadFailureReason"):
        note_parts.append(f"modelLoadFailure={artifacts.get('loadFailureReason')}.")
    if artifacts.get("modelVersion"):
        note_parts.append(f"modelVersion={artifacts.get('modelVersion')}.")
    if artifacts.get("featureSchemaVersion"):
        note_parts.append(f"featureSchemaVersion={artifacts.get('featureSchemaVersion')}.")
    if summary:
        note_parts.append(f"peakLoad={summary.get('peakLoad', 0.0)}.")
    if fallback_reason:
        note_parts.append(f"fallbackReason={fallback_reason}.")

    payload = {
        "runId": _run_id(),
        "scenarioId": "power-118",
        "generatedAt": _utc_now_iso(),
        "metrics": {
            "solveTimeMs": float(max(elapsed_ms, 0.0)),
            "infeasibilityRate": 0.0,
            "suboptimality": 0.0,
        },
        "strategies": [],
        "trend": [{"label": f"H{idx + 1}", "value": float(value)} for idx, value in enumerate(total_load_by_hour)],
        "comparison": generator_capacity_preview[:4],
        "adapterMode": "compat",
        "adapterNote": " ".join(note_parts),
        "generatorDispatchByHour": None,
        "unitCommitmentByHour": None,
    }
    return _decorate_payload(
        payload,
        requested_run_mode=requested_run_mode,
        solver_mode_used=solver_mode_used,
        runtime_ms=elapsed_ms,
        objective_value=None,
        feasible=False,
        model_artifacts=artifacts,
        fallback_reason=fallback_reason,
        status_name="COMPAT",
        status_code=None,
        solution_count=0,
        terminated_by_time_limit=False,
        optimal=False,
        has_incumbent=False,
        hybrid_strategy_requested=hybrid_strategy_requested,
        hybrid_strategy_used=hybrid_strategy_used,
        constraint_aware_hybrid_used=constraint_aware_hybrid_used,
        reduced_solve_applied=reduced_solve_applied,
        fixed_commitment_count=fixed_commitment_count,
        predicted_active_constraint_count=predicted_active_constraint_count,
        constraint_confidence=constraint_confidence,
        repair_after_reduced_solve=repair_after_reduced_solve,
        reduced_solve_fallback_reason=reduced_solve_fallback_reason,
        constraint_scoring_used=constraint_scoring_used,
        critical_constraint_count=critical_constraint_count,
        deferred_constraint_count=deferred_constraint_count,
        constraint_reactivation_count=constraint_reactivation_count,
        staged_solve_rounds=staged_solve_rounds,
        constraint_aware_reduction_mode=constraint_aware_reduction_mode,
        reduced_model_validated=reduced_model_validated,
        reduction_rejected_reason=reduction_rejected_reason,
    )


def _real_run_payload(
    result: Dict[str, Any],
    requested_run_mode: str,
    solver_mode_used: str,
    model_artifacts: dict[str, Any] | None = None,
    ml_confidence: float | None = None,
    repair_applied: bool | None = None,
    fallback_reason: str | None = None,
    hybrid_strategy_requested: str | None = None,
    hybrid_strategy_used: str | None = None,
    constraint_aware_hybrid_used: bool | None = None,
    reduced_solve_applied: bool | None = None,
    fixed_commitment_count: int | None = None,
    predicted_active_constraint_count: int | None = None,
    constraint_confidence: float | None = None,
    repair_after_reduced_solve: bool | None = None,
    reduced_solve_fallback_reason: str | None = None,
    constraint_scoring_used: bool | None = None,
    critical_constraint_count: int | None = None,
    deferred_constraint_count: int | None = None,
    constraint_reactivation_count: int | None = None,
    staged_solve_rounds: int | None = None,
    constraint_aware_reduction_mode: str | None = None,
    reduced_model_validated: bool | None = None,
    reduction_rejected_reason: str | None = None,
) -> Dict[str, Any]:
    objective = float(result.get("objective") or 0.0)
    top_generators = result.get("topGenerators") if isinstance(result.get("topGenerators"), list) else []
    total_load_by_hour = result.get("totalLoadByHour") if isinstance(result.get("totalLoadByHour"), list) else []
    peak_line_flow_by_hour = result.get("peakLineFlowByHour") if isinstance(result.get("peakLineFlowByHour"), list) else []
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    artifacts = model_artifacts or _base_model_artifacts()

    strategy_name = {
        "exact": "SCUC-118 Exact Schedule",
        "hybrid": "SCUC-118 Hybrid Schedule",
        "ml": "SCUC-118 ML Schedule",
    }.get(solver_mode_used, "SCUC-118 Schedule")

    note_parts = [
        f"power-118 real mode: {strategy_name} completed.",
        "objective is mapped to strategies[0].cost.",
        "solveTimeMs is mapped from solver or inference elapsed time.",
        "trend is mapped from hourly system load totals.",
        "comparison is mapped from dispatch totals.",
        f"Workbook={_power118_data_label()}.",
        f"Script={_power118_script_label()}.",
        f"modelPath={artifacts.get('modelPath')}.",
        f"modelLoadStatus={artifacts.get('loadStatus')}.",
    ]
    if artifacts.get("loadFailureReason"):
        note_parts.append(f"modelLoadFailure={artifacts.get('loadFailureReason')}.")
    if artifacts.get("modelVersion"):
        note_parts.append(f"modelVersion={artifacts.get('modelVersion')}.")
    if artifacts.get("featureSchemaVersion"):
        note_parts.append(f"featureSchemaVersion={artifacts.get('featureSchemaVersion')}.")
    if result.get("statusName"):
        note_parts.append(f"status={result.get('statusName')}.")
    if runtime:
        note_parts.append(f"runtimeStage={runtime.get('stage', 'unknown')}.")
    if summary:
        note_parts.append(f"peakLoad={summary.get('peakLoad', 0.0)}.")
    if peak_line_flow_by_hour:
        note_parts.append(f"peakLineFlow={max(peak_line_flow_by_hour)}.")
    if result.get("warmStartUsed"):
        note_parts.append("solver received ML warm-start values.")
    if fallback_reason:
        note_parts.append(f"fallbackReason={fallback_reason}.")

    payload = {
        "runId": _run_id(),
        "scenarioId": "power-118",
        "generatedAt": _utc_now_iso(),
        "metrics": {
            "solveTimeMs": float(result.get("solveTimeMs") or 0.0),
            "infeasibilityRate": 0.0 if result.get("feasible", True) else 1.0,
            "suboptimality": 0.0,
        },
        "strategies": [
            {
                "id": "strategy-1",
                "name": strategy_name,
                "feasible": bool(result.get("feasible", True)),
                "cost": objective,
                "rank": 1,
            }
        ],
        "trend": [{"label": f"H{idx + 1}", "value": float(value)} for idx, value in enumerate(total_load_by_hour)],
        "comparison": top_generators[:4],
        "adapterMode": "real",
        "adapterNote": " ".join(note_parts),
        "generatorDispatchByHour": result.get("generatorDispatchByHour"),
        "unitCommitmentByHour": result.get("unitCommitmentByHour"),
    }
    return _decorate_payload(
        payload,
        requested_run_mode=requested_run_mode,
        solver_mode_used=solver_mode_used,
        runtime_ms=float(result.get("solveTimeMs") or 0.0),
        objective_value=objective,
        feasible=bool(result.get("feasible", True)),
        model_artifacts=artifacts,
        ml_confidence=ml_confidence,
        repair_applied=repair_applied,
        fallback_reason=fallback_reason,
        status_name=str(result.get("statusName") or "UNKNOWN"),
        status_code=int(result.get("statusCode")) if result.get("statusCode") is not None else None,
        solution_count=int(result.get("solutionCount") or 0),
        terminated_by_time_limit=bool(result.get("terminatedByTimeLimit", False)),
        optimal=bool(result.get("optimal", False)),
        has_incumbent=bool(result.get("hasIncumbent", bool(result.get("feasible", True)))),
        hybrid_strategy_requested=hybrid_strategy_requested,
        hybrid_strategy_used=hybrid_strategy_used,
        constraint_aware_hybrid_used=constraint_aware_hybrid_used,
        reduced_solve_applied=reduced_solve_applied,
        fixed_commitment_count=fixed_commitment_count if fixed_commitment_count is not None else result.get("fixedCommitmentCount"),
        predicted_active_constraint_count=predicted_active_constraint_count,
        constraint_confidence=constraint_confidence,
        repair_after_reduced_solve=repair_after_reduced_solve,
        reduced_solve_fallback_reason=reduced_solve_fallback_reason,
        constraint_scoring_used=constraint_scoring_used,
        critical_constraint_count=critical_constraint_count,
        deferred_constraint_count=deferred_constraint_count,
        constraint_reactivation_count=constraint_reactivation_count,
        staged_solve_rounds=staged_solve_rounds,
        constraint_aware_reduction_mode=constraint_aware_reduction_mode,
        reduced_model_validated=reduced_model_validated,
        reduction_rejected_reason=reduction_rejected_reason,
    )


def _ml_run_payload(
    prediction: Dict[str, Any],
    elapsed_ms: float,
    requested_run_mode: str,
    solver_mode_used: str,
    model_artifacts: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
    hybrid_strategy_requested: str | None = None,
    hybrid_strategy_used: str | None = None,
    constraint_aware_hybrid_used: bool | None = None,
    reduced_solve_applied: bool | None = None,
    fixed_commitment_count: int | None = None,
    predicted_active_constraint_count: int | None = None,
    constraint_confidence: float | None = None,
    repair_after_reduced_solve: bool | None = None,
    reduced_solve_fallback_reason: str | None = None,
    constraint_scoring_used: bool | None = None,
    critical_constraint_count: int | None = None,
    deferred_constraint_count: int | None = None,
    constraint_reactivation_count: int | None = None,
    staged_solve_rounds: int | None = None,
    constraint_aware_reduction_mode: str | None = None,
    reduced_model_validated: bool | None = None,
    reduction_rejected_reason: str | None = None,
) -> Dict[str, Any]:
    artifacts = model_artifacts or _base_model_artifacts()
    note_parts = [
        "power-118 ML mode: schedule predicted from trained model artifact.",
        "trend is mapped from hourly system load totals.",
        "comparison is mapped from predicted generator dispatch totals.",
        "no exact power-flow feasibility guarantee is implied beyond lightweight repair checks.",
        f"Workbook={_power118_data_label()}.",
        f"modelPath={artifacts.get('modelPath')}.",
        f"modelLoadStatus={artifacts.get('loadStatus')}.",
    ]
    if artifacts.get("modelVersion"):
        note_parts.append(f"modelVersion={artifacts.get('modelVersion')}.")
    if artifacts.get("featureSchemaVersion"):
        note_parts.append(f"featureSchemaVersion={artifacts.get('featureSchemaVersion')}.")
    if fallback_reason:
        note_parts.append(f"fallbackReason={fallback_reason}.")

    payload = {
        "runId": _run_id(),
        "scenarioId": "power-118",
        "generatedAt": _utc_now_iso(),
        "metrics": {
            "solveTimeMs": float(max(elapsed_ms, 0.0)),
            "infeasibilityRate": 0.0 if prediction.get("feasible") else 1.0,
            "suboptimality": 0.0,
        },
        "strategies": [
            {
                "id": "strategy-1",
                "name": "SCUC-118 ML Schedule",
                "feasible": bool(prediction.get("feasible")),
                "cost": float(prediction.get("objective") or 0.0),
                "rank": 1,
            }
        ],
        "trend": [
            {"label": f"H{idx + 1}", "value": float(value)}
            for idx, value in enumerate(prediction.get("totalLoadByHour", []))
        ],
        "comparison": list(prediction.get("topGenerators", []))[:4],
        "adapterMode": "real",
        "adapterNote": " ".join(note_parts),
        "generatorDispatchByHour": prediction.get("generatorDispatchByHour"),
        "unitCommitmentByHour": prediction.get("unitCommitmentByHour"),
    }
    return _decorate_payload(
        payload,
        requested_run_mode=requested_run_mode,
        solver_mode_used=solver_mode_used,
        runtime_ms=elapsed_ms,
        objective_value=float(prediction.get("objective") or 0.0),
        feasible=bool(prediction.get("feasible")),
        model_artifacts=artifacts,
        ml_confidence=float(prediction.get("mlConfidence") or 0.0),
        repair_applied=bool(prediction.get("repairApplied")),
        fallback_reason=fallback_reason,
        status_name="ML_PREDICTED",
        status_code=None,
        solution_count=None,
        terminated_by_time_limit=False,
        optimal=False,
        has_incumbent=bool(prediction.get("feasible")),
        hybrid_strategy_requested=hybrid_strategy_requested,
        hybrid_strategy_used=hybrid_strategy_used,
        constraint_aware_hybrid_used=constraint_aware_hybrid_used,
        reduced_solve_applied=reduced_solve_applied,
        fixed_commitment_count=fixed_commitment_count,
        predicted_active_constraint_count=predicted_active_constraint_count,
        constraint_confidence=constraint_confidence,
        repair_after_reduced_solve=repair_after_reduced_solve,
        reduced_solve_fallback_reason=reduced_solve_fallback_reason,
        constraint_scoring_used=constraint_scoring_used,
        critical_constraint_count=critical_constraint_count,
        deferred_constraint_count=deferred_constraint_count,
        constraint_reactivation_count=constraint_reactivation_count,
        staged_solve_rounds=staged_solve_rounds,
        constraint_aware_reduction_mode=constraint_aware_reduction_mode,
        reduced_model_validated=reduced_model_validated,
        reduction_rejected_reason=reduction_rejected_reason,
    )


def _run_solver(
    module: Any,
    data_path: Path,
    overrides: dict[str, Any] | None = None,
    initial_unit_commitment: list[list[float]] | None = None,
    initial_dispatch: list[list[float]] | None = None,
    time_limit_ms: int | None = None,
    fixed_commitment_mask: list[list[bool]] | None = None,
    active_ramp_constraint_ids: list[str] | None = None,
    active_line_constraint_ids: list[str] | None = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    if not hasattr(module, "solve_scuc_118"):
        return None, "solve_scuc_118 is missing"

    try:
        solve_kwargs = {
            "data_path": data_path,
            "write_output": False,
            "overrides": overrides,
            "initial_unit_commitment": initial_unit_commitment,
            "initial_dispatch": initial_dispatch,
            "time_limit_s": (float(time_limit_ms) / 1000.0 if time_limit_ms else None),
        }
        if fixed_commitment_mask is not None:
            solve_kwargs["fixed_commitment_mask"] = fixed_commitment_mask
        if active_ramp_constraint_ids is not None:
            solve_kwargs["active_ramp_constraint_ids"] = active_ramp_constraint_ids
        if active_line_constraint_ids is not None:
            solve_kwargs["active_line_constraint_ids"] = active_line_constraint_ids
        result = module.solve_scuc_118(
            **solve_kwargs,
        )
    except Exception as exc:  # pragma: no cover - depends on solver/runtime
        return None, str(exc)

    if not isinstance(result, dict):
        return None, "solve_scuc_118 returned a non-dict result"
    return result, None


def _is_real_result(result: Dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    status = str(result.get("statusName") or "").upper()
    feasible = bool(result.get("feasible", status in {"OPTIMAL", "SUBOPTIMAL"}))
    has_incumbent = bool(result.get("hasIncumbent", result.get("solutionCount", 0)))
    optimal = bool(result.get("optimal", status == "OPTIMAL"))
    return bool(feasible and (optimal or has_incumbent or status in {"OPTIMAL", "SUBOPTIMAL"}))


def _predict_with_model(
    preview: dict[str, Any],
    model_artifacts: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        prediction = predict_power118_schedule(
            preview,
            model_artifacts["modelBundle"],
            metadata=model_artifacts.get("metadata"),
        )
    except Exception as exc:
        return None, str(exc)
    return prediction, None


def _predict_constraints_with_model(
    preview: dict[str, Any],
    model_artifacts: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        prediction = predict_power118_constraints(
            preview,
            model_artifacts["modelBundle"],
            metadata=model_artifacts.get("metadata"),
        )
    except Exception as exc:
        return None, str(exc)
    return prediction, None


def _predict_constraint_scores_with_model(
    preview: dict[str, Any],
    schedule_prediction: dict[str, Any],
    model_artifacts: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        prediction = predict_power118_constraint_scores(
            preview,
            schedule_prediction,
            model_artifacts["modelBundle"],
            metadata=model_artifacts.get("metadata"),
        )
    except Exception as exc:
        return None, str(exc)
    return prediction, None


def _build_constraint_aware_fixing_plan(
    schedule_prediction: dict[str, Any],
    constraint_prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    commitment_scores = schedule_prediction.get("commitmentScores")
    commitment_confidence = schedule_prediction.get("commitmentConfidence")
    unit_commitment = schedule_prediction.get("unitCommitmentByHour")
    if not isinstance(commitment_scores, list) or not isinstance(commitment_confidence, list) or not isinstance(unit_commitment, list):
        return {
            "fixedCommitmentMask": None,
            "fixedCommitmentCount": 0,
            "constraintAwareHybridUsed": False,
            "reducedSolveApplied": False,
            "constraintConfidence": float(constraint_prediction.get("constraintConfidence", 0.0)) if isinstance(constraint_prediction, dict) else None,
            "predictedActiveConstraintCount": int(constraint_prediction.get("predictedActiveConstraintCount", 0) or 0) if isinstance(constraint_prediction, dict) else 0,
        }

    num_generators = len(unit_commitment)
    horizon = len(unit_commitment[0]) if unit_commitment else 0
    if num_generators == 0 or horizon == 0:
        return {
            "fixedCommitmentMask": None,
            "fixedCommitmentCount": 0,
            "constraintAwareHybridUsed": False,
            "reducedSolveApplied": False,
            "constraintConfidence": float(constraint_prediction.get("constraintConfidence", 0.0)) if isinstance(constraint_prediction, dict) else None,
            "predictedActiveConstraintCount": int(constraint_prediction.get("predictedActiveConstraintCount", 0) or 0) if isinstance(constraint_prediction, dict) else 0,
        }

    total_binary_count = num_generators * horizon
    predicted_active_count = int(constraint_prediction.get("predictedActiveConstraintCount", 0) or 0) if isinstance(constraint_prediction, dict) else 0
    constraint_confidence = float(constraint_prediction.get("constraintConfidence", 0.0) or 0.0) if isinstance(constraint_prediction, dict) else 0.0
    fix_mask_scores = constraint_prediction.get("predictedFixedCommitmentMaskScores") if isinstance(constraint_prediction, dict) else None
    if not isinstance(fix_mask_scores, list) or len(fix_mask_scores) != num_generators:
        fix_mask_scores = [[1.0 for _ in range(horizon)] for _ in range(num_generators)]

    active_ratio = min(max(predicted_active_count / float(max(total_binary_count, 1)), 0.0), 1.0)
    target_fix_ratio = max(0.05, min(0.35, 0.25 - (0.12 * active_ratio)))
    min_combined_score = 0.72

    candidate_entries: list[tuple[float, int, int]] = []
    for gen_index in range(num_generators):
        for hour_index in range(horizon):
            combined_score = float(commitment_confidence[gen_index][hour_index]) * float(fix_mask_scores[gen_index][hour_index])
            candidate_entries.append((combined_score, gen_index, hour_index))

    candidate_entries.sort(key=lambda item: item[0], reverse=True)
    max_fix_count = max(1, int(round(total_binary_count * target_fix_ratio)))
    fixed_mask = [[False for _ in range(horizon)] for _ in range(num_generators)]
    fixed_count = 0
    for combined_score, gen_index, hour_index in candidate_entries:
        if fixed_count >= max_fix_count:
            break
        if combined_score < min_combined_score:
            break
        fixed_mask[gen_index][hour_index] = True
        fixed_count += 1

    return {
        "fixedCommitmentMask": fixed_mask if fixed_count > 0 else None,
        "fixedCommitmentCount": fixed_count,
        "constraintAwareHybridUsed": True,
        "reducedSolveApplied": fixed_count > 0,
        "constraintConfidence": constraint_confidence,
        "predictedActiveConstraintCount": predicted_active_count,
    }


def _build_constraint_scoring_plan(
    constraint_scoring: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(constraint_scoring, dict):
        return {
            "criticalConstraintIds": None,
            "deferredConstraintIds": None,
            "criticalConstraintCount": 0,
            "deferredConstraintCount": 0,
            "constraintConfidence": None,
        }

    critical_ids = [str(value) for value in constraint_scoring.get("topKConstraintIds", [])]
    deferred_ids = [
        str(item.get("constraintId"))
        for item in constraint_scoring.get("predictedReducibleConstraints", [])
        if isinstance(item, dict) and item.get("constraintId")
    ]
    return {
        "criticalConstraintIds": critical_ids,
        "deferredConstraintIds": deferred_ids,
        "criticalConstraintCount": int(constraint_scoring.get("criticalConstraintCount", len(critical_ids))),
        "deferredConstraintCount": int(constraint_scoring.get("deferredConstraintCount", len(deferred_ids))),
        "constraintConfidence": float(constraint_scoring.get("constraintConfidence", 0.0) or 0.0),
    }


def _violated_deferred_constraint_ids(
    result: dict[str, Any] | None,
    deferred_constraint_ids: list[str] | None,
    violation_tolerance: float = 1e-4,
) -> list[str]:
    if not isinstance(result, dict) or not deferred_constraint_ids:
        return []
    diagnostics = result.get("constraintDiagnostics") if isinstance(result.get("constraintDiagnostics"), dict) else {}
    slack_records = diagnostics.get("slackRecords") if isinstance(diagnostics.get("slackRecords"), dict) else {}
    deferred_id_set = {str(value) for value in deferred_constraint_ids}
    violated_ids: list[str] = []
    for records in slack_records.values():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            constraint_id = str(record.get("constraintId") or "")
            if constraint_id not in deferred_id_set:
                continue
            raw_slack = float(record.get("rawSlack", record.get("slack", 0.0)) or 0.0)
            if raw_slack < -violation_tolerance:
                violated_ids.append(constraint_id)
    return violated_ids


def run_power118_once(
    run_mode: str = "exact",
    time_limit_ms: int | None = None,
    fallback_to_exact: bool = True,
    overrides: dict[str, Any] | None = None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    hybrid_strategy: str = "warm_start",
) -> Dict[str, Any]:
    start = perf_counter()
    requested_mode, requested_hybrid_strategy = _normalize_power118_modes(run_mode, hybrid_strategy)

    module = _load_module()
    data_path = _power118_data()

    preview, preview_error = _load_preview(module, data_path, overrides=overrides)
    if preview is None:
        try:
            preview = load_power118_data(data_path=data_path, overrides=overrides)
        except Exception:
            preview = None

    runtime = _detect_runtime(module)

    if requested_mode == "exact":
        model_artifacts = _base_model_artifacts(model_path=model_path, metadata_path=metadata_path)
        if not runtime.get("available"):
            elapsed_ms = (perf_counter() - start) * 1000.0
            return _compat_run_payload(
                elapsed_ms=elapsed_ms,
                reason=_translate_runtime_reason(runtime),
                requested_run_mode=requested_mode,
                solver_mode_used="exact",
                preview=preview,
                preview_error=preview_error,
                model_artifacts=model_artifacts,
            )

        result, solve_error = _run_solver(module, data_path, overrides=overrides, time_limit_ms=time_limit_ms)
        if solve_error:
            elapsed_ms = (perf_counter() - start) * 1000.0
            return _compat_run_payload(
                elapsed_ms=elapsed_ms,
                reason=f"external solve failed after runtime ready: {solve_error}",
                requested_run_mode=requested_mode,
                solver_mode_used="exact",
                preview=preview,
                preview_error=preview_error,
                model_artifacts=model_artifacts,
            )

        if preview and result is not None:
            result.setdefault("summary", preview.get("summary", {}))
            result.setdefault("totalLoadByHour", preview.get("totalLoadByHour", []))
            result.setdefault("generatorCapacityPreview", preview.get("generatorCapacityPreview", []))
        if result is not None:
            result.setdefault("runtime", runtime)

        if not _is_real_result(result):
            elapsed_ms = float(result.get("solveTimeMs") or (perf_counter() - start) * 1000.0)
            return _compat_run_payload(
                elapsed_ms=elapsed_ms,
                reason=f"solver returned non-real result status={result.get('statusName', 'UNKNOWN')}",
                requested_run_mode=requested_mode,
                solver_mode_used="exact",
                preview=preview,
                preview_error=preview_error,
                model_artifacts=model_artifacts,
            )

        return _real_run_payload(
            result,
            requested_run_mode=requested_mode,
            solver_mode_used="exact",
            model_artifacts=model_artifacts,
        )

    model_artifacts = load_power118_model_artifacts(model_path=model_path, metadata_path=metadata_path)
    if not model_artifacts["loadSuccess"]:
        fallback_reason = str(model_artifacts.get("loadFailureReason") or "ml artifacts unavailable")
        if fallback_to_exact and runtime.get("available"):
            result, solve_error = _run_solver(module, data_path, overrides=overrides, time_limit_ms=time_limit_ms)
            if solve_error is None and _is_real_result(result):
                assert result is not None
                result.setdefault("runtime", runtime)
                return _real_run_payload(
                    result,
                    requested_run_mode=requested_mode,
                    solver_mode_used="exact",
                    model_artifacts=model_artifacts,
                    fallback_reason=fallback_reason,
                )

        elapsed_ms = (perf_counter() - start) * 1000.0
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=fallback_reason,
            requested_run_mode=requested_mode,
            solver_mode_used=requested_mode,
            preview=preview,
            preview_error=preview_error,
            fallback_reason=fallback_reason,
            model_artifacts=model_artifacts,
        )

    if preview is None:
        elapsed_ms = (perf_counter() - start) * 1000.0
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason="preview data unavailable for ML feature generation",
            requested_run_mode=requested_mode,
            solver_mode_used=requested_mode,
            preview=preview,
            preview_error=preview_error,
            model_artifacts=model_artifacts,
        )

    prediction, prediction_error = _predict_with_model(preview, model_artifacts)
    if prediction_error:
        fallback_reason = f"ml inference blocked: {prediction_error}"
        if fallback_to_exact and runtime.get("available"):
            result, solve_error = _run_solver(module, data_path, overrides=overrides, time_limit_ms=time_limit_ms)
            if solve_error is None and _is_real_result(result):
                assert result is not None
                result.setdefault("runtime", runtime)
                return _real_run_payload(
                    result,
                    requested_run_mode=requested_mode,
                    solver_mode_used="exact",
                    model_artifacts=model_artifacts,
                    fallback_reason=fallback_reason,
                )

        elapsed_ms = (perf_counter() - start) * 1000.0
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=fallback_reason,
            requested_run_mode=requested_mode,
            solver_mode_used=requested_mode,
            preview=preview,
            preview_error=preview_error,
            fallback_reason=fallback_reason,
            model_artifacts=model_artifacts,
        )

    assert prediction is not None

    if requested_mode == "ml":
        if not prediction.get("feasible") and fallback_to_exact and runtime.get("available"):
            fallback_reason = "ml prediction failed lightweight feasibility checks; fell back to exact"
            result, solve_error = _run_solver(module, data_path, overrides=overrides, time_limit_ms=time_limit_ms)
            if solve_error is None and _is_real_result(result):
                assert result is not None
                result.setdefault("runtime", runtime)
                return _real_run_payload(
                    result,
                    requested_run_mode=requested_mode,
                    solver_mode_used="exact",
                    model_artifacts=model_artifacts,
                    ml_confidence=float(prediction.get("mlConfidence") or 0.0),
                    repair_applied=bool(prediction.get("repairApplied")),
                    fallback_reason=fallback_reason,
                )

        elapsed_ms = (perf_counter() - start) * 1000.0
        return _ml_run_payload(
            prediction,
            elapsed_ms=elapsed_ms,
            requested_run_mode=requested_mode,
            solver_mode_used="ml",
            model_artifacts=model_artifacts,
        )

    constraint_prediction = None
    constraint_prediction_error = None
    constraint_scoring = None
    constraint_scoring_error = None
    fixing_plan = {
        "fixedCommitmentMask": None,
        "fixedCommitmentCount": 0,
        "constraintAwareHybridUsed": False,
        "reducedSolveApplied": False,
        "constraintConfidence": None,
        "predictedActiveConstraintCount": 0,
    }
    scoring_plan = {
        "criticalConstraintIds": None,
        "deferredConstraintIds": None,
        "criticalConstraintCount": 0,
        "deferredConstraintCount": 0,
        "constraintConfidence": None,
    }
    active_ramp_constraint_ids = None
    active_line_constraint_ids = None

    if requested_mode == "hybrid_constraint_aware_v2":
        constraint_prediction, constraint_prediction_error = _predict_constraints_with_model(preview, model_artifacts)
        if constraint_prediction_error is None and constraint_prediction is not None:
            fixing_plan = _build_constraint_aware_fixing_plan(prediction, constraint_prediction)
        else:
            fixing_plan["constraintConfidence"] = 0.0

    if requested_mode == "hybrid_constraint_aware_v3":
        constraint_scoring, constraint_scoring_error = _predict_constraint_scores_with_model(preview, prediction, model_artifacts)
        if constraint_scoring_error is None and constraint_scoring is not None:
            scoring_plan = _build_constraint_scoring_plan(constraint_scoring)
            critical_ids = scoring_plan["criticalConstraintIds"] or []
            active_ramp_constraint_ids = [constraint_id for constraint_id in critical_ids if str(constraint_id).startswith("ramp:")]
            active_line_constraint_ids = [constraint_id for constraint_id in critical_ids if str(constraint_id).startswith("line:")]
        else:
            scoring_plan["constraintConfidence"] = 0.0

    constraint_aware_hybrid_used = requested_mode in {"hybrid_constraint_aware_v2", "hybrid_constraint_aware_v3"}
    reduced_solve_applied = bool(fixing_plan["reducedSolveApplied"]) or bool(scoring_plan["criticalConstraintIds"])
    predicted_active_constraint_count = (
        int(fixing_plan["predictedActiveConstraintCount"])
        if requested_hybrid_strategy == "constraint_aware_v2"
        else int(scoring_plan["criticalConstraintCount"])
    )
    constraint_confidence_value = (
        fixing_plan["constraintConfidence"]
        if requested_hybrid_strategy == "constraint_aware_v2"
        else scoring_plan["constraintConfidence"]
    )
    constraint_scoring_used = requested_hybrid_strategy == "constraint_aware_v3"
    critical_constraint_count = int(scoring_plan["criticalConstraintCount"])
    deferred_constraint_count = int(scoring_plan["deferredConstraintCount"])
    constraint_reactivation_count = 0
    staged_solve_rounds = 1
    reduced_model_validated = None
    reduction_rejected_reason = constraint_prediction_error or constraint_scoring_error
    constraint_aware_reduction_mode = (
        "fixed_commitment_mask"
        if requested_hybrid_strategy == "constraint_aware_v2"
        else "critical_constraint_subset"
        if requested_hybrid_strategy == "constraint_aware_v3"
        else "warm_start_only"
    )

    if not runtime.get("available"):
        elapsed_ms = (perf_counter() - start) * 1000.0
        if prediction.get("feasible"):
            return _ml_run_payload(
                prediction,
                elapsed_ms=elapsed_ms,
                requested_run_mode=requested_mode,
                solver_mode_used="ml",
                model_artifacts=model_artifacts,
                fallback_reason=f"hybrid requested but {_translate_runtime_reason(runtime)}",
                hybrid_strategy_requested=requested_hybrid_strategy,
                hybrid_strategy_used=requested_hybrid_strategy,
                constraint_aware_hybrid_used=constraint_aware_hybrid_used,
                reduced_solve_applied=reduced_solve_applied,
                fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
                predicted_active_constraint_count=predicted_active_constraint_count,
                constraint_confidence=constraint_confidence_value,
                reduced_solve_fallback_reason=constraint_prediction_error or constraint_scoring_error,
                constraint_scoring_used=constraint_scoring_used,
                critical_constraint_count=critical_constraint_count,
                deferred_constraint_count=deferred_constraint_count,
                constraint_reactivation_count=constraint_reactivation_count,
                staged_solve_rounds=staged_solve_rounds,
                constraint_aware_reduction_mode=constraint_aware_reduction_mode,
                reduced_model_validated=reduced_model_validated,
                reduction_rejected_reason=reduction_rejected_reason,
            )
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=_translate_runtime_reason(runtime),
            requested_run_mode=requested_mode,
            solver_mode_used="hybrid",
            preview=preview,
            preview_error=preview_error,
            model_artifacts=model_artifacts,
            hybrid_strategy_requested=requested_hybrid_strategy,
            hybrid_strategy_used=requested_hybrid_strategy,
            constraint_aware_hybrid_used=constraint_aware_hybrid_used,
            reduced_solve_applied=reduced_solve_applied,
            fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
            predicted_active_constraint_count=predicted_active_constraint_count,
            constraint_confidence=constraint_confidence_value,
            reduced_solve_fallback_reason=constraint_prediction_error or constraint_scoring_error,
            constraint_scoring_used=constraint_scoring_used,
            critical_constraint_count=critical_constraint_count,
            deferred_constraint_count=deferred_constraint_count,
            constraint_reactivation_count=constraint_reactivation_count,
            staged_solve_rounds=staged_solve_rounds,
            constraint_aware_reduction_mode=constraint_aware_reduction_mode,
            reduced_model_validated=reduced_model_validated,
            reduction_rejected_reason=reduction_rejected_reason,
        )

    reduced_solve_fallback_reason = None
    result, solve_error = _run_solver(
        module,
        data_path,
        overrides=overrides,
        initial_unit_commitment=prediction.get("unitCommitmentByHour"),
        initial_dispatch=prediction.get("generatorDispatchByHour"),
        time_limit_ms=time_limit_ms,
        fixed_commitment_mask=fixing_plan["fixedCommitmentMask"],
        active_ramp_constraint_ids=active_ramp_constraint_ids,
        active_line_constraint_ids=active_line_constraint_ids,
    )
    if solve_error is None and requested_hybrid_strategy == "constraint_aware_v3" and _is_real_result(result):
        violated_deferred_ids = _violated_deferred_constraint_ids(result, scoring_plan["deferredConstraintIds"])
        if violated_deferred_ids:
            constraint_reactivation_count = len(violated_deferred_ids)
            staged_solve_rounds = 2
            expanded_critical_ids = list((scoring_plan["criticalConstraintIds"] or []) + violated_deferred_ids)
            active_ramp_constraint_ids = [constraint_id for constraint_id in expanded_critical_ids if str(constraint_id).startswith("ramp:")]
            active_line_constraint_ids = [constraint_id for constraint_id in expanded_critical_ids if str(constraint_id).startswith("line:")]
            second_result, second_error = _run_solver(
                module,
                data_path,
                overrides=overrides,
                initial_unit_commitment=prediction.get("unitCommitmentByHour"),
                initial_dispatch=prediction.get("generatorDispatchByHour"),
                time_limit_ms=time_limit_ms,
                active_ramp_constraint_ids=active_ramp_constraint_ids,
                active_line_constraint_ids=active_line_constraint_ids,
            )
            if second_error is None and _is_real_result(second_result):
                result = second_result
                remaining_violations = _violated_deferred_constraint_ids(result, scoring_plan["deferredConstraintIds"])
                if remaining_violations:
                    reduction_rejected_reason = "deferred constraints still violated after staged reactivation"
                    reduced_model_validated = False
                else:
                    reduced_model_validated = True
                    reduction_rejected_reason = None
            else:
                solve_error = second_error or "staged reactivation solve failed"
                reduced_model_validated = False
                reduction_rejected_reason = solve_error
        else:
            reduced_model_validated = True
            reduction_rejected_reason = None
    if solve_error:
        fallback_reason = f"hybrid {requested_hybrid_strategy} solve failed: {solve_error}"
        reduced_solve_fallback_reason = fallback_reason
        if fallback_to_exact:
            if requested_hybrid_strategy in {"constraint_aware_v2", "constraint_aware_v3"}:
                warm_result, warm_error = _run_solver(
                    module,
                    data_path,
                    overrides=overrides,
                    initial_unit_commitment=prediction.get("unitCommitmentByHour"),
                    initial_dispatch=prediction.get("generatorDispatchByHour"),
                    time_limit_ms=time_limit_ms,
                )
                if warm_error is None and _is_real_result(warm_result):
                    assert warm_result is not None
                    warm_result.setdefault("runtime", runtime)
                    return _real_run_payload(
                        warm_result,
                        requested_run_mode=requested_mode,
                        solver_mode_used="hybrid",
                        model_artifacts=model_artifacts,
                        ml_confidence=float(prediction.get("mlConfidence") or 0.0),
                        repair_applied=bool(prediction.get("repairApplied")),
                        hybrid_strategy_requested=requested_hybrid_strategy,
                        hybrid_strategy_used="warm_start",
                        constraint_aware_hybrid_used=constraint_aware_hybrid_used,
                        reduced_solve_applied=reduced_solve_applied,
                        fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
                        predicted_active_constraint_count=predicted_active_constraint_count,
                        constraint_confidence=constraint_confidence_value,
                        repair_after_reduced_solve=True,
                        reduced_solve_fallback_reason=reduced_solve_fallback_reason,
                        constraint_scoring_used=constraint_scoring_used,
                        critical_constraint_count=critical_constraint_count,
                        deferred_constraint_count=deferred_constraint_count,
                        constraint_reactivation_count=constraint_reactivation_count,
                        staged_solve_rounds=staged_solve_rounds,
                        constraint_aware_reduction_mode=constraint_aware_reduction_mode,
                        reduced_model_validated=reduced_model_validated,
                        reduction_rejected_reason=reduction_rejected_reason,
                    )
            fallback_result, exact_error = _run_solver(module, data_path, overrides=overrides, time_limit_ms=time_limit_ms)
            if exact_error is None and _is_real_result(fallback_result):
                assert fallback_result is not None
                fallback_result.setdefault("runtime", runtime)
                return _real_run_payload(
                    fallback_result,
                    requested_run_mode=requested_mode,
                    solver_mode_used="exact",
                    model_artifacts=model_artifacts,
                    ml_confidence=float(prediction.get("mlConfidence") or 0.0),
                    repair_applied=bool(prediction.get("repairApplied")),
                    fallback_reason=fallback_reason,
                    hybrid_strategy_requested=requested_hybrid_strategy,
                    hybrid_strategy_used="exact",
                    constraint_aware_hybrid_used=constraint_aware_hybrid_used,
                    reduced_solve_applied=reduced_solve_applied,
                    fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
                    predicted_active_constraint_count=predicted_active_constraint_count,
                    constraint_confidence=constraint_confidence_value,
                    repair_after_reduced_solve=True,
                    reduced_solve_fallback_reason=reduced_solve_fallback_reason,
                    constraint_scoring_used=constraint_scoring_used,
                    critical_constraint_count=critical_constraint_count,
                    deferred_constraint_count=deferred_constraint_count,
                    constraint_reactivation_count=constraint_reactivation_count,
                    staged_solve_rounds=staged_solve_rounds,
                    constraint_aware_reduction_mode=constraint_aware_reduction_mode,
                    reduced_model_validated=reduced_model_validated,
                    reduction_rejected_reason=reduction_rejected_reason,
                )
        elapsed_ms = (perf_counter() - start) * 1000.0
        if prediction.get("feasible"):
            return _ml_run_payload(
                prediction,
                elapsed_ms=elapsed_ms,
                requested_run_mode=requested_mode,
                solver_mode_used="ml",
                model_artifacts=model_artifacts,
                fallback_reason=fallback_reason,
                hybrid_strategy_requested=requested_hybrid_strategy,
                hybrid_strategy_used="ml",
                constraint_aware_hybrid_used=constraint_aware_hybrid_used,
                reduced_solve_applied=reduced_solve_applied,
                fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
                predicted_active_constraint_count=predicted_active_constraint_count,
                constraint_confidence=constraint_confidence_value,
                repair_after_reduced_solve=True,
                reduced_solve_fallback_reason=reduced_solve_fallback_reason,
                constraint_scoring_used=constraint_scoring_used,
                critical_constraint_count=critical_constraint_count,
                deferred_constraint_count=deferred_constraint_count,
                constraint_reactivation_count=constraint_reactivation_count,
                staged_solve_rounds=staged_solve_rounds,
                constraint_aware_reduction_mode=constraint_aware_reduction_mode,
                reduced_model_validated=reduced_model_validated,
                reduction_rejected_reason=reduction_rejected_reason,
            )
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=fallback_reason,
            requested_run_mode=requested_mode,
            solver_mode_used="hybrid",
            preview=preview,
            preview_error=preview_error,
            fallback_reason=fallback_reason,
            model_artifacts=model_artifacts,
            hybrid_strategy_requested=requested_hybrid_strategy,
            hybrid_strategy_used=requested_hybrid_strategy,
            constraint_aware_hybrid_used=constraint_aware_hybrid_used,
            reduced_solve_applied=reduced_solve_applied,
            fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
            predicted_active_constraint_count=predicted_active_constraint_count,
            constraint_confidence=constraint_confidence_value,
            repair_after_reduced_solve=True,
            reduced_solve_fallback_reason=reduced_solve_fallback_reason,
            constraint_scoring_used=constraint_scoring_used,
            critical_constraint_count=critical_constraint_count,
            deferred_constraint_count=deferred_constraint_count,
            constraint_reactivation_count=constraint_reactivation_count,
            staged_solve_rounds=staged_solve_rounds,
            constraint_aware_reduction_mode=constraint_aware_reduction_mode,
            reduced_model_validated=reduced_model_validated,
            reduction_rejected_reason=reduction_rejected_reason,
        )

    if result is not None:
        result.setdefault("runtime", runtime)
    if preview and result is not None:
        result.setdefault("summary", preview.get("summary", {}))
        result.setdefault("totalLoadByHour", preview.get("totalLoadByHour", []))
        result.setdefault("generatorCapacityPreview", preview.get("generatorCapacityPreview", []))

    if _is_real_result(result):
        assert result is not None
        return _real_run_payload(
            result,
            requested_run_mode=requested_mode,
            solver_mode_used="hybrid",
            model_artifacts=model_artifacts,
            ml_confidence=float(prediction.get("mlConfidence") or 0.0),
            repair_applied=bool(prediction.get("repairApplied")),
            hybrid_strategy_requested=requested_hybrid_strategy,
            hybrid_strategy_used=requested_hybrid_strategy,
            constraint_aware_hybrid_used=constraint_aware_hybrid_used,
            reduced_solve_applied=reduced_solve_applied,
            fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
            predicted_active_constraint_count=predicted_active_constraint_count,
            constraint_confidence=constraint_confidence_value,
            repair_after_reduced_solve=bool(constraint_reactivation_count > 0),
            reduced_solve_fallback_reason=reduced_solve_fallback_reason,
            constraint_scoring_used=constraint_scoring_used,
            critical_constraint_count=critical_constraint_count,
            deferred_constraint_count=deferred_constraint_count,
            constraint_reactivation_count=constraint_reactivation_count,
            staged_solve_rounds=staged_solve_rounds,
            constraint_aware_reduction_mode=constraint_aware_reduction_mode,
            reduced_model_validated=reduced_model_validated,
            reduction_rejected_reason=reduction_rejected_reason,
        )

    elapsed_ms = float(result.get("solveTimeMs") or (perf_counter() - start) * 1000.0) if result else (perf_counter() - start) * 1000.0
    if prediction.get("feasible"):
        return _ml_run_payload(
            prediction,
            elapsed_ms=elapsed_ms,
            requested_run_mode=requested_mode,
            solver_mode_used="ml",
            model_artifacts=model_artifacts,
            fallback_reason="hybrid solve returned non-real status; returned ML schedule",
            hybrid_strategy_requested=requested_hybrid_strategy,
            hybrid_strategy_used="ml",
            constraint_aware_hybrid_used=constraint_aware_hybrid_used,
            reduced_solve_applied=reduced_solve_applied,
            fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
            predicted_active_constraint_count=predicted_active_constraint_count,
            constraint_confidence=constraint_confidence_value,
            repair_after_reduced_solve=True,
            reduced_solve_fallback_reason="hybrid solve returned non-real status; returned ML schedule",
            constraint_scoring_used=constraint_scoring_used,
            critical_constraint_count=critical_constraint_count,
            deferred_constraint_count=deferred_constraint_count,
            constraint_reactivation_count=constraint_reactivation_count,
            staged_solve_rounds=staged_solve_rounds,
            constraint_aware_reduction_mode=constraint_aware_reduction_mode,
            reduced_model_validated=reduced_model_validated,
            reduction_rejected_reason=reduction_rejected_reason,
        )

    return _compat_run_payload(
        elapsed_ms=elapsed_ms,
        reason=f"hybrid solve returned non-real result status={result.get('statusName', 'UNKNOWN') if result else 'UNKNOWN'}",
        requested_run_mode=requested_mode,
        solver_mode_used="hybrid",
        preview=preview,
        preview_error=preview_error,
        model_artifacts=model_artifacts,
        hybrid_strategy_requested=requested_hybrid_strategy,
        hybrid_strategy_used=requested_hybrid_strategy,
        constraint_aware_hybrid_used=constraint_aware_hybrid_used,
        reduced_solve_applied=reduced_solve_applied,
        fixed_commitment_count=int(fixing_plan["fixedCommitmentCount"]),
        predicted_active_constraint_count=predicted_active_constraint_count,
        constraint_confidence=constraint_confidence_value,
        repair_after_reduced_solve=True,
        reduced_solve_fallback_reason="hybrid solve returned non-real result",
        constraint_scoring_used=constraint_scoring_used,
        critical_constraint_count=critical_constraint_count,
        deferred_constraint_count=deferred_constraint_count,
        constraint_reactivation_count=constraint_reactivation_count,
        staged_solve_rounds=staged_solve_rounds,
        constraint_aware_reduction_mode=constraint_aware_reduction_mode,
        reduced_model_validated=reduced_model_validated,
        reduction_rejected_reason=reduction_rejected_reason or "hybrid solve returned non-real result",
    )
