from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

from backend_adapter.services.power118_dataset import load_power118_data
from backend_adapter.services.power118_ml_model import (
    load_power118_model_artifacts,
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
) -> Dict[str, Any]:
    artifacts = model_artifacts or _base_model_artifacts()
    payload["requestedRunMode"] = requested_run_mode
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
    )


def _real_run_payload(
    result: Dict[str, Any],
    requested_run_mode: str,
    solver_mode_used: str,
    model_artifacts: dict[str, Any] | None = None,
    ml_confidence: float | None = None,
    repair_applied: bool | None = None,
    fallback_reason: str | None = None,
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
    )


def _ml_run_payload(
    prediction: Dict[str, Any],
    elapsed_ms: float,
    requested_run_mode: str,
    solver_mode_used: str,
    model_artifacts: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
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
    )


def _run_solver(
    module: Any,
    data_path: Path,
    overrides: dict[str, Any] | None = None,
    initial_unit_commitment: list[list[float]] | None = None,
    initial_dispatch: list[list[float]] | None = None,
    time_limit_ms: int | None = None,
) -> tuple[Dict[str, Any] | None, str | None]:
    if not hasattr(module, "solve_scuc_118"):
        return None, "solve_scuc_118 is missing"

    try:
        result = module.solve_scuc_118(
            data_path=data_path,
            write_output=False,
            overrides=overrides,
            initial_unit_commitment=initial_unit_commitment,
            initial_dispatch=initial_dispatch,
            time_limit_s=(float(time_limit_ms) / 1000.0 if time_limit_ms else None),
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
    return status in {"OPTIMAL", "SUBOPTIMAL"} and feasible


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


def run_power118_once(
    run_mode: str = "exact",
    time_limit_ms: int | None = None,
    fallback_to_exact: bool = True,
    overrides: dict[str, Any] | None = None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Dict[str, Any]:
    start = perf_counter()
    requested_mode = (run_mode or "exact").strip().lower()
    if requested_mode not in {"exact", "hybrid", "ml"}:
        requested_mode = "exact"

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
            )
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=_translate_runtime_reason(runtime),
            requested_run_mode=requested_mode,
            solver_mode_used="hybrid",
            preview=preview,
            preview_error=preview_error,
            model_artifacts=model_artifacts,
        )

    result, solve_error = _run_solver(
        module,
        data_path,
        overrides=overrides,
        initial_unit_commitment=prediction.get("unitCommitmentByHour"),
        initial_dispatch=prediction.get("generatorDispatchByHour"),
        time_limit_ms=time_limit_ms,
    )
    if solve_error:
        fallback_reason = f"hybrid warm-start solve failed: {solve_error}"
        if fallback_to_exact:
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
        )

    return _compat_run_payload(
        elapsed_ms=elapsed_ms,
        reason=f"hybrid solve returned non-real result status={result.get('statusName', 'UNKNOWN') if result else 'UNKNOWN'}",
        requested_run_mode=requested_mode,
        solver_mode_used="hybrid",
        preview=preview,
        preview_error=preview_error,
        model_artifacts=model_artifacts,
    )
