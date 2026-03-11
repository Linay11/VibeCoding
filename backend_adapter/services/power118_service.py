from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict


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


def _load_preview(module: Any, data_path: Path) -> tuple[Dict[str, Any] | None, str | None]:
    if not hasattr(module, "load_power118_data"):
        return None, "load_power118_data is missing"

    try:
        preview = module.load_power118_data(data_path=data_path)
    except Exception as exc:  # pragma: no cover - depends on optional deps/runtime
        return None, str(exc)

    return preview if isinstance(preview, dict) else None, None


def _translate_runtime_reason(runtime: Dict[str, Any]) -> str:
    stage = runtime.get("stage") or "unknown"
    reason = str(runtime.get("reason") or "unknown runtime failure").strip()
    return f"runtime blocked at {stage}: {reason}"


def _compat_run_payload(
    elapsed_ms: float,
    reason: str,
    preview: Dict[str, Any] | None = None,
    preview_error: str | None = None,
) -> Dict[str, Any]:
    preview = preview or {}
    total_load_by_hour = preview.get("totalLoadByHour") if isinstance(preview.get("totalLoadByHour"), list) else []
    generator_capacity_preview = (
        preview.get("generatorCapacityPreview")
        if isinstance(preview.get("generatorCapacityPreview"), list)
        else []
    )
    summary = preview.get("summary") if isinstance(preview.get("summary"), dict) else {}

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
    if summary:
        note_parts.append(f"peakLoad={summary.get('peakLoad', 0.0)}.")

    return {
        "runId": _run_id(),
        "scenarioId": "power-118",
        "generatedAt": _utc_now_iso(),
        "metrics": {
            "solveTimeMs": float(max(elapsed_ms, 0.0)),
            "infeasibilityRate": 0.0,
            "suboptimality": 0.0,
        },
        "strategies": [],
        "trend": [
            {"label": f"H{idx + 1}", "value": float(value)}
            for idx, value in enumerate(total_load_by_hour)
        ],
        "comparison": generator_capacity_preview[:4],
        "adapterMode": "compat",
        "adapterNote": " ".join(note_parts),
    }


def _real_run_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    objective = float(result.get("objective") or 0.0)
    top_generators = result.get("topGenerators") if isinstance(result.get("topGenerators"), list) else []
    total_load_by_hour = result.get("totalLoadByHour") if isinstance(result.get("totalLoadByHour"), list) else []
    peak_line_flow_by_hour = result.get("peakLineFlowByHour") if isinstance(result.get("peakLineFlowByHour"), list) else []
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}

    note_parts = [
        "power-118 real mode: external SCUC solver completed.",
        "objective is mapped to strategies[0].cost.",
        "solveTimeMs is mapped from solver elapsed time.",
        "trend is mapped from hourly system load totals.",
        "comparison is mapped from solved generator dispatch totals.",
        "Only one strategy is exposed because the external solver returns a single schedule, not multiple candidates.",
        f"Workbook={_power118_data_label()}.",
        f"Script={_power118_script_label()}.",
        f"status={result.get('statusName')}.",
    ]
    if runtime:
        note_parts.append(f"runtimeStage={runtime.get('stage', 'unknown')}.")
    if summary:
        note_parts.append(f"peakLoad={summary.get('peakLoad', 0.0)}.")
    if peak_line_flow_by_hour:
        note_parts.append(f"peakLineFlow={max(peak_line_flow_by_hour)}.")

    return {
        "runId": _run_id(),
        "scenarioId": "power-118",
        "generatedAt": _utc_now_iso(),
        "metrics": {
            "solveTimeMs": float(result.get("solveTimeMs") or 0.0),
            "infeasibilityRate": 0.0,
            "suboptimality": 0.0,
        },
        "strategies": [
            {
                "id": "strategy-1",
                "name": "SCUC-118 Schedule",
                "feasible": True,
                "cost": objective,
                "rank": 1,
            }
        ],
        "trend": [
            {"label": f"H{idx + 1}", "value": float(value)}
            for idx, value in enumerate(total_load_by_hour)
        ],
        "comparison": top_generators[:4],
        "adapterMode": "real",
        "adapterNote": " ".join(note_parts),
    }


def _run_solver(module: Any, data_path: Path) -> tuple[Dict[str, Any] | None, str | None]:
    if not hasattr(module, "solve_scuc_118"):
        return None, "solve_scuc_118 is missing"

    try:
        result = module.solve_scuc_118(data_path=data_path, write_output=False)
    except Exception as exc:  # pragma: no cover - depends on solver/runtime
        return None, str(exc)

    if not isinstance(result, dict):
        return None, "solve_scuc_118 returned a non-dict result"
    return result, None


def run_power118_once() -> Dict[str, Any]:
    start = perf_counter()
    module = _load_module()
    data_path = _power118_data()

    preview, preview_error = _load_preview(module, data_path)
    runtime = _detect_runtime(module)
    if not runtime.get("available"):
        elapsed_ms = (perf_counter() - start) * 1000.0
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=_translate_runtime_reason(runtime),
            preview=preview,
            preview_error=preview_error,
        )

    result, solve_error = _run_solver(module, data_path)
    if solve_error:
        elapsed_ms = (perf_counter() - start) * 1000.0
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=f"external solve failed after runtime ready: {solve_error}",
            preview=preview,
            preview_error=preview_error,
        )

    if preview and result is not None:
        result.setdefault("summary", preview.get("summary", {}))
        result.setdefault("totalLoadByHour", preview.get("totalLoadByHour", []))
        result.setdefault("generatorCapacityPreview", preview.get("generatorCapacityPreview", []))
    if result is not None:
        result.setdefault("runtime", runtime)

    status = str(result.get("statusName") or "").upper()
    feasible = bool(result.get("feasible", status in {"OPTIMAL", "SUBOPTIMAL"}))
    if status not in {"OPTIMAL", "SUBOPTIMAL"} or not feasible:
        elapsed_ms = float(result.get("solveTimeMs") or (perf_counter() - start) * 1000.0)
        return _compat_run_payload(
            elapsed_ms=elapsed_ms,
            reason=f"solver returned non-real result status={status or 'UNKNOWN'} feasible={feasible}",
            preview=preview,
            preview_error=preview_error,
        )

    return _real_run_payload(result)
