from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if parsed != parsed:  # NaN guard
            return default
        return parsed
    except Exception:
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    solve_time_ms = max(
        0.0,
        _to_float(metrics.get("solveTimeMs", raw.get("solveTimeMs", raw.get("timeMs", 0.0))), default=0.0),
    )
    infeasibility_rate = _to_float(metrics.get("infeasibilityRate", raw.get("infeasibilityRate", 0.0)), default=0.0)
    infeasibility_rate = min(max(infeasibility_rate, 0.0), 1.0)
    suboptimality = max(
        0.0,
        _to_float(metrics.get("suboptimality", raw.get("suboptimality", 0.0)), default=0.0),
    )
    return {
        "solveTimeMs": solve_time_ms,
        "infeasibilityRate": infeasibility_rate,
        "suboptimality": suboptimality,
    }


def _normalize_strategies(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    strategies = raw.get("strategies")
    if not isinstance(strategies, list):
        return []

    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(strategies):
        item = row if isinstance(row, dict) else {}
        rank = _to_int(item.get("rank"), index + 1)
        rows.append(
            {
                "id": str(item.get("id") or f"strategy-{index + 1}"),
                "name": str(item.get("name") or f"Strategy {index + 1}"),
                "feasible": bool(item.get("feasible", False)),
                "cost": _to_float(item.get("cost", 0.0), default=0.0),
                "rank": max(1, rank),
            }
        )
    rows.sort(key=lambda x: (x["rank"], x["id"]))
    return rows


def _normalize_trend(raw: Dict[str, Any], solve_time_ms: float) -> List[Dict[str, float]]:
    trend = raw.get("trend")
    points: List[Dict[str, float]] = []
    if isinstance(trend, list):
        for index, row in enumerate(trend):
            item = row if isinstance(row, dict) else {}
            value = _to_float(item.get("value", item.get("solveTimeMs", item.get("solve", 0.0))), default=0.0)
            if value < 0.0:
                continue
            points.append(
                {
                    "label": str(item.get("label") or f"R-{index + 1}"),
                    "value": value,
                }
            )
    if points:
        return points

    if solve_time_ms <= 0.0:
        return []

    multipliers = [1.15, 1.08, 1.03, 1.00, 0.95, 0.90]
    return [
        {
            "label": f"R-{len(multipliers) - idx}",
            "value": max(0.1, solve_time_ms * m),
        }
        for idx, m in enumerate(multipliers)
    ]


def _normalize_comparison(raw: Dict[str, Any], strategies: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    comparison = raw.get("comparison")
    if not isinstance(comparison, list):
        comparison = raw.get("comparisons")

    rows: List[Dict[str, float]] = []
    if isinstance(comparison, list):
        for index, row in enumerate(comparison):
            item = row if isinstance(row, dict) else {}
            value = _to_float(item.get("value", item.get("cost", 0.0)), default=0.0)
            rows.append(
                {
                    "label": str(item.get("label") or f"Item {index + 1}"),
                    "value": value,
                }
            )
    if rows:
        return rows

    return [{"label": row["name"], "value": row["cost"]} for row in strategies[:4]]


def normalize_run_payload(payload: Any, scenario_id: str) -> Dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}

    normalized_scenario_id = str(raw.get("scenarioId") or scenario_id)
    generated_at = raw.get("generatedAt")
    if not isinstance(generated_at, str) or not generated_at.strip():
        generated_at = _utc_now_iso()

    metrics = _normalize_metrics(raw)
    strategies = _normalize_strategies(raw)
    trend = _normalize_trend(raw, metrics["solveTimeMs"])
    comparison = _normalize_comparison(raw, strategies)

    mode = str(raw.get("adapterMode") or "compat").strip().lower()
    if mode not in {"real", "compat"}:
        mode = "compat"

    note = raw.get("adapterNote")
    if not isinstance(note, str) or not note.strip():
        note = "Real backend execution completed." if mode == "real" else "Compatibility mode run from backend adapter."

    run_id = raw.get("runId") or raw.get("id")
    if not isinstance(run_id, str) or not run_id.strip():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        run_id = f"run-{normalized_scenario_id}-{ts}"

    return {
        "runId": run_id,
        "scenarioId": normalized_scenario_id,
        "generatedAt": generated_at,
        "metrics": metrics,
        "strategies": strategies,
        "trend": trend,
        "comparison": comparison,
        "adapterMode": mode,
        "adapterNote": note,
    }
