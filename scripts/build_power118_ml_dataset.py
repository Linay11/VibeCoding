from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend_adapter.services.power118_data_augment import generate_power118_override_set
from backend_adapter.services.power118_dataset import (
    build_power118_constraint_label_record,
    build_power118_constraint_candidate_records,
    build_power118_feature_record,
    build_power118_fixing_label_record,
    build_power118_metadata_record,
    build_power118_target_record,
    load_power118_data,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend_adapter" / "data" / "power118_dataset"
DEFAULT_DATASET_FILENAME = "power118_ml_dataset.pkl"
DEFAULT_SUMMARY_FILENAME = "dataset_summary.json"
DEFAULT_IMPACT_PROBE_GROUPS = 3
DEFAULT_IMPACT_PROBE_GROUP_SIZE = 12
DEFAULT_IMPACT_OBJECTIVE_DELTA_THRESHOLD = 0.002
DEFAULT_IMPACT_RUNTIME_DELTA_THRESHOLD = 0.15
DEFAULT_IMPACT_RUNTIME_DELTA_MS_THRESHOLD = 50.0
DEFAULT_IMPACT_PROBE_TIME_LIMIT_S = 3.0


def _load_power118_module():
    from backend_adapter.services.power118_dataset import _load_power118_module  # type: ignore[attr-defined]

    return _load_power118_module()


def _write_dataset_summary(
    output_dir: Path,
    summary: dict[str, Any],
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
) -> Path:
    summary_path = output_dir / summary_filename
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _chunked(sequence: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return []
    return [sequence[index : index + chunk_size] for index in range(0, len(sequence), chunk_size)]


def _relative_delta(probe_value: float | None, base_value: float | None) -> float | None:
    if probe_value is None or base_value is None:
        return None
    denominator = max(abs(base_value), 1.0)
    return float((probe_value - base_value) / denominator)


def _derive_impact_probe_time_limit_s(
    baseline_result: dict[str, Any],
    requested_time_limit_s: float | None,
    impact_probe_time_limit_s: float | None,
) -> float | None:
    if impact_probe_time_limit_s is not None:
        return max(0.0, float(impact_probe_time_limit_s))
    if requested_time_limit_s is not None:
        return max(0.0, float(requested_time_limit_s))
    baseline_runtime_ms = float(baseline_result.get("solveTimeMs") or 0.0)
    if baseline_runtime_ms <= 0.0:
        return DEFAULT_IMPACT_PROBE_TIME_LIMIT_S
    return max(1.0, min(DEFAULT_IMPACT_PROBE_TIME_LIMIT_S, (baseline_runtime_ms / 1000.0) * 1.25))


def _build_constraint_impact_records(
    external_module: Any,
    baseline_result: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    data_path: str | Path | None,
    overrides: dict[str, Any] | None,
    time_limit_s: float | None,
    impact_probe_groups: int,
    impact_probe_group_size: int,
    impact_probe_time_limit_s: float | None,
    objective_delta_threshold: float = DEFAULT_IMPACT_OBJECTIVE_DELTA_THRESHOLD,
    runtime_delta_threshold: float = DEFAULT_IMPACT_RUNTIME_DELTA_THRESHOLD,
    runtime_delta_ms_threshold: float = DEFAULT_IMPACT_RUNTIME_DELTA_MS_THRESHOLD,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    summary: dict[str, Any] = {
        "impactProbeGroupAttemptedCount": 0,
        "impactProbeGroupEvaluatedCount": 0,
        "impactProbeGroupFailedCount": 0,
        "impactProbeConstraintAttemptedCount": 0,
        "impactProbeConstraintEvaluatedCount": 0,
        "impactProbeConstraintFallbackCount": 0,
        "impactCriticalExactCount": 0,
        "impactProbeEnabled": bool(impact_probe_groups > 0 and impact_probe_group_size > 0),
    }
    if impact_probe_groups <= 0 or impact_probe_group_size <= 0:
        return {}, summary

    diagnostics = (
        baseline_result.get("constraintDiagnostics")
        if isinstance(baseline_result.get("constraintDiagnostics"), dict)
        else {}
    )
    ramp_active_ids = {
        str(value)
        for value in diagnostics.get("rampActiveIndices", [])
        if str(value).startswith("ramp:")
    }
    line_active_ids = {
        str(value)
        for value in diagnostics.get("lineActiveIndices", [])
        if str(value).startswith("line:")
    }
    if not ramp_active_ids and not line_active_ids:
        return {}, summary

    reducible_rows = [
        row
        for row in candidate_records
        if isinstance(row, dict)
        and float(row.get("canBeReduced", 0.0) or 0.0) >= 0.5
        and float(row.get("labelActive", 0.0) or 0.0) >= 0.5
        and row.get("constraintId")
    ]
    reducible_rows.sort(key=lambda row: float(row.get("labelRankScore", 0.0) or 0.0), reverse=True)

    selected_constraint_ids: list[str] = []
    seen_constraint_ids: set[str] = set()
    max_constraints = impact_probe_groups * impact_probe_group_size
    for row in reducible_rows:
        constraint_id = str(row.get("constraintId") or "")
        if not constraint_id or constraint_id in seen_constraint_ids:
            continue
        if not (constraint_id.startswith("ramp:") or constraint_id.startswith("line:")):
            continue
        selected_constraint_ids.append(constraint_id)
        seen_constraint_ids.add(constraint_id)
        if len(selected_constraint_ids) >= max_constraints:
            break
    if not selected_constraint_ids:
        return {}, summary

    constraint_groups = _chunked(selected_constraint_ids, impact_probe_group_size)[:impact_probe_groups]
    baseline_objective = (
        float(baseline_result["objective"])
        if baseline_result.get("objective") is not None
        else None
    )
    baseline_runtime_ms = (
        float(baseline_result["solveTimeMs"])
        if baseline_result.get("solveTimeMs") is not None
        else None
    )
    baseline_feasible = bool(baseline_result.get("feasible"))
    probe_time_limit_s = _derive_impact_probe_time_limit_s(
        baseline_result=baseline_result,
        requested_time_limit_s=time_limit_s,
        impact_probe_time_limit_s=impact_probe_time_limit_s,
    )

    impact_records: dict[str, dict[str, Any]] = {}
    for group_index, group_constraint_ids in enumerate(constraint_groups, start=1):
        group_id = f"impact-group-{group_index:02d}"
        summary["impactProbeGroupAttemptedCount"] += 1
        summary["impactProbeConstraintAttemptedCount"] += len(group_constraint_ids)
        removed_ramp = {constraint_id for constraint_id in group_constraint_ids if constraint_id.startswith("ramp:")}
        removed_line = {constraint_id for constraint_id in group_constraint_ids if constraint_id.startswith("line:")}
        active_ramp_ids = sorted(ramp_active_ids - removed_ramp)
        active_line_ids = sorted(line_active_ids - removed_line)
        if len(active_ramp_ids) == len(ramp_active_ids) and len(active_line_ids) == len(line_active_ids):
            summary["impactProbeGroupFailedCount"] += 1
            for constraint_id in group_constraint_ids:
                impact_records[constraint_id] = {
                    "probeGroupId": group_id,
                    "probeEvaluated": 0.0,
                    "probeStatus": "skipped_not_reducible",
                    "exactLabelAvailable": 0.0,
                    "labelCriticalExact": 0.0,
                    "fallbackLabelSource": "impact_probe_unavailable_fallback",
                }
            summary["impactProbeConstraintFallbackCount"] += len(group_constraint_ids)
            continue

        try:
            probe_result = external_module.solve_scuc_118(
                data_path=data_path,
                write_output=False,
                overrides=overrides,
                time_limit_s=probe_time_limit_s,
                initial_unit_commitment=baseline_result.get("unitCommitmentByHour"),
                initial_dispatch=baseline_result.get("generatorDispatchByHour"),
                active_ramp_constraint_ids=active_ramp_ids,
                active_line_constraint_ids=active_line_ids,
            )
        except Exception as exc:
            summary["impactProbeGroupFailedCount"] += 1
            for constraint_id in group_constraint_ids:
                impact_records[constraint_id] = {
                    "probeGroupId": group_id,
                    "probeEvaluated": 0.0,
                    "probeStatus": f"probe_failed:{exc}",
                    "exactLabelAvailable": 0.0,
                    "labelCriticalExact": 0.0,
                    "fallbackLabelSource": "impact_probe_failed_fallback",
                }
            summary["impactProbeConstraintFallbackCount"] += len(group_constraint_ids)
            continue

        summary["impactProbeGroupEvaluatedCount"] += 1
        summary["impactProbeConstraintEvaluatedCount"] += len(group_constraint_ids)
        probe_feasible = bool(probe_result.get("feasible"))
        probe_objective = float(probe_result["objective"]) if probe_result.get("objective") is not None else None
        probe_runtime_ms = float(probe_result["solveTimeMs"]) if probe_result.get("solveTimeMs") is not None else None
        impact_feasibility_delta = 1.0 if baseline_feasible and not probe_feasible else 0.0
        impact_objective_delta_ratio = _relative_delta(probe_objective, baseline_objective)
        impact_runtime_delta_ratio = _relative_delta(probe_runtime_ms, baseline_runtime_ms)
        impact_runtime_delta_ms = (
            float(probe_runtime_ms - baseline_runtime_ms)
            if probe_runtime_ms is not None and baseline_runtime_ms is not None
            else None
        )
        objective_critical = bool(
            impact_objective_delta_ratio is not None and impact_objective_delta_ratio >= objective_delta_threshold
        )
        runtime_critical = bool(
            impact_runtime_delta_ratio is not None
            and impact_runtime_delta_ratio >= runtime_delta_threshold
            and impact_runtime_delta_ms is not None
            and impact_runtime_delta_ms >= runtime_delta_ms_threshold
        )
        critical_exact = 1.0 if (impact_feasibility_delta >= 0.5 or objective_critical or runtime_critical) else 0.0
        impact_score = 1.0 if critical_exact >= 0.5 else 0.0
        if impact_feasibility_delta >= 0.5:
            impact_score = 1.0
        else:
            objective_component = max(0.0, (impact_objective_delta_ratio or 0.0) / max(objective_delta_threshold, 1e-6))
            runtime_component = max(0.0, (impact_runtime_delta_ratio or 0.0) / max(runtime_delta_threshold, 1e-6))
            impact_score = float(min(1.0, 0.6 * objective_component + 0.4 * runtime_component))

        if critical_exact >= 0.5:
            summary["impactCriticalExactCount"] += len(group_constraint_ids)

        for constraint_id in group_constraint_ids:
            impact_records[constraint_id] = {
                "probeGroupId": group_id,
                "probeEvaluated": 1.0,
                "probeStatus": str(probe_result.get("statusName") or ("FEASIBLE" if probe_feasible else "INFEASIBLE")),
                "exactLabelAvailable": 1.0,
                "labelCriticalExact": critical_exact,
                "labelSource": "impact_exact_group_probe",
                "impactFeasibilityDelta": impact_feasibility_delta,
                "impactObjectiveDeltaRatio": impact_objective_delta_ratio,
                "impactRuntimeDeltaRatio": impact_runtime_delta_ratio,
                "impactRuntimeDeltaMs": impact_runtime_delta_ms,
                "impactScoreExact": impact_score,
            }

    return impact_records, summary


def build_dataset(
    num_samples: int,
    seed: int,
    output_dir: Path,
    dataset_filename: str,
    time_limit_s: float | None,
    data_path: str | Path | None = None,
    impact_probe_groups: int = DEFAULT_IMPACT_PROBE_GROUPS,
    impact_probe_group_size: int = DEFAULT_IMPACT_PROBE_GROUP_SIZE,
    impact_probe_time_limit_s: float | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    external_module = _load_power118_module()
    runtime = external_module.check_gurobi_runtime()
    if not runtime.get("available"):
        raise RuntimeError(f"Cannot build dataset without exact solver runtime: {runtime.get('reason')}")

    base_data = load_power118_data(data_path=data_path)
    overrides_list = generate_power118_override_set(base_data=base_data, n_samples=num_samples, seed=seed)

    feature_rows: list[dict] = []
    target_rows: list[dict] = []
    constraint_label_rows: list[dict] = []
    constraint_candidate_rows: list[dict] = []
    fixing_label_rows: list[dict] = []
    metadata_rows: list[dict] = []
    dropped_infeasible_count = 0
    dropped_no_incumbent_count = 0
    dropped_by_status: dict[str, int] = {}
    constraint_label_missing_count = 0
    impact_probe_group_attempted_total = 0
    impact_probe_group_evaluated_total = 0
    impact_probe_group_failed_total = 0
    impact_probe_constraint_attempted_total = 0
    impact_probe_constraint_evaluated_total = 0
    impact_probe_constraint_fallback_total = 0
    impact_critical_exact_total = 0

    for sample_index, overrides in enumerate(overrides_list, start=1):
        power_data = load_power118_data(data_path=data_path, overrides=overrides)
        result = external_module.solve_scuc_118(
            data_path=data_path,
            write_output=False,
            overrides=overrides,
            time_limit_s=time_limit_s,
        )
        if not result.get("feasible"):
            status_name = str(result.get("statusName") or "UNKNOWN")
            dropped_by_status[status_name] = dropped_by_status.get(status_name, 0) + 1
            if result.get("hasIncumbent"):
                dropped_infeasible_count += 1
            else:
                dropped_no_incumbent_count += 1
            continue

        sample_id = f"power118-{sample_index:05d}"
        feature_rows.append(build_power118_feature_record(power_data))
        target_rows.append(build_power118_target_record(result))
        constraint_record = build_power118_constraint_label_record(result)
        fixing_record = build_power118_fixing_label_record(result)
        heuristic_candidates = build_power118_constraint_candidate_records(
            power_data=power_data,
            result=result,
            sample_id=sample_id,
        )
        impact_records, impact_summary = _build_constraint_impact_records(
            external_module=external_module,
            baseline_result=result,
            candidate_records=heuristic_candidates,
            data_path=data_path,
            overrides=overrides,
            time_limit_s=time_limit_s,
            impact_probe_groups=impact_probe_groups,
            impact_probe_group_size=impact_probe_group_size,
            impact_probe_time_limit_s=impact_probe_time_limit_s,
        )
        constraint_candidate_rows.extend(
            build_power118_constraint_candidate_records(
                power_data=power_data,
                result=result,
                sample_id=sample_id,
                impact_records_by_constraint_id=impact_records,
            )
        )
        constraint_label_rows.append(constraint_record)
        fixing_label_rows.append(fixing_record)
        impact_probe_group_attempted_total += int(impact_summary["impactProbeGroupAttemptedCount"])
        impact_probe_group_evaluated_total += int(impact_summary["impactProbeGroupEvaluatedCount"])
        impact_probe_group_failed_total += int(impact_summary["impactProbeGroupFailedCount"])
        impact_probe_constraint_attempted_total += int(impact_summary["impactProbeConstraintAttemptedCount"])
        impact_probe_constraint_evaluated_total += int(impact_summary["impactProbeConstraintEvaluatedCount"])
        impact_probe_constraint_fallback_total += int(impact_summary["impactProbeConstraintFallbackCount"])
        impact_critical_exact_total += int(impact_summary["impactCriticalExactCount"])
        if not constraint_record.get("constraintLabelAvailable"):
            constraint_label_missing_count += 1
        metadata_rows.append(
            build_power118_metadata_record(
                power_data=power_data,
                overrides=overrides,
                sample_id=sample_id,
                split="train",
                result=result,
            )
        )

    dataset_bundle = {
        "features": pd.DataFrame(feature_rows),
        "targets": pd.DataFrame(target_rows),
        "constraint_labels": pd.DataFrame(constraint_label_rows),
        "constraint_candidates": pd.DataFrame(constraint_candidate_rows),
        "fixing_labels": pd.DataFrame(fixing_label_rows),
        "metadata": pd.DataFrame(metadata_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / dataset_filename
    pd.to_pickle(dataset_bundle, dataset_path)

    constraint_frame = pd.DataFrame(constraint_label_rows) if constraint_label_rows else pd.DataFrame()
    candidate_frame = pd.DataFrame(constraint_candidate_rows) if constraint_candidate_rows else pd.DataFrame()
    dataset_summary = {
        "datasetPath": str(dataset_path),
        "summaryPath": str(output_dir / DEFAULT_SUMMARY_FILENAME),
        "inputDataPath": str(data_path or base_data.get("dataPath") or ""),
        "seed": seed,
        "requestedSampleCount": int(num_samples),
        "keptSampleCount": int(len(feature_rows)),
        "droppedInfeasibleCount": int(dropped_infeasible_count),
        "droppedNoIncumbentCount": int(dropped_no_incumbent_count),
        "droppedByStatus": dropped_by_status,
        "constraintLabelMissingCount": int(constraint_label_missing_count),
        "constraintLabelGenerated": bool(len(constraint_label_rows) == len(feature_rows) and len(feature_rows) > 0),
        "timeLimitS": float(time_limit_s) if time_limit_s is not None else None,
        "exactBaselineUsed": bool(runtime.get("available")),
        "runtimeStage": runtime.get("stage"),
        "runtimeReason": runtime.get("reason"),
        "impactLabelMode": "group_probe_v1",
        "impactProbeEnabled": bool(impact_probe_groups > 0 and impact_probe_group_size > 0),
        "impactProbeGroupsPerSample": int(max(impact_probe_groups, 0)),
        "impactProbeGroupSize": int(max(impact_probe_group_size, 0)),
        "impactProbeTimeLimitS": (
            float(impact_probe_time_limit_s)
            if impact_probe_time_limit_s is not None
            else None
        ),
        "impactProbeGroupAttemptedTotal": int(impact_probe_group_attempted_total),
        "impactProbeGroupEvaluatedTotal": int(impact_probe_group_evaluated_total),
        "impactProbeGroupFailedTotal": int(impact_probe_group_failed_total),
        "impactProbeConstraintAttemptedTotal": int(impact_probe_constraint_attempted_total),
        "impactProbeConstraintEvaluatedTotal": int(impact_probe_constraint_evaluated_total),
        "impactProbeConstraintFallbackTotal": int(impact_probe_constraint_fallback_total),
        "impactCriticalExactTotal": int(impact_critical_exact_total),
    }
    if not constraint_frame.empty:
        dataset_summary["avgActiveConstraintCount"] = float(
            pd.to_numeric(constraint_frame["constraint_totalActiveConstraintCount"], errors="coerce").mean()
        )
        dataset_summary["avgActiveGeneratorLimitRatio"] = float(
            pd.to_numeric(constraint_frame["constraint_activeGeneratorLimitRatio"], errors="coerce").mean()
        )
        dataset_summary["avgActiveRampRatio"] = float(
            pd.to_numeric(constraint_frame["constraint_activeRampRatio"], errors="coerce").mean()
        )
        dataset_summary["avgActiveLineRatio"] = float(
            pd.to_numeric(constraint_frame["constraint_activeLineRatio"], errors="coerce").mean()
        )
    if not candidate_frame.empty:
        dataset_summary["constraintCandidateCount"] = int(len(candidate_frame))
        dataset_summary["constraintCandidateTypeCounts"] = {
            str(key): int(value)
            for key, value in candidate_frame["constraintType"].value_counts(dropna=False).to_dict().items()
        }
        dataset_summary["constraintCandidateActiveRatio"] = float(
            pd.to_numeric(candidate_frame["labelActive"], errors="coerce").mean()
        )
        dataset_summary["constraintCandidateTightRatio"] = float(
            pd.to_numeric(candidate_frame["labelTight"], errors="coerce").mean()
        )
        dataset_summary["constraintCandidateTopRankCoverage"] = float(
            pd.to_numeric(candidate_frame["labelRankScore"], errors="coerce").ge(0.5).mean()
        )
        if "labelCriticalExactAvailable" in candidate_frame.columns:
            dataset_summary["constraintCandidateExactLabelCoverage"] = float(
                pd.to_numeric(candidate_frame["labelCriticalExactAvailable"], errors="coerce").mean()
            )
        if "labelCritical" in candidate_frame.columns:
            dataset_summary["constraintCandidateCriticalRate"] = float(
                pd.to_numeric(candidate_frame["labelCritical"], errors="coerce").mean()
            )
        if "labelSource" in candidate_frame.columns:
            dataset_summary["constraintCandidateLabelSourceCounts"] = {
                str(key): int(value)
                for key, value in candidate_frame["labelSource"].fillna("").astype(str).value_counts(dropna=False).to_dict().items()
            }
    summary_path = _write_dataset_summary(output_dir, dataset_summary)
    return dataset_path, summary_path, dataset_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build supervised dataset for power-118 SCUC ML experiments.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of perturbed SCUC samples to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for data augmentation.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional override for the source power118 workbook path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for dataset artifacts.",
    )
    parser.add_argument(
        "--dataset-filename",
        type=str,
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset pickle filename written into the output directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Deprecated direct output path. If provided, it overrides output-dir and dataset-filename.",
    )
    parser.add_argument(
        "--time-limit-s",
        type=float,
        default=None,
        help="Optional exact SCUC solver time limit in seconds.",
    )
    parser.add_argument(
        "--impact-probe-groups",
        type=int,
        default=DEFAULT_IMPACT_PROBE_GROUPS,
        help="Number of reduced-constraint probe groups per sample for impact-aware labels.",
    )
    parser.add_argument(
        "--impact-probe-group-size",
        type=int,
        default=DEFAULT_IMPACT_PROBE_GROUP_SIZE,
        help="Constraints per reduced-constraint probe group.",
    )
    parser.add_argument(
        "--impact-probe-time-limit-s",
        type=float,
        default=None,
        help="Optional time limit per impact probe solve in seconds.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    dataset_filename = args.dataset_filename
    if args.output_path is not None:
        output_dir = args.output_path.resolve().parent
        dataset_filename = args.output_path.name

    dataset_path, summary_path, dataset_summary = build_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=output_dir,
        dataset_filename=dataset_filename,
        time_limit_s=args.time_limit_s,
        data_path=args.data_path.resolve() if args.data_path is not None else None,
        impact_probe_groups=args.impact_probe_groups,
        impact_probe_group_size=args.impact_probe_group_size,
        impact_probe_time_limit_s=args.impact_probe_time_limit_s,
    )
    print("Power118 dataset build")
    print(f"- Input source: {dataset_summary['inputDataPath']}")
    print(f"- Output dataset: {dataset_path}")
    print(f"- Output summary: {summary_path}")
    print(f"- Requested samples: {args.num_samples}")
    print(f"- Kept samples: {dataset_summary['keptSampleCount']}")
    print(f"- Dropped infeasible with incumbent: {dataset_summary['droppedInfeasibleCount']}")
    print(f"- Dropped with no incumbent: {dataset_summary['droppedNoIncumbentCount']}")
    print(f"- Constraint labels generated: {'YES' if dataset_summary['constraintLabelGenerated'] else 'NO'}")
    print(f"- Constraint label missing count: {dataset_summary['constraintLabelMissingCount']}")
    print(f"- Seed: {args.seed}")
    print(f"- Exact baseline runtime available: {'YES' if dataset_summary['exactBaselineUsed'] else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
