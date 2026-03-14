from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


GENERATOR_FEATURE_KEYS = (
    "busIndex",
    "pMin",
    "pMax",
    "a2",
    "a1",
    "a0",
    "rampUp",
    "rampDown",
    "startCost",
    "shutCost",
    "minUpTime",
    "minDownTime",
)

CONSTRAINT_LABEL_SCHEMA_VERSION = "power118-constraint-label-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _power118_script() -> Path:
    return _repo_root() / "external" / "power118" / "SCUC_118_new.py"


def _load_power118_module():
    script_path = _power118_script()
    if not script_path.exists():
        raise FileNotFoundError(f"power118 script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("external.power118.scuc_118_new", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_power118_data(
    data_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = _load_power118_module()
    if not hasattr(module, "load_power118_data"):
        raise AttributeError("external power118 module is missing load_power118_data")
    return module.load_power118_data(data_path=data_path, overrides=overrides)


def build_power118_feature_record(power_data: dict[str, Any]) -> dict[str, float]:
    generators = power_data.get("generators", [])
    branches = power_data.get("branches", [])
    total_load_by_hour = power_data.get("totalLoadByHour", [])
    summary = power_data.get("summary", {})

    feature_row: dict[str, float] = {
        "timeHorizon": float(power_data.get("timeHorizon", 0)),
        "numBus": float(summary.get("numBus", 0)),
        "numLine": float(summary.get("numLine", 0)),
        "numGen": float(summary.get("numGen", 0)),
        "numLoad": float(summary.get("numLoad", 0)),
        "peakLoad": float(summary.get("peakLoad", 0.0)),
        "totalDailyLoad": float(summary.get("totalDailyLoad", 0.0)),
        "minHourlyLoad": float(min(total_load_by_hour) if total_load_by_hour else 0.0),
        "avgHourlyLoad": float(sum(total_load_by_hour) / len(total_load_by_hour) if total_load_by_hour else 0.0),
        "totalGeneratorPMax": float(sum(generator["pMax"] for generator in generators)),
        "totalGeneratorPMin": float(sum(generator["pMin"] for generator in generators)),
        "totalStartCost": float(sum(generator["startCost"] for generator in generators)),
        "totalShutCost": float(sum(generator["shutCost"] for generator in generators)),
        "avgRampUp": float(sum(generator["rampUp"] for generator in generators) / len(generators) if generators else 0.0),
        "avgRampDown": float(
            sum(generator["rampDown"] for generator in generators) / len(generators) if generators else 0.0
        ),
        "avgBranchCapacity": float(sum(branch["capacity"] for branch in branches) / len(branches) if branches else 0.0),
        "avgBranchReactance": float(sum(branch["x"] for branch in branches) / len(branches) if branches else 0.0),
    }

    peak_load = feature_row["peakLoad"] or 1.0
    feature_row["reserveMarginPeak"] = max(
        0.0,
        (feature_row["totalGeneratorPMax"] - peak_load) / peak_load,
    )

    for hour_index, load_value in enumerate(total_load_by_hour, start=1):
        feature_row[f"hourlyLoad_{hour_index}"] = float(load_value)

    for gen_index, generator in enumerate(generators, start=1):
        feature_row[f"gen{gen_index}_genId"] = float(generator["genId"])
        for key in GENERATOR_FEATURE_KEYS:
            feature_row[f"gen{gen_index}_{key}"] = float(generator[key])

    return feature_row


def build_power118_feature_frame(power_data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([build_power118_feature_record(power_data)])


def build_power118_target_record(result: dict[str, Any]) -> dict[str, float]:
    target_row: dict[str, float] = {
        "objective": float(result.get("objective") or 0.0),
        "solveTimeMs": float(result.get("solveTimeMs") or 0.0),
        "feasible": 1.0 if result.get("feasible") else 0.0,
    }

    unit_commitment = result.get("unitCommitmentByHour", [])
    generator_dispatch = result.get("generatorDispatchByHour", [])

    for gen_index, hourly_values in enumerate(unit_commitment, start=1):
        for hour_index, value in enumerate(hourly_values, start=1):
            target_row[f"unitCommitment_g{gen_index}_h{hour_index}"] = float(value)

    for gen_index, hourly_values in enumerate(generator_dispatch, start=1):
        for hour_index, value in enumerate(hourly_values, start=1):
            target_row[f"dispatch_g{gen_index}_h{hour_index}"] = float(value)

    return target_row


def build_power118_target_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([build_power118_target_record(result)])


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def build_power118_constraint_label_record(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("constraintDiagnostics") if isinstance(result.get("constraintDiagnostics"), dict) else {}
    counts = diagnostics.get("bindingConstraintCounts") if isinstance(diagnostics.get("bindingConstraintCounts"), dict) else {}
    slack_summary = diagnostics.get("constraintSlackSummary") if isinstance(diagnostics.get("constraintSlackSummary"), dict) else {}

    total_generator_limit_constraints = float(diagnostics.get("totalGeneratorLimitConstraintCount", 0) or 0)
    total_ramp_constraints = float(diagnostics.get("totalRampConstraintCount", 0) or 0)
    total_line_constraints = float(diagnostics.get("totalLineConstraintCount", 0) or 0)
    total_balance_constraints = float(diagnostics.get("totalBalanceConstraintCount", 0) or 0)

    active_generator_limit_count = float(result.get("activeGeneratorLimitCount", counts.get("generatorLimit", 0)) or 0)
    active_ramp_count = float(result.get("activeRampConstraintCount", counts.get("ramp", 0)) or 0)
    active_line_count = float(result.get("activeLineConstraintCount", counts.get("line", 0)) or 0)
    active_balance_count = float(result.get("activeBalanceConstraintCount", counts.get("balance", 0)) or 0)

    record: dict[str, Any] = {
        "constraintLabelSchemaVersion": CONSTRAINT_LABEL_SCHEMA_VERSION,
        "constraintLabelAvailable": 1.0 if diagnostics else 0.0,
        "constraint_activeGeneratorLimitCount": active_generator_limit_count,
        "constraint_activeRampConstraintCount": active_ramp_count,
        "constraint_activeLineConstraintCount": active_line_count,
        "constraint_activeBalanceConstraintCount": active_balance_count,
        "constraint_totalActiveConstraintCount": active_generator_limit_count
        + active_ramp_count
        + active_line_count
        + active_balance_count,
        "constraint_activeGeneratorLimitRatio": (
            active_generator_limit_count / total_generator_limit_constraints if total_generator_limit_constraints > 0 else 0.0
        ),
        "constraint_activeRampRatio": active_ramp_count / total_ramp_constraints if total_ramp_constraints > 0 else 0.0,
        "constraint_activeLineRatio": active_line_count / total_line_constraints if total_line_constraints > 0 else 0.0,
        "constraint_activeBalanceRatio": (
            active_balance_count / total_balance_constraints if total_balance_constraints > 0 else 0.0
        ),
        "constraint_generatorLimitActiveIndicesJson": json.dumps(_json_list(diagnostics.get("generatorLimitActiveIndices"))),
        "constraint_rampActiveIndicesJson": json.dumps(_json_list(diagnostics.get("rampActiveIndices"))),
        "constraint_lineActiveIndicesJson": json.dumps(_json_list(diagnostics.get("lineActiveIndices"))),
        "constraint_balanceActiveIndicesJson": json.dumps(_json_list(diagnostics.get("balanceActiveIndices"))),
        "constraint_topTightConstraintsJson": json.dumps(diagnostics.get("topTightConstraints", {})),
        "constraint_slackSummaryJson": json.dumps(slack_summary),
    }
    return record


def build_power118_constraint_label_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([build_power118_constraint_label_record(result)])


def _decode_index_entries(raw_json: str) -> list[str]:
    try:
        parsed = json.loads(raw_json)
    except Exception:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def build_power118_fixing_label_record(result: dict[str, Any]) -> dict[str, float]:
    unit_commitment = result.get("unitCommitmentByHour", [])
    if not isinstance(unit_commitment, list) or not unit_commitment:
        return {}

    diagnostics = build_power118_constraint_label_record(result)
    generator_limit_active = set(_decode_index_entries(str(diagnostics["constraint_generatorLimitActiveIndicesJson"])))
    ramp_active = set(_decode_index_entries(str(diagnostics["constraint_rampActiveIndicesJson"])))

    num_generators = len(unit_commitment)
    horizon = len(unit_commitment[0]) if unit_commitment else 0
    fixing_record: dict[str, float] = {}
    fixed_count = 0.0

    for gen_index in range(num_generators):
        for hour_index in range(horizon):
            current = 1 if float(unit_commitment[gen_index][hour_index]) >= 0.5 else 0
            prev_value = current if hour_index == 0 else 1 if float(unit_commitment[gen_index][hour_index - 1]) >= 0.5 else 0
            next_value = current if hour_index == horizon - 1 else 1 if float(unit_commitment[gen_index][hour_index + 1]) >= 0.5 else 0
            stable_segment = current == prev_value == next_value
            switching_boundary = current != prev_value or current != next_value
            generator_limit_binding = (
                f"genLimit:g{gen_index + 1}:h{hour_index + 1}:pMin" in generator_limit_active
                or f"genLimit:g{gen_index + 1}:h{hour_index + 1}:pMax" in generator_limit_active
            )
            ramp_binding = (
                f"ramp:g{gen_index + 1}:h{hour_index + 1}:up" in ramp_active
                or f"ramp:g{gen_index + 1}:h{hour_index + 1}:down" in ramp_active
                or (hour_index > 0 and f"ramp:g{gen_index + 1}:h{hour_index}:up" in ramp_active)
                or (hour_index > 0 and f"ramp:g{gen_index + 1}:h{hour_index}:down" in ramp_active)
            )
            fixable = stable_segment and not switching_boundary and not generator_limit_binding and not ramp_binding
            fixing_record[f"fixCommitment_g{gen_index + 1}_h{hour_index + 1}"] = 1.0 if fixable else 0.0
            fixed_count += 1.0 if fixable else 0.0

    total_binaries = float(max(num_generators * horizon, 1))
    fixing_record["fixingLabelCount"] = fixed_count
    fixing_record["fixingLabelRatio"] = fixed_count / total_binaries
    return fixing_record


def build_power118_fixing_label_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([build_power118_fixing_label_record(result)])


def _constraint_type_code(constraint_type: str) -> float:
    mapping = {
        "generatorLimit": 0.0,
        "ramp": 1.0,
        "line": 2.0,
        "balance": 3.0,
    }
    return float(mapping.get(constraint_type, 99.0))


def _constraint_subtype_code(subtype: str) -> float:
    mapping = {
        "pMin": 0.0,
        "pMax": 1.0,
        "up": 2.0,
        "down": 3.0,
        "absCap": 4.0,
        "equality": 5.0,
    }
    return float(mapping.get(subtype, 99.0))


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {})


def build_power118_constraint_candidate_records(
    power_data: dict[str, Any],
    result: dict[str, Any] | None = None,
    schedule_prediction: dict[str, Any] | None = None,
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    summary = power_data.get("summary", {})
    total_load_by_hour = list(power_data.get("totalLoadByHour", []))
    generators = list(power_data.get("generators", []))
    branches = list(power_data.get("branches", []))
    load_at_bus = list(power_data.get("loadAtBus", []))
    horizon = int(power_data.get("timeHorizon", 0))

    result = result or {}
    diagnostics = result.get("constraintDiagnostics") if isinstance(result.get("constraintDiagnostics"), dict) else {}
    slack_records = diagnostics.get("slackRecords") if isinstance(diagnostics.get("slackRecords"), dict) else {}
    top_tight_by_type = diagnostics.get("topTightConstraints") if isinstance(diagnostics.get("topTightConstraints"), dict) else {}
    top_tight_ids = {
        str(constraint_type): {str(item.get("constraintId")) for item in items if isinstance(item, dict)}
        for constraint_type, items in top_tight_by_type.items()
        if isinstance(items, list)
    }

    unit_commitment = result.get("unitCommitmentByHour") if isinstance(result.get("unitCommitmentByHour"), list) else None
    dispatch = result.get("generatorDispatchByHour") if isinstance(result.get("generatorDispatchByHour"), list) else None
    if unit_commitment is None and schedule_prediction is not None:
        unit_commitment = schedule_prediction.get("unitCommitmentByHour")
    if dispatch is None and schedule_prediction is not None:
        dispatch = schedule_prediction.get("generatorDispatchByHour")

    candidate_records: list[dict[str, Any]] = []
    for constraint_type, records in slack_records.items():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            hour_index = int(record.get("hourIndex", 0))
            generator_index = record.get("generatorIndex")
            line_index = record.get("lineIndex")
            bus_index = record.get("busIndex")
            subtype = str(record.get("constraintSubtype") or "")
            slack_value = float(record.get("slack") or 0.0)
            active_label = 1.0 if record.get("active") else 0.0
            tight_label = 1.0 if slack_value <= 1e-3 else 0.0
            rank_score = 1.0 / (1.0 + max(slack_value, 0.0))
            if str(record.get("constraintId")) in top_tight_ids.get(str(constraint_type), set()):
                rank_score += 0.25

            hour_load = float(total_load_by_hour[hour_index]) if 0 <= hour_index < len(total_load_by_hour) else 0.0
            next_hour_load = (
                float(total_load_by_hour[hour_index + 1])
                if 0 <= hour_index + 1 < len(total_load_by_hour)
                else hour_load
            )
            previous_hour_load = (
                float(total_load_by_hour[hour_index - 1])
                if hour_index > 0 and hour_index - 1 < len(total_load_by_hour)
                else hour_load
            )
            instance_features = {
                "inst_peakLoad": float(summary.get("peakLoad", 0.0)),
                "inst_totalDailyLoad": float(summary.get("totalDailyLoad", 0.0)),
                "inst_hourLoad": hour_load,
                "inst_hourLoadRatio": hour_load / max(float(summary.get("peakLoad", 0.0)), 1.0),
                "inst_hourLoadDeltaNext": next_hour_load - hour_load,
                "inst_hourLoadDeltaPrev": hour_load - previous_hour_load,
                "inst_slack": slack_value,
            }
            abstract_features = {
                "abs_constraintTypeCode": _constraint_type_code(str(constraint_type)),
                "abs_constraintSubtypeCode": _constraint_subtype_code(subtype),
                "abs_hourIndex": float(hour_index),
                "abs_hourNorm": float(hour_index + 1) / max(float(horizon), 1.0),
            }

            if generator_index is not None and 0 <= int(generator_index) < len(generators):
                generator = generators[int(generator_index)]
                dispatch_value = 0.0
                commitment_value = 0.0
                prev_dispatch = 0.0
                if isinstance(dispatch, list) and int(generator_index) < len(dispatch):
                    dispatch_row = dispatch[int(generator_index)]
                    if hour_index < len(dispatch_row):
                        dispatch_value = float(dispatch_row[hour_index])
                    if hour_index > 0 and hour_index - 1 < len(dispatch_row):
                        prev_dispatch = float(dispatch_row[hour_index - 1])
                if isinstance(unit_commitment, list) and int(generator_index) < len(unit_commitment):
                    commitment_row = unit_commitment[int(generator_index)]
                    if hour_index < len(commitment_row):
                        commitment_value = float(commitment_row[hour_index])

                instance_features.update(
                    {
                        "inst_dispatch": dispatch_value,
                        "inst_commitment": commitment_value,
                        "inst_dispatchToPmaxRatio": dispatch_value / max(float(generator["pMax"]), 1.0),
                        "inst_dispatchMarginPmax": float(generator["pMax"]) - dispatch_value,
                        "inst_dispatchMarginPmin": dispatch_value - float(generator["pMin"]),
                        "inst_rampPressure": abs(dispatch_value - prev_dispatch) / max(float(generator["rampUp"]), 1.0),
                    }
                )
                abstract_features.update(
                    {
                        "abs_generatorIndex": float(int(generator_index)),
                        "abs_generatorIndexNorm": float(int(generator_index) + 1) / max(float(len(generators)), 1.0),
                        "abs_busIndex": float(generator["busIndex"]),
                        "abs_pMin": float(generator["pMin"]),
                        "abs_pMax": float(generator["pMax"]),
                        "abs_rampUp": float(generator["rampUp"]),
                        "abs_rampDown": float(generator["rampDown"]),
                        "abs_startCost": float(generator["startCost"]),
                        "abs_minUpTime": float(generator["minUpTime"]),
                        "abs_minDownTime": float(generator["minDownTime"]),
                    }
                )

            if line_index is not None and 0 <= int(line_index) < len(branches):
                branch = branches[int(line_index)]
                from_bus = int(branch["fromBusIndex"])
                to_bus = int(branch["toBusIndex"])
                from_load = float(load_at_bus[hour_index][from_bus]) if hour_index < len(load_at_bus) else 0.0
                to_load = float(load_at_bus[hour_index][to_bus]) if hour_index < len(load_at_bus) else 0.0
                instance_features.update(
                    {
                        "inst_lineEndpointLoad": from_load + to_load,
                        "inst_lineEndpointLoadRatio": (from_load + to_load) / max(float(branch["capacity"]), 1.0),
                    }
                )
                abstract_features.update(
                    {
                        "abs_lineIndex": float(int(line_index)),
                        "abs_lineIndexNorm": float(int(line_index) + 1) / max(float(len(branches)), 1.0),
                        "abs_fromBusIndex": float(from_bus),
                        "abs_toBusIndex": float(to_bus),
                        "abs_branchCapacity": float(branch["capacity"]),
                        "abs_branchReactance": float(branch["x"]),
                    }
                )

            if bus_index is not None:
                bus_idx = int(bus_index)
                bus_load = float(load_at_bus[hour_index][bus_idx]) if hour_index < len(load_at_bus) else 0.0
                instance_features["inst_busLoad"] = bus_load
                abstract_features["abs_busIndex"] = float(bus_idx)

            candidate_record = {
                "sampleId": sample_id or "",
                "constraintId": str(record.get("constraintId") or ""),
                "constraintType": str(constraint_type),
                "constraintSubtype": subtype,
                "instanceFeaturesJson": _safe_json_dumps(instance_features),
                "abstractFeaturesJson": _safe_json_dumps(abstract_features),
                "labelActive": active_label,
                "labelTight": tight_label,
                "labelRankScore": float(rank_score),
                "canBeReduced": 1.0 if str(constraint_type) in {"ramp", "line"} else 0.0,
                "canBeDeferred": 1.0 if str(constraint_type) in {"ramp", "line"} else 0.0,
                "canBeFixed": 1.0 if str(constraint_type) == "generatorLimit" else 0.0,
            }
            for feature_name, feature_value in instance_features.items():
                candidate_record[feature_name] = float(feature_value)
            for feature_name, feature_value in abstract_features.items():
                candidate_record[feature_name] = float(feature_value)
            candidate_records.append(candidate_record)

    return candidate_records


def build_power118_constraint_candidate_frame(
    power_data: dict[str, Any],
    result: dict[str, Any] | None = None,
    schedule_prediction: dict[str, Any] | None = None,
    sample_id: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        build_power118_constraint_candidate_records(
            power_data=power_data,
            result=result,
            schedule_prediction=schedule_prediction,
            sample_id=sample_id,
        )
    )


def build_power118_metadata_record(
    power_data: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    sample_id: str | None = None,
    split: str = "train",
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = power_data.get("summary", {})
    total_load_by_hour = power_data.get("totalLoadByHour", [])
    constraint_record = build_power118_constraint_label_record(result or {})
    fixing_record = build_power118_fixing_label_record(result or {})
    return {
        "sampleId": sample_id or "",
        "split": split,
        "dataPath": str(power_data.get("dataPath") or ""),
        "numBus": int(summary.get("numBus", 0)),
        "numLine": int(summary.get("numLine", 0)),
        "numGen": int(summary.get("numGen", 0)),
        "numLoad": int(summary.get("numLoad", 0)),
        "peakLoad": float(summary.get("peakLoad", 0.0)),
        "totalDailyLoad": float(summary.get("totalDailyLoad", 0.0)),
        "totalLoadByHourJson": json.dumps(total_load_by_hour),
        "overridesJson": json.dumps(overrides or {}),
        "constraintLabelAvailable": float(constraint_record.get("constraintLabelAvailable", 0.0)),
        "constraintTotalActiveCount": float(constraint_record.get("constraint_totalActiveConstraintCount", 0.0)),
        "constraintLabelSchemaVersion": str(constraint_record.get("constraintLabelSchemaVersion", "")),
        "fixingLabelCount": float(fixing_record.get("fixingLabelCount", 0.0)),
        "fixingLabelRatio": float(fixing_record.get("fixingLabelRatio", 0.0)),
    }
