from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

DEFAULT_DATA_FILE = Path(__file__).with_name("118_data.xls")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("SCUC_result.xlsx")


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(value)


def _status_name(grb_module: Any, status_code: int) -> str:
    status_map = {
        grb_module.OPTIMAL: "OPTIMAL",
        grb_module.SUBOPTIMAL: "SUBOPTIMAL",
        grb_module.INFEASIBLE: "INFEASIBLE",
        grb_module.INF_OR_UNBD: "INF_OR_UNBD",
        grb_module.UNBOUNDED: "UNBOUNDED",
        grb_module.TIME_LIMIT: "TIME_LIMIT",
    }
    return status_map.get(status_code, f"STATUS_{status_code}")


def check_gurobi_runtime() -> dict[str, Any]:
    try:
        import gurobipy as gp
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": False,
            "stage": "import",
            "reason": f"gurobipy import failed: {exc}",
        }

    try:
        model = gp.Model("power118_runtime_check")
        model.setParam("OutputFlag", 0)
        model.update()
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": False,
            "stage": "model_init",
            "reason": f"gurobi model init failed: {exc}",
        }

    return {
        "available": True,
        "stage": "ready",
        "reason": "gurobi runtime ready",
    }


def load_power118_data(data_path: str | Path | None = None) -> dict[str, Any]:
    import pandas as pd

    resolved_data_path = Path(data_path or DEFAULT_DATA_FILE).resolve()
    if not resolved_data_path.exists():
        raise FileNotFoundError(f"power118 data file not found: {resolved_data_path}")

    df_bus = pd.read_excel(resolved_data_path, sheet_name="Bus")
    df_branch = pd.read_excel(resolved_data_path, sheet_name="Branch")
    df_gen = pd.read_excel(resolved_data_path, sheet_name="Generator")
    df_load = pd.read_excel(resolved_data_path, sheet_name="Hourly Load")

    bus = df_bus.iloc[:, 0].tolist()
    branch_rows = df_branch.values.tolist()
    gen_rows = df_gen.values.tolist()
    load_rows = df_load.values.tolist()

    T = 24
    num_bus = len(bus)
    num_line = len(branch_rows)
    num_gen = len(gen_rows)
    num_load = len(load_rows)

    bus_id_map = {bus[idx]: idx for idx in range(num_bus)}

    load_at_bus = [[0.0 for _ in range(num_bus)] for _ in range(T)]
    for load_row in load_rows:
        load_bus_id = load_row[1]
        bus_idx = bus_id_map[load_bus_id]
        for hour in range(T):
            load_at_bus[hour][bus_idx] += _as_float(load_row[2 + hour])

    generators: list[dict[str, Any]] = []
    for row in gen_rows:
        generators.append(
            {
                "genId": _as_int(row[0]),
                "busId": _as_int(row[1]),
                "busIndex": bus_id_map[row[1]],
                "pMin": _as_float(row[2]),
                "pMax": _as_float(row[3]),
                "a2": _as_float(row[4]),
                "a1": _as_float(row[5]),
                "a0": _as_float(row[6]),
                "rampUp": _as_float(row[7]),
                "rampDown": _as_float(row[8]),
                "startCost": _as_float(row[9]),
                "shutCost": _as_float(row[10]),
                "minUpTime": _as_int(row[11]),
                "minDownTime": _as_int(row[12]),
            }
        )

    branches: list[dict[str, Any]] = []
    for row in branch_rows:
        branches.append(
            {
                "fromBusId": _as_int(row[0]),
                "toBusId": _as_int(row[1]),
                "fromBusIndex": bus_id_map[row[0]],
                "toBusIndex": bus_id_map[row[1]],
                "x": _as_float(row[2]),
                "capacity": _as_float(row[3]),
            }
        )

    total_load_by_hour = [float(sum(hour_values)) for hour_values in load_at_bus]
    generator_capacity_preview = sorted(
        [
            {
                "label": f"Gen {generator['genId']}",
                "value": float(generator["pMax"]),
            }
            for generator in generators
        ],
        key=lambda item: item["value"],
        reverse=True,
    )

    return {
        "dataPath": str(resolved_data_path),
        "timeHorizon": T,
        "bus": bus,
        "branchRows": branch_rows,
        "genRows": gen_rows,
        "loadRows": load_rows,
        "busIdMap": bus_id_map,
        "loadAtBus": load_at_bus,
        "generators": generators,
        "branches": branches,
        "summary": {
            "numBus": num_bus,
            "numLine": num_line,
            "numGen": num_gen,
            "numLoad": num_load,
            "peakLoad": float(max(total_load_by_hour) if total_load_by_hour else 0.0),
            "totalDailyLoad": float(sum(total_load_by_hour)),
        },
        "totalLoadByHour": total_load_by_hour,
        "generatorCapacityPreview": generator_capacity_preview,
    }


def solve_scuc_118(
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
    write_output: bool = False,
) -> dict[str, Any]:
    import pandas as pd

    runtime = check_gurobi_runtime()
    if not runtime.get("available"):
        raise RuntimeError(str(runtime.get("reason") or "gurobi runtime unavailable"))

    import gurobipy as gp
    from gurobipy import GRB

    data = load_power118_data(data_path)
    T = data["timeHorizon"]
    num_bus = data["summary"]["numBus"]
    num_line = data["summary"]["numLine"]
    num_gen = data["summary"]["numGen"]
    load_at_bus = data["loadAtBus"]
    generators = data["generators"]
    branches = data["branches"]

    model = gp.Model("SCUC_118")

    P = model.addVars(num_gen, T, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P")
    u = model.addVars(num_gen, T, vtype=GRB.BINARY, name="u")
    svar = model.addVars(num_gen, T, vtype=GRB.BINARY, name="s")
    dvar = model.addVars(num_gen, T, vtype=GRB.BINARY, name="d")
    theta = model.addVars(num_bus, T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
    flow = model.addVars(num_line, T, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="flow")

    objective = gp.QuadExpr()
    for g, generator in enumerate(generators):
        for hour in range(T):
            objective.add(P[g, hour] * P[g, hour], generator["a2"])
            objective.add(P[g, hour], generator["a1"])
            objective.add(u[g, hour], generator["a0"])
            objective.add(svar[g, hour], generator["startCost"])
            objective.add(dvar[g, hour], generator["shutCost"])
    model.setObjective(objective, GRB.MINIMIZE)

    for g, generator in enumerate(generators):
        for hour in range(T):
            model.addConstr(P[g, hour] >= generator["pMin"] * u[g, hour], name=f"Pmin_{g}_{hour}")
            model.addConstr(P[g, hour] <= generator["pMax"] * u[g, hour], name=f"Pmax_{g}_{hour}")

    for g, generator in enumerate(generators):
        for hour in range(T - 1):
            model.addConstr(P[g, hour + 1] - P[g, hour] <= generator["rampUp"], name=f"rampUp_{g}_{hour}")
            model.addConstr(P[g, hour] - P[g, hour + 1] <= generator["rampDown"], name=f"rampDown_{g}_{hour}")

    for g in range(num_gen):
        model.addConstr(u[g, 0] <= svar[g, 0], name=f"startup_{g}_0")
        model.addConstr(-u[g, 0] <= dvar[g, 0], name=f"shutdown_{g}_0")
        for hour in range(1, T):
            model.addConstr(u[g, hour] - u[g, hour - 1] <= svar[g, hour], name=f"startup_{g}_{hour}")
            model.addConstr(u[g, hour - 1] - u[g, hour] <= dvar[g, hour], name=f"shutdown_{g}_{hour}")

    for g, generator in enumerate(generators):
        up = max(1, generator["minUpTime"])
        down = max(1, generator["minDownTime"])
        for hour in range(T - up + 1):
            model.addConstr(
                gp.quicksum(u[g, tau] for tau in range(hour, hour + up)) >= up * svar[g, hour],
                name=f"minUp_{g}_{hour}",
            )
        for hour in range(T - up + 1, T):
            model.addConstr(
                gp.quicksum(u[g, tau] - svar[g, hour] for tau in range(hour, T)) >= 0,
                name=f"minUp_{g}_{hour}_border",
            )
        for hour in range(T - down + 1):
            model.addConstr(
                gp.quicksum(u[g, tau] for tau in range(hour, hour + down)) >= down * dvar[g, hour],
                name=f"minDown_{g}_{hour}",
            )
        for hour in range(T - down + 1, T):
            model.addConstr(
                gp.quicksum(1 - u[g, tau] - dvar[g, hour] for tau in range(hour, T)) >= 0,
                name=f"minDown_{g}_{hour}_border",
            )

    for line_idx, branch in enumerate(branches):
        for hour in range(T):
            model.addConstr(
                flow[line_idx, hour]
                == (1 / branch["x"]) * (theta[branch["fromBusIndex"], hour] - theta[branch["toBusIndex"], hour]),
                name=f"dcFlow_{line_idx}_{hour}",
            )
            model.addConstr(flow[line_idx, hour] * 100 <= branch["capacity"], name=f"fmaxPos_{line_idx}_{hour}")
            model.addConstr(flow[line_idx, hour] * 100 >= -branch["capacity"], name=f"fmaxNeg_{line_idx}_{hour}")

    for bus_idx in range(num_bus):
        for hour in range(T):
            gen_sum = gp.quicksum(P[g, hour] for g, generator in enumerate(generators) if generator["busIndex"] == bus_idx)
            flow_expr = gp.LinExpr()
            for line_idx, branch in enumerate(branches):
                if branch["fromBusIndex"] == bus_idx:
                    flow_expr.add(flow[line_idx, hour], 1.0)
                elif branch["toBusIndex"] == bus_idx:
                    flow_expr.add(flow[line_idx, hour], -1.0)
            model.addConstr(gen_sum - load_at_bus[hour][bus_idx] == flow_expr * 100, name=f"balance_{bus_idx}_{hour}")

    for hour in range(T):
        model.addConstr(theta[0, hour] == 0, name=f"refAngle_{hour}")

    model.setParam("MIPGap", 1e-4)
    solve_start = perf_counter()
    model.optimize()
    solve_time_ms = (perf_counter() - solve_start) * 1000.0

    status_code = int(model.status)
    status_name = _status_name(GRB.Status, status_code)
    feasible = status_name in {"OPTIMAL", "SUBOPTIMAL"}

    generator_dispatch_by_hour = [
        [float(P[g, hour].X) if feasible else 0.0 for hour in range(T)]
        for g in range(num_gen)
    ]
    unit_commitment_by_hour = [
        [float(u[g, hour].X) if feasible else 0.0 for hour in range(T)]
        for g in range(num_gen)
    ]
    line_flow_by_hour = [
        [float(flow[line_idx, hour].X) if feasible else 0.0 for hour in range(T)]
        for line_idx in range(num_line)
    ]
    bus_angle_by_hour = [
        [float(theta[bus_idx, hour].X) if feasible else 0.0 for hour in range(T)]
        for bus_idx in range(num_bus)
    ]

    total_dispatch_by_generator = []
    for g, generator in enumerate(generators):
        total_dispatch_by_generator.append(
            {
                "label": f"Gen {generator['genId']}",
                "value": float(sum(generator_dispatch_by_hour[g])),
            }
        )
    top_generators = sorted(total_dispatch_by_generator, key=lambda item: item["value"], reverse=True)[:4]

    peak_line_flow_by_hour = []
    for hour in range(T):
        peak_line_flow_by_hour.append(
            float(max(abs(line_flow_by_hour[line_idx][hour]) * 100 for line_idx in range(num_line)) if num_line else 0.0)
        )

    if feasible and write_output:
        resolved_output_path = Path(output_path or DEFAULT_OUTPUT_FILE).resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_P = pd.DataFrame(generator_dispatch_by_hour, columns=[f"Hour{hour}" for hour in range(T)])
        df_u = pd.DataFrame(unit_commitment_by_hour, columns=[f"Hour{hour}" for hour in range(T)])
        df_flow = pd.DataFrame(line_flow_by_hour, columns=[f"Hour{hour}" for hour in range(T)])
        df_theta = pd.DataFrame(bus_angle_by_hour, columns=[f"Hour{hour}" for hour in range(T)])
        df_P.index = [generator["genId"] for generator in generators]
        df_u.index = [generator["genId"] for generator in generators]
        with pd.ExcelWriter(resolved_output_path) as writer:
            df_P.to_excel(writer, sheet_name="GenOutput")
            df_u.to_excel(writer, sheet_name="GenStatus")
            df_flow.to_excel(writer, sheet_name="LineFlow")
            df_theta.to_excel(writer, sheet_name="BusAngle")
    else:
        resolved_output_path = None

    return {
        "dataPath": data["dataPath"],
        "outputPath": str(resolved_output_path) if resolved_output_path else None,
        "runtime": runtime,
        "statusCode": status_code,
        "statusName": status_name,
        "feasible": feasible,
        "objective": float(model.ObjVal) if feasible else None,
        "solveTimeMs": float(solve_time_ms),
        "summary": data["summary"],
        "totalLoadByHour": data["totalLoadByHour"],
        "generatorCapacityPreview": data["generatorCapacityPreview"],
        "topGenerators": top_generators,
        "peakLineFlowByHour": peak_line_flow_by_hour,
        "generatorDispatchByHour": generator_dispatch_by_hour,
        "unitCommitmentByHour": unit_commitment_by_hour,
        "lineFlowByHour": line_flow_by_hour,
        "busAngleByHour": bus_angle_by_hour,
    }


if __name__ == "__main__":
    result = solve_scuc_118(write_output=True)
    print(f"power118 solve status: {result['statusName']}")
    if result.get("objective") is not None:
        print(f"objective: {result['objective']}")
    if result.get("outputPath"):
        print(f"result workbook: {result['outputPath']}")
