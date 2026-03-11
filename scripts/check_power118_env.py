from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    critical: bool = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_path() -> Path:
    return _repo_root() / "external" / "power118" / "SCUC_118_new.py"


def _data_path() -> Path:
    return _repo_root() / "external" / "power118" / "118_data.xls"


def _result(name: str, status: str, detail: str, critical: bool = True) -> CheckResult:
    return CheckResult(name=name, status=status, detail=detail, critical=critical)


def _load_external_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("external.power118.scuc_118_new", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks() -> dict[str, Any]:
    results: list[CheckResult] = []
    context: dict[str, Any] = {
        "repo_root": _repo_root(),
        "script_path": _script_path(),
        "data_path": _data_path(),
    }

    py = sys.version_info
    py_status = PASS if (py.major, py.minor) >= (3, 10) else WARN
    results.append(_result("Python version", py_status, f"{sys.version.split()[0]}"))

    pandas_mod = None
    try:
        import pandas as pd

        pandas_mod = pd
        results.append(_result("pandas import", PASS, getattr(pd, "__version__", "unknown")))
    except Exception as exc:
        results.append(_result("pandas import", FAIL, str(exc)))

    xlrd_mod = None
    try:
        import xlrd  # noqa: F401

        xlrd_mod = xlrd
        results.append(_result("xlrd import", PASS, getattr(xlrd, "__version__", "unknown")))
    except Exception as exc:
        results.append(_result("xlrd import", FAIL, str(exc)))

    gurobi_mod = None
    try:
        import gurobipy as gp

        gurobi_mod = gp
        results.append(_result("gurobipy import", PASS, getattr(gp, "__version__", "unknown")))
    except Exception as exc:
        results.append(_result("gurobipy import", FAIL, str(exc)))

    if gurobi_mod is not None:
        try:
            model = gurobi_mod.Model("power118_env_check")
            model.setParam("OutputFlag", 0)
            model.update()
            dispose = getattr(model, "dispose", None)
            if callable(dispose):
                dispose()
            results.append(_result("Gurobi model init", PASS, "model initialized successfully"))
            context["gurobi_model_init"] = True
        except Exception as exc:
            results.append(_result("Gurobi model init", FAIL, str(exc)))
            context["gurobi_model_init"] = False
    else:
        results.append(_result("Gurobi model init", WARN, "skipped because gurobipy import failed"))
        context["gurobi_model_init"] = False

    script_path = context["script_path"]
    if script_path.exists():
        results.append(_result("SCUC script exists", PASS, str(script_path)))
    else:
        results.append(_result("SCUC script exists", FAIL, str(script_path)))

    data_path = context["data_path"]
    if data_path.exists():
        results.append(_result("118_data.xls exists", PASS, str(data_path)))
    else:
        results.append(_result("118_data.xls exists", FAIL, str(data_path)))

    workbook_readable = False
    if pandas_mod is not None and xlrd_mod is not None and data_path.exists():
        try:
            xls = pandas_mod.ExcelFile(data_path)
            required = {"Bus", "Branch", "Generator", "Hourly Load"}
            missing = sorted(required.difference(set(xls.sheet_names)))
            if missing:
                results.append(_result("118_data.xls readable", FAIL, f"missing sheets: {', '.join(missing)}"))
            else:
                results.append(_result("118_data.xls readable", PASS, f"sheets={', '.join(xls.sheet_names)}"))
                workbook_readable = True
        except Exception as exc:
            results.append(_result("118_data.xls readable", FAIL, str(exc)))
    else:
        results.append(_result("118_data.xls readable", WARN, "skipped because pandas/xlrd/data file is unavailable"))
    context["workbook_readable"] = workbook_readable

    module = None
    if script_path.exists():
        try:
            module = _load_external_module(script_path)
            results.append(_result("SCUC script import", PASS, "module import succeeded"))
        except Exception as exc:
            results.append(_result("SCUC script import", FAIL, str(exc)))
    else:
        results.append(_result("SCUC script import", WARN, "skipped because script file is missing"))
    context["module"] = module

    runtime_info: dict[str, Any] | None = None
    if module is not None and hasattr(module, "check_gurobi_runtime"):
        results.append(_result("check_gurobi_runtime available", PASS, "function found"))
        try:
            runtime_info = module.check_gurobi_runtime()
            if isinstance(runtime_info, dict):
                if runtime_info.get("available"):
                    results.append(
                        _result(
                            "check_gurobi_runtime call",
                            PASS,
                            f"stage={runtime_info.get('stage', 'unknown')}; reason={runtime_info.get('reason', '')}",
                        )
                    )
                else:
                    results.append(
                        _result(
                            "check_gurobi_runtime call",
                            FAIL,
                            f"stage={runtime_info.get('stage', 'unknown')}; reason={runtime_info.get('reason', '')}",
                        )
                    )
            else:
                results.append(_result("check_gurobi_runtime call", FAIL, "returned non-dict result"))
        except Exception as exc:
            results.append(_result("check_gurobi_runtime call", FAIL, str(exc)))
    elif module is not None:
        results.append(_result("check_gurobi_runtime available", FAIL, "function missing"))
        results.append(_result("check_gurobi_runtime call", WARN, "skipped because function is missing"))
    else:
        results.append(_result("check_gurobi_runtime available", WARN, "skipped because module import failed"))
        results.append(_result("check_gurobi_runtime call", WARN, "skipped because module import failed"))
    context["runtime_info"] = runtime_info

    if module is not None and hasattr(module, "solve_scuc_118"):
        results.append(_result("solve_scuc_118 available", PASS, "function found"))
    elif module is not None:
        results.append(_result("solve_scuc_118 available", FAIL, "function missing"))
    else:
        results.append(_result("solve_scuc_118 available", WARN, "skipped because module import failed"))

    can_attempt_real = bool(
        module is not None
        and hasattr(module, "solve_scuc_118")
        and workbook_readable
        and isinstance(runtime_info, dict)
        and runtime_info.get("available")
    )

    if can_attempt_real:
        results.append(_result("real-run preconditions", PASS, "environment is ready to attempt solve_scuc_118"))
    else:
        blockers = []
        if module is None:
            blockers.append("SCUC script import")
        if not workbook_readable:
            blockers.append("118_data.xls readable")
        if not isinstance(runtime_info, dict) or not runtime_info.get("available"):
            blockers.append("Gurobi runtime")
        if module is not None and not hasattr(module, "solve_scuc_118"):
            blockers.append("solve_scuc_118 available")
        if not blockers:
            blockers.append("unknown precondition")
        results.append(_result("real-run preconditions", FAIL, f"missing: {', '.join(blockers)}"))

    critical_failures = [item.name for item in results if item.critical and item.status == FAIL]
    summary = {
        "real_mode_ready": len(critical_failures) == 0 and can_attempt_real,
        "critical_failures": critical_failures,
        "results": results,
    }
    return summary


def _print_results(summary: dict[str, Any]) -> None:
    results: list[CheckResult] = summary["results"]
    for item in results:
        print(f"[{item.status:<4}] {item.name:<28} {item.detail}")

    pass_count = sum(1 for item in results if item.status == PASS)
    warn_count = sum(1 for item in results if item.status == WARN)
    fail_count = sum(1 for item in results if item.status == FAIL)

    print("")
    print("Summary")
    print(f"- PASS: {pass_count}")
    print(f"- WARN: {warn_count}")
    print(f"- FAIL: {fail_count}")
    print(f"- Real mode ready: {'YES' if summary['real_mode_ready'] else 'NO'}")
    if summary["critical_failures"]:
        print(f"- Missing / blocked: {', '.join(summary['critical_failures'])}")
    else:
        print("- Missing / blocked: none")


def main() -> int:
    summary = run_checks()
    _print_results(summary)
    return 0 if summary["real_mode_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
