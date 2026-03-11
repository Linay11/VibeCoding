from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_check_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "check_power118_env.py"
    spec = importlib.util.spec_from_file_location("scripts.check_power118_env", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_checks_returns_structured_summary() -> None:
    module = _load_check_module()
    summary = module.run_checks()

    assert isinstance(summary, dict)
    assert "real_mode_ready" in summary
    assert "critical_failures" in summary
    assert "results" in summary
    assert isinstance(summary["results"], list)
    assert any(item.name == "Python version" for item in summary["results"])
    assert any(item.name == "real-run preconditions" for item in summary["results"])
