from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class RunStore:
    """File-based latest-run store, keyed by scenario id."""

    def __init__(self, store_file: Path):
        self._file = store_file
        self._lock = Lock()
        self._file.parent.mkdir(parents=True, exist_ok=True)
        if not self._file.exists():
            self._file.write_text("{}", encoding="utf-8")

    def _read_all(self) -> Dict[str, Any]:
        try:
            content = self._file.read_text(encoding="utf-8").strip()
            if not content:
                return {}
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _write_all(self, payload: Dict[str, Any]) -> None:
        self._file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_latest(self, scenario_id: str, run_payload: Dict[str, Any]) -> None:
        with self._lock:
            data = self._read_all()
            data[scenario_id] = run_payload
            self._write_all(data)

    def get_latest(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._read_all()
            value = data.get(scenario_id)
            return value if isinstance(value, dict) else None

