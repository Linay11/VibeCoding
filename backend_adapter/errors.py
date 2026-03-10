from __future__ import annotations

from dataclasses import dataclass


@dataclass
class APIError(Exception):
    status_code: int
    code: str
    message: str

