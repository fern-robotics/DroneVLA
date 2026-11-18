"""Task JSON and reference-path helpers for UAV-Flow-Sim."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def load_task(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def reference_path_cm(task: dict[str, Any]) -> list[list[float]]:
    path = task.get("reference_path_raw") or []
    if not path:
        raise ValueError("task has no reference_path_raw")
    return [[float(v) for v in pose[:3]] for pose in path]
