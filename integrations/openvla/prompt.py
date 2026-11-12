"""Thesis OpenVLA prompt template."""
from __future__ import annotations

import json
from typing import Any


def build_uav_prompt(instruction: str, state: Any = None, progress: Any = None) -> str:
    state_json = json.dumps(state if state is not None else [])
    progress_json = json.dumps(progress if progress is not None else 0)
    return (
        f"In: What UAV action should be taken to {instruction}? "
        f"State: {state_json}. Progress: {progress_json}. Out:"
    )
