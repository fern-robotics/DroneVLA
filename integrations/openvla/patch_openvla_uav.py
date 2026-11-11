#!/usr/bin/env python3
"""Register the exported UAV RLDS dataset in the official OpenVLA repo.

Usage: python integrations/openvla/patch_openvla_uav.py /path/to/openvla
This edits dataset registry/config files only; model, tokenizer and loss are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

CONFIG_ENTRY = '''    "uav_flow_sim": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
'''

MIXTURE_ENTRY = '''    "uav_flow_sim": [
        ("uav_flow_sim", 1.0),
    ],
'''

TRANSFORM = '''

def uav_flow_sim_dataset_transform(trajectory):
    """UAV rows are 6-D state / 7-D action (4 UAV dims + zero padding)."""
    return trajectory

'''


def insert_before_final_brace(path: Path, text: str) -> None:
    s = path.read_text()
    idx = s.rfind("}")
    if idx == -1:
        raise SystemExit(f"no closing brace in {path}")
    path.write_text(s[:idx] + text + s[idx:])


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "openvla")
    configs = root / "prismatic/vla/datasets/rlds/oxe/configs.py"
    mixtures = root / "prismatic/vla/datasets/rlds/oxe/mixtures.py"
    transforms = root / "prismatic/vla/datasets/rlds/oxe/transforms.py"
    for p in (configs, mixtures, transforms):
        if not p.exists():
            raise SystemExit(f"missing {p}; pass the OpenVLA repo root")

    if '"uav_flow_sim"' not in configs.read_text():
        insert_before_final_brace(configs, CONFIG_ENTRY)
    if '"uav_flow_sim"' not in mixtures.read_text():
        insert_before_final_brace(mixtures, MIXTURE_ENTRY)
    t = transforms.read_text()
    if "uav_flow_sim_dataset_transform" not in t:
        marker = "# === Registry ==="
        t = t.replace(marker, TRANSFORM + "\n" + marker)
        t = t.replace("OXE_STANDARDIZATION_TRANSFORMS = {", 'OXE_STANDARDIZATION_TRANSFORMS = {\n    "uav_flow_sim": uav_flow_sim_dataset_transform,')
        transforms.write_text(t)
    print("patched", root)


if __name__ == "__main__":
    main()
