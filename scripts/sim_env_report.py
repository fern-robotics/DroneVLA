#!/usr/bin/env python3
"""Print and validate the native UAV-Flow simulator configuration."""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/uavflow_sim_env.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    scene = Path(cfg["scene_dir"])
    binary = scene / cfg["binary"]
    cfg["binary_exists"] = binary.exists()
    cfg["display_env"] = os.environ.get("DISPLAY")
    cfg["nvidia_smi"] = shutil.which("nvidia-smi") is not None
    cfg["vulkan_icd"] = sorted(str(p) for p in Path("/usr/share/vulkan/icd.d").glob("*.json"))
    print(json.dumps(cfg, indent=2))
    if not cfg["binary_exists"]:
        raise SystemExit("scene binary missing")


if __name__ == "__main__":
    main()
