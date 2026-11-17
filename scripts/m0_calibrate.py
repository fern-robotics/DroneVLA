#!/usr/bin/env python3
"""Calibrate the M0 UAV decoder from the training split only.

M0 keeps the base VLA weights and replaces only the action statistics/decoder
with UAV training statistics. The raw JSONL stores world positions, yaw and
body-FRD actions; states are expressed relative to each episode's first frame.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _stats(x: np.ndarray) -> dict:
    return {
        "mean": np.mean(x, axis=0).round(8).tolist(),
        "std": np.std(x, axis=0).round(8).tolist(),
        "q01": np.percentile(x, 1, axis=0).round(8).tolist(),
        "q99": np.percentile(x, 99, axis=0).round(8).tolist(),
        "min": np.min(x, axis=0).round(8).tolist(),
        "max": np.max(x, axis=0).round(8).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/uavflow_sim.jsonl")
    ap.add_argument("--out", default="data/m0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    actions, states, count = [], [], 0
    for line in Path(args.jsonl).open():
        e = json.loads(line)
        if not e.get("actions") or not e.get("positions"):
            continue
        pos = np.asarray(e["positions"], dtype=np.float64)
        yaw = np.asarray(e["yaw"], dtype=np.float64)
        n = min(len(pos), len(yaw), len(e["actions"]))
        pos0, yaw0 = pos[0], yaw[0]
        rel = np.stack([
            pos[:n, 0] - pos0[0],
            pos[:n, 1] - pos0[1],
            pos[:n, 2] - pos0[2],
            (yaw[:n] - yaw0 + math.pi) % (2 * math.pi) - math.pi,
        ], axis=1)
        states.append(rel)
        actions.append(np.asarray(e["actions"][:n], dtype=np.float64))
        count += 1
    if not actions:
        raise SystemExit("no usable episodes")
    a = np.concatenate(actions, axis=0)
    s = np.concatenate(states, axis=0)
    action_stats = _stats(a)
    state_stats = _stats(s)
    stats = {"source": str(args.jsonl), "episodes": count, "frames": int(a.shape[0]),
             "action": action_stats, "state": state_stats}
    (out / "m0_stats.json").write_text(json.dumps(stats, indent=2))
    # OpenPI: one NormStats object per key, asset dir is assets/<repo_id>.
    openpi = {
        "state": {k: state_stats[k] for k in ("mean", "std", "q01", "q99")},
        "actions": {k: action_stats[k] for k in ("mean", "std", "q01", "q99")},
    }
    (out / "openpi_norm_stats.json").write_text(json.dumps(openpi, indent=2))
    # OpenVLA: UAV 4-D decoder; unused dims are neutralized.
    openvla = {
        "action": {
            **{k: action_stats[k] for k in ("mean", "std", "q01", "q99", "min", "max")},
            "mask": [True, True, True, True],
        }
    }
    (out / "openvla_norm_stats.json").write_text(json.dumps(openvla, indent=2))
    print(json.dumps({"episodes": count, "frames": int(a.shape[0]),
                      "q01": action_stats["q01"], "q99": action_stats["q99"]}, indent=2))


if __name__ == "__main__":
    main()
