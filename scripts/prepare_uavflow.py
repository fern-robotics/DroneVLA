#!/usr/bin/env python3
"""Convert a UAV-Flow-Sim parquet shard into compact DroneVL training JSONL.

Follows the official OpenVLA-UAV loader:
  raw_logs columns are [x, y, z, roll, yaw_deg, pitch] (positions in cm)
  action_t = local-frame transform of pose_{t+1} into pose_t (metres, radians)
"""
import json, math, os, re, sys
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

SRC = sys.argv[1] if len(sys.argv) > 1 else "uavflow_data/train-00000.parquet"
OUT = sys.argv[2] if len(sys.argv) > 2 else "dronevla-rs/data"
os.makedirs(OUT, exist_ok=True)
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "900"))

tbl = pq.read_table(SRC, columns=["id", "frame_idx", "log"])
ids = tbl.column("id").to_pylist()
frames = tbl.column("frame_idx").to_pylist()
logs = tbl.column("log").to_pylist()
by_id = {}
for eid, f, log in zip(ids, frames, logs):
    if eid not in by_id:
        by_id[eid] = (f, log)
print("episodes in shard:", len(by_id))

def actions_from_raw(raw):
    raw = np.asarray(raw, dtype=np.float64)
    pos = raw[:, [0, 1, 2]] / 100.0
    yaw = np.deg2rad(raw[:, 4])
    acts = []
    for i in range(len(raw) - 1):
        c, n, cy, ny = pos[i], pos[i + 1], yaw[i], yaw[i + 1]
        R = np.array([[math.cos(cy), -math.sin(cy), 0.0],
                      [math.sin(cy),  math.cos(cy), 0.0],
                      [0.0, 0.0, 1.0]])
        lp = R.T @ (n - c)
        dy = (ny - cy + math.pi) % (2 * math.pi) - math.pi
        acts.append([float(lp[0]), float(lp[1]), float(lp[2]), float(dy)])
    acts.append([0.0, 0.0, 0.0, 0.0])
    return pos.tolist(), yaw.tolist(), acts

def intent_tag(text):
    t = text.lower()
    if "orbit" in t or "circle" in t: return "orbit"
    if "pass" in t or "fly through" in t or "through" in t: return "pass"
    if "approach" in t or "progress toward" in t or "move to the" in t or "go to the" in t: return "approach"
    if "hold" in t or "hover" in t or "stay" in t: return "hold"
    if "turn" in t or "rotate" in t or "yaw" in t: return "yaw"
    if "ascend" in t or "climb" in t or "descend" in t: return "altitude"
    if "move" in t or "forward" in t or "backward" in t or "left" in t or "right" in t: return "move"
    return "other"

def parse_params(text):
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*met(?:er|re)s?", t)
    dist = float(m.group(1)) if m else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*degrees?", t)
    ang = float(m.group(1)) if m else None
    side = -1 if "left" in t else (1 if "right" in t else 0)
    cw = 0
    if "counterclockwise" in t or "counter-clockwise" in t or "ccw" in t: cw = -1
    elif "clockwise" in t or "cw" in t: cw = 1
    return {"distance_m": dist, "angle_deg": ang, "side": side, "orbit_dir": cw}

episodes = []
for eid, (f, log) in by_id.items():
    try:
        log = json.loads(log)
        raw = log.get("raw_logs")
        if not raw or len(raw) < 10:
            continue
        pos, yaw, acts = actions_from_raw(raw)
        if not all(math.isfinite(v) for a in acts for v in a):
            continue
        if max(abs(a[0]) + abs(a[1]) for a in acts) > 5.0:
            continue
        inst = log.get("instruction") or ""
        episodes.append({
            "id": eid,
            "instruction": inst,
            "instruction_unified": log.get("instruction_unified") or inst,
            "intent_tag": intent_tag(inst),
            "params": parse_params(inst),
            "positions": pos,
            "yaw": yaw,
            "actions": acts,
        })
    except Exception:
        continue

counts = Counter(e["intent_tag"] for e in episodes)
print("tag counts:", dict(counts))
episodes.sort(key=lambda e: e["id"])
per_tag_cap = max(60, MAX_EPISODES // max(1, len([c for c in counts.values() if c > 0])))
selected = []
tag_used = Counter()
for e in episodes:
    if len(selected) >= MAX_EPISODES:
        break
    if tag_used[e["intent_tag"]] >= per_tag_cap and len(episodes) > MAX_EPISODES:
        continue
    selected.append(e)
    tag_used[e["intent_tag"]] += 1

with open(os.path.join(OUT, "uavflow_sim.jsonl"), "w") as f:
    for e in selected:
        f.write(json.dumps(e, separators=(",", ":")) + "\n")

all_actions = np.concatenate([np.asarray(e["actions"]) for e in selected]) if selected else np.zeros((1, 4))
stats = {
    "source": SRC,
    "episodes_in_shard": len(by_id),
    "episodes_converted": len(episodes),
    "episodes_selected": len(selected),
    "frames": int(all_actions.shape[0]),
    "action_units": "metres, radians; body FRD per 0.1 s frame",
    "q01": np.percentile(all_actions, 1, axis=0).round(6).tolist(),
    "q99": np.percentile(all_actions, 99, axis=0).round(6).tolist(),
    "mean": all_actions.mean(axis=0).round(6).tolist(),
    "std": all_actions.std(axis=0).round(6).tolist(),
    "abs_mean": np.abs(all_actions).mean(axis=0).round(6).tolist(),
    "tag_counts_selected": dict(tag_used),
}
with open(os.path.join(OUT, "uavflow_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
