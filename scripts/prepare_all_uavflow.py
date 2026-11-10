#!/usr/bin/env python3
"""Convert ALL UAV-Flow-Sim parquet shards to one compact DroneVL JSONL on the pod."""
import json, math, os, re, subprocess, sys
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 21
os.makedirs(OUT_DIR, exist_ok=True)
BASE = "https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim/resolve/main/train-{i:05d}-of-{n:05d}.parquet"
TMP = "/tmp/uavflow_shard.parquet"

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
    if "pass" in t or "through" in t: return "pass"
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

seen = set()
episodes = []
for i in range(SHARDS):
    url = BASE.format(i=i, n=SHARDS)
    print(f"[{i+1}/{SHARDS}] downloading", flush=True)
    subprocess.run(["curl", "-L", "--retry", "3", "-sS", "-o", TMP, url], check=True)
    tbl = pq.read_table(TMP, columns=["id", "frame_idx", "log"])
    ids = tbl.column("id").to_pylist()
    logs = tbl.column("log").to_pylist()
    by_id = {}
    for eid, log in zip(ids, logs):
        by_id.setdefault(eid, log)
    del tbl
    os.remove(TMP)
    added = 0
    for eid, log in by_id.items():
        if eid in seen:
            continue
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
            seen.add(eid)
            added += 1
        except Exception:
            continue
    print(f"  added {added}, total {len(episodes)}", flush=True)

episodes.sort(key=lambda e: e["id"])
with open(os.path.join(OUT_DIR, "uavflow_sim.jsonl"), "w") as f:
    for e in episodes:
        f.write(json.dumps(e, separators=(",", ":")) + "\n")
all_actions = np.concatenate([np.asarray(e["actions"]) for e in episodes])
tags = Counter(e["intent_tag"] for e in episodes)
stats = {
    "source": "wangxiangyu0814/UAV-Flow-Sim",
    "shards": SHARDS,
    "episodes_selected": len(episodes),
    "frames": int(all_actions.shape[0]),
    "action_units": "metres, radians; body FRD per 0.1 s frame",
    "q01": np.percentile(all_actions, 1, axis=0).round(6).tolist(),
    "q99": np.percentile(all_actions, 99, axis=0).round(6).tolist(),
    "mean": all_actions.mean(axis=0).round(6).tolist(),
    "std": all_actions.std(axis=0).round(6).tolist(),
    "abs_mean": np.abs(all_actions).mean(axis=0).round(6).tolist(),
    "tag_counts_selected": dict(tags),
}
with open(os.path.join(OUT_DIR, "uavflow_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2), flush=True)
