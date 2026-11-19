#!/usr/bin/env python3
"""Build one cached expert trajectory per training task from UAV-Flow-Sim parquet."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

# OpenVLA-UAV action bins/stats are loaded from the checkpoint at training time;
# this script only stores raw expert state/action/image data.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/uavflow_sim_full")
    ap.add_argument("--tasks", required=True, help="manifest JSON or task JSON")
    ap.add_argument("--out", default="/workspace/runs/eggrpo/expert_cache")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    if args.tasks.endswith(".json"):
        obj = json.loads(Path(args.tasks).read_text())
        ids = [Path(t).stem for t in obj["tasks"]] if isinstance(obj, dict) and "tasks" in obj else [Path(args.tasks).stem]
    else:
        ids = [Path(args.tasks).stem]
    ids = set(ids[: args.limit])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    done = set()
    for shard in sorted(Path(args.data).glob("*.parquet")):
        for row in pq.read_table(shard, columns=["id", "frame_idx", "image", "log"]).to_pylist():
            tid = row["id"]
            if tid not in ids or tid in done:
                continue
            log = json.loads(row["log"])
            raw = log.get("raw_logs") or []
            prep = log.get("preprocessed_logs") or []
            if len(raw) < 2:
                continue
            d = out / tid; d.mkdir(parents=True, exist_ok=True)
            img = row["image"]
            if isinstance(img, dict) and img.get("bytes"):
                (d / f"{int(row['frame_idx']):06d}.jpg").write_bytes(img["bytes"])
            (d / "expert.json").write_text(json.dumps({
                "id": tid, "instruction": log.get("instruction"),
                "raw_logs": raw, "preprocessed_logs": prep,
            }))
            done.add(tid)
            print(tid, flush=True)
            if len(done) >= len(ids):
                break
        if len(done) >= len(ids):
            break
    print(json.dumps({"experts": len(done), "out": str(out)}))


if __name__ == "__main__":
    main()
