#!/usr/bin/env python3
"""Build one cached expert trajectory per training task from UAV-Flow-Sim parquet."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/uavflow_sim_full")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", default="/workspace/runs/eggrpo/expert_cache")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    obj = json.loads(Path(args.tasks).read_text())
    ids = {Path(t).stem for t in obj["tasks"]} if isinstance(obj, dict) and "tasks" in obj else {Path(args.tasks).stem}
    ids = set(list(ids)[: args.limit])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta, frames = {}, {tid: 0 for tid in ids}
    for shard in sorted(Path(args.data).glob("*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=2000, columns=["id", "frame_idx", "image", "log"]):
            for row in batch.to_pylist():
                tid = row["id"]
                if tid not in ids:
                    continue
                if tid not in meta:
                    log = json.loads(row["log"])
                    meta[tid] = {"id": tid, "instruction": log.get("instruction"),
                                 "raw_logs": log.get("raw_logs"), "preprocessed_logs": log.get("preprocessed_logs")}
                d = out / tid; d.mkdir(parents=True, exist_ok=True)
                img = row["image"]
                if isinstance(img, dict) and img.get("bytes"):
                    (d / f"{int(row['frame_idx']):06d}.jpg").write_bytes(img["bytes"])
                    frames[tid] += 1
        print("scanned", shard.name, "have", len(meta), flush=True)
        if len(meta) >= len(ids):
            break
    for tid, m in meta.items():
        (out / tid / "expert.json").write_text(json.dumps(m))
    print(json.dumps({"experts": len(meta), "frames": sum(frames.values()), "out": str(out)}))


if __name__ == "__main__":
    main()
