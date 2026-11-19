#!/usr/bin/env python3
"""Convert UAV-Flow-Sim parquet trajectories to native task JSONs with references."""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/uavflow_sim_full")
    ap.add_argument("--out", default="/workspace/uavflow_train_tasks")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    test = {Path(p).stem for p in glob.glob("/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons/*.json")}
    done = set()
    count = 0
    for shard in sorted(glob.glob(str(Path(args.data) / "*.parquet"))):
        table = pq.read_table(shard, columns=["id", "log"])
        for row in table.to_pylist():
            tid = row["id"]
            if tid in test or tid in done:
                continue
            log = json.loads(row["log"])
            raw = log.get("raw_logs") or []
            prep = log.get("preprocessed_logs") or []
            if len(raw) < 2:
                continue
            task = {
                "id": tid,
                "instruction": log.get("instruction", ""),
                "instruction_unified": log.get("instruction_unified", log.get("instruction", "")),
                "initial_pos": raw[0],
                "end_pos": raw[-1],
                "target_pos": raw[-1][:3],
                "reference_path_raw": raw,
                "reference_path_preprocessed": prep,
                "obj_id": None,
                "use_obj": None,
            }
            (out / f"{tid}.json").write_text(json.dumps(task))
            done.add(tid); count += 1
            if args.limit and count >= args.limit:
                break
        if args.limit and count >= args.limit:
            break
    print(json.dumps({"trajectories": count, "out": str(out)}))


if __name__ == "__main__":
    main()
