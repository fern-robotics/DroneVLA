#!/usr/bin/env python3
"""Single-pass 10% UAV-Flow-Sim builder (one shard downloaded at a time)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import validate
from scripts.dataset.select import assign_splits, select_stratified
from scripts.dataset.uavflow import compute_stats, ensure_shard, export_episodes, parse_shards, scan_metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="all")
    ap.add_argument("--download-dir", default="data/uavflow")
    ap.add_argument("--out", default="data/uav_flow_10pct")
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--formats", default="jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    indices = parse_shards(args.shards)
    selected = []
    written: set[str] = set()
    for i in indices:
        p = ensure_shard(i, Path(args.download_dir))
        metas = scan_metadata([p])
        shard_sel = select_stratified(metas, args.fraction, args.seed + i)
        export_episodes([p], {m.id for m in shard_sel}, out, {m.id: m for m in shard_sel},
                        image_size=args.image_size, append=True, written=written, require_all=False)
        selected.extend(shard_sel)
        p.unlink(missing_ok=True)
        print(json.dumps({"shard": i, "scanned": len(metas), "selected": len(shard_sel), "total": len(selected)}), flush=True)
    splits, flags, ood_templates = assign_splits(selected, args.seed)
    selected = [dataclasses.replace(m, split=splits[m.id], ood=flags[m.id]) for m in selected]
    by_id = {m.id: m for m in selected}
    records = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    with (out / "episodes.jsonl").open("w") as f:
        for e in records:
            m = by_id[e["id"]]
            e["split"], e["ood"] = m.split, m.ood
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    stats = compute_stats(out / "episodes.jsonl", out / "stats.json")
    manifest = {
        "schema_version": "1.0", "source": "hf", "shards": indices,
        "fraction": args.fraction, "seed": args.seed, "image_size": args.image_size,
        "action_contract": "delta4 [dx,dy,dz,dyaw] metres/radians, body FRD, 10 Hz",
        "counts": stats, "ood_templates": ood_templates,
        "episodes": [{k: e[k] for k in ("id", "instruction", "intent_tag", "template", "split", "ood", "n_frames")}
                     for e in records],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "splits.json").write_text(json.dumps(
        {e["id"]: {"split": e["split"], "ood": e["ood"], "template": e["template"]} for e in records}, indent=2))
    report = validate(manifest, out, records)
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"validation": report}), flush=True)
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    if "lerobot" in formats:
        from scripts.dataset.lerobot_writer import write_lerobot
        print(json.dumps({"lerobot": write_lerobot(out / "episodes.jsonl", out / "lerobot", image_size=args.image_size)}), flush=True)
    if "rlds" in formats:
        shutil.copy(Path(__file__).parent / "dataset/rlds_builder.py", out / "uav_flow_sim_dataset_builder.py")
        (out / "rlds_info.json").write_text(json.dumps({"builder": str(out / "uav_flow_sim_dataset_builder.py"),
                                                        "export_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
