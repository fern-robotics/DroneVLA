#!/usr/bin/env python3
"""Build the DroneVL UAV-Flow-Sim subset.

Stops at data: scan, stratified selection, ID/OOD splits, image/JSONL export,
normalization statistics, and optional LeRobot/RLDS materialization.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dataset.select import assign_splits, select_stratified
from scripts.dataset.uavflow import compute_stats, ensure_shard, ensure_shards, export_episodes, parse_shards, scan_metadata


def validate(manifest: dict, out: Path, eps: list[dict]) -> dict:
    ids = [e["id"] for e in eps]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode ids")
    splits = {e["id"]: e["split"] for e in eps}
    if len(set(splits.values()) - {"train", "dev", "test"}):
        raise ValueError("invalid split name")
    ood_templates = set(manifest["ood_templates"])
    train_templates = {e["template"] for e in eps if e["split"] == "train"}
    if ood_templates & train_templates:
        raise ValueError("OOD template leaked into train")
    for e in eps:
        if len(e["images"]) != len(e["actions"]) or len(e["actions"]) != len(e["states"]):
            raise ValueError(f"length mismatch in {e['id']}")
        if not all(Path(out / p).exists() for p in e["images"]):
            raise ValueError(f"missing image in {e['id']}")
    counts = {s: sum(1 for e in eps if e["split"] == s) for s in ("train", "dev", "test")}
    ood = sum(1 for e in eps if e["ood"])
    return {"episodes": len(eps), "splits": counts, "ood": ood,
            "templates": len({e["template"] for e in eps}),
            "intents": sorted({e["intent_tag"] for e in eps})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf", help="'hf', a parquet file, or a directory of parquet shards")
    ap.add_argument("--shards", default="0")
    ap.add_argument("--download-dir", default="data/uavflow")
    ap.add_argument("--out", default="data/uav_flow_10pct")
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--formats", default="jsonl", help="comma list: jsonl,lerobot,rlds")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stream", action="store_true", help="download one shard at a time and delete it (50GB disk safe)")
    ap.add_argument("--online", action="store_true", help="select/export 10%% per shard during the single download pass")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.dry_run:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        indices = parse_shards(args.shards)
        if args.stream:
            paths = [ensure_shard(i, Path(args.download_dir)) for i in indices[:1]]
        else:
            paths = ensure_shards(indices, Path(args.download_dir))
    else:
        src = Path(args.source)
        paths = [src] if src.is_file() else sorted(src.glob("*.parquet"))
    if not paths:
        raise SystemExit("no parquet shards found")
    print(json.dumps({"shards": [p.name for p in paths]}), flush=True)

    if args.stream and args.source == "hf":
        metas = []
        for i in indices:
            p = ensure_shard(i, Path(args.download_dir))
            metas.extend(scan_metadata([p]))
            p.unlink(missing_ok=True)
            print(json.dumps({"scanned": len(metas), "shard": i}), flush=True)
    else:
        metas = scan_metadata(paths, max_episodes=args.max_episodes)
    print(json.dumps({"scanned_episodes": len(metas)}), flush=True)
    selected = select_stratified(metas, args.fraction, args.seed, max_episodes=args.max_episodes)
    splits, flags, ood_templates = assign_splits(selected, args.seed)
    selected = [dataclasses.replace(m, split=splits[m.id], ood=flags[m.id]) for m in selected]
    print(json.dumps({"selected": len(selected), "splits": {s: sum(1 for m in selected if m.split == s) for s in ("train", "dev", "test")},
                      "ood": sum(m.ood for m in selected)}), flush=True)
    if args.dry_run:
        print(json.dumps([m.id for m in selected[:5]]))
        return

    if args.online and args.source == "hf":
        # Single-pass: select and export ~fraction of each shard, then assign global splits.
        selected = []
        written: set[str] = set()
        for i in indices:
            p = ensure_shard(i, Path(args.download_dir))
            shard_metas = scan_metadata([p])
            shard_sel = select_stratified(shard_metas, args.fraction, args.seed + i)
            export_episodes([p], {m.id for m in shard_sel}, out, {m.id: m for m in shard_sel},
                            image_size=args.image_size, append=True, written=written, require_all=False)
            selected.extend(shard_sel)
            p.unlink(missing_ok=True)
            print(json.dumps({"online_shard": i, "selected": len(shard_sel), "total": len(selected)}), flush=True)
        splits, flags, ood_templates = assign_splits(selected, args.seed)
        selected = [dataclasses.replace(m, split=splits[m.id], ood=flags[m.id]) for m in selected]
        # Rewrite split/ood metadata in the already-exported JSONL.
        records = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
        by_id = {m.id: m for m in selected}
        with (out / "episodes.jsonl").open("w") as f:
            for e in records:
                m = by_id[e["id"]]
                e["split"], e["ood"] = m.split, m.ood
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
        jsonl = out / "episodes.jsonl"
    meta_map = {m.id: m for m in selected}
    if args.online and args.source == "hf":
        pass
    elif args.stream and args.source == "hf":
        written: set[str] = set()
        for i in indices:
            p = ensure_shard(i, Path(args.download_dir))
            export_episodes([p], set(meta_map), out, meta_map, image_size=args.image_size,
                            append=True, written=written, require_all=False)
            p.unlink(missing_ok=True)
        missing = set(meta_map) - written
        if missing:
            raise RuntimeError(f"failed to export {len(missing)} selected episodes, e.g. {sorted(missing)[:3]}")
        jsonl = out / "episodes.jsonl"
    else:
        jsonl = export_episodes(paths, {m.id for m in selected}, out, meta_map, image_size=args.image_size)
    stats = compute_stats(jsonl, out / "stats.json")
    episodes = [json.loads(line) for line in jsonl.read_text().splitlines()]
    manifest = {
        "schema_version": "1.0",
        "source": args.source,
        "shards": [p.name for p in paths],
        "fraction": args.fraction,
        "seed": args.seed,
        "image_size": args.image_size,
        "action_contract": "delta4 [dx,dy,dz,dyaw] metres/radians, body FRD, 10 Hz",
        "counts": stats,
        "ood_templates": ood_templates,
        "episodes": [
            {k: e[k] for k in ("id", "instruction", "intent_tag", "template", "split", "ood", "n_frames")}
            for e in episodes
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "splits.json").write_text(json.dumps(
        {e["id"]: {"split": e["split"], "ood": e["ood"], "template": e["template"]} for e in episodes}, indent=2))
    report = validate(manifest, out, episodes)
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"validation": report}), flush=True)

    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    if "lerobot" in formats:
        from scripts.dataset.lerobot_writer import write_lerobot
        info = write_lerobot(jsonl, out / "lerobot", image_size=args.image_size)
        print(json.dumps({"lerobot": info}), flush=True)
    if "rlds" in formats:
        shutil.copy(Path(__file__).parent / "dataset/rlds_builder.py", out / "uav_flow_sim_dataset_builder.py")
        (out / "rlds_info.json").write_text(json.dumps({
            "builder": str(out / "uav_flow_sim_dataset_builder.py"),
            "data_dir": str(out / "rlds"),
            "export_dir": str(out),
            "note": "Run from the export dir: UAV_EXPORT_DIR=<export> tfds build --data_dir <export>/rlds --overwrite; then apply integrations/openvla/patch_openvla_uav.py to the official repo.",
        }, indent=2))
        print(json.dumps({"rlds": "builder written; run tfds build"}), flush=True)


if __name__ == "__main__":
    main()
