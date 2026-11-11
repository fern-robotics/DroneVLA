"""UAV-Flow-Sim scanning, episode export and normalization statistics."""
from __future__ import annotations

import io
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

HF_REPO = "wangxiangyu0814/UAV-Flow-Sim"
N_SHARDS = 21


def parse_shards(spec: str) -> list[int]:
    if spec in ("all", "*"):
        return list(range(N_SHARDS))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < N_SHARDS)


def shard_url(i: int) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/train-{i:05d}-of-{N_SHARDS:05d}.parquet"


def _valid_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        import pyarrow.parquet as pq
        pq.ParquetFile(path)
        return True
    except Exception:
        return False


def ensure_shard(i: int, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    p = download_dir / f"train-{i:05d}-of-{N_SHARDS:05d}.parquet"
    if _valid_parquet(p):
        return p
    p.unlink(missing_ok=True)
    tmp = p.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    print(f"downloading shard {i} -> {p}", flush=True)
    urllib.request.urlretrieve(shard_url(i), tmp)
    if not _valid_parquet(tmp):
        raise RuntimeError(f"downloaded shard {i} is not a valid parquet file")
    tmp.replace(p)
    return p


def ensure_shards(indices: Iterable[int], download_dir: Path) -> list[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in indices:
        p = download_dir / f"train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        if not p.exists() or p.stat().st_size < 1024:
            print(f"downloading shard {i} -> {p}", flush=True)
            urllib.request.urlretrieve(shard_url(i), p)
        paths.append(p)
    return paths


def actions_from_raw(raw: list) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Official OpenVLA-UAV convention: cm -> m, local FRD delta4."""
    raw = np.asarray(raw, dtype=np.float64)
    pos = raw[:, [0, 1, 2]] / 100.0
    yaw = np.deg2rad(raw[:, 4])
    acts = []
    for i in range(len(raw) - 1):
        c, n, cy, ny = pos[i], pos[i + 1], yaw[i], yaw[i + 1]
        r = np.array([[math.cos(cy), -math.sin(cy), 0.0],
                      [math.sin(cy), math.cos(cy), 0.0],
                      [0.0, 0.0, 1.0]])
        lp = r.T @ (n - c)
        dy = (ny - cy + math.pi) % (2 * math.pi) - math.pi
        acts.append([float(lp[0]), float(lp[1]), float(lp[2]), float(dy)])
    acts.append([0.0, 0.0, 0.0, 0.0])
    return pos.tolist(), yaw.tolist(), acts


def scan_metadata(paths: list[Path], max_episodes: int | None = None) -> list:
    """Read only id/frame_idx/log to build EpisodeMeta without loading images."""
    import pyarrow.parquet as pq
    from scripts.dataset.select import EpisodeMeta, intent_tag, length_bucket, template_of

    metas: dict[str, EpisodeMeta] = {}
    for path in paths:
        table = pq.read_table(path, columns=["id", "frame_idx", "log"])
        ids = table.column("id").to_pylist()
        frames = table.column("frame_idx").to_pylist()
        logs = table.column("log").to_pylist()
        for eid, frame, raw_log in zip(ids, frames, logs):
            if eid in metas:
                continue
            try:
                log = json.loads(raw_log)
                raw = log.get("raw_logs") or []
                instruction = (log.get("instruction") or "").strip()
                if len(raw) < 10 or not instruction:
                    continue
                n = max(int(frame) + 1, len(raw))
                metas[eid] = EpisodeMeta(
                    id=eid,
                    instruction=instruction,
                    intent_tag=intent_tag(instruction),
                    template=template_of(instruction),
                    n_frames=n,
                    length_bucket=length_bucket(n),
                    shard=path.name,
                )
            except Exception:
                continue
            if max_episodes is not None and len(metas) >= max_episodes:
                return sorted(metas.values(), key=lambda m: m.id)
    return sorted(metas.values(), key=lambda m: m.id)


def export_episodes(paths: list[Path], selected_ids: set[str], out_dir: Path,
                    meta_by_id: dict, image_size: int = 256, jpeg_quality: int = 95,
                    append: bool = False, written: set[str] | None = None,
                    require_all: bool = True) -> Path:
    """Export selected episodes to images/ + episodes.jsonl."""
    import pyarrow.parquet as pq
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / "images"
    images_root.mkdir(exist_ok=True)
    jsonl_path = out_dir / "episodes.jsonl"
    written = set() if written is None else written
    with jsonl_path.open("a" if append else "w") as out:
        for path in paths:
            # Stream row groups/batches: loading a full shard's images at once
            # can exceed a small data-pod's RAM.
            parquet = pq.ParquetFile(path)
            by_id: dict[str, list[dict]] = defaultdict(list)
            for batch in parquet.iter_batches(batch_size=64, columns=["id", "frame_idx", "image", "log"]):
                for r in batch.to_pylist():
                    if r["id"] in selected_ids and r["id"] not in written:
                        by_id[r["id"]].append(r)
            for eid, frames in by_id.items():
                frames.sort(key=lambda r: int(r["frame_idx"]))
                try:
                    log = json.loads(frames[0]["log"])
                    pos, yaw, acts = actions_from_raw(log["raw_logs"])
                except Exception:
                    continue
                ep_dir = images_root / eid
                ep_dir.mkdir(parents=True, exist_ok=True)
                image_paths = []
                for r in frames:
                    idx = int(r["frame_idx"])
                    img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").resize((image_size, image_size))
                    rel = f"images/{eid}/{idx:04d}.jpg"
                    img.save(out_dir / rel, quality=jpeg_quality)
                    image_paths.append(rel)
                m = meta_by_id[eid]
                pos0, yaw0 = pos[0], yaw[0]
                states = []
                for i in range(len(frames)):
                    states.append([
                        pos[i][0] - pos0[0], pos[i][1] - pos0[1], pos[i][2] - pos0[2],
                        (yaw[i] - yaw0 + math.pi) % (2 * math.pi) - math.pi,
                    ])
                record = {
                    "id": eid,
                    "instruction": m.instruction,
                    "intent_tag": m.intent_tag,
                    "template": m.template,
                    "split": getattr(m, "split", "train"),
                    "ood": bool(getattr(m, "ood", False)),
                    "n_frames": len(frames),
                    "image_size": image_size,
                    "images": image_paths,
                    "states": states,
                    "actions": acts,
                    "source_shard": path.name,
                }
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                written.add(eid)
    missing = selected_ids - written
    if require_all and missing:
        raise RuntimeError(f"failed to export {len(missing)} selected episodes, e.g. {sorted(missing)[:3]}")
    return jsonl_path


def compute_stats(episodes_jsonl: Path, out_path: Path) -> dict:
    acts, states, lengths = [], [], []
    with episodes_jsonl.open() as f:
        for line in f:
            e = json.loads(line)
            acts.extend(e["actions"])
            states.extend(e["states"])
            lengths.append(e["n_frames"])
    a = np.asarray(acts, dtype=np.float64)
    s = np.asarray(states, dtype=np.float64)
    stats = {
        "episodes": len(lengths),
        "frames": int(a.shape[0]),
        "action_units": "metres and radians; body FRD per 0.1 s",
        "action": {
            "q01": np.percentile(a, 1, axis=0).round(6).tolist(),
            "q99": np.percentile(a, 99, axis=0).round(6).tolist(),
            "mean": a.mean(axis=0).round(6).tolist(),
            "std": a.std(axis=0).round(6).tolist(),
            "abs_mean": np.abs(a).mean(axis=0).round(6).tolist(),
        },
        "state": {
            "q01": np.percentile(s, 1, axis=0).round(6).tolist(),
            "q99": np.percentile(s, 99, axis=0).round(6).tolist(),
            "mean": s.mean(axis=0).round(6).tolist(),
            "std": s.std(axis=0).round(6).tolist(),
        },
        "length": {
            "min": int(min(lengths)), "max": int(max(lengths)),
            "mean": round(float(np.mean(lengths)), 3),
        },
    }
    out_path.write_text(json.dumps(stats, indent=2))
    return stats
