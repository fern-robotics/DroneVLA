"""Write an exported UAV episode directory to LeRobot datasets, one repo per split."""
from __future__ import annotations

import json
from pathlib import Path


def write_lerobot(episodes_jsonl: Path, out_root: Path, repo_prefix: str = "uav_flow_sim",
                  image_size: int = 256) -> dict:
    try:
        import numpy as np
        from PIL import Image
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        raise RuntimeError("LeRobot writer requires lerobot, numpy and pillow in the active environment") from e

    episodes = [json.loads(line) for line in episodes_jsonl.read_text().splitlines() if line.strip()]
    by_split: dict[str, list[dict]] = {}
    for e in episodes:
        by_split.setdefault(e["split"], []).append(e)

    out_root.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, eps in sorted(by_split.items()):
        repo_id = f"{repo_prefix}_{split}"
        ds = LeRobotDataset.create(
            repo_id=repo_id, robot_type="uav", fps=10,
            features={
                "image": {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (4,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (4,), "names": ["actions"]},
            },
            image_writer_threads=4, image_writer_processes=2,
        )
        for e in eps:
            base = episodes_jsonl.parent
            for img_rel, state, action in zip(e["images"], e["states"], e["actions"]):
                img = np.asarray(Image.open(base / img_rel).convert("RGB"))
                ds.add_frame({"image": img, "state": np.asarray(state, dtype=np.float32),
                              "actions": np.asarray(action, dtype=np.float32), "task": e["instruction"]})
            ds.save_episode()
        info[split] = {"repo_id": repo_id, "episodes": len(eps)}
    (out_root / "lerobot_info.json").write_text(json.dumps(info, indent=2))
    return info
