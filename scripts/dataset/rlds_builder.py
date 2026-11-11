"""TFDS/RLDS builder for an exported UAV episode directory.

The builder stores a 6-D state and 7-D action (4 UAV dims + zero padding) so the
official OpenVLA action tokenizer can be used unchanged. Decode only the first
four action dimensions at inference.

Run:
  cp scripts/dataset/rlds_builder.py <export>/rlds_builder.py
  cp -r <export> <export>/uav_export   # or set UAV_EXPORT_DIR
  tfds build --data_dir <rlds_data> <export>/rlds_builder.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial UAV-Flow-Sim subset."}
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name="uav_flow_sim", description="UAV-Flow-Sim intent subset")]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="UAV-Flow-Sim episodes exported by DroneVL.",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Sequence({
                    "observation": {
                        "image": tfds.features.Image(shape=(256, 256, 3)),
                        "state": tfds.features.Tensor(shape=(6,), dtype=tf.float32),
                    },
                    "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    "language_instruction": tfds.features.Text(),
                    "is_first": tfds.features.Tensor(shape=(), dtype=tf.bool),
                    "is_last": tfds.features.Tensor(shape=(), dtype=tf.bool),
                    "is_terminal": tfds.features.Tensor(shape=(), dtype=tf.bool),
                    "reward": tfds.features.Tensor(shape=(), dtype=tf.float32),
                    "discount": tfds.features.Tensor(shape=(), dtype=tf.float32),
                }),
                "episode_id": tfds.features.Text(),
                "intent_tag": tfds.features.Text(),
            }),
        )

    def _split_generators(self, dl_manager):
        return [
            tfds.core.SplitGenerator(name=split, gen_kwargs={"split": split})
            for split in ("train", "dev", "test")
        ]

    def _generate_examples(self, split: str):
        root = Path(os.environ["UAV_EXPORT_DIR"])
        with (root / "episodes.jsonl").open() as f:
            for line in f:
                e = json.loads(line)
                if e["split"] != split:
                    continue
                steps = []
                n = len(e["actions"])
                for i in range(n):
                    steps.append({
                        "observation": {
                            "image": str(root / e["images"][i]),
                            "state": list(e["states"][i]) + [0.0, 0.0],
                        },
                        "action": list(e["actions"][i]) + [0.0, 0.0, 0.0],
                        "language_instruction": e["instruction"],
                        "is_first": i == 0,
                        "is_last": i == n - 1,
                        "is_terminal": i == n - 1,
                        "reward": 0.0,
                        "discount": 1.0,
                    })
                yield e["id"], {"steps": steps, "episode_id": e["id"], "intent_tag": e["intent_tag"]}
