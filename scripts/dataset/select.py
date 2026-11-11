"""Stratified episode selection and ID/OOD splits for UAV-Flow-Sim."""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field


def intent_tag(text: str) -> str:
    t = text.lower()
    if "orbit" in t or "circle" in t:
        return "orbit"
    if "pass" in t or "through" in t:
        return "pass"
    if "approach" in t or "progress toward" in t or "move to the" in t or "go to the" in t:
        return "approach"
    if "hold" in t or "hover" in t or "stay" in t:
        return "hold"
    if "turn" in t or "rotate" in t or "yaw" in t:
        return "yaw"
    if "ascend" in t or "climb" in t or "descend" in t:
        return "altitude"
    if "move" in t or "forward" in t or "backward" in t or "left" in t or "right" in t:
        return "move"
    return "other"


def template_of(text: str) -> str:
    """Normalize numbers and object nouns so instruction templates can be split."""
    t = text.lower().strip().rstrip(".")
    t = re.sub(r"\d+(?:\.\d+)?", "<n>", t)
    for noun in (
        "tree", "sculpture", "streetlight", "sunshade", "umbrella", "dog",
        "building", "tower", "bench", "car", "marker", "landmark",
    ):
        t = re.sub(rf"\b{noun}\b", "<obj>", t)
    t = re.sub(r"\s+", " ", t)
    return t


def length_bucket(n_frames: int) -> str:
    if n_frames < 20:
        return "short"
    if n_frames <= 40:
        return "medium"
    return "long"


@dataclass(frozen=True)
class EpisodeMeta:
    id: str
    instruction: str
    intent_tag: str
    template: str
    n_frames: int
    length_bucket: str
    shard: str
    split: str = "train"
    ood: bool = False


@dataclass
class Selection:
    episodes: list[EpisodeMeta] = field(default_factory=list)
    fraction: float = 0.1
    seed: int = 0
    ood_templates: list[str] = field(default_factory=list)


def select_stratified(
    metas: list[EpisodeMeta],
    fraction: float,
    seed: int,
    min_per_stratum: int = 1,
    max_episodes: int | None = None,
) -> list[EpisodeMeta]:
    """Select a deterministic stratified subset.

    Strata are (intent_tag, length_bucket). Every non-empty stratum gets at
    least `min_per_stratum`, then the remainder is sampled proportionally.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if not metas:
        return []
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[EpisodeMeta]] = defaultdict(list)
    for m in metas:
        by_stratum[(m.intent_tag, m.length_bucket)].append(m)

    target = max(len(by_stratum) * min_per_stratum, math.ceil(len(metas) * fraction))
    if max_episodes is not None:
        target = min(target, max_episodes)
    target = min(target, len(metas))

    chosen: list[EpisodeMeta] = []
    for key in sorted(by_stratum):
        group = list(by_stratum[key])
        rng.shuffle(group)
        quota = max(min_per_stratum, round(len(group) / len(metas) * target))
        quota = min(quota, len(group))
        chosen.extend(group[:quota])

    # Trim or top up deterministically to reach the target.
    if len(chosen) > target:
        rng.shuffle(chosen)
        chosen = chosen[:target]
    if len(chosen) < target:
        chosen_ids = {m.id for m in chosen}
        rest = [m for m in metas if m.id not in chosen_ids]
        rng.shuffle(rest)
        chosen.extend(rest[: target - len(chosen)])
    return sorted(chosen, key=lambda m: m.id)


def assign_splits(
    metas: list[EpisodeMeta],
    seed: int,
    ood_template_fraction: float = 0.10,
    train_frac: float = 0.90,
    dev_frac: float = 0.10,
) -> tuple[dict[str, str], dict[str, bool], list[str]]:
    """Assign train/dev/test and ID/OOD flags by instruction template.

    Templates held out from training become test-only OOD episodes. Remaining
    episodes are split by template into train/dev/test ID.
    """
    rng = random.Random(seed)
    templates = sorted({m.template for m in metas})
    rng.shuffle(templates)
    n_ood = max(1, round(len(templates) * ood_template_fraction)) if len(templates) > 1 else 0
    ood = set(templates[:n_ood])
    id_templates = templates[n_ood:]

    # Split ID templates, keeping at least one template per split when possible.
    rng.shuffle(id_templates)
    n_train = max(1, int(len(id_templates) * train_frac)) if id_templates else 0
    n_dev = max(0, int(len(id_templates) * dev_frac))
    train_t = set(id_templates[:n_train])
    dev_t = set(id_templates[n_train : n_train + n_dev])
    test_t = set(id_templates[n_train + n_dev :])

    splits: dict[str, str] = {}
    flags: dict[str, bool] = {}
    for m in metas:
        if m.template in ood:
            splits[m.id] = "test"
            flags[m.id] = True
        elif m.template in train_t:
            splits[m.id] = "train"
            flags[m.id] = False
        elif m.template in dev_t:
            splits[m.id] = "dev"
            flags[m.id] = False
        else:
            splits[m.id] = "test"
            flags[m.id] = False
    return splits, flags, sorted(ood)
