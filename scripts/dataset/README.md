# UAV-Flow-Sim data pipeline

Everything here stops at data: no model training.

## Build a subset

```bash
# 10% stratified subset, streaming shards (fits a 30-50 GB disk)
PYTHONPATH=. python scripts/build_dataset.py \
  --source hf --shards all --stream \
  --out data/uav_flow_10pct --fraction 0.10 --formats jsonl

# Add LeRobot materialization (requires OpenPI's lerobot environment)
... --formats jsonl,lerobot

# Add the RLDS builder source (requires TFDS to build)
... --formats jsonl,rlds
```

Outputs:

- `images/<episode_id>/*.jpg` — 256×256 RGB
- `episodes.jsonl` — action/state/image paths, instruction, split, OOD flag
- `manifest.json` — counts, shards, seed, fraction, OOD templates
- `splits.json` — episode → train/dev/test, OOD
- `stats.json` — action/state q01/q99/mean/std
- `validation.json` — split disjointness, image/action length checks
- optional `lerobot/` or `uav_flow_sim_dataset_builder.py`

## Splits

- Episode-level, deterministic by seed.
- Instruction templates are normalized (numbers/objects) before splitting.
- A held-out template set is test-only OOD; it never appears in train.
- Default target: ~90% train / 10% dev, with ~10% of templates OOD test.
  The external evaluation manifest is separate (30 AirSimNH episodes).

## RLDS for OpenVLA

```bash
cd data/uav_flow_10pct
UAV_EXPORT_DIR=$(pwd) tfds build --data_dir $(pwd)/rlds --overwrite
python /path/to/dronevla/integrations/openvla/patch_openvla_uav.py /path/to/openvla
```

The builder pads UAV `state` (4→6) and `action` (4→7) with zeros so the
official OpenVLA tokenizer can be used unchanged. Decode only the first four
action dimensions at inference.

## Notes

- Raw shards are deleted after each streaming step; re-downloading is expected.
- Never mix test episodes into SFT or RL rollouts.
- Use the same subset and splits for OpenVLA and π₀-FAST.
