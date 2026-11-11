# UAV-Flow-Sim 10% build (RunPod)

Built on a CPU data pod with `scripts/build_dataset_online.py` (single shard
download pass, images streamed in batches).

## Result

- Selected: **936 episodes**, **30,763 frames**, ~1.0 GB JPEG export.
- Full dataset: **312,240 frames**, ~10,400 episodes, 21 shards.
- Fraction: ~9.0% of episodes (per-shard 10% stratified).
- Splits: **train 765 / dev 90 / test 81**.
- OOD: **81 test episodes** from held-out instruction templates.
- Templates: 378; intents: altitude, approach, move, orbit, other, pass, yaw.
- Validation passed: no split overlap, no OOD template in train, image/action/state lengths consistent.

Action q01/q99 (metres, radians per 0.1 s): `[-0.378, -0.360, -0.200, -0.052]`
to `[0.506, 0.346, 0.200, 0.052]`.

## RLDS verification

`tfds` built from the export with `UAV_EXPORT_DIR`:

- splits: train 765, dev 90, test 81
- one sample episode: action `(T, 7)`, state `(T, 6)`, image `(T, 256, 256, 3)`

The 4-D UAV action and 4-D state are zero-padded to 7-D/6-D so the official
OpenVLA tokenizer works unchanged; decode the first four action dimensions.

## Commands

```bash
PYTHONPATH=. python scripts/build_dataset_online.py \
  --shards all --out data/uav_flow_10pct --fraction 0.10 --formats jsonl

cd data/uav_flow_10pct
UAV_EXPORT_DIR=$(pwd) tfds build --data_dir $(pwd)/rlds --overwrite
python /path/to/dronevla/integrations/openvla/patch_openvla_uav.py /path/to/openvla
```

The LeRobot split writer is implemented in `scripts/dataset/lerobot_writer.py`;
it was not re-run on this CPU pod because LeRobot's torch dependency exceeds the
pod's memory budget. The earlier OpenPI 2-step smoke used the same LeRobot
features (image/state/actions/task) and passed.
