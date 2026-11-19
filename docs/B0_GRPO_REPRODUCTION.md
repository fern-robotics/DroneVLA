# B0 + GRPO reproduction

This is the thesis-facing, reproducible native UAV-Flow pipeline. B0 is the
released `wangxiangyu0814/OpenVLA-UAV` checkpoint. B0 is **not** M1.

## Principle

- B0 is already SFT on UAV-Flow-Sim. Do not SFT it again.
- Use the 10,109 training trajectories as the GRPO task/reference pool.
- Keep the 273 official test trajectories held out.
- Train/test overlap must be zero.

## Data

```text
/workspace/uavflow_sim_full/              # 21 parquet shards, 10,109 trajectories
/workspace/uavflow_train_tasks/            # generated task JSONs (reference paths)
/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons/  # 273 held-out tasks
```

Build tasks:

```bash
python scripts/build_uavflow_tasks.py \
  --data /workspace/uavflow_sim_full \
  --out /workspace/uavflow_train_tasks
```

Validate:

```bash
python scripts/validate_uavflow_pool.py \
  --train-dir /workspace/uavflow_train_tasks \
  --test-dir /workspace/UAV-Flow/UAV-Flow-Eval/test_jsons
```

Expected:

```json
{"train": 10109, "test": 273, "overlap": 0, "bad_reference": 0}
```

## Contract

- prompt: `In: Current State: x,y,z,yaw, What action should the uav take to <instruction>?\nOut:`
- state: local `x,y,z,yaw_degrees`, centimetres
- action: native absolute local pose
- action scale: 1.0

## Collect

Start the B0 token server and proxy (`:8000` and `:5007`) using
`scripts/serve_openvla_actions.py --prompt-style openvla_uav --absolute-output`.
Then:

```bash
python scripts/sample_uavflow_tasks.py \
  --train-dir /workspace/uavflow_train_tasks \
  --test-dir /workspace/UAV-Flow/UAV-Flow-Eval/test_jsons \
  --out /workspace/runs/b0_grpo_full/manifest.json --count 100 --seed 7
```

Collect 4 rollouts per sampled task with `scripts/collect_uavflow_groups.py`.
Each rollout stores `episode.jsonl`, frames, tokens, log-probs, and a
reference-trajectory reward.

## Train

```bash
python scripts/aggregate_uavflow_groups.py \
  --run /workspace/runs/b0_grpo_full \
  --manifest /workspace/runs/b0_grpo_full/manifest.json \
  --out /workspace/runs/b0_grpo_full/groups.json

python scripts/train_b0_grpo.py \
  --groups /workspace/runs/b0_grpo_full/groups.json \
  --out /workspace/runs/b0_grpo_full/adapter \
  --max-groups 100 --lr 1e-5
```

## Evaluate

Serve `/workspace/runs/b0_grpo_full/adapter` and evaluate the held-out 273
tasks. Use the official `test_jsons`; never train on them.

## One-command runner

```bash
bash scripts/reproduce_b0_grpo.sh
```

## Artifacts

```text
/workspace/runs/b0_grpo_full/
  manifest.json
  groups.json
  <task_id>/gX_rY/{episode.jsonl,frames/,runner.log}
  adapter/{adapter_config.json,adapter_model.safetensors}
```

## Current limitations

- 100-task collection is expensive; start with 8 tasks and scale.
- The native evaluator needs the Linux env patch in
  `integrations/uavflow_sim/patch_official_env.py`.
- NDTW/success-rate reporting is separate from the training reward.
