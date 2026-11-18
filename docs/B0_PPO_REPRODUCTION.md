# B0 + PPO reproduction

Mirrors the B0+GRPO pipeline, but the actor uses per-decision critic values and
GAE. The base and prompt/action contract are identical to B0+GRPO.

## Data

- 10,109 training tasks: `/workspace/uavflow_train_tasks`
- 273 held-out test tasks: `/workspace/UAV-Flow/UAV-Flow-Eval/test_jsons`
- deterministic 100-task manifest: `scripts/sample_uavflow_tasks.py`

## Collection

Three native engines per group:

```bash
python scripts/collect_uavflow_parallel.py \
  --manifest /workspace/runs/b0_ppo_100/manifest.json \
  --out /workspace/runs/b0_ppo_100 --workers 3 --steps 10 --timeout 60
```

## Training

```bash
python scripts/train_b0_ppo.py \
  --groups /workspace/runs/b0_ppo_100/groups.json \
  --out /workspace/runs/b0_ppo_100/adapter \
  --max-groups 100 --lr 1e-5
```

Outputs:

```text
adapter/adapter_model.safetensors
adapter/value_head.pt
```

## Evaluation

Serve `adapter` on `:8000`, then:

```bash
python scripts/eval_uavflow_test.py \
  --test-dir /workspace/UAV-Flow/UAV-Flow-Eval/test_jsons \
  --out /workspace/runs/eval_b0_ppo_100 \
  --workers 3 --steps 20 --rollouts 1
```

The evaluator reports success, NDTW, and final target distance on the same 273
tasks used for GRPO.

## One command

```bash
bash scripts/reproduce_b0_ppo.sh
```
