# OpenVLA-7B: official SFT + UAV adapter

**Rule:** SFT uses the official OpenVLA repository unchanged. Do not reimplement the
model, action tokenizer, LoRA layers, loss, or optimizer in this repo.

Official source: https://github.com/openvla/openvla

## 1. Install

```bash
git clone https://github.com/openvla/openvla.git
cd openvla
# follow the official README; pin PyTorch 2.2, transformers 4.40.1,
# tokenizers 0.19.1, timm 0.9.10, flash-attn 2.5.5
pip install -e .
```

Use a pinned commit in experiment records. Do not patch `prismatic/` model code.

## 2. Data

The official `vla-scripts/finetune.py` expects an RLDS dataset registered in:

- `prismatic/vla/datasets/rlds/oxe/configs.py`
- `prismatic/vla/datasets/rlds/oxe/transforms.py`

Convert UAV-Flow-Sim to RLDS with the official dataset builder, then register the
UAV dataset name. This is **config/dataset registration only**, not a model change.
Keep the official 4-D action convention (`dx, dy, dz, dyaw`, metres/radians).

If a full RLDS conversion is too expensive, use the released
`wangxiangyu0814/OpenVLA-UAV` checkpoint as a third-party M1 baseline and record
that deviation explicitly.

## 3. Official LoRA-SFT command

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name uav_flow_sim \
  --run_root_dir /workspace/runs/openvla_sft \
  --adapter_tmp_dir /workspace/runs/openvla_sft/adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 2000
```

- Reference recipe: rank 32, batch 16, lr 5e-4, A100 80GB.
- On A100 80GB use batch 8–16; on A40 use batch 1–2.
- Record steps, effective batch, and samples seen. Do not compare raw steps across models.

## 4. UAV rollout adapter

`scripts/airsim_rollout.py` is the only homebrew component. It:

- captures RGB, state, instruction and verified progress,
- sends the observation to the policy server,
- decodes the returned 4-D action chunk,
- computes the intent reward,
- logs **every action, token id, log-probability, reward and terminal flag** to JSONL.

For OpenVLA, the policy server must return the decoded action tokens and their
log-probabilities; request them explicitly from the serving wrapper.

## 5. GRPO / PPO

Do not implement these in the OpenVLA repo. Use veRL (see `../verl/README.md`)
with the OpenVLA/OpenVLA-OFT worker path demonstrated by SimpleVLA-RL. PPO needs
a value function; GRPO is critic-free.
