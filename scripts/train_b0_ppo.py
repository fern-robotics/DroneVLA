#!/usr/bin/env python3
"""PPO for B0: LoRA actor + per-decision value head, GAE over native rollouts."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rl.ppo import compute_gae, ppo_loss


def action_prompt(instruction: str, state: list[float]) -> str:
    vals = ",".join(str(round(float(v), 1)) for v in state[:4])
    return f"In: Current State: {vals}, What action should the uav take to {instruction}?\nOut:"


class ValueHead(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, 1)

    def forward(self, h):
        return self.proj(h).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--base", default="/workspace/openvla-uav")
    ap.add_argument("--out", default="/workspace/runs/b0_ppo")
    ap.add_argument("--max-groups", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    args = ap.parse_args()
    groups = json.load(open(args.groups))[: args.max_groups]
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    hidden = int(model.config.text_config.hidden_size)
    value_head = ValueHead(hidden).to("cuda")
    params = [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    model.train(); value_head.train()

    def forward_decision(image, instruction, state, tokens):
        inputs = processor(action_prompt(instruction, state), image)
        ids = inputs["input_ids"]
        if ids.ndim == 2:
            ids = ids[0]
        ids = ids.to("cuda", dtype=torch.long)
        toks = torch.tensor(tokens, dtype=torch.long, device="cuda")
        full = torch.cat([ids, toks[:-1]]).unsqueeze(0)
        kwargs = {"input_ids": full, "output_hidden_states": True, "return_dict": True}
        if inputs.get("pixel_values") is not None:
            kwargs["pixel_values"] = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
        out = model(**kwargs)
        start = ids.numel() - 1
        logits = out.logits[0, start:start + toks.numel()]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, toks[:, None]).squeeze(-1)
        h = out.hidden_states[-1][0, ids.numel() - 1]
        value = value_head(h)
        return logp, value

    total = 0.0; updates = 0
    for group in groups:
        task = json.load(open(group["task"]))
        rows_by_rollout = []
        for rollout in group["rollouts"]:
            ep = Path(rollout["dir"]) / "episode.jsonl"
            if not ep.exists():
                continue
            rows_by_rollout.append([json.loads(l) for l in ep.read_text().splitlines() if l.strip()])
        if not rows_by_rollout:
            continue
        opt.zero_grad(set_to_none=True)
        all_cur, all_old, all_adv, all_val, all_ret = [], [], [], [], []
        for rows in rows_by_rollout:
            states = [list(r["local_xyz"]) + [math.degrees(r["yaw"])] for r in rows]
            imgs = [Image.open(Path(r["frame"])).convert("RGB").resize((224, 224)) for r in rows]
            tokens = [r["tokens"] for r in rows]
            old = [r["logprobs"] for r in rows]
            logps, values = [], []
            for im, st, tk in zip(imgs, states, tokens):
                lp, v = forward_decision(im, task["instruction"], st, tk)
                logps.append(lp); values.append(v)
            vals = torch.stack(values)
            # Dense native reward: progress in inverse distance, terminal at the end.
            d = [float(r.get("target_distance_m", 0.0)) for r in rows]
            rewards = []
            for t in range(len(rows)):
                prev = d[t - 1] if t > 0 else d[0]
                rewards.append(prev - d[t])
            rewards[-1] -= d[-1]
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device="cuda")
            dones = torch.zeros_like(rewards_t); dones[-1] = 1.0
            adv, ret = compute_gae(rewards_t, vals, dones, args.gamma, args.lam)
            for lp, old_lp, a, v, r in zip(logps, old, adv, vals, ret):
                n = len(old_lp)
                all_cur.append(lp[:n]); all_old.append(torch.tensor(old_lp, device="cuda"))
                all_adv.append(a.expand(n)); all_val.append(v.expand(1)); all_ret.append(r.expand(1))
        cur = torch.stack(all_cur); old = torch.stack(all_old)
        mask = torch.ones_like(cur)
        adv = torch.stack(all_adv)
        vals = torch.stack(all_val).reshape(-1); ret = torch.stack(all_ret).reshape(-1)
        loss = ppo_loss(cur, old, adv, mask, values=vals, returns=ret,
                        clip_range=args.clip, value_coef=args.value_coef,
                        entropy_coef=args.entropy_coef)
        loss.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        updates += 1; total += float(loss.loss.detach())
        print("update", updates, "loss", float(loss.loss.detach()),
              "policy", float(loss.policy_loss.detach()), "value", float(loss.value_loss.detach()), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    torch.save(value_head.state_dict(), Path(args.out) / "value_head.pt")
    print(json.dumps({"updates": updates, "mean_loss": total / max(1, updates)}))


if __name__ == "__main__":
    main()
