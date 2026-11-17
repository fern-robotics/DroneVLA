#!/usr/bin/env python3
"""Minimal teacher-forced GRPO step for B0 (released OpenVLA-UAV)."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rl.grpo import grpo_loss


def action_prompt(instruction: str, state: list[float]) -> str:
    vals = ",".join(str(round(float(v), 1)) for v in state[:4])
    return f"In: Current State: {vals}, What action should the uav take to {instruction}?\nOut:"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--base", default="/workspace/openvla-uav")
    ap.add_argument("--out", default="/workspace/runs/b0_grpo")
    ap.add_argument("--max-groups", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    groups = json.load(open(args.groups))[: args.max_groups]
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to("cuda")
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_loss = 0.0
    updates = 0
    for group in groups:
        rewards = torch.tensor([r["reward"] for r in group["rollouts"]], dtype=torch.float32)
        adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        cur_rows, old_rows, adv_rows = [], [], []
        for rollout, advantage in zip(group["rollouts"], adv):
            rows = [json.loads(l) for l in (Path(rollout["dir"]) / "episode.jsonl").read_text().splitlines() if l.strip()]
            task = json.load(open(group["task"]))
            instruction = task["instruction"]
            for r in rows:
                tokens = r.get("tokens")
                old = r.get("logprobs")
                if not tokens or not old or len(tokens) != len(old):
                    continue
                img_path = Path(rollout["dir"]) / r["frame"]
                image = Image.open(img_path).convert("RGB").resize((224, 224))
                state = list(r["local_xyz"]) + [math.degrees(r["yaw"])]
                inputs = processor(action_prompt(instruction, state), image)
                ids = inputs["input_ids"]
                if ids.ndim == 2:
                    ids = ids[0]
                ids = ids.to(dtype=torch.long)
                toks = torch.tensor(tokens, dtype=torch.long)
                prefix = ids.numel()
                full = torch.cat([ids, toks[:-1]]).unsqueeze(0).to("cuda")
                kwargs = {"input_ids": full}
                if inputs.get("pixel_values") is not None:
                    kwargs["pixel_values"] = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
                with torch.no_grad():
                    logits = model(**kwargs).logits[0]
                start = prefix - 1
                selected = logits[start:start + toks.numel()]
                cur = torch.log_softmax(selected, dim=-1).gather(-1, toks[:, None].to("cuda")).squeeze(-1)
                cur_rows.append(cur)
                old_rows.append(torch.tensor(old, dtype=torch.float32, device="cuda"))
                adv_rows.append(torch.full_like(old_rows[-1], float(advantage)))
        if not cur_rows:
            continue
        # Recompute with gradients per row (small smoke batch).
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for rollout, advantage in zip(group["rollouts"], adv):
            rows = [json.loads(l) for l in (Path(rollout["dir"]) / "episode.jsonl").read_text().splitlines() if l.strip()]
            task = json.load(open(group["task"]))
            for r in rows:
                tokens, old = r.get("tokens"), r.get("logprobs")
                if not tokens or not old or len(tokens) != len(old):
                    continue
                image = Image.open(Path(rollout["dir"]) / r["frame"]).convert("RGB").resize((224, 224))
                state = list(r["local_xyz"]) + [math.degrees(r["yaw"])]
                inputs = processor(action_prompt(task["instruction"], state), image)
                ids = inputs["input_ids"][0] if inputs["input_ids"].ndim == 2 else inputs["input_ids"]
                ids = ids.to("cuda")
                toks = torch.tensor(tokens, dtype=torch.long, device="cuda")
                full = torch.cat([ids, toks[:-1]]).unsqueeze(0).to("cuda")
                kwargs = {"input_ids": full}
                if inputs.get("pixel_values") is not None:
                    kwargs["pixel_values"] = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
                logits = model(**kwargs).logits[0]
                selected = logits[ids.numel() - 1:ids.numel() - 1 + toks.numel()]
                cur = torch.log_softmax(selected, dim=-1).gather(-1, toks[:, None]).squeeze(-1).unsqueeze(0)
                old_t = torch.tensor(old, dtype=torch.float32, device="cuda").unsqueeze(0)
                mask = torch.ones_like(cur)
                loss = grpo_loss(cur, old_t, torch.tensor([advantage], device="cuda"),
                                 mask, clip_range=0.2, beta=0.0).loss
                loss_sum = loss_sum + loss
        (loss_sum / max(1, len(cur_rows))).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        updates += 1
        total_loss += float(loss_sum.detach()) / max(1, len(cur_rows))
        print("update", updates, "loss", float(loss_sum.detach()) / max(1, len(cur_rows)), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(json.dumps({"updates": updates, "mean_loss": total_loss / max(1, updates)}))


if __name__ == "__main__":
    main()
