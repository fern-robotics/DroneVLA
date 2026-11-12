#!/usr/bin/env python3
"""HTTP policy server for OpenVLA that also returns action tokens and log-probs.

The model, processor, action tokenizer and un-normalization are the official
OpenVLA implementation; this wrapper only exposes the generated token ids and
their log-probabilities so the UAV adapter can log them for GRPO/PPO.

Untested here because it requires the 7B checkpoint and a CUDA GPU. Run it on
the OpenVLA pod after installing the official repo:

  python scripts/serve_openvla_actions.py --vla-path openvla/openvla-7b \
      --unnorm-key uav_flow_sim --port 8000
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image
from integrations.openvla.prompt import build_uav_prompt


def load_model(vla_path: str, device: str):
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    model.eval()
    return processor, model


def decode_actions(model, token_ids: np.ndarray, unnorm_key: str) -> np.ndarray:
    bins = model.bin_centers
    discretized = model.vocab_size - token_ids
    discretized = np.clip(discretized - 1, 0, bins.shape[0] - 1)
    normalized = bins[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


class Handler(BaseHTTPRequestHandler):
    processor = None
    model = None
    device = "cuda"
    unnorm_key = "uav_flow_sim"
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        image = Image.open(io.BytesIO(base64.b64decode(request["image_png_b64"]))).convert("RGB")
        prompt = build_uav_prompt(
            request["instruction"],
            request.get("state"),
            request.get("progress", 0),
        )
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        if int(input_ids[0, -1]) != 29871:
            input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1)
        n = self.model.get_action_dim(self.unnorm_key)
        with self.lock, torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids, max_new_tokens=n, output_scores=True,
                return_dict_in_generate=True, do_sample=False,
            )
        token_ids = generated.sequences[0, -n:].cpu().numpy()
        logprobs = []
        for score, token in zip(generated.scores, token_ids):
            logprobs.append(float(torch.log_softmax(score[0], dim=-1)[int(token)].item()))
        actions = decode_actions(self.model, token_ids, self.unnorm_key)
        body = json.dumps({
            "actions": [actions[:4].tolist()],
            "tokens": token_ids.tolist(),
            "logprobs": logprobs,
            "action_dim": int(len(actions)),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla-path", default="openvla/openvla-7b")
    ap.add_argument("--unnorm-key", default="uav_flow_sim")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    Handler.processor, Handler.model = load_model(args.vla_path, args.device)
    Handler.device = args.device
    Handler.unnorm_key = args.unnorm_key
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving OpenVLA on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
