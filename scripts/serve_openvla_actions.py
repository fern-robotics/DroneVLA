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
import queue
import io
from pathlib import Path
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image
from integrations.openvla.prompt import build_uav_prompt


def install_batch_generation(model) -> None:
    """OpenVLA-UAV guards batch>1; install the equivalent unguarded method."""
    from typing import Any, Dict, List, Optional
    import torch

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        })
        return model_inputs

    type(model).prepare_inputs_for_generation = prepare_inputs_for_generation

    # Remove the remote forward's batch-size assertion via AST (not text hacks).
    import ast, inspect, sys, textwrap
    mod = sys.modules[type(model).__module__]
    tree = ast.parse(textwrap.dedent(inspect.getsource(type(model).forward)))

    class RemoveBatchAssert(ast.NodeTransformer):
        def visit_Assert(self, node):
            if node.msg is not None and "batch size of 1" in ast.unparse(node.msg):
                return None
            return node

    tree = RemoveBatchAssert().visit(tree)
    ns = mod.__dict__
    exec(compile(ast.unparse(tree), "<openvla_batch_forward>", "exec"), ns)
    type(model).forward = ns["forward"]


def load_model(vla_path: str, device: str, adapter_path: str = ""):
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
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
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
    prompt_style = "dronevl"
    absolute_output = False
    lock = threading.Lock()
    batch_size = 4
    batch_timeout = 0.05
    queue = queue.Queue()
    worker_started = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        image = Image.open(io.BytesIO(base64.b64decode(request["image_png_b64"]))).convert("RGB")
        state = request.get("state", [0, 0, 0, 0])
        if self.prompt_style == "openvla_uav":
            if not Handler.worker_started:
                Handler.worker_started = True
                threading.Thread(target=Handler._batch_worker, daemon=True).start()
            event = threading.Event()
            item = {"request": request, "event": event, "result": None, "error": None}
            Handler.queue.put(item)
            event.wait()
            if item["error"]:
                body = json.dumps({"error": item["error"]}).encode()
                self.send_response(500); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            else:
                self._send(json.dumps(item["result"]).encode())
            return
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
                input_ids=input_ids,
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=n,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
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

    @classmethod
    def _batch_worker(cls) -> None:
        while True:
            first = cls.queue.get()
            items = [first]
            deadline = time.time() + cls.batch_timeout
            while len(items) < cls.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    items.append(cls.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results = cls._predict_batch([it["request"] for it in items])
                for it, result in zip(items, results):
                    it["result"] = result
            except Exception as exc:
                import traceback
                traceback.print_exc()
                for it in items:
                    it["error"] = str(exc)
            finally:
                for it in items:
                    it["event"].set()

    @classmethod
    def _predict_batch(cls, requests: list[dict]) -> list[dict]:
        images, prompts, states = [], [], []
        for request in requests:
            images.append(Image.open(io.BytesIO(base64.b64decode(request["image_png_b64"]))).convert("RGB"))
            state = request.get("state", [0, 0, 0, 0])
            s4 = [round(float(v), 1) for v in list(state)[:4]]
            states.append(s4)
            prompts.append(f"In: Current State: {','.join(map(str, s4))}, What action should the uav take to {request['instruction']}?\nOut:")
        inputs = cls.processor(prompts, images, padding=True, return_tensors="pt").to(cls.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        eos = torch.full((input_ids.shape[0], 1), 29871, device=input_ids.device)
        input_ids = torch.cat([input_ids, eos], dim=1)
        attention = torch.ones_like(input_ids)
        n = cls.model.get_action_dim(cls.unnorm_key)
        with cls.lock, torch.no_grad():
            generated = cls.model.generate(
                input_ids=input_ids, pixel_values=inputs.get("pixel_values"),
                attention_mask=attention, max_new_tokens=n, output_scores=True,
                return_dict_in_generate=True, do_sample=False,
            )
        token_ids = generated.sequences[:, -n:].cpu().numpy()
        out = []
        for i, s4 in enumerate(states):
            logprobs = [float(torch.log_softmax(score[i], dim=-1)[int(tok)].item())
                        for score, tok in zip(generated.scores, token_ids[i])]
            actions = np.asarray(decode_actions(cls.model, token_ids[i], cls.unnorm_key)[:4], dtype=float)
            if cls.absolute_output:
                cy = np.deg2rad(float(s4[3])); c, sn = np.cos(cy), np.sin(cy)
                actions[:3] = np.asarray(s4[:3]) + np.array([c*actions[0]-sn*actions[1], sn*actions[0]+c*actions[1], actions[2]])
                actions[3] = actions[3] + cy
            out.append({"actions": [actions.tolist()], "tokens": token_ids[i].tolist(), "logprobs": logprobs})
        return out

    def _send(self, body: bytes) -> None:
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
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--norm-stats", default="", help="JSON action stats for the calibrated M0 decoder")
    ap.add_argument("--adapter", default="", help="LoRA adapter directory")
    ap.add_argument("--prompt-style", choices=["dronevl", "openvla_uav"], default="dronevl")
    ap.add_argument("--absolute-output", action="store_true", help="OpenVLA-UAV native local pose output")
    args = ap.parse_args()
    Handler.processor, Handler.model = load_model(args.vla_path, args.device, args.adapter)
    install_batch_generation(Handler.model)
    Handler.device = args.device
    Handler.unnorm_key = args.unnorm_key
    Handler.prompt_style = args.prompt_style
    Handler.absolute_output = args.absolute_output
    Handler.batch_size = args.batch_size
    if args.norm_stats:
        stats = json.loads(Path(args.norm_stats).read_text())
        Handler.model.norm_stats[args.unnorm_key] = stats
        print(f"loaded M0 norm stats {args.norm_stats} -> {args.unnorm_key}", flush=True)
    Handler.unnorm_key = args.unnorm_key
    Handler.prompt_style = args.prompt_style
    Handler.absolute_output = args.absolute_output
    Handler.batch_size = args.batch_size
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving OpenVLA on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
