#!/usr/bin/env python3
"""Serve an OpenPI pi0-FAST policy over HTTP for the AirSim evaluator."""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as _config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--norm-stats", default="")
    ap.add_argument("--port", type=int, default=8004)
    args = ap.parse_args()
    cfg = _config.get_config(args.config)
    norm = None
    if args.norm_stats:
        norm = normalize.deserialize_json(Path(args.norm_stats).read_text())
    policy = policy_config.create_trained_policy(
        cfg, args.dir, sample_kwargs={"return_logprobs": True}, norm_stats=norm
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n))
            image = np.asarray(Image.open(io.BytesIO(base64.b64decode(req["image_png_b64"]))).convert("RGB"))
            result = policy.infer({
                "observation/image": image,
                "observation/state": np.asarray(req.get("state", [0, 0, 0, 0]), dtype=np.float32),
                "prompt": req["instruction"],
            })
            tokens = result.get("tokens")
            logprobs = result.get("logprobs")
            body = json.dumps({
                "actions": np.asarray(result["actions"]).tolist(),
                "tokens": None if tokens is None else np.asarray(tokens).tolist(),
                "logprobs": None if logprobs is None else np.asarray(logprobs).tolist(),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"serving pi0 HTTP on {args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
