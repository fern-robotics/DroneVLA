#!/usr/bin/env python3
"""HTTP bridge for the OpenPI websocket policy.

Runs in the OpenPI client environment (msgpack>=1) while the AirSim evaluator
runs in its own environment (msgpack<1 for msgpack-rpc).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=8002)
    ap.add_argument("--port", type=int, default=8003)
    args = ap.parse_args()
    client = websocket_client_policy.WebsocketClientPolicy(host=args.upstream_host, port=args.upstream_port)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n))
            image = np.asarray(Image.open(io.BytesIO(base64.b64decode(req["image_png_b64"]))).convert("RGB"))
            result = client.infer({
                "observation/image": image,
                "observation/state": np.asarray(req.get("state", [0, 0, 0, 0]), dtype=np.float32),
                "prompt": req["instruction"],
            })
            body = json.dumps({
                "actions": np.asarray(result["actions"]).tolist(),
                "tokens": result.get("tokens"),
                "logprobs": result.get("logprobs"),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"bridging ws:{args.upstream_port} -> http:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
