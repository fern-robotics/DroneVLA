"""Compatibility proxy from UAV-Flow-Eval's /predict API to DroneVL's OpenVLA HTTP API."""
from __future__ import annotations

import base64
import io
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


class Proxy:
    def __init__(self, upstream_url: str, timeout_s: float = 120.0) -> None:
        self.upstream_url = upstream_url
        self.timeout_s = timeout_s

    def predict(self, payload: dict) -> dict:
        image = base64.b64decode(payload["image"])
        from PIL import Image
        import io as _io
        _im = Image.open(_io.BytesIO(image)).convert("RGB")
        if _im.size != (224, 224):
            _im = _im.resize((224, 224))
            _buf = _io.BytesIO(); _im.save(_buf, format="PNG"); image = _buf.getvalue()
        body = json.dumps({
            "image_png_b64": base64.b64encode(image).decode(),
            "state": payload.get("proprio", [0, 0, 0, 0, 0, 0]),
            "progress": payload.get("progress", 0.0),
            "instruction": payload.get("instr", ""),
        }).encode()
        request = urllib.request.Request(
            self.upstream_url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read())
        return {"action": result.get("actions", []), "done": False}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=5007)
    args = ap.parse_args()
    proxy = Proxy(args.upstream)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path == "/reset":
                body = b'{"ok": true}'
            else:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                body = json.dumps(proxy.predict(payload)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"proxy ws-openvla {args.upstream} -> :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
