"""Thin policy adapters for the UAV eval suite.

The evaluator only needs an object with ``act(observation, instruction)``
returning a dict with ``actions`` and optional ``tokens``/``logprobs``.  Model
servers (OpenVLA, pi0-FAST) live outside this repository; this module is the
small bridge and never reimplements the model.
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

import numpy as np


class MockVLAPolicy:
    """Deterministic scripted policy wrapped in the VLA interface (for tests)."""

    def __init__(self, scripted: Any) -> None:
        self.scripted = scripted

    def act(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        action, _ = self.scripted.act(observation)
        return {"actions": [action], "tokens": [0], "logprobs": [0.0], "policy_latency_ms": 0.0}


class WebsocketPolicy:
    """OpenPI websocket policy (pi0-FAST). Actions only; tokens may be absent."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        try:
            import websockets.sync.client as wsc
            _orig = wsc.connect

            def _connect(uri, **kw):
                kw.setdefault("ping_interval", None)
                kw.setdefault("ping_timeout", None)
                kw.setdefault("open_timeout", 60)
                return _orig(uri, **kw)

            wsc.connect = _connect
            from openpi_client import websocket_client_policy
        except Exception as exc:
            raise RuntimeError("openpi_client/websockets are required for --policy websocket") from exc
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def act(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        t0 = time.time()
        result = self.client.infer({
            "observation/image": observation.get("rgb"),
            "observation/state": np.asarray(observation.get("state", [0, 0, 0, 0]), dtype=np.float32),
            "prompt": instruction,
        })
        return {
            "actions": np.asarray(result["actions"]).tolist(),
            "tokens": result.get("tokens"),
            "logprobs": result.get("logprobs"),
            "policy_latency_ms": (time.time() - t0) * 1000.0,
        }


class HttpPolicy:
    """Generic HTTP JSON policy server.

    Request: {"image_png_b64", "state", "progress", "instruction"}.
    Response: {"actions": [[dx,dy,dz,dyaw], ...], "tokens": [...], "logprobs": [...]}.
    """

    def __init__(self, url: str, timeout_s: float = 60.0) -> None:
        import urllib.request
        self.url = url
        self.timeout_s = timeout_s
        self._request = urllib.request.Request

    def act(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        from PIL import Image
        rgb = observation.get("rgb")
        buf = io.BytesIO()
        if rgb is None:
            Image.new("RGB", (256, 256), (0, 0, 0)).save(buf, format="PNG")
        else:
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
        payload = json.dumps({
            "image_png_b64": base64.b64encode(buf.getvalue()).decode(),
            "state": [float(v) for v in observation.get("state", [0, 0, 0, 0])],
            "progress": float(observation.get("progress", 0.0)),
            "instruction": instruction,
        }).encode()
        req = self._request(self.url, data=payload, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with __import__("urllib.request").request.urlopen(req, timeout=self.timeout_s) as resp:
            result = json.loads(resp.read())
        result["policy_latency_ms"] = (time.time() - t0) * 1000.0
        return result


def make_policy(spec: str, *, scripted: Any = None, host: str = "127.0.0.1", port: int = 8000,
                url: str = "") -> Any:
    if spec == "mock":
        return MockVLAPolicy(scripted)
    if spec == "websocket":
        return WebsocketPolicy(host=host, port=port)
    if spec == "http":
        if not url:
            raise ValueError("--policy http requires --policy-url")
        return HttpPolicy(url)
    raise ValueError(f"unknown policy {spec!r}")
