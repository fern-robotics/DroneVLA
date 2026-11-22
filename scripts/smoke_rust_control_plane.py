#!/usr/bin/env python3
"""Smoke test the Rust experiment control plane without Unreal or model weights."""
from __future__ import annotations
import json, os, socket, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("DRONEVLCTL", ROOT / "target/debug/dronevlctl"))


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def run(*args, check=True, timeout=30):
    r = subprocess.run([str(BIN), *map(str, args)], cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"FAIL {' '.join(map(str,args))}\nstdout={r.stdout}\nstderr={r.stderr}")
    return r


def write_result(d: Path, task: str, success: bool, ndtw: float, steps: int):
    d.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "success": int(success), "ndtW": ndtw,
               "final_target_m": 1.0, "rollouts": [{"steps": steps, "success": success, "ndtW": ndtw}]}
    (d / f"{task}.json").write_text(json.dumps(payload))


def smoke_eval(tmp: Path):
    a, b = tmp / "m0" / "results", tmp / "online" / "results"
    write_result(a, "t1", False, 0.2, 50); write_result(a, "t2", True, 0.8, 200)
    write_result(b, "t1", True, 0.3, 50); write_result(b, "t2", True, 0.9, 200)
    out = tmp / "summary.json"
    run("eval-summary", "--dir", a, "--out", out)
    s = json.loads(out.read_text())["results"]
    assert s["tasks"] == 2 and s["successes"] == 1, s
    out2 = tmp / "paired.json"
    run("eval-compare", "--baseline", a, "--candidate", b, "--out", out2)
    p = json.loads(out2.read_text())
    assert p["candidate_wins"] == 1 and p["baseline_wins"] == 0, p
    return p


def smoke_plan(tmp: Path):
    run("plan", "--out-dir", tmp / "plan")
    runs = json.loads((tmp / "plan" / "experiment_plan.json").read_text())
    ids = {r["experiment"] for r in runs}
    assert ids == set(range(1, 9)), ids
    assert len(runs) >= 20, len(runs)
    return runs


def smoke_cache(tmp: Path):
    src = tmp / "src"; (src / "frames").mkdir(parents=True)
    (src / "episode.jsonl").write_text("{}\n"); (src / "frames" / "f.jpg").write_bytes(b"x")
    run("cache", "--src", src, "--out", tmp / "cache")
    m = json.loads((tmp / "cache" / "cache_manifest.json").read_text())
    assert m["files"] == 2 and (tmp / "cache" / "frames" / "f.jpg").exists(), m
    return m


def smoke_run():
    ok = json.loads(run("run", "--", "sh", "-c", "exit 0").stdout)
    assert ok["success"] is True
    bad = run("run", "--timeout-s", "1", "--", "sh", "-c", "sleep 5", check=False)
    assert bad.returncode == 1, bad
    assert json.loads(bad.stdout)["timed_out"] is True, bad.stdout
    return bad.stdout


def smoke_proxy():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
            body = json.dumps({"actions": [[1, 2, 3, 4]], "tokens": [1, 2, 3, 4], "logprobs": [-0.1]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    up_port = free_port()
    up = HTTPServer(("127.0.0.1", up_port), H)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    bind_port = free_port()
    p = subprocess.Popen([str(BIN), "proxy", "--bind", f"127.0.0.1:{bind_port}",
                          "--upstream", f"http://127.0.0.1:{up_port}/predict"],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", bind_port), timeout=0.2).close(); break
            except OSError: time.sleep(0.1)
        # Valid 1x1 PNG; the Rust proxy must decode and resize it to 224x224.
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGPgEpEDAABoAD1UCKP3AAAAAElFTkSuQmCC"
        body_payload = json.dumps({"image": tiny_png, "proprio": [0,0,0,0,0,0], "progress": 0.0, "instr": "x"}).encode()
        req = Request(f"http://127.0.0.1:{bind_port}/predict", data=body_payload,
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            body = json.loads(urlopen(req, timeout=5).read())
        except Exception as e:
            detail = getattr(e, "read", lambda: b"")()
            raise RuntimeError(f"proxy error: {e} body={detail!r}")
        assert body["action"] == [[1, 2, 3, 4]], body
    finally:
        p.terminate(); p.wait(timeout=5); up.shutdown()
    return body


def smoke_unrealcv():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1); port = srv.getsockname()[1]
    def serve():
        import struct
        c, _ = srv.accept()
        header = c.recv(8)
        magic, n = struct.unpack("<II", header)
        assert magic == 0x9E2B83C1, hex(magic)
        c.recv(n)
        payload = b"10.0 20.0 30.0\n"
        c.sendall(struct.pack("<II", 0x9E2B83C1, len(payload)) + payload)
        c.close()
    threading.Thread(target=serve, daemon=True).start()
    r = run("unrealcv-probe", "--addr", f"127.0.0.1:{port}", "--command", "vget /camera/0/lit")
    assert "10.0 20.0 30.0" in r.stdout, r.stdout
    return r.stdout.strip()


def main():
    if not BIN.exists():
        raise SystemExit(f"missing binary {BIN}; run `cargo build` first")
    with tempfile.TemporaryDirectory(prefix="dronevl-smoke-") as td:
        tmp = Path(td)
        print("eval:", json.dumps(smoke_eval(tmp), sort_keys=True)[:160])
        print("plan:", len(smoke_plan(tmp)), "runs")
        print("cache:", smoke_cache(tmp)["files"], "files")
        print("run:", smoke_run().strip()[:120])
        print("proxy:", smoke_proxy())
        print("unrealcv:", smoke_unrealcv())
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
