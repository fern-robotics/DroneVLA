#!/usr/bin/env python3
"""Run the real ablation/data300/schedule shell scripts against fake GPU tools.

This catches control-flow bugs (wrong flags, bad merges, missing markers) without
loading OpenVLA, Unreal, or a dataset. It is intentionally a mock pipeline, not
a model result.
"""
from __future__ import annotations
import json, os, shutil, signal, stat, subprocess, sys, tempfile, textwrap, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPS = REPO / "scripts" / "ops"
FAKE = r'''#!/usr/bin/env python3
import json, os, socket, sys, time
from pathlib import Path

def arg(name, default=None):
    a=sys.argv
    return a[a.index(name)+1] if name in a else default

def main():
    script=Path(sys.argv[1]).name if len(sys.argv)>1 else ""
    if script == "train_b0_eggrpo.py":
        out=Path(arg("--out")); out.mkdir(parents=True, exist_ok=True)
        (out/"adapter_model.safetensors").write_bytes(b"adapter")
        (out/"checkpoint-1").mkdir(exist_ok=True)
    elif script == "serve_openvla_actions.py":
        port=int(arg("--port", "8010"))
        s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port)); s.listen(5)
        while True:
            c,_=s.accept(); c.close()
    elif script == "policy_proxy.py":
        port=int(arg("--port", "5007"))
        s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port)); s.listen(5)
        while True:
            c,_=s.accept(); c.close()
    elif script == "eval_uavflow_test_5090.py":
        out=Path(arg("--out")); out.mkdir(parents=True, exist_ok=True)
        n=int(arg("--limit", "50"))
        res=out/"results"; res.mkdir(exist_ok=True)
        rows=[]
        for i in range(n):
            tid=f"task_{i:03d}"
            p={"task":tid,"success":1 if i%4==0 else 0,"ndtW":0.5,"final_target_m":2.0,
               "rollouts":[{"steps":50,"success":i%4==0,"ndtW":0.5,"final_target_m":2.0}]}
            (res/f"{tid}.json").write_text(json.dumps(p)); rows.append(p)
        rate=sum(x["success"] for x in rows)/n
        (out/"summary.json").write_text(json.dumps({"tasks":n,"success":rate,"success_rate":rate,
            "pass_at_k":rate,"k":1,"mean_ndtW":0.5,"mean_final_target_m":2.0}))
    elif script == "collect_uavflow_queue.py":
        manifest=json.load(open(arg("--manifest")))["tasks"]; out=Path(arg("--out")); out.mkdir(parents=True, exist_ok=True)
        groups=[]
        for i,t in enumerate(manifest):
            tid=Path(t).stem; d=out/tid; (d/"r0").mkdir(parents=True, exist_ok=True)
            (d/"r0"/"episode.jsonl").write_text("")
            g={"group":i,"task":t,"rollouts":[{"dir":str(d/"r0"),"reward":0.0,"progress":0.0,"cross_track_m":0.0,"terminal_m":0.0}]*4}
            (d/"group.json").write_text(json.dumps(g)); groups.append(g)
        (out/"groups.json").write_text(json.dumps(groups))
    elif script == "build_expert_cache.py":
        Path(arg("--out")).mkdir(parents=True, exist_ok=True)
    else:
        raise SystemExit(f"unhandled fake script {script}")
if __name__=="__main__": main()
'''

ONLINE = r'''#!/usr/bin/env bash
set -u
DATA=/mnt/data/skripsi-ryan-adidaru
TAG=${TAG:-online}; R=${N_ROUNDS:-1}
mkdir -p "$DATA/runs/${TAG}_adapter_r$(printf %02d "$R")"
touch "$DATA/runs/${TAG}_adapter_r$(printf %02d "$R")/adapter_model.safetensors"
echo "ONLINE_DONE" >> "$DATA/${TAG}.launch.log"
'''

def prep(sandbox: Path):
    root=sandbox/"ws"; data=sandbox/"data"
    for p in [root/"envs/openvla/bin", root/"envs/uavflow/bin", root/"repo/scripts/eval", root/"repo/integrations/uavflow_sim",
              root/"UAV-Flow/UAV-Flow-Eval/test_jsons", data/"runs/eval_offline150_244_gpu/results",
              data/"runs/online150", data/"runs", data/"UnrealEnv", data/"openvla-uav", data/"uavflow_train_tasks"]:
        p.mkdir(parents=True, exist_ok=True)
    fake=root/"fake.py"; fake.write_text(FAKE); fake.chmod(0o755)
    for p in [root/"envs/openvla/bin/python", root/"envs/uavflow/bin/python"]:
        p.write_text(fake.read_text()); p.chmod(0o755)
    for name in ["train_b0_eggrpo.py","serve_openvla_actions.py","collect_uavflow_queue.py","eval_uavflow_test_5090.py","policy_proxy.py","build_expert_cache.py"]:
        (root/"repo/scripts"/name).write_text(f"# fake\n")
    (root/"run_online_eggrpo.sh").write_text(ONLINE.replace("/mnt/data/skripsi-ryan-adidaru", str(data))); (root/"run_online_eggrpo.sh").chmod(0o755)
    # data fixtures
    tasks=[str(data/"uavflow_train_tasks"/f"t{i:03d}.json") for i in range(300)]
    for t in tasks: Path(t).write_text("{}")
    groups=[]
    for i,t in enumerate(tasks[:150]):
        d=data/"fake_groups"/Path(t).stem/"r0"; d.mkdir(parents=True, exist_ok=True)
        groups.append({"group":i,"task":t,"rollouts":[{"dir":str(d),"reward":0.0,"progress":0.0,"cross_track_m":0.0,"terminal_m":0.0}]*4})
    (data/"groups_offline_150.json").write_text(json.dumps(groups))
    (data/"manifest_150.json").write_text(json.dumps({"tasks":tasks[:150]}))
    (data/"manifest_extra150.json").write_text(json.dumps({"tasks":tasks[150:]}))
    for i in range(50):
        tid=f"task_{i:03d}"
        (data/"runs/eval_offline150_244_gpu/results"/f"{tid}.json").write_text(json.dumps({"task":tid,"success":1 if i%4==0 else 0,"ndtW":0.5,"final_target_m":2.0,"rollouts":[{"steps":50}]}))
    # existing online groups: 15 x 10
    for r in range(1,16):
        rd=data/"runs/online150"/f"roundr{r:02d}"; rd.mkdir(parents=True, exist_ok=True)
        gs=[]
        for j in range(10):
            t=tasks[(r-1)*10+j]; gs.append({"group":j,"task":t,"rollouts":[]})
        (rd/"groups.json").write_text(json.dumps(gs))
    (data/"expert_cache_150").mkdir(exist_ok=True); (data/"expert_cache_300").mkdir(exist_ok=True)
    return root,data

def copy_ops(root: Path):
    for src in OPS.glob("*.sh"):
        s=src.read_text().replace("/home/digilab/ws/skripsi-ryan-adidaru", str(root)).replace("/mnt/data/skripsi-ryan-adidaru", str(root.parent/"data"))
        (root/src.name).write_text(s); (root/src.name).chmod(0o755)

def run_until(script, root, marker, logpath, timeout=120):
    p=subprocess.Popen(["bash", str(root/script)], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
    deadline=time.time()+timeout
    while time.time()<deadline:
        text=Path(logpath).read_text() if Path(logpath).exists() else ""
        if marker in text or "FAIL" in text:
            break
        if p.poll() is not None:
            break
        time.sleep(0.2)
    text=Path(logpath).read_text() if Path(logpath).exists() else ""
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    p.wait(timeout=5)
    if marker not in text:
        raise SystemExit(f"missing {marker} in {logpath}\n{text[-3000:]}")
    return text

def main():
    with tempfile.TemporaryDirectory(prefix="dronevl-pipeline-") as td:
        sandbox=Path(td); root,data=prep(sandbox); copy_ops(root)
        log=run_until("run_ablation_queue.sh", root, "ABLATION_QUEUE_DONE", data/"ablation_queue.log", 180)
        assert "ABLATION_QUEUE_DONE" in log
        rows=(data/"runs/ablation_results.csv").read_text().strip().splitlines()
        if len(rows) < 7:
            print("ABLATION_LOG:\n"+log[-5000:])
        assert len(rows)>=7, rows
        bclog=run_until("run_bc244_queue.sh", root, "BC244_DONE", data/"bc244_queue.log", 180)
        assert "BC244_DONE" in bclog and "BC244_FAIL" not in bclog
        assert len((data/"runs/bc244_results.csv").read_text().splitlines())>=3
        dlog=run_until("run_data300_queue.sh", root, "DATA300_DONE", data/"data300_queue.log", 180)
        assert "DATA300_DONE" in dlog and "DATA300_FAIL" not in dlog
        assert len(json.load(open(data/"groups_offline_300.json")))>=300
        assert len(json.load(open(data/"groups_online_300.json")))>=300
        slog=run_until("run_schedules_queue.sh", root, "SCHEDULES_DONE", data/"schedules_queue.log", 180)
        assert "SCHEDULES_DONE" in slog and "SCHEDULE_FAIL" not in slog
        assert len((data/"runs/ablation_results.csv").read_text().splitlines())>=10
        print("PIPELINE_MOCK_OK")
        print("ablation rows:",len(rows)-1)
        print("data300 groups:",len(json.load(open(data/"groups_offline_300.json"))))
        print("schedule adapters:",[p.name for p in (data/"runs").glob("online_sched_*_adapter_*")])

if __name__=="__main__": main()
