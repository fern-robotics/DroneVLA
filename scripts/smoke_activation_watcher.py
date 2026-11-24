#!/usr/bin/env python3
"""Smoke-test scripts/ops/activate_nvme_pipeline.sh without real scene/model/GPU."""
from __future__ import annotations
import os, subprocess, tempfile, time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix="dronevl-act-", ignore_cleanup_errors=True) as td:
        t=Path(td); src=t/"data"/"skripsi"; nv=t/"nv"
        src.mkdir(parents=True); nv.mkdir()
        for d in ["UnrealEnv","openvla-uav","expert_cache_300","uavflow_train_tasks"]:
            (nv/d).mkdir(parents=True)
        eng=nv/"UnrealEnv/Collection_v4_LinuxNoEditor/Collection/Binaries/Linux/Collection"
        eng.parent.mkdir(parents=True); eng.write_text("#!/bin/sh\n"); eng.chmod(0o755)
        (nv/"openvla-uav/config.json").write_text("{}")
        (nv/"expert_cache_300/x").mkdir()
        (src/"uavflow_train_tasks").mkdir(); (src/"uavflow_train_tasks/t.json").write_text("{}")
        (src/"groups_offline_150.json").write_text("[]"); (src/"manifest_extra150.json").write_text('{"tasks":[]}')
        (t/"migrate.log").write_text("MIGRATION_COPY_DONE\n")
        (src/"run_data300_queue.sh").write_text(f'#!/bin/bash\necho START >> "{t}/fake_start.log"\necho FAKE_DATA300 >> "{t}/fake_data.log"\n')
        (src/"run_data300_queue.sh").chmod(0o755)
        (src/"watch_schedules.sh").write_text(f'#!/bin/bash\necho FAKE_SCHEDULES >> "{t}/fake_sched.log"\n')
        (src/"watch_schedules.sh").chmod(0o755)
        text=(REPO/"scripts/ops/activate_nvme_pipeline.sh").read_text()
        text=text.replace("/mnt/data/skripsi-ryan-adidaru",str(src))
        text=text.replace("/home/digilab/ws/skripsi-ryan-adidaru",str(src))
text=text.replace("/home/digilab/dronevl_nvme",str(nv))
        text=text.replace("/home/digilab/dronevl_migrate.log",str(t/"migrate.log"))
        text=text.replace("/home/digilab/dronevl_activate.log",str(t/"activate.log"))
        text=text.replace("/home/digilab/ws/skripsi-ryan-adidaru/watch_schedules.sh",str(src/"watch_schedules.sh"))
        script=src/"activate.sh"; script.write_text(text); script.chmod(0o755)
        r=subprocess.run(["bash",str(script)],cwd=src,text=True,capture_output=True)
        if r.returncode!=0:
            raise SystemExit(f"activation failed rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}\nlog={(t/'activate.log').read_text() if (t/'activate.log').exists() else ''}")
        assert (src/"UnrealEnv").is_symlink(), "UnrealEnv not symlinked"
        assert (src/"openvla-uav").is_symlink(), "model not symlinked"
        assert (src/"runs").is_symlink(), "runs not symlinked"
        assert (src/"uavflow_train_tasks").is_symlink()
        for _ in range(100):
            if (t/"fake_data.log").exists() and (t/"fake_sched.log").exists(): break
            time.sleep(0.1)
        assert (t/"fake_start.log").exists(), "fake data300 never started"
        assert "FAKE_DATA300" in (t/"fake_data.log").read_text()
        assert "FAKE_SCHEDULES" in (t/"fake_sched.log").read_text()
        print("ACTIVATION_SMOKE_OK")
        print((t/"activate.log").read_text().strip().splitlines()[-2:])

# srcreplaceWS
if __name__=="__main__": main()
