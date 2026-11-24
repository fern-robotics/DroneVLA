#!/usr/bin/env bash
set -uo pipefail
SRC=/mnt/data/skripsi-ryan-adidaru
NV=/home/digilab/dronevl_nvme
LOG=/home/digilab/dronevl_activate.log
: > "$LOG"
while ! grep -q MIGRATION_COPY_DONE /home/digilab/dronevl_migrate.log 2>/dev/null; do
  if grep -q RSYNC_FAIL /home/digilab/dronevl_migrate.log 2>/dev/null; then echo MIGRATION_FAIL | tee -a "$LOG"; exit 1; fi
  sleep 30
done
for d in UnrealEnv openvla-uav expert_cache_300; do
  [ -d "$NV/$d" ] || { echo "MISSING $NV/$d" | tee -a "$LOG"; exit 1; }
done
[ -x "$NV/UnrealEnv/Collection_v4_LinuxNoEditor/Collection/Binaries/Linux/Collection" ] || { echo NO_ENGINE | tee -a "$LOG"; exit 1; }
[ -f "$NV/openvla-uav/config.json" ] || { echo NO_MODEL | tee -a "$LOG"; exit 1; }
[ -f "$NV/expert_cache_300/2025-06-26_17-33-34/expert.json" ] || true
# Copy task metadata and small manifests to NVMe, then replace them with symlinks.
rsync -a "$SRC/uavflow_train_tasks" "$NV/" >> "$LOG" 2>&1 || exit 1
for f in "$SRC"/groups_*.json "$SRC"/manifest_*.json; do [ -f "$f" ] && cp -a "$f" "$NV/"; done
for d in UnrealEnv openvla-uav expert_cache_300 uavflow_train_tasks; do
  if [ -e "$SRC/$d" ] && [ ! -L "$SRC/$d" ]; then mv "$SRC/$d" "$SRC/$d.hdd"; fi
  [ -e "$SRC/$d" ] || ln -s "$NV/$d" "$SRC/$d"
done
if [ -e "$SRC/runs" ] && [ ! -L "$SRC/runs" ]; then mv "$SRC/runs" "$SRC/runs.hdd"; fi
[ -e "$SRC/runs" ] || ln -s "$NV/runs" "$SRC/runs"
for f in "$SRC"/groups_*.json "$SRC"/manifest_*.json; do
  b=$(basename "$f")
  [ -L "$SRC/$b" ] && continue
  [ -f "$SRC/$b" ] && mv "$SRC/$b" "$SRC/$b.hdd" || true
  ln -s "$NV/$b" "$SRC/$b" 2>/dev/null || true
done
echo ACTIVE_NVME | tee -a "$LOG"
nohup bash "$SRC/run_data300_queue.sh" >> "$SRC/data300_launch.log" 2>&1 &
nohup bash "$SRC/watch_schedules.sh" >> "$SRC/schedules_launch.log" 2>&1 &
echo DATA300_STARTED | tee -a "$LOG"
