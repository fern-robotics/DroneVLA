#!/bin/bash
DATA=/mnt/data/skripsi-ryan-adidaru
LOG=$DATA/ablation_queue.log
while ! grep -q ABLATION_QUEUE_DONE "$LOG" 2>/dev/null; do sleep 60; done
if grep -q FULL244_ABLATION_FAIL "$DATA/full244_ablation_queue.log" 2>/dev/null; then exit 1; fi
bash /home/digilab/ws/skripsi-ryan-adidaru/run_full244_ablation_queue.sh >> "$DATA/full244_ablation_launch.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo FULL244_ABLATION_FAIL >> "$DATA/full244_ablation_queue.log"; exit $rc; fi
bash /home/digilab/ws/skripsi-ryan-adidaru/run_data300_queue.sh >> "$DATA/data300_launch.log" 2>&1
