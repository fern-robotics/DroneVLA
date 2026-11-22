#!/bin/bash
DATA=/mnt/data/skripsi-ryan-adidaru
while ! grep -q ABLATION_QUEUE_DONE "$DATA/ablation_queue.log" 2>/dev/null; do sleep 60; done
if grep -q BC244_DONE "$DATA/bc244_queue.log" 2>/dev/null; then
  bash /home/digilab/ws/skripsi-ryan-adidaru/run_data300_queue.sh >> "$DATA/data300_launch.log" 2>&1
  exit $?
fi
bash /home/digilab/ws/skripsi-ryan-adidaru/run_bc244_queue.sh >> "$DATA/bc244_launch.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  bash /home/digilab/ws/skripsi-ryan-adidaru/run_data300_queue.sh >> "$DATA/data300_launch.log" 2>&1
  rc=$?
fi
exit $rc
