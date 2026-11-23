# Experiment operations pipeline

The long-running 5090 chain is split into three marker-driven stages.

```text
scripts/ops/run_ablation_queue.sh
  experiments 3–6: six offline arms + fixed-50 evals
  marker: ABLATION_QUEUE_DONE

watch_full244.sh
  waits for ABLATION_QUEUE_DONE
  runs full-244 evals for all six ablation arms
  marker: FULL244_ABLATION_DONE or FULL244_ABLATION_FAIL
  then starts data300

watch_data300.sh
  waits for ABLATION_QUEUE_DONE
  collects 150 extra offline groups
  merges groups_offline_150 → groups_offline_300
  trains offline300
  collects 150 extra online groups with r15
  trains online300
  marker: DATA300_DONE or DATA300_FAIL

watch_schedules.sh
  waits for DATA300_DONE
  runs 5×30 and 30×5
  marker: SCHEDULES_DONE
```

All result rows append to:

```text
/mnt/data/skripsi-ryan-adidaru/runs/ablation_results.csv
```

## Mock verification

`scripts/smoke_experiment_pipeline.py` runs the real shell scripts against fake
Python/model/simulator commands. It checks:

- all six ablation arms are launched and evaluated
- `groups_offline_300` has ≥300 groups
- `groups_online_300` has ≥300 groups
- 5×30 and 30×5 adapters are produced
- no `TRAIN_FAIL`, `SCHEDULE_FAIL`, or `DATA300_FAIL` marker appears

```bash
python3 scripts/smoke_experiment_pipeline.py
```

The mock caught two real shell bugs:

```bash
# broken under set -u
local exp=$1 id=$2 adapter=$3 eout=$DATA/.../$id
local id=$1 adapter=$2 eout=$DATA/.../${id}_244
```

`$id` was expanded before assignment in both the schedule and BC-244 scripts,
aborting each as soon as it started. Both now use separate `local eout=...`
lines.
