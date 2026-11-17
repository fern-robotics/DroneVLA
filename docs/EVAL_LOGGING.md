# Logged AirSim evaluation

Every live episode logs one JSONL line per executed action:

- absolute `position_ned_m` before the action
- absolute `position_after_ned_m`
- `yaw_rad`, relative `state`, `progress`
- full `vla_actions` chunk and the executed `action`
- `tokens`, `logprobs`, `latency_ms`, collision/intervention flags

Run:

```bash
scripts/eval/run_logged_airsim_eval.sh
```

Outputs:

- `runs/logged_eval/episodes/<id>.jsonl`
- `runs/logged_eval/episode.mp4`
- `runs/logged_eval/summary.json`

MP4 is written from the AirSim camera at 10 fps.
