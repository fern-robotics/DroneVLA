# Real-VLA AirSim evaluation

```bash
# 1. start AirSimNH (VNC optional)
scripts/eval/airsim_setup.sh

# 2. start the OpenVLA/pi0 HTTP server (separate GPU/env)
python scripts/serve_openvla_actions.py --vla-path wangxiangyu0814/OpenVLA-UAV --unnorm-key uav_flow_sim --port 8000

# 3. run one real-policy episode and record MP4
scripts/eval/run_real_vla_airsim.sh
```

Outputs:

- `runs/real_vla/real_vla.mp4`
- `runs/real_vla/episodes/<id>.jsonl`
- `runs/real_vla/summary.json`

`evaluate_vla` uses the same 30-episode manifest, marker provisioning and
intent evaluator as the mock smoke. `--capture-image` is required for a real
VLA; the MP4 is written from the AirSim camera frames at 10 fps.
