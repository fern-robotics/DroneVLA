# π₀-FAST UAV pilot

This is the thesis-faithful π₀-FAST path, started after the OpenVLA-UAV baseline.

## Status

- OpenPI (JAX, CUDA) installed on a separate 50 GB A40 pod.
- UAV-Flow-Sim shard 0 converted to a LeRobot dataset: 300 episodes, 9,437 frames, 4-D action/state.
- Custom `UAVInputs`/`UAVOutputs` and `LeRobotUAVDataConfig` added to OpenPI.
- Training config: `pi0_fast_uav_low_mem` — π₀-FAST base, `gemma_2b_lora`, `action_dim=4`, `action_horizon=8`, batch 4, 600 steps.
- Norm statistics: `assets/pi0_fast_uav_low_mem/uav_flow_sim`.
- Checkpoints: `checkpoints/pi0_fast_uav_low_mem/uav_pilot/<step>`.
- AirSimNH evaluation client: `scripts/pi0_airsim_intent.py`, using the OpenPI websocket policy server.

## Environment

The exact commands and patches are in:

- `integrations/openpi/uav_policy.py`
- `integrations/openpi/patch_config.py`
- `scripts/pi0_airsim_intent.py`

The dataset conversion script lives at `examples/uav/convert_uavflow_to_lerobot.py` on the training pod.

## Caveat

This is a LoRA fine-tune of π₀-FAST on one UAV-Flow-Sim shard, not the final thesis training budget. It is the first real π₀-FAST UAV adaptation checkpoint in this project.
