# Development results

These are development/pilot measurements, not the thesis main experiment.

## AirSimNH environment validation

- AirSimNH (Microsoft AirSim 1.8.1, UE4/Vulkan) on an A40 RunPod, noVNC desktop.
- out-and-back waypoint episode: SUCCESS.
- square episode: SUCCESS (landing cleanup warning).
- visual-grounding marker missions: 2/2 SUCCESS.
- five-family suite: 8/10 completed before the compatibility run was stopped.

## Surrogate Rust experiment (UAV-Flow-Sim SFT)

- SFT data: 494 UAV-Flow-Sim episodes / 15,035 frames; action `[dx,dy,dz,dyaw]` in metres/radians.
- Test split: 38 AirSimNH-style intent episodes, 30 ID / 8 held-out compositions; 3 seeds per arm.
- M1 success 65.8% (OpenVLA codec) and 67.5% (FAST codec); M0 26.3% and 21.1%.
- M2 (GRPO) was mixed: 57.0% and 68.4% in the second configuration. The first configuration showed a larger degradation; the rollout-noise fix reduced it.
- Paired bootstrap CIs and per-intent tables are in the generated `report.md`.

Caveat: the policy is a 48→4 linear goal-conditioned controller, not OpenVLA/π₀-FAST. These numbers validate the harness, not the thesis hypothesis.

## OpenVLA-UAV in actual AirSimNH

- Checkpoint: `wangxiangyu0814/OpenVLA-UAV` (official UAV-Flow OpenVLA-7B baseline).
- Loaded bf16 with eager attention; ~16.5 GB VRAM; inference 0.23–0.37 s after warm-up.
- Instruction: `Move to the red circle on the front side`.
- Provisioned red marker at 10 m ahead; the VLA approached from 10.0 m to 3.38 m in 26 decisions and satisfied the 3.5 m approach predicate.
- Predicted actions were almost constant forward (approx. 18–37 cm / 0.5 s execution), near-zero lateral and yaw.
- This is a real OpenVLA-UAV closed-loop intent run in AirSimNH. It is not an M0/M1/M2 comparison and not a statistical result.
