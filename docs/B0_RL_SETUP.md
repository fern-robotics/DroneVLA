# B0 native RL setup

B0 is the released `wangxiangyu0814/OpenVLA-UAV` checkpoint, not M1.

## Contract

- prompt: `In: Current State: x,y,z,yaw, What action should the uav take to <instruction>?\nOut:`
- state: local `x,y,z,yaw_degrees`, centimetres
- output: native absolute local pose
- action scale: 1.0

## Matrix

| Arm | Groups | Group size |
|---|---:|---:|
| B0 + GRPO | 100 | 4 |
| B0 + PPO | 100 | 4 |

Both arms use identical task, seed, initial pose, reward, and max steps.

## Missing before launch

- native group collector that stores images, tokens, log-probs
- B0 trainable LoRA + PPO value head
- side-aware pass reward
