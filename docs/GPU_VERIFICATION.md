# GPU verification (OpenVLA, A100 80GB)

Run on a single A100 80GB, 2026-11-11.

## Result

The official OpenVLA LoRA-SFT path completed a training step on the UAV RLDS
dataset and saved a LoRA adapter:

- dataset: 80 train / 8 dev / 11 test RLDS episodes
- LoRA rank 32, batch 1, 1 step
- adapter: `adapter_model.safetensors`, **484 MB**
- run directory: `runs/openvla_smoke/`

The final fused 7B save was stopped during CPU-side LoRA merging; the adapter
and training step are the verification artifacts. Do not treat the missing fused
checkpoint as a training failure.

## Environment pins that were required

OpenVLA's stack targets Python 3.10/3.11, not 3.12:

- Python **3.11**
- `torch==2.2.0`, `torchvision==0.17.0`, `torchaudio==2.2.0`
- `transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`
- `tensorflow==2.15.0`, `tensorflow-datasets==4.9.3`
- `tensorflow-metadata==1.14.0`, `protobuf==3.20.3`
- `wandb==0.16.6`
- `dlimp` from `moojink/dlimp_openvla`, installed `--no-deps`
- `tensorflow-graphics==2021.12.3`, installed `--no-deps`
- `numpy<2`

## Failures found and fixed

1. **NumPy 2 vs torch 2.2** — segfault/NumPy init failure. Pin `numpy<2`.
2. **torchrun c10d rendezvous segfault** — avoid torchrun; set
   `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK` and launch the script
   directly.
3. **dlimp missing** — `uv pip install --no-deps "dlimp @ git+..."`.
4. **tensorflow-graphics missing** — `uv pip install --no-deps tensorflow-graphics`.
5. **TFDS 4.9.10 SkipDecoding error** — pin TFDS 4.9.3, tensorflow-metadata 1.14,
   protobuf 3.20.3.
6. **RLDS `steps` feature** — must be `tfds.features.Dataset`, not `Sequence`;
   fixed in PR #6.
7. **RLDS root** — `--data_root_dir` must point at the directory containing
   `uav_flow_sim/1.0.0`, not the parent `rlds/` directory.

## Remaining

- π₀-FAST token log-prob server: code merged in PR #4; runtime test still needed.
- Model-specific GRPO/PPO optimizer loops.
- Live AirSim evaluation (needs a simulator pod).

## π₀-FAST log-prob runtime

The patch and server compile against a fresh OpenPI checkout. Runtime was not
executed:

- community A100-40 never published SSH;
- secure A40/A100/H100 capacities were unavailable at creation time;
- the CPU pod has a 4 GB cgroup limit, too small for the 3B checkpoint.

Next run, once a GPU is available:

```bash
bash /tmp/pi0_run.sh   # installs OpenPI, applies the patch, runs the droid observation test
```

Expected output includes `tokens`, `logprobs` and `token_mask` with the same
decoding length, then `PI0_VERIFY_OK`.

## π₀-FAST log-prob runtime — verified

Verified on an A40 48 GB with the base checkpoint and `pi0_fast_droid`:

- output keys: `actions`, `tokens`, `logprobs`, `token_mask`, `state`
- `tokens` / `logprobs` / `token_mask`: `(256,)` each
- first log-probs: `0, 0, 0, -3.59, -2.84`
- action/token arrays came from the FAST decoder

Two inference bugs were found and fixed:

1. `token` is `(B, 1)`, so `token[:, None, None]` produced a 4-D gather index.
   Fixed with `reshape((B, 1, 1))`.
2. `return_logprobs` was passed into a jitted function, so the Python `if`
   became a traced boolean. The patched sampler now always returns
   `(tokens, logprobs, mask)` and `Policy.infer` detects the tuple.

The websocket server uses the same patched policy; the droid observation test
exercises the log-prob path directly.


## Trained-LoRA UAV server — verified

After a 1-step UAV LoRA-SFT (`loss 13.23`, checkpoint `.../uav_1step/0`), the
websocket server returned:

```text
keys ['actions','logprobs','policy_timing','server_timing','token_mask','tokens']
actions     (10, 4)
tokens      (256,)
logprobs    (256,)
token_mask  (256,)
UAV_SERVER_OK
```

The client needed websocket keepalive disabled because the first JAX inference
exceeds the default ping timeout. That workaround is already in
`scripts/eval/vla_policy.py`.
