# UAV pi0-FAST policy server

After SFT produces a LoRA checkpoint, serve it with token log-probs:

```bash
OPENPI_DIR=/workspace/openpi \
  scripts/run_uav_pi0_server.sh /workspace/openpi/checkpoints/pi0_fast_uav_low_mem/uav_pilot/10000
```

The server runs:

```bash
uv run python scripts/serve_pi0_logprobs.py \
  --config pi0_fast_uav_low_mem \
  --dir <checkpoint> \
  --port 8000
```

The checkpoint must use `paligemma_variant="gemma_2b_lora"`, matching
`pi0_fast_uav_low_mem`. A full `gemma_2b` base checkpoint will not load under
that config; use `pi0_fast_uav_base` for base-only smoke tests.

Each inference returns `actions` (10×4), `tokens`, `logprobs` and `token_mask`
for GRPO/PPO collection.
