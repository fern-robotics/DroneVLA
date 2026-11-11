#!/usr/bin/env python3
"""Patch OpenPI pi0-FAST to expose FAST action tokens and per-token log-probs.

Usage: python integrations/openpi/patch_pi0_fast_logprobs.py /workspace/openpi

This modifies only the sampling/inference path; SFT, the FAST tokenizer, the
model weights and the loss are untouched. OpenVLA does not use this patch.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_model(path: Path) -> None:
    s = path.read_text()
    if "return_logprobs" in s:
        print("model already patched")
        return
    s = s.replace(
        "        max_decoding_steps: int | at.Int[at.Array, \"\"] = 256,\n        temperature: float = 0.0,\n    ) -> _model.Actions:",
        "        max_decoding_steps: int | at.Int[at.Array, \"\"] = 256,\n        temperature: float = 0.0,\n        return_logprobs: bool = False,\n    ) -> _model.Actions:",
    )
    s = s.replace(
        "        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))",
        "        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))\n        output_logprobs = jnp.zeros_like(output_tokens)",
    )
    s = s.replace(
        "            rng, last_logit, output_tokens, cache, _, step = carry",
        "            rng, last_logit, output_tokens, output_logprobs, cache, _, step = carry",
    )
    s = s.replace(
        "            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)",
        "            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)\n"
        "            token_logp = jnp.take_along_axis(\n"
        "                jax.nn.log_softmax(last_logit, axis=-1),\n"
        "                token.reshape((token.shape[0], 1, 1)),\n"
        "                axis=-1,\n"
        "            ).reshape((token.shape[0], 1))\n"
        "            output_logprobs = put_along_last_axis(\n"
        "                output_logprobs, jnp.broadcast_to(step, (token.shape[0], 1)), token_logp\n"
        "            )",
    )
    s = s.replace(
        "            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1\n\n        def cond(carry):\n            _, _, _, _, all_eos, step = carry",
        "            return rng, last_logit, output_tokens, output_logprobs, kv_cache, all_eos, step + 1\n\n        def cond(carry):\n            _, _, _, _, _, all_eos, step = carry",
    )
    s = s.replace(
        "        _, _, output_tokens, _, _, _ = jax.lax.while_loop(\n            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)\n        )\n        return output_tokens",
        "        _, _, output_tokens, output_logprobs, _, _, _ = jax.lax.while_loop(\n            cond, step, (rng, last_logit, output_tokens, output_logprobs, kv_cache, False, 0)\n        )\n"
        "        eos_mask = output_tokens != PALIGEMMA_EOS_TOKEN\n"
        "        return output_tokens, output_logprobs, eos_mask",
    )
    path.write_text(s)
    print("patched", path)


def patch_policy(path: Path) -> None:
    s = path.read_text()
    if "sample_has_logprobs" in s:
        print("policy already patched")
        return
    s = s.replace(
        "        outputs = {\n            \"state\": inputs[\"state\"],\n            \"actions\": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),\n        }",
        "        sample = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)\n"
        "        sample_has_logprobs = isinstance(sample, tuple)\n"
        "        if sample_has_logprobs:\n"
        "            tokens, logprobs, token_mask = sample\n"
        "            outputs = {\"state\": inputs[\"state\"], \"actions\": tokens, \"tokens\": tokens, \"logprobs\": logprobs, \"token_mask\": token_mask}\n"
        "        else:\n"
        "            outputs = {\"state\": inputs[\"state\"], \"actions\": sample}",
    )
    path.write_text(s)
    print("patched", path)


def patch_uav_outputs(path: Path) -> None:
    s = path.read_text()
    if "data.get(\"tokens\")" in s:
        print("uav outputs already patched")
        return
    s = s.replace(
        "    def __call__(self, data: dict) -> dict:\n        return {\"actions\": np.asarray(data[\"actions\"][..., :4])}",
        "    def __call__(self, data: dict) -> dict:\n"
        "        out = {\"actions\": np.asarray(data[\"actions\"][..., :4])}\n"
        "        for key in (\"tokens\", \"logprobs\", \"token_mask\"):\n"
        "            if key in data:\n"
        "                out[key] = np.asarray(data[key])\n"
        "        return out",
    )
    path.write_text(s)
    print("patched", path)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/openpi")
    patch_model(root / "src/openpi/models/pi0_fast.py")
    patch_policy(root / "src/openpi/policies/policy.py")
    uav = root / "src/openpi/policies/uav_policy.py"
    if uav.exists():
        patch_uav_outputs(uav)
    else:
        print("uav_policy.py not present; copy integrations/openpi/uav_policy.py first")


if __name__ == "__main__":
    main()
