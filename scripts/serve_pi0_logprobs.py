#!/usr/bin/env python3
"""Serve a pi0-FAST checkpoint over websockets with FAST token log-probs.

Requires the OpenPI repo patched with integrations/openpi/patch_pi0_fast_logprobs.py.
The normal OpenPI server remains the default for evaluation; this server is for
GRPO/PPO rollouts that need per-token log-probabilities.
"""
from __future__ import annotations

import argparse
import logging
import socket

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as _config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="OpenPI TrainConfig name, e.g. pi0_fast_uav_low_mem")
    ap.add_argument("--dir", required=True, help="checkpoint directory")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    train_config = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(
        train_config,
        args.dir,
        sample_kwargs={"return_logprobs": True},
    )
    logging.info("serving pi0-FAST with token log-probs on %s:%s", args.host, args.port)
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
