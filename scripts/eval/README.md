# AirSimNH UAV intent evaluation suite

A frozen, deterministic 30-episode evaluation suite for instruction-conditioned
UAV flight intents:

- **22 ID episodes** and **8 OOD episodes**;
- ID/OOD are separated by **intent template**, with explicit `split`/`ood`
  fields and disjoint template whitelists;
- families: approach, pass-left/right, orbit-cw/ccw, hold-at-distance,
  climb/descend, and composed sequences;
- world coordinates are **NED** (`x` north, `y` east, `z` down); yaw is
  clockwise-positive around `+z`, matching AirSim;
- no randomness: the manifest is generated from constants and checked in as
  `scripts/eval/manifests/uav_intent_30.json`.

## Files

| Path | Purpose |
| --- | --- |
| `scripts/eval/manifest.py` | Build/validate/load the 30-episode manifest. |
| `scripts/eval/manifests/uav_intent_30.json` | Frozen manifest (22 ID + 8 OOD). |
| `scripts/eval/intents.py` | Geometric predicates and `evaluate_episode()`. |
| `scripts/eval/evaluate.py` | CLI, deterministic mock env/policy, AirSim hook. |
| `scripts/eval/airsim_env.py` | Thin AirSimNH adapter; lazy `airsim` import. |
| `tests/test_eval_suite.py` | Manifest, predicate, schema, and mock tests. |

## Setup

The suite and its tests need only `numpy` and `Pillow`; AirSim is optional and
is imported lazily only by `--airsim`.

```bash
python3 -m venv .venv-eval
.venv-eval/bin/pip install numpy pillow
```

If the repository already has a suitable virtual environment, use it instead.

## Manifest

Generate or check the frozen manifest:

```bash
.venv-eval/bin/python -m scripts.eval.manifest --write
.venv-eval/bin/python -m scripts.eval.manifest --check
```

Each episode contains `episode_id`, `split`, `ood`, `family`, `template`,
`instruction`, `markers` with world-NED positions, `start_pose`,
`success_predicates`, `max_steps`, `dt_s`, and a nominal `reference_path_m`.
`manifest.py` validates counts, unique IDs, template disjointness, marker
references, finite geometry, and predicate parameters. Loading a missing
manifest fails with a message to run `--write`.

## Run the mock evaluation

```bash
.venv-eval/bin/python -m scripts.eval.evaluate --mock
```

Run a subset:

```bash
.venv-eval/bin/python -m scripts.eval.evaluate --mock \
  --episode-id id-approach-01 --episode-id id-orbit-cw-01 \
  --out runs/eval/subset
```

`--mock` uses a deterministic kinematic point-mass environment and a scripted
waypoint policy derived from the success predicates. The scripted policy is a
reproducibility baseline, not a VLA. It succeeds on all 30 episodes, which makes
the suite useful as an evaluator smoke test.

Outputs under `--out` (default `runs/eval/uav_intent_30`):

```text
episodes/<episode_id>.jsonl   # observed step log for each episode
per_episode.jsonl             # one evaluated episode summary per line
summary.csv                   # flat per-episode metrics table
summary.json                  # aggregate metrics + per-episode summaries
```

Per-step JSONL fields include `position_ned_m`, `yaw_rad`, `action` (the
4-D `[dx, dy, dz, dyaw]`), `collision`, `intervention`, `invalid_action`, and
`policy_latency_ms`. The summary reports success, intent-adherence score,
subgoal completion, collisions, interventions, invalid actions, latency
(mean/median/p95), path length, and path efficiency. `summary.json` also splits
ID and OOD stats.

## Run in AirSimNH

Install the `airsim` package in the same environment and start AirSimNH with
its RPC server reachable (default `127.0.0.1:41451`):

```bash
.venv-eval/bin/pip install airsim
.venv-eval/bin/python -m scripts.eval.evaluate --airsim \
  --airsim-ip 127.0.0.1 --airsim-port 41451 \
  --airsim-vehicle Drone0 --airsim-camera 0 \
  --out runs/eval/airsim
```

`AirSimEnv` resets the vehicle to each manifest start pose, captures RGB on
demand, steps a 4-D delta action kinematically with `simSetVehiclePose`, and
exposes NED position/yaw. It does **not** spawn markers: markers are scene
assets described by the manifest and must already exist in the AirSimNH scene.
If the `airsim` package or RPC server is unavailable, the adapter raises
`AirSimUnavailableError` with installation/connection guidance.

The adapter's action is a kinematic delta for deterministic evaluation, not a
replacement for AirSim's velocity/position controllers. For real dynamic
flight, use the existing rollout/OpenVLA/pi0 bridges in `scripts/`.

## Predicate semantics

All predicates are evaluator-only and scan the observed trajectory.

- `distance_to_marker`: the stand-off distance is within
  `distance_m ± tolerance_m` (horizontal by default; `axis: "3d"` is also
  accepted). This is used for approach and as a composed-sequence subgoal.
- `pass_side`: the marker's transverse plane is defined by the horizontal
  direction from the episode start pose to the marker. Success requires a
  near-to-far crossing with lateral offset at least `clearance_m` on the
  requested side. Positive lateral is right of the approach direction.
- `orbit`: bearing is `atan2(east, north)`; increasing bearing is clockwise.
  Success requires the accumulated bearing sweep, restricted to samples inside
  `[min_radius_m, radius_m + radius_tolerance_m]`, to reach
  `min_angle_deg` in the requested direction.
- `hold_at_distance`: a continuous run inside the stand-off annulus must last
  at least `duration_s` (timestamps come from `t_s` or `dt_s`).
- `altitude`: the trajectory must reach `target_z_m ± tolerance_m`. In NED,
  climbing means a more negative `z`.

`evaluate_episode(episode, trajectory)` checks composed predicates in order on
successive trajectory slices. It returns:

- `success`: every subgoal satisfied **and** no collision;
- `intent_score` / `intent_adherence_score`: mean per-subgoal progress;
- `subgoal_completion`, `completed_subgoals`, `total_subgoals`, `subgoals`;
- `collision`, `invalid_action_count`;
- `path_length_m`, `reference_path_m`, `path_efficiency`.

Path efficiency is the nominal manifest route length divided by the observed
path length, capped at `1.0`; it is `None` when no reference is available.

## Tests

```bash
.venv-eval/bin/python -m unittest tests.test_eval_suite -v
```

The tests cover:

- 22/8 counts, explicit ID/OOD flags, and template disjointness;
- frozen-manifest equality with the generator;
- each geometric predicate (distance, pass plane/side, orbit angle/direction,
  hold duration, altitude);
- `evaluate_episode` metrics, collision and invalid-action flags;
- mock evaluator output schema and per-episode JSONL/CSV/JSON files;
- a successful mock approach episode and a deterministic 30/30 mock suite;
- lazy AirSim import and the helpful missing-package error.

No test imports the `airsim` package at test time.
