# UAV-Flow-Sim native migration

Branch: `feature/uavflow-sim-native`

## Why

UAV-Flow-Sim is an **Unreal Engine campus built with UnrealCV**, not AirSimNH.
It uses 6-DoF pose/position control, while DroneVL currently uses a 4-D body-FRD
delta `[dx,dy,dz,dyaw]`. Moving to the native simulator removes the cross-engine
gap but changes the action contract, evaluator and checkpoint format.

## Scope

- native 6-DoF pose ingestion and rollout
- explicit world-NED <-> body-FRD conversion
- native UAV-Flow-Sim success/NDTW evaluation
- separate checkpoint/evaluation config; do not reuse AirSimNH numbers

## Current status

- conversion module: `integrations/uavflow_sim/action.py`
- round-trip tests: `tests/test_uavflow_sim_action.py`
- interactive UnrealCV client: not yet implemented

## Next

1. Obtain/run the UAV-Flow-Sim UnrealCV scene and control API.
2. Implement a UE client that returns native pose plus RGB.
3. Decide whether the policy predicts 6-DoF pose or 4-D delta.
4. Rebuild SFT targets and re-run M0/M1/M2 with the native action contract.
