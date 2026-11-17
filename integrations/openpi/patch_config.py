"""Patch a fresh OpenPI checkout with the DroneVL UAV data config.

Run on the OpenPI repo before training/serving:
    python integrations/openpi/patch_config.py /workspace/openpi
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/openpi")
PATH = ROOT / "src/openpi/training/config.py"
s = PATH.read_text()

if "import os" not in s.split("\n\n")[0]:
    s = "import os\n" + s
if "import openpi.policies.uav_policy as uav_policy" not in s:
    # Place after the last openpi import near the top.
    anchor = "from openpi.policies import policy_config as _policy_config"
    if anchor in s:
        s = s.replace(anchor, anchor + "\nimport openpi.policies.uav_policy as uav_policy", 1)
    else:
        s = s.replace("import openpi.models.model as _model", "import openpi.models.model as _model\nimport openpi.policies.uav_policy as uav_policy", 1)

if "class LeRobotUAVDataConfig" not in s:
    marker = "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:"
    block = '''@dataclasses.dataclass(frozen=True)
class LeRobotUAVDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[uav_policy.UAVInputs(model_type=model_config.model_type)],
            outputs=[uav_policy.UAVOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:'''
    if marker not in s:
        raise SystemExit("TrainConfig marker not found")
    s = s.replace(marker, block, 1)

if 'name="pi0_fast_uav_low_mem"' not in s:
    cfg = '''    TrainConfig(
        name="pi0_fast_uav_low_mem",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=4, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotUAVDataConfig(
            repo_id=os.environ.get("UAV_REPO_ID", "uav_flow_sim_train"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=10000,
        batch_size=4,
        save_interval=2000,
        keep_period=10000,
        num_workers=4,
        ema_decay=None,
        wandb_enabled=False,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=4, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
    ),
    TrainConfig(
        name="pi0_fast_uav_base",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=4, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b"
        ),
        data=LeRobotUAVDataConfig(
            repo_id=os.environ.get("UAV_REPO_ID", "uav_flow_sim_train"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=1,
        batch_size=1,
        save_interval=1,
        num_workers=0,
        ema_decay=None,
        wandb_enabled=False,
    ),
'''
    idx = s.rfind("\n]")
    if idx < 0:
        raise SystemExit("config list end not found")
    s = s[:idx] + "\n" + cfg + "\n]" + s[idx + 2 :]

PATH.write_text(s)
print("patched", PATH)
