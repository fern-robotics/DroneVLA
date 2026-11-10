from pathlib import Path

p = Path("/workspace/openpi/src/openpi/training/config.py")
s = p.read_text()
if "uav_policy" in s:
    print("already patched")
    raise SystemExit(0)
s = s.replace(
    "from openpi.policies import policy_config as _policy_config",
    "from openpi.policies import policy_config as _policy_config\nfrom openpi.policies import uav_policy",
)
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
s = s.replace(marker, block, 1)
cfg = '''    TrainConfig(
        name="pi0_fast_uav_low_mem",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=4, action_horizon=8, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotUAVDataConfig(
            repo_id="uav_flow_sim",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=600,
        batch_size=4,
        save_interval=300,
        num_workers=4,
        ema_decay=None,
        wandb_enabled=False,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=4, action_horizon=8, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
    ),
]'''
idx = s.rfind("\n]")
s = s[:idx] + "\n" + cfg + s[idx + 2:]
p.write_text(s)
print("patched")
