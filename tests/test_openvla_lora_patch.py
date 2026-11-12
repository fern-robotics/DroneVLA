import subprocess
import sys
import tempfile
from pathlib import Path


def test_patch_lora_config():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "vla-scripts").mkdir()
        f = root / "vla-scripts" / "finetune.py"
        f.write_text('LoraConfig(target_modules="all-linear", lora_alpha=min(cfg.lora_rank, 16))')
        subprocess.run([sys.executable, "integrations/openvla/patch_lora_config.py", str(root)], check=True)
        s = f.read_text()
        assert 'target_modules=["q_proj", "v_proj"]' in s
        assert "lora_alpha=64" in s
