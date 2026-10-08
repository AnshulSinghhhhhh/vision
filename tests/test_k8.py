import os
import sys
import importlib.util
from pathlib import Path
import torch
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

# Load k8 run module dynamically
k8_path = repo_root / "kaggle" / "kernels" / "k8-m7-v2" / "run.py"
spec = importlib.util.spec_from_file_location("k8_run", k8_path)
k8_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k8_mod)


def test_k8_suite_generation_clean():
    clean_t = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
    suite = k8_mod.build_m7_v2_suite(clean_t, clean_t, condition="clean", severity=0, image_id="img_001")
    expected_keys = [
        "A_orig448", "B_filtered448",
        "B_g0.5", "B_g0.866", "B_g1.5", "B_g2.5",
        "C_down224", "D_up448"
    ]
    for k in expected_keys:
        assert k in suite, f"Missing key {k} in clean suite"
    assert "E_noise_matched" not in suite  # Only for gaussian noise
    assert suite["A_orig448"].shape == (1, 3, 448, 448)
    assert suite["C_down224"].shape == (1, 3, 224, 224)
    assert suite["D_up448"].shape == (1, 3, 448, 448)


def test_k8_suite_generation_noise():
    clean_t = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
    corr_t = clean_t + 0.1
    suite = k8_mod.build_m7_v2_suite(clean_t, corr_t, condition="gaussian_noise", severity=3, image_id="img_002")
    assert "E_noise_matched" in suite
    assert suite["E_noise_matched"].shape == (1, 3, 448, 448)
    # Check determinism: same seed gives identical tensor
    suite2 = k8_mod.build_m7_v2_suite(clean_t, corr_t, condition="gaussian_noise", severity=3, image_id="img_002")
    assert torch.allclose(suite["E_noise_matched"], suite2["E_noise_matched"])
