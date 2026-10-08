import sys
import importlib.util
from pathlib import Path
import torch
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

k10_path = repo_root / "kaggle" / "kernels" / "k10-resize-ablation" / "run.py"
spec = importlib.util.spec_from_file_location("k10_run", k10_path)
k10_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k10_mod)


def test_resize_operators_output_shapes():
    dummy = torch.randn(2, 3, 448, 448)
    for op in ["bilinear_antialias", "bilinear_no_antialias", "bicubic_antialias", "area"]:
        for r in [224, 320, 384]:
            out = k10_mod.apply_resize_operator(dummy, op, r)
            assert out.shape == (2, 3, r, r)

    out_gauss = k10_mod.apply_resize_operator(dummy, "gaussian_prefilter_448", 448)
    assert out_gauss.shape == (2, 3, 448, 448)

    out_ident = k10_mod.apply_resize_operator(dummy, "identity_448", 448)
    assert out_ident.shape == (2, 3, 448, 448)
    assert torch.allclose(out_ident, dummy)


def test_resize_operator_noise_gains():
    device = torch.device("cpu")
    gain_bl_aa = k10_mod.compute_operator_noise_gain("bilinear_antialias", 224, device)
    gain_bl_no_aa = k10_mod.compute_operator_noise_gain("bilinear_no_antialias", 224, device)

    # Bilinear antialias at 224 should be approx 0.313
    assert 0.29 < gain_bl_aa < 0.33, f"Bilinear AA gain {gain_bl_aa} outside expected range [0.29, 0.33]"
    # Without antialias, gain should be significantly higher (~0.50)
    assert gain_bl_no_aa > gain_bl_aa, "Non-antialiased downsampling should pass more noise variance"
