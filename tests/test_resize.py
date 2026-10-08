"""Tests for resize operators and M7 filter-matched information control."""

import numpy as np
import torch
import pytest
from knobs.resize import (
    resize_image_np,
    resize_tensor_torch,
    generate_m7_suite,
    effective_noise_gain,
    apply_gaussian_lowpass,
)


def test_constant_image_stays_constant():
    """Downsampling a constant image preserves pixel values exactly."""
    for val in [0, 128, 255]:
        const_img = np.full((448, 448, 3), val, dtype=np.uint8)
        for target_res in [384, 320, 224]:
            resized = resize_image_np(const_img, target_res)
            assert np.all(resized == val), f"Constant value {val} changed at resolution {target_res}"


def test_antialias_noise_reduction():
    """Resizing a white-noise image with antialias downsampling attenuates noise standard deviation."""
    rng = np.random.RandomState(999)
    noise = rng.normal(128, 20, (448, 448, 3)).clip(0, 255).astype(np.uint8)
    
    orig_std = np.std(noise.astype(float))
    
    # 448 -> 224 (scale factor 0.5)
    down_224 = resize_image_np(noise, 224)
    down_std = np.std(down_224.astype(float))
    
    # Downsampling acts as a low-pass filter; std should drop significantly
    assert down_std < orig_std * 0.8, f"Antialiasing did not attenuate noise std: orig={orig_std}, down={down_std}"


def test_m7_suite_structure_and_properties():
    """Verifies shapes and properties of M7 conditions A, B, C, D and Gaussian cuts."""
    tensor_448 = torch.randn(2, 3, 448, 448)
    suite = generate_m7_suite(tensor_448)
    
    assert suite["A_orig448"].shape == (2, 3, 448, 448)
    assert suite["B_filtered448"].shape == (2, 3, 448, 448)
    assert suite["C_down224"].shape == (2, 3, 224, 224)
    assert suite["D_up448"].shape == (2, 3, 448, 448)
    
    # Check default Gaussian filter arms exist
    for sigma in [0.5, 0.866, 1.5, 2.5]:
        key = f"B_g{sigma}"
        assert key in suite
        assert suite[key].shape == (2, 3, 448, 448)
    
    # Filtered B should have lower high-frequency variance than original A
    var_a = torch.var(suite["A_orig448"][:, :, 1:, :] - suite["A_orig448"][:, :, :-1, :]).item()
    var_b = torch.var(suite["B_filtered448"][:, :, 1:, :] - suite["B_filtered448"][:, :, :-1, :]).item()
    assert var_b < var_a, "Condition B (low-pass) did not reduce spatial derivative variance"
    
    # Gaussian filters should monotonically reduce high-frequency derivative variance
    vars_g = [
        torch.var(suite[f"B_g{s}"][:, :, 1:, :] - suite[f"B_g{s}"][:, :, :-1, :]).item()
        for s in [0.5, 0.866, 1.5, 2.5]
    ]
    for i in range(len(vars_g) - 1):
        assert vars_g[i] > vars_g[i + 1], f"Gaussian filtering variance not monotonic: {vars_g}"


def test_effective_noise_gain_analytic():
    """Verifies analytic noise gain against reference values expected from PyTorch antialias filter."""
    expected = {
        224: 0.313,
        320: 0.481,
        384: 0.565,
        448: 1.000,
    }
    for res, exp_val in expected.items():
        gain = effective_noise_gain(res, base_res=448)
        assert abs(gain - exp_val) <= 0.003, f"Resolution {res}: gain {gain:.4f} != expected {exp_val:.3f}"


def test_effective_noise_gain_empirical():
    """Verifies empirical noise gain on 64x3x448x448 N(0,1) matches analytic within 3%."""
    torch.manual_seed(42)
    noise = torch.randn(64, 3, 448, 448)
    in_std = noise.std().item()
    
    for res in [224, 320, 384, 448]:
        out = resize_tensor_torch(noise, res)
        measured_ratio = out.std().item() / in_std
        analytic_ratio = effective_noise_gain(res, base_res=448)
        rel_diff = abs(measured_ratio - analytic_ratio) / analytic_ratio
        assert rel_diff < 0.03, f"Resolution {res}: measured {measured_ratio:.4f} vs analytic {analytic_ratio:.4f} exceeds 3% (rel_diff={rel_diff:.3%})"
