"""Tests for resize operators and M7 filter-matched information control."""

import numpy as np
import torch
import pytest
from knobs.resize import resize_image_np, resize_tensor_torch, generate_m7_suite


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
    """Verifies shapes and properties of M7 conditions A, B, C, D."""
    tensor_448 = torch.randn(2, 3, 448, 448)
    suite = generate_m7_suite(tensor_448)
    
    assert suite["A_orig448"].shape == (2, 3, 448, 448)
    assert suite["B_filtered448"].shape == (2, 3, 448, 448)
    assert suite["C_down224"].shape == (2, 3, 224, 224)
    assert suite["D_up448"].shape == (2, 3, 448, 448)
    
    # Filtered B should have lower high-frequency variance than original A
    var_a = torch.var(suite["A_orig448"][:, :, 1:, :] - suite["A_orig448"][:, :, :-1, :]).item()
    var_b = torch.var(suite["B_filtered448"][:, :, 1:, :] - suite["B_filtered448"][:, :, :-1, :]).item()
    assert var_b < var_a, "Condition B (low-pass) did not reduce spatial derivative variance"
