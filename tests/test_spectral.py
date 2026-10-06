"""Tests for spectral feature extractors and zero-parameter gate."""

import numpy as np
import pytest
from knobs.spectral import compute_spectral_features
from knobs.selector import ZeroParamSpectralGate, compute_oracles


def test_spectral_features_extraction():
    """Extracts features on clean vs noisy synthetic image and verifies noise sigma responds."""
    clean = np.full((448, 448, 3), 128, dtype=np.uint8)
    noisy = np.clip(clean.astype(float) + np.random.normal(0, 20, clean.shape), 0, 255).astype(np.uint8)
    
    clean_feats = compute_spectral_features(clean)
    noisy_feats = compute_spectral_features(noisy)
    
    assert "noise_sigma" in clean_feats
    assert "laplacian_var" in clean_feats
    assert "psd_slope" in clean_feats
    assert noisy_feats["noise_sigma"] > clean_feats["noise_sigma"] * 5


def test_zero_param_spectral_gate():
    """Verifies that high noise selects downsampling to 224."""
    gate = ZeroParamSpectralGate(noise_thresh=0.04, blur_thresh=0.002)
    
    res_high_noise = gate.select_resolution({"noise_sigma": 0.08, "laplacian_var": 0.01})
    assert res_high_noise == 224
    
    res_clean = gate.select_resolution({"noise_sigma": 0.01, "laplacian_var": 0.01})
    assert res_clean == 448


def test_oracles_calculation():
    """Verifies condition-level and per-image oracle calculation."""
    records = [
        {"image_id": "1", "condition": "c1", "correct_by_res": {224: True, 320: False, 384: False, 448: False}},
        {"image_id": "2", "condition": "c1", "correct_by_res": {224: False, 320: False, 384: False, 448: True}},
        {"image_id": "3", "condition": "c2", "correct_by_res": {224: True, 320: True, 384: True, 448: True}},
    ]
    oracles = compute_oracles(records, resolutions=[224, 320, 384, 448])
    
    assert oracles["per_image_oracle_upper_bound"] == 1.0  # every image was correct at some resolution
    assert oracles["best_fixed_acc"] == 2.0 / 3.0
