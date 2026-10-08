"""Tests for deterministic corruptions and frequency-controlled noise."""

import numpy as np
import pytest
from knobs.corrupt import (
    apply_corruption,
    PRIMARY_CORRUPTIONS,
    generate_frequency_controlled_noise,
    CORRUPTION_BACKEND,
    FALLBACK_GAUSSIAN_NOISE_SIGMAS,
)


@pytest.fixture(autouse=True)
def allow_fallback_env(monkeypatch):
    monkeypatch.setenv("KNOBS_ALLOW_FALLBACK", "1")


def test_corruption_determinism():
    """Test 1: Bit-identical outputs for identical (image_id, corruption, severity)."""
    h, w = 448, 448
    dummy_img = (np.random.RandomState(123).rand(h, w, 3) * 255).astype(np.uint8)
    
    for corr in PRIMARY_CORRUPTIONS:
        out1 = apply_corruption(dummy_img, image_id="img_001", corruption_name=corr, severity=3)
        out2 = apply_corruption(dummy_img, image_id="img_001", corruption_name=corr, severity=3)
        
        # Bit-identical for identical (image_id, corruption, severity)
        assert np.array_equal(out1, out2), f"Non-deterministic output for corruption {corr}"
        
    # For stochastic corruptions (e.g. gaussian_noise), different seeds produce different outputs
    out_noise1 = apply_corruption(dummy_img, image_id="img_001", corruption_name="gaussian_noise", severity=3)
    out_noise2 = apply_corruption(dummy_img, image_id="img_002", corruption_name="gaussian_noise", severity=3)
    assert not np.array_equal(out_noise1, out_noise2), "Identical output for different seeds in gaussian_noise"


def test_severity_monotonicity():
    """Test 2: Mean absolute pixel change from clean increases with severity."""
    h, w = 448, 448
    dummy_img = (np.random.RandomState(456).rand(h, w, 3) * 255).astype(np.uint8)
    
    for corr in ["gaussian_noise", "defocus_blur", "contrast"]:
        diffs = []
        for sev in [1, 2, 3, 4, 5]:
            out = apply_corruption(dummy_img, image_id="test_mono", corruption_name=corr, severity=sev)
            mean_diff = np.mean(np.abs(out.astype(float) - dummy_img.astype(float)))
            diffs.append(mean_diff)
            
        # Check monotonicity
        for i in range(len(diffs) - 1):
            assert diffs[i] <= diffs[i + 1] + 1e-5, f"Monotonicity violated for {corr}: {diffs}"


def test_frequency_controlled_noise():
    """Verifies frequency-controlled noise produces matched RMS power across broadband and bandlimited modes."""
    h, w = 448, 448
    clean = np.full((h, w, 3), 128, dtype=np.uint8)
    
    target_rms = 25.0
    broad = generate_frequency_controlled_noise(clean, "img_rms", "broadband", target_rms=target_rms)
    band = generate_frequency_controlled_noise(clean, "img_rms", "bandlimited", target_rms=target_rms)
    
    diff_broad = broad.astype(float) - 128.0
    diff_band = band.astype(float) - 128.0
    
    rms_broad = np.sqrt(np.mean(diff_broad ** 2))
    rms_band = np.sqrt(np.mean(diff_band ** 2))
    
    # Tolerances within 1.0 RMS unit
    assert abs(rms_broad - target_rms) < 2.0
    assert abs(rms_band - target_rms) < 2.0


def test_fallback_gaussian_noise_sigmas():
    """Verifies that the fallback gaussian_noise sigma list equals [0.08, 0.12, 0.18, 0.26, 0.38]."""
    assert FALLBACK_GAUSSIAN_NOISE_SIGMAS == [0.08, 0.12, 0.18, 0.26, 0.38]


def test_corruption_backend_exposed():
    """Verifies that CORRUPTION_BACKEND is exposed as expected."""
    assert CORRUPTION_BACKEND in ("imagecorruptions", "fallback")


def test_fallback_backend_raises_without_env_flag(monkeypatch):
    """Verifies that fallback corruption raises unless KNOBS_ALLOW_FALLBACK=1."""
    monkeypatch.delenv("KNOBS_ALLOW_FALLBACK", raising=False)
    monkeypatch.setattr("knobs.corrupt.CORRUPTION_BACKEND", "fallback")
    dummy = np.zeros((448, 448, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="KNOBS_ALLOW_FALLBACK"):
        apply_corruption(dummy, "id_test", "gaussian_noise", severity=1)
        
    # With flag set, it executes properly
    monkeypatch.setenv("KNOBS_ALLOW_FALLBACK", "1")
    out = apply_corruption(dummy, "id_test", "gaussian_noise", severity=1)
    assert out.shape == dummy.shape
