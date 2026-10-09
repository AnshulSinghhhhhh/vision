"""Test backend safety guard and fallback divergence detection."""

import os
import numpy as np
import pytest

from knobs.corrupt import (
    apply_corruption,
    _fallback_corrupt,
    CORRUPTION_BACKEND,
    IMAGECORRUPTIONS_VERSION,
)


def test_fallback_raises_without_env_flag(monkeypatch):
    """Asserts that the fallback backend raises a RuntimeError unless KNOBS_ALLOW_FALLBACK=1."""
    monkeypatch.delenv("KNOBS_ALLOW_FALLBACK", raising=False)
    monkeypatch.setattr("knobs.corrupt.CORRUPTION_BACKEND", "fallback")
    
    dummy = np.zeros((448, 448, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="KNOBS_ALLOW_FALLBACK"):
        apply_corruption(dummy, "test_id", "gaussian_noise", severity=3)

    # When explicitly allowed, fallback executes
    monkeypatch.setenv("KNOBS_ALLOW_FALLBACK", "1")
    out = apply_corruption(dummy, "test_id", "gaussian_noise", severity=3)
    assert out.shape == dummy.shape


def test_defocus_blur_fallback_differs_from_imagecorruptions():
    """Asserts defocus_blur output differs between imagecorruptions and fallback on a fixed random image."""
    if CORRUPTION_BACKEND != "imagecorruptions":
        pytest.skip("imagecorruptions is not installed in the current environment")

    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (448, 448, 3), dtype=np.uint8)

    ic_out = apply_corruption(img, image_id="fixed_guard_test", corruption_name="defocus_blur", severity=3)
    fb_out = _fallback_corrupt(img, corruption_name="defocus_blur", severity=3)

    # The real ImageNet-C disk defocus kernel differs from the scipy gaussian_filter fallback
    assert not np.array_equal(ic_out, fb_out), (
        "Silent fallback detected! imagecorruptions defocus_blur output is identical to fallback."
    )
    max_diff = np.max(np.abs(ic_out.astype(int) - fb_out.astype(int)))
    assert max_diff > 0, f"Defocus output difference should be positive, got max diff {max_diff}"
