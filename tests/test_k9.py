import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from knobs.corrupt import (
    generate_fft_bandlimited_noise,
    apply_fft_frequency_noise,
    FREQ_BANDS,
)


def test_mask_partition_of_unity():
    """Verify that annular band masks partition the frequency space [0, 224] cyc/img."""
    h, w = 448, 448
    u = np.fft.fftfreq(h) * h
    v = np.fft.fftfreq(w) * w
    U, V = np.meshgrid(u, v, indexing="ij")
    R = np.sqrt(U**2 + V**2)

    bands = ["low", "mid", "high"]
    masks = []
    for band in bands:
        r_lo, r_hi = FREQ_BANDS[band]
        if r_hi < h // 2:
            m = (R >= r_lo) & (R < r_hi)
        else:
            m = (R >= r_lo) & (R <= r_hi)
        masks.append(m)

    # Disjointness: no two masks overlap
    assert not (masks[0] & masks[1]).any(), "Overlap between low and mid masks"
    assert not (masks[1] & masks[2]).any(), "Overlap between mid and high masks"
    assert not (masks[0] & masks[2]).any(), "Overlap between low and high masks"

    # Union covers all radii up to 224
    union_mask = masks[0] | masks[1] | masks[2]
    within_nyquist = R <= (h // 2)
    assert np.array_equal(union_mask[within_nyquist], np.ones_like(union_mask[within_nyquist], dtype=bool))


def test_parseval_and_rms_invariance_across_bands():
    """Verify Parseval energy conservation and spatial RMS invariance across frequency bands.
    
    By Parseval's theorem, spatial energy equals normalized spectral energy.
    Because independent Fourier coefficients with random phases yield spatial Gaussian
    marginals with standard deviation target_rms, spatial clipping acts identically on
    the 1D marginal distributions regardless of frequency band, keeping post-clip RMS equal.
    """
    shape = (448, 448, 3)
    target_rms = 45.9  # severity 3 target
    clean = np.full(shape, 128, dtype=np.uint8)

    post_clip_rms_values = []
    for band in ["low", "mid", "high"]:
        noise, pre_rms = generate_fft_bandlimited_noise(shape=shape, band=band, target_rms=target_rms, seed=42)
        # 1. Parseval check: spatial variance equals target_rms^2 within 1%
        spatial_rms = np.sqrt(np.mean(noise**2))
        assert abs(spatial_rms - target_rms) / target_rms < 0.01

        # 2. Check spatial Gaussian marginal kurtosis ~ 3 (normal distribution)
        kurtosis = np.mean(noise**4) / (np.mean(noise**2)**2)
        assert abs(kurtosis - 3.0) < 0.2, f"Band {band} non-Gaussian marginal (kurtosis={kurtosis:.2f})"

        # 3. Simulate image corruption addition + clip
        corrupted = np.clip(clean.astype(float) + noise, 0, 255)
        post_rms = np.sqrt(np.mean((corrupted - clean)**2))
        post_clip_rms_values.append(post_rms)

    # Across low, mid, high: post-clip RMS should be within 0.5 units of each other (~41.5)
    max_diff = max(post_clip_rms_values) - min(post_clip_rms_values)
    assert max_diff < 0.5, f"Post-clip RMS differs across bands by {max_diff:.3f} > 0.5: {post_clip_rms_values}"


def test_annular_power_distribution():
    """Verify that each band has >= 95% of its power inside its annulus."""
    shape = (448, 448, 3)
    h, w, c = shape
    u = np.fft.fftfreq(h) * h
    v = np.fft.fftfreq(w) * w
    U, V = np.meshgrid(u, v, indexing="ij")
    R = np.sqrt(U**2 + V**2)

    for band, (r_lo, r_hi) in FREQ_BANDS.items():
        if band == "broadband":
            continue
        noise, pre_rms = generate_fft_bandlimited_noise(
            shape=shape,
            band=band,
            target_rms=45.9,
            seed=12345,
        )
        # Compute 2D Fourier power
        W = np.fft.fft2(noise, axes=(0, 1))
        power = np.abs(W)**2
        total_power = np.sum(power)

        if r_hi < h // 2:
            annulus_mask = (R >= r_lo) & (R < r_hi)
        else:
            annulus_mask = (R >= r_lo) & (R <= r_hi)

        in_power = np.sum(power[annulus_mask])
        power_frac = in_power / total_power
        assert power_frac >= 0.95, f"Band {band} power fraction {power_frac:.4f} < 0.95"


def test_pre_clip_rms_accuracy():
    """Verify that pre-clip RMS is within 1% of the target value."""
    targets = [45.9, 96.9]
    for target in targets:
        for band in ["low", "mid", "high", "broadband"]:
            noise, pre_rms = generate_fft_bandlimited_noise(
                shape=(448, 448, 3),
                band=band,
                target_rms=target,
                seed=999,
            )
            measured_rms = float(np.sqrt(np.mean(noise**2)))
            rel_diff = abs(measured_rms - target) / target
            assert rel_diff < 0.01, f"Band {band} target {target} rel_diff {rel_diff:.4f} >= 0.01"


def test_per_image_seeding_determinism():
    """Verify same image ID produces identical noise, different IDs produce different noise."""
    dummy_img = np.zeros((448, 448, 3), dtype=np.uint8)
    
    corr1, pre1, post1 = apply_fft_frequency_noise(dummy_img, image_id="img_100", band="high", severity=3)
    corr2, pre2, post2 = apply_fft_frequency_noise(dummy_img, image_id="img_100", band="high", severity=3)
    corr3, pre3, post3 = apply_fft_frequency_noise(dummy_img, image_id="img_200", band="high", severity=3)

    assert np.array_equal(corr1, corr2), "Same image ID must produce identical corrupted array"
    assert not np.array_equal(corr1, corr3), "Different image IDs must produce different corrupted array"
