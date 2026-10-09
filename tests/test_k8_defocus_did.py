"""Unit tests for paired K8 Difference-in-Differences analysis."""

import pytest
from analysis.k8_defocus_did import compute_k8_defocus_did
from analysis.common import get_out_dir


def test_k8_defocus_and_noise_did():
    out_dir = get_out_dir()
    df, tex_str = compute_k8_defocus_did(out_dir)

    assert len(df) == 6
    assert (df["N"] == 2000).all()

    # Targets from specification:
    # Defocus DiD: DeiT −6.75 ±1.96, EfficientNet −2.70 ±2.13, FlexiViT(F-p) −0.70 ±1.40
    # Noise s3 DiD: DeiT −3.75, EfficientNet −21.80, FlexiViT −7.50
    # Tolerance: ±0.05 pp on point estimates
    targets = {
        ("deit_base", "standard", "defocus_blur"): (-6.75, 1.96),
        ("efficientnet_b3", "standard", "defocus_blur"): (-2.70, 2.13),
        ("flexivit_base", "F-p", "defocus_blur"): (-0.70, 1.40),
        ("deit_base", "standard", "gaussian_noise"): (-3.75, 1.88),
        ("efficientnet_b3", "standard", "gaussian_noise"): (-21.80, 2.41),
        ("flexivit_base", "F-p", "gaussian_noise"): (-7.50, 1.73),
    }

    for (m, arm, cond), (expected_did, expected_hw) in targets.items():
        row = df[(df["model"] == m) & (df["arm"] == arm) & (df["condition"] == cond)].iloc[0]
        assert abs(row["did_pp"] - expected_did) <= 0.05, (
            f"Point estimate mismatch for {m} ({arm}) under {cond}: got {row['did_pp']}, expected {expected_did}"
        )
        assert abs(row["ci_half_width"] - expected_hw) <= 0.05, (
            f"CI half-width mismatch for {m} ({arm}) under {cond}: got {row['ci_half_width']}, expected {expected_hw}"
        )
