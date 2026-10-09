"""Unit tests for adaptive resolution selector oracle headroom analysis."""

import pytest
from analysis.oracle_headroom import compute_oracle_headroom
from analysis.common import get_out_dir


def test_oracle_headroom_targets():
    out_dir = get_out_dir()
    df, tex_str = compute_oracle_headroom(out_dir)

    assert len(df) == 4

    # 1. Clean + s3 pool (50k images x 5 conditions = 250k evaluations per model)
    # DeiT: best fixed (224) 78.46 vs condition-oracle 78.50
    # EfficientNet: best fixed (384) 77.86 vs condition-oracle 78.04
    pool1_deit = df[(df["pool"] == "clean_plus_s3_50k") & (df["model"] == "deit_base")].iloc[0]
    assert pool1_deit["best_fixed_res"] == 224
    assert abs(pool1_deit["best_fixed_acc"] - 78.46) <= 0.05
    assert abs(pool1_deit["condition_oracle_acc"] - 78.50) <= 0.05
    assert pool1_deit["N_evaluations"] == 250000

    pool1_eff = df[(df["pool"] == "clean_plus_s3_50k") & (df["model"] == "efficientnet_b3")].iloc[0]
    assert pool1_eff["best_fixed_res"] == 384
    assert abs(pool1_eff["best_fixed_acc"] - 77.86) <= 0.05
    assert abs(pool1_eff["condition_oracle_acc"] - 78.04) <= 0.05
    assert pool1_eff["N_evaluations"] == 250000

    # 2. Severity mixtures pool on pilot images
    # DeiT +0.01 pp, EfficientNet +0.46 pp
    pool2_deit = df[(df["pool"] == "severity_mixture_pilot") & (df["model"] == "deit_base")].iloc[0]
    assert abs(pool2_deit["condition_oracle_gain_pp"] - 0.01) <= 0.02
    assert pool2_deit["N_evaluations"] == 35000

    pool2_eff = df[(df["pool"] == "severity_mixture_pilot") & (df["model"] == "efficientnet_b3")].iloc[0]
    assert abs(pool2_eff["condition_oracle_gain_pp"] - 0.46) <= 0.02
    assert pool2_eff["N_evaluations"] == 35000

    # 3. Verify per-image oracle is clearly labeled as not achievable
    for _, row in df.iterrows():
        assert "not achievable" in row["per_image_achievable"].lower()
        # Per-image oracle gives theoretical ~ +5 pp
        assert 4.0 <= row["per_image_oracle_gain_pp"] <= 7.0
