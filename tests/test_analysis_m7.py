import os
import pytest
from analysis.m7 import load_m7_results
from analysis.common import get_out_dir


def test_m7_decomposition():
    out_dir = get_out_dir()
    df = load_m7_results(out_dir)

    expected_cols = ["model", "arm", "condition", "severity", "suite_condition", "accuracy", "N", "ci_low", "ci_high"]
    assert list(df.columns) == expected_cols
    assert len(df) > 0

    # Verify no mock flags or mock columns exist
    assert "is_mock" not in df.columns
    assert "data_source" not in df.columns

    # Verify all records have N = 2000
    assert (df["N"] == 2000).all()

    # Verify suite conditions exist in measured dataset
    suites = set(df["suite_condition"].unique())
    assert {"A_orig448", "B_filtered448", "C_down224", "D_up448", "E_noise_matched"}.issubset(suites)

    # Under gaussian_noise s5, filter B rescues EfficientNet substantially (> 40 pp)
    eff_s5_a = df[(df["model"] == "efficientnet_b3") & (df["condition"] == "gaussian_noise") & (df["severity"] == 5) & (df["suite_condition"] == "A_orig448")].iloc[0]
    eff_s5_b = df[(df["model"] == "efficientnet_b3") & (df["condition"] == "gaussian_noise") & (df["severity"] == 5) & (df["suite_condition"] == "B_filtered448")].iloc[0]
    assert eff_s5_b["accuracy"] - eff_s5_a["accuracy"] > 40.0

    # Under gaussian_noise s3, filter B rescues EfficientNet substantially (> 15 pp)
    eff_s3_a = df[(df["model"] == "efficientnet_b3") & (df["condition"] == "gaussian_noise") & (df["severity"] == 3) & (df["suite_condition"] == "A_orig448")].iloc[0]
    eff_s3_b = df[(df["model"] == "efficientnet_b3") & (df["condition"] == "gaussian_noise") & (df["severity"] == 3) & (df["suite_condition"] == "B_filtered448")].iloc[0]
    assert eff_s3_b["accuracy"] - eff_s3_a["accuracy"] > 15.0

    # Confidence intervals are valid (ci_low <= accuracy <= ci_high)
    assert (df["ci_low"] <= df["accuracy"] + 1e-4).all()
    assert (df["accuracy"] <= df["ci_high"] + 1e-4).all()
