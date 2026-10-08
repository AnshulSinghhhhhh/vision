import os
import pytest
from analysis.transitions import compute_transitions_table
from analysis.common import get_out_dir


def test_transitions_table():
    out_dir = get_out_dir()
    df = compute_transitions_table(out_dir)

    expected_cols = [
        "model", "corruption", "severity", "N",
        "clean_helped", "clean_hurt", "clean_inv_corr", "clean_inv_inc",
        "deg_helped", "deg_hurt", "deg_inv_corr", "deg_inv_inc",
        "delta_clean_pp", "delta_deg_pp", "did_transition_pp", "did_ztest_pp", "diff_match"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) >= 12

    # Assert all rows satisfy exact mathematical identity
    assert (df["diff_match"] < 1e-6).all()

    # Assert partitions sum to N
    clean_sums = df["clean_helped"] + df["clean_hurt"] + df["clean_inv_corr"] + df["clean_inv_inc"]
    deg_sums = df["deg_helped"] + df["deg_hurt"] + df["deg_inv_corr"] + df["deg_inv_inc"]
    assert (clean_sums == df["N"]).all()
    assert (deg_sums == df["N"]).all()

    # Check DeiT-B gaussian noise s3 DiD
    row = df[(df["model"] == "deit_base") & (df["corruption"] == "gaussian_noise") & (df["severity"] == 3)].iloc[0]
    assert abs(row["did_transition_pp"] - (-4.084)) < 0.001
