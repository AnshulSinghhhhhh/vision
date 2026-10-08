import os
import pytest
from analysis.m7 import load_m7_results
from analysis.common import get_out_dir


def test_m7_decomposition():
    out_dir = get_out_dir()
    df = load_m7_results(out_dir)

    expected_cols = ["condition", "filter_suite", "deit_acc", "effnet_acc", "is_mock", "data_source", "N"]
    assert list(df.columns) == expected_cols
    assert len(df) > 0

    # Verify measured conditions exist
    measured = df[df["is_mock"] == False]
    assert set(measured["filter_suite"]) == {"A_orig448", "B_filtered448", "C_down224", "D_up448"}

    # Under gaussian_noise, filter B rescues EfficientNet substantially (> 20 pp)
    gn_a = df[(df["condition"] == "gaussian_noise") & (df["filter_suite"] == "A_orig448")].iloc[0]
    gn_b = df[(df["condition"] == "gaussian_noise") & (df["filter_suite"] == "B_filtered448")].iloc[0]
    assert gn_b["effnet_acc"] - gn_a["effnet_acc"] > 20.0

    # Verify mock filters are clearly labeled
    mocked = df[df["is_mock"] == True]
    assert len(mocked) > 0
    assert (mocked["data_source"].str.contains("pending K8")).all()
