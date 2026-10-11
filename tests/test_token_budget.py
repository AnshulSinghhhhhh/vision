"""Unit test for token budget analysis table."""

import os
import pandas as pd
import pytest


def test_table_token_budget_exists_and_valid():
    csv_path = os.path.join("analysis", "out", "table_token_budget.csv")
    assert os.path.exists(csv_path), f"{csv_path} does not exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 64
    expected_cols = {"experiment", "contrast", "condition", "N", "estimate_pp", "ci95_halfwidth_pp"}
    assert expected_cols.issubset(df.columns)

    # Check key empirical benchmarks
    # 1. K15 FlexiViT patch32@448 minus patch16@224 at noise s3 and s5
    sub_s3 = df[(df.experiment == "FlexiViT patch grid (K15)") & 
                (df.contrast == "patch32@448 minus patch16@224 (same 196 tokens)") & 
                (df.condition == "gaussian_noise_s3")]
    assert len(sub_s3) == 1
    assert abs(sub_s3.iloc[0]["estimate_pp"] - (-0.37)) < 0.05

    sub_s5 = df[(df.experiment == "FlexiViT patch grid (K15)") & 
                (df.contrast == "patch32@448 minus patch16@224 (same 196 tokens)") & 
                (df.condition == "gaussian_noise_s5")]
    assert len(sub_s5) == 1
    assert abs(sub_s5.iloc[0]["estimate_pp"] - (-4.67)) < 0.05

    # 2. ToMe matched FLOPs res-320 vs tome-r64@448
    tome_s5 = df[(df.experiment == "Matched FLOPs (K12)") & 
                 (df.contrast == "res_320_base minus tome_r64_448, minus clean gap") & 
                 (df.condition == "gaussian_noise_s5")]
    assert len(tome_s5) == 1
    assert abs(tome_s5.iloc[0]["estimate_pp"] - 19.86) < 0.05
