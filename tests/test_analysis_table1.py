import pytest
import pandas as pd
from analysis.table1_did import generate_table1


REFERENCE_TABLE1 = {
    ("deit_base", "gaussian_noise"): {"DiD_pp": -4.084, "Clean_224": 81.54, "Clean_448": 79.94, "Deg_224": 78.02, "Deg_448": 72.33},
    ("deit_base", "defocus_blur"): {"DiD_pp": -5.556, "Clean_224": 81.54, "Clean_448": 79.94, "Deg_224": 75.44, "Deg_448": 68.28},
    ("deit_base", "jpeg_compression"): {"DiD_pp": -5.406, "Clean_224": 81.54, "Clean_448": 79.94, "Deg_224": 77.38, "Deg_448": 70.37},
    ("deit_base", "contrast"): {"DiD_pp": -1.274, "Clean_224": 81.54, "Clean_448": 79.94, "Deg_224": 79.91, "Deg_448": 77.04},
    ("efficientnet_b3", "gaussian_noise"): {"DiD_pp": -21.916, "Clean_224": 78.29, "Clean_448": 82.71, "Deg_224": 71.37, "Deg_448": 53.87},
    ("efficientnet_b3", "defocus_blur"): {"DiD_pp": -1.938, "Clean_224": 78.29, "Clean_448": 82.71, "Deg_224": 69.47, "Deg_448": 71.95},
    ("efficientnet_b3", "jpeg_compression"): {"DiD_pp": -0.600, "Clean_224": 78.29, "Clean_448": 82.71, "Deg_224": 74.32, "Deg_448": 78.13},
    ("efficientnet_b3", "contrast"): {"DiD_pp": 1.162, "Clean_224": 78.29, "Clean_448": 82.71, "Deg_224": 75.86, "Deg_448": 81.44},
    ("flexivit_base", "gaussian_noise"): {"DiD_pp": -7.208, "Clean_224": 83.55, "Clean_448": 84.27, "Deg_224": 80.31, "Deg_448": 73.82},
    ("flexivit_base", "defocus_blur"): {"DiD_pp": -1.262, "Clean_224": 83.55, "Clean_448": 84.27, "Deg_224": 77.58, "Deg_448": 77.04},
    ("flexivit_base", "jpeg_compression"): {"DiD_pp": -2.452, "Clean_224": 83.55, "Clean_448": 84.27, "Deg_224": 80.46, "Deg_448": 78.72},
    ("flexivit_base", "contrast"): {"DiD_pp": 0.700, "Clean_224": 83.55, "Clean_448": 84.27, "Deg_224": 83.20, "Deg_448": 84.62},
}


def test_table1_matches_reference():
    df, latex_str = generate_table1()
    assert len(df) == 12
    assert "begin{table*}" in latex_str
    
    for (model, corr), ref in REFERENCE_TABLE1.items():
        sub = df[(df["Model"] == model) & (df["Corruption"] == corr)]
        assert len(sub) == 1, f"Missing row for {model}, {corr}"
        row = sub.iloc[0]
        
        for col, expected_val in ref.items():
            actual_val = row[col]
            diff = abs(actual_val - expected_val)
            assert diff <= 0.01, f"Mismatch in {model} {corr} {col}: actual={actual_val}, expected={expected_val}, diff={diff}"
