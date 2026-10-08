import pytest
from analysis.common import load_clean_and_corrupted, get_out_dir, get_cache_dir


def test_load_clean_and_corrupted_structure():
    df = load_clean_and_corrupted(corruption="gaussian_noise", severity=3, model="deit_base")
    expected_cols = [
        "image_id", "model", "condition", "severity", "arm",
        "c_224", "c_320", "c_384", "c_448",
        "d_224", "d_320", "d_384", "d_448"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 50000
    assert not df["image_id"].duplicated().any()

    # Verify DeiT-B clean accuracy is 81.54% at 224
    acc_c224 = df["c_224"].mean() * 100.0
    assert abs(acc_c224 - 81.54) < 0.01


def test_directories_exist():
    assert get_out_dir().endswith("out")
    assert get_cache_dir().endswith(".cache")
