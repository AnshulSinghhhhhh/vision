"""Unit tests for paired 95% confidence interval tables across K8-K13."""

import os
import pytest
import pandas as pd
from analysis.ci_tables import generate_all_ci_tables
from analysis.common import get_out_dir


def test_ci_tables_generation_and_integrity():
    out_dir = get_out_dir()
    results = generate_all_ci_tables(out_dir)

    expected_tables = [
        "m7_decomposition_ci",
        "table_defocus_did_k8_ci",
        "k9_freqnoise_ci",
        "k10_resize_ablation_ci",
        "table_tome_matched_ci",
        "k13_bn_recal_ci",
    ]

    for tbl_name in expected_tables:
        assert tbl_name in results, f"Missing table {tbl_name}"
        df = results[tbl_name]
        assert len(df) > 0, f"Table {tbl_name} is empty"

        # Verify required CI columns exist
        assert "ci_low" in df.columns, f"Missing ci_low in {tbl_name}"
        assert "ci_high" in df.columns, f"Missing ci_high in {tbl_name}"
        assert "N" in df.columns, f"Missing N in {tbl_name}"

        # Verify CIs bracket the point estimate (or within rounding margin)
        if "accuracy" in df.columns:
            assert (df["ci_low"] <= df["accuracy"] + 0.05).all()
            assert (df["accuracy"] <= df["ci_high"] + 0.05).all()
        elif "acc" in df.columns:
            assert (df["ci_low"] <= df["acc"] + 0.05).all()
            assert (df["acc"] <= df["ci_high"] + 0.05).all()
        elif "did_pp" in df.columns:
            assert (df["ci_low"] <= df["did_pp"] + 0.05).all()
            assert (df["did_pp"] <= df["ci_high"] + 0.05).all()
