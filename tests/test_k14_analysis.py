"""Unit test for K14 analysis on synthetic shards."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from analysis.k14_matched_noise import compute_k14_table


def test_k14_analysis_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        synth_data = [
            {"image_id": f"img_{i}", "model": "deit_base", "arm": "standard",
             "condition": "gaussian_noise", "severity": 3, "suite_condition": "A_orig448",
             "label": 1, "pred": 1 if i % 2 == 0 else 0, "correct": (i % 2 == 0)}
            for i in range(100)
        ]
        df_synth = pd.DataFrame(synth_data)
        shard_path = Path(tmpdir) / "synth_k14.parquet"
        df_synth.to_parquet(shard_path, index=False)

        res_df, tex_str = compute_k14_table(parquet_path=str(shard_path), out_dir=tmpdir)
        assert len(res_df) == 1
        assert res_df.iloc[0]["N"] == 100
        assert res_df.iloc[0]["accuracy"] == 50.0
        assert res_df.iloc[0]["ci_low"] < 50.0 < res_df.iloc[0]["ci_high"]
        assert "50.00" in tex_str
