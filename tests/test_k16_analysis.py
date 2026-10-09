"""Unit test for K16 analysis on synthetic shards."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from analysis.k16_bn_vs_ln import compute_k16_table


def test_k16_analysis_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        synth_data = []
        for i in range(100):
            synth_data.append({
                "image_id": f"img_{i}",
                "model": "resnet50",
                "norm_type": "BatchNorm",
                "condition": "gaussian_noise",
                "severity": 3,
                "suite_condition": "A_orig448",
                "label": 1,
                "pred": 1 if i % 2 == 0 else 0,
                "correct": (i % 2 == 0),
            })
            synth_data.append({
                "image_id": f"img_{i}",
                "model": "convnext_base",
                "norm_type": "LayerNorm",
                "condition": "gaussian_noise",
                "severity": 3,
                "suite_condition": "A_orig448",
                "label": 1,
                "pred": 1 if i % 4 != 0 else 0,
                "correct": (i % 4 != 0),
            })
        df_synth = pd.DataFrame(synth_data)
        shard_path = Path(tmpdir) / "synth_k16.parquet"
        df_synth.to_parquet(shard_path, index=False)

        res_df, tex_str = compute_k16_table(parquet_path=str(shard_path), out_dir=tmpdir)
        assert len(res_df) == 2
        
        bn_row = res_df[res_df["norm_type"] == "BatchNorm"].iloc[0]
        assert bn_row["N"] == 100
        assert bn_row["accuracy"] == 50.0
        assert bn_row["ci_low"] < 50.0 < bn_row["ci_high"]

        ln_row = res_df[res_df["norm_type"] == "LayerNorm"].iloc[0]
        assert ln_row["N"] == 100
        assert ln_row["accuracy"] == 75.0
        assert ln_row["ci_low"] < 75.0 < ln_row["ci_high"]

        assert "50.00" in tex_str
        assert "75.00" in tex_str
        assert "BatchNorm" in tex_str
        assert "LayerNorm" in tex_str
