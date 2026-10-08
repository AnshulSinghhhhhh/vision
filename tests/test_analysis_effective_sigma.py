import os
import pytest
from analysis.effective_sigma import compute_effective_sigma_table
from analysis.common import get_out_dir


def test_effective_sigma_analysis():
    out_dir = get_out_dir()
    df = compute_effective_sigma_table(out_dir)

    expected_cols = [
        "model", "R", "sigma_injected", "noise_gain",
        "sigma_eff", "measured_acc", "matched_sigma_acc", "excess_loss_pp"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 12

    # Noise gains
    gains = df[df["model"] == "efficientnet_b3"].set_index("R")["noise_gain"]
    assert abs(gains[224] - 0.3125) < 0.005
    assert abs(gains[320] - 0.4807) < 0.005
    assert abs(gains[384] - 0.5658) < 0.005
    assert abs(gains[448] - 1.0000) < 0.005

    # EfficientNet at 448: excess loss is small compared to the full 28.8 pp collapse
    eff448 = df[(df["model"] == "efficientnet_b3") & (df["R"] == 448)].iloc[0]
    assert abs(eff448["excess_loss_pp"]) < 6.0

    # DeiT-B at 448: positive excess retention
    deit448 = df[(df["model"] == "deit_base") & (df["R"] == 448)].iloc[0]
    assert deit448["excess_loss_pp"] > 0.0
