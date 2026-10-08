import os
import pytest
import pandas as pd
from analysis.matched_tables import generate_all_matched_tables
from analysis.common import get_out_dir


def test_matched_tables_generation():
    out_dir = get_out_dir()
    t2, t3, t4, t5 = generate_all_matched_tables(out_dir)

    # 1. Table 2: FlexiViT F-p vs F-t
    assert os.path.exists(os.path.join(out_dir, "table2.csv"))
    assert len(t2) == 20  # 5 conditions * 4 resolutions
    assert (t2["N"] == 5000).all()
    assert (t2["Acc_Fp"] >= 0).all() and (t2["Acc_Fp"] <= 100).all()
    assert (t2["Acc_Ft"] >= 0).all() and (t2["Acc_Ft"] <= 100).all()

    # 2. Table 3: DeiT-384 native
    assert os.path.exists(os.path.join(out_dir, "table3.csv"))
    assert len(t3) == 5  # 5 conditions
    assert (t3["N"] == 5000).all()
    assert (t3["Acc_384_native"] >= 0).all() and (t3["Acc_384_native"] <= 100).all()

    # 3. Table 4: ToMe controls
    assert os.path.exists(os.path.join(out_dir, "table4.csv"))
    assert len(t4) == 10  # 5 conditions * 2 resolutions
    assert (t4["N"] == 5000).all()
    assert (t4["Acc_Standard"] >= 0).all()

    # 4. Table 5: BN recalibration
    assert os.path.exists(os.path.join(out_dir, "table5.csv"))
    assert len(t5) == 4  # 4 conditions (noise s1/s5, blur s1/s5)
    assert (t5["N"] > 0).all()
