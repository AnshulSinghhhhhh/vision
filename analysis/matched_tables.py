"""Matched tables for controls (Tables 2, 3, 4, 5).

Critical check: asserts that comparing F-p to F-t or standard to ToMe evaluates
on the exact same 5,000 images per condition.
Outputs:
- analysis/out/table2.csv (FlexiViT F-p vs F-t)
- analysis/out/table3.csv (DeiT-384 native vs interpolated)
- analysis/out/table4.csv (ToMe token merging controls)
- analysis/out/table5.csv (BatchNorm recalibration controls)
"""

import os
import pandas as pd
from typing import Dict, List, Tuple
from analysis.common import load_all_raw_data, get_out_dir


PRIMARY_CONDITIONS = [
    ("clean", 0),
    ("gaussian_noise", 3),
    ("defocus_blur", 3),
    ("jpeg_compression", 3),
    ("contrast", 3),
]


def generate_table2_flexivit(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Table 2: FlexiViT F-p (patch 16, variable tokens) vs F-t (256 tokens, variable patch)."""
    rows = []
    resolutions = [224, 320, 384, 448]

    for cond, sev in PRIMARY_CONDITIONS:
        for r in resolutions:
            sub_fp = df[(df["model"] == "flexivit_base") & (df["arm"] == "F-p") &
                        (df["condition"] == cond) & (df["severity"] == sev) & (df["resolution"] == r)]
            sub_ft = df[(df["model"] == "flexivit_base") & (df["arm"] == "F-t") &
                        (df["condition"] == cond) & (df["severity"] == sev) & (df["resolution"] == r)]

            fp_s = sub_fp.set_index("image_id")["correct"].astype(float)
            ft_s = sub_ft.set_index("image_id")["correct"].astype(float)

            common_ids = fp_s.index.intersection(ft_s.index)
            if len(fp_s) != len(ft_s) or len(common_ids) != len(fp_s):
                print(f"[Warning Table 2] Mismatch in {cond} s{sev} @ {r}: F-p={len(fp_s)}, F-t={len(ft_s)}, common={len(common_ids)}")

            # Critical assertion: must evaluate on matched 5,000 images
            assert len(common_ids) == 5000, f"Expected exactly 5,000 matched images for {cond} s{sev} @ {r}, got {len(common_ids)}"
            assert fp_s.loc[common_ids].index.equals(ft_s.loc[common_ids].index)

            acc_fp = float(fp_s.loc[common_ids].mean() * 100.0)
            acc_ft = float(ft_s.loc[common_ids].mean() * 100.0)
            tokens_fp = (r // 16) ** 2

            rows.append({
                "Condition": cond,
                "Severity": sev,
                "Resolution": r,
                "Tokens_Fp": tokens_fp,
                "Acc_Fp": acc_fp,
                "Tokens_Ft": 256,
                "Acc_Ft": acc_ft,
                "Diff_Fp_Ft": acc_fp - acc_ft,
                "N": len(common_ids),
            })

    df_t2 = pd.DataFrame(rows)
    p = os.path.join(out_dir, "table2.csv")
    df_t2.to_csv(p, index=False)
    print(f"Generated Table 2 (FlexiViT F-p vs F-t) -> {p}")
    return df_t2


def generate_table3_deit384(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Table 3: DeiT-384 native checkpoint vs standard DeiT-B interpolated."""
    rows = []

    for cond, sev in PRIMARY_CONDITIONS:
        sub_std = df[(df["model"] == "deit_base") & (df["arm"] == "standard") &
                     (df["condition"] == cond) & (df["severity"] == sev)]
        sub_384 = df[(df["model"] == "deit_base_384") & (df["arm"] == "native_384") &
                     (df["condition"] == cond) & (df["severity"] == sev)]

        std_224 = sub_std[sub_std["resolution"] == 224].set_index("image_id")["correct"].astype(float)
        std_384 = sub_std[sub_std["resolution"] == 384].set_index("image_id")["correct"].astype(float)
        std_448 = sub_std[sub_std["resolution"] == 448].set_index("image_id")["correct"].astype(float)
        nat_384 = sub_384[sub_384["resolution"] == 384].set_index("image_id")["correct"].astype(float)

        common_ids = std_224.index.intersection(std_384.index).intersection(std_448.index).intersection(nat_384.index)
        if len(common_ids) != len(nat_384):
            print(f"[Warning Table 3] Dropped images in {cond} s{sev}: native={len(nat_384)}, common={len(common_ids)}")

        # Critical assertion
        assert len(common_ids) == 5000, f"Expected exactly 5,000 matched images for {cond} s{sev}, got {len(common_ids)}"

        acc_224 = float(std_224.loc[common_ids].mean() * 100.0)
        acc_384_int = float(std_384.loc[common_ids].mean() * 100.0)
        acc_448 = float(std_448.loc[common_ids].mean() * 100.0)
        acc_384_nat = float(nat_384.loc[common_ids].mean() * 100.0)

        rows.append({
            "Condition": cond,
            "Severity": sev,
            "Acc_224_std": acc_224,
            "Acc_384_interp": acc_384_int,
            "Acc_384_native": acc_384_nat,
            "Acc_448_std": acc_448,
            "Diff_native_minus_interp_384": acc_384_nat - acc_384_int,
            "N": len(common_ids),
        })

    df_t3 = pd.DataFrame(rows)
    p = os.path.join(out_dir, "table3.csv")
    df_t3.to_csv(p, index=False)
    print(f"Generated Table 3 (DeiT-384 native) -> {p}")
    return df_t3


def generate_table4_tome(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Table 4: ToMe Token Merging controls (r=4, r=8) vs standard DeiT-B."""
    rows = []

    for cond, sev in PRIMARY_CONDITIONS:
        for r in [224, 448]:
            sub_std = df[(df["model"] == "deit_base") & (df["arm"] == "standard") &
                         (df["condition"] == cond) & (df["severity"] == sev) & (df["resolution"] == r)]
            sub_r4 = df[(df["model"] == "deit_base_tome_r4") & (df["arm"] == "tome_r4") &
                        (df["condition"] == cond) & (df["severity"] == sev) & (df["resolution"] == r)]
            sub_r8 = df[(df["model"] == "deit_base_tome_r8") & (df["arm"] == "tome_r8") &
                        (df["condition"] == cond) & (df["severity"] == sev) & (df["resolution"] == r)]

            s_std = sub_std.set_index("image_id")["correct"].astype(float)
            s_r4 = sub_r4.set_index("image_id")["correct"].astype(float)
            s_r8 = sub_r8.set_index("image_id")["correct"].astype(float)

            common_ids = s_std.index.intersection(s_r4.index).intersection(s_r8.index)
            if len(common_ids) != 5000:
                print(f"[Warning Table 4] Dropped images in {cond} s{sev} @ {r}: std={len(s_std)}, r4={len(s_r4)}, common={len(common_ids)}")

            # Critical assertion
            assert len(common_ids) == 5000, f"Expected exactly 5,000 matched images for {cond} s{sev} @ {r}, got {len(common_ids)}"

            acc_std = float(s_std.loc[common_ids].mean() * 100.0)
            acc_r4 = float(s_r4.loc[common_ids].mean() * 100.0)
            acc_r8 = float(s_r8.loc[common_ids].mean() * 100.0)

            rows.append({
                "Condition": cond,
                "Severity": sev,
                "Resolution": r,
                "Acc_Standard": acc_std,
                "Acc_ToMe_r4": acc_r4,
                "Acc_ToMe_r8": acc_r8,
                "Diff_r4_minus_std": acc_r4 - acc_std,
                "Diff_r8_minus_std": acc_r8 - acc_std,
                "N": len(common_ids),
            })

    df_t4 = pd.DataFrame(rows)
    p = os.path.join(out_dir, "table4.csv")
    df_t4.to_csv(p, index=False)
    print(f"Generated Table 4 (ToMe controls) -> {p}")
    return df_t4


def generate_table5_bn_recal(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Table 5: BatchNorm Recalibration controls for EfficientNet-B3."""
    rows = []
    bn_conditions = [
        ("gaussian_noise", 1),
        ("gaussian_noise", 5),
        ("defocus_blur", 1),
        ("defocus_blur", 5),
    ]

    for cond, sev in bn_conditions:
        sub_std = df[(df["model"] == "efficientnet_b3") & (df["arm"] == "standard") &
                     (df["condition"] == cond) & (df["severity"] == sev)]
        sub_ebn = df[(df["model"] == "efficientnet_b3_ebn") & (df["arm"] == "bn_recal") &
                     (df["condition"] == cond) & (df["severity"] == sev)]

        std_224 = sub_std[sub_std["resolution"] == 224].set_index("image_id")["correct"].astype(float)
        std_448 = sub_std[sub_std["resolution"] == 448].set_index("image_id")["correct"].astype(float)
        ebn_224 = sub_ebn[sub_ebn["resolution"] == 224].set_index("image_id")["correct"].astype(float)
        ebn_448 = sub_ebn[sub_ebn["resolution"] == 448].set_index("image_id")["correct"].astype(float)

        common_ids = std_224.index.intersection(std_448.index).intersection(ebn_224.index).intersection(ebn_448.index)
        if len(common_ids) < len(ebn_224):
            dropped = len(ebn_224) - len(common_ids)
            print(f"[Warning Table 5] Dropped {dropped} images in {cond} s{sev} during inner join (retained {len(common_ids)})")

        assert len(common_ids) > 0, f"No common images found for BN recalibration on {cond} s{sev}"

        acc_std_224 = float(std_224.loc[common_ids].mean() * 100.0)
        acc_std_448 = float(std_448.loc[common_ids].mean() * 100.0)
        delta_std = acc_std_448 - acc_std_224

        acc_ebn_224 = float(ebn_224.loc[common_ids].mean() * 100.0)
        acc_ebn_448 = float(ebn_448.loc[common_ids].mean() * 100.0)
        delta_ebn = acc_ebn_448 - acc_ebn_224

        rows.append({
            "Condition": cond,
            "Severity": sev,
            "Acc_224_std": acc_std_224,
            "Acc_448_std": acc_std_448,
            "Delta_std": delta_std,
            "Acc_224_ebn": acc_ebn_224,
            "Acc_448_ebn": acc_ebn_448,
            "Delta_ebn": delta_ebn,
            "Rescue_Diff": delta_ebn - delta_std,
            "N": len(common_ids),
        })

    df_t5 = pd.DataFrame(rows)
    p = os.path.join(out_dir, "table5.csv")
    df_t5.to_csv(p, index=False)
    print(f"Generated Table 5 (BN Recalibration) -> {p}")
    return df_t5


def generate_all_matched_tables(out_dir: str = None):
    if out_dir is None:
        out_dir = get_out_dir()

    df = load_all_raw_data().copy()
    if "resolution" in df.columns:
        df = df.dropna(subset=["resolution"])
        df["resolution"] = df["resolution"].astype(int)

    t2 = generate_table2_flexivit(df, out_dir)
    t3 = generate_table3_deit384(df, out_dir)
    t4 = generate_table4_tome(df, out_dir)
    t5 = generate_table5_bn_recal(df, out_dir)
    return t2, t3, t4, t5


if __name__ == "__main__":
    generate_all_matched_tables()
