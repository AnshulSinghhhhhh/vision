"""Paired per-image Difference-in-Differences (DiD) on verified K8 ImageNet-C corruptions.

Evaluates:
- defocus_blur severity 3 (real ImageNet-C disk defocus at 448 vs downsampled to 224)
- gaussian_noise severity 3
Across three models:
- DeiT-B (arm: standard)
- EfficientNet-B3 (arm: standard)
- FlexiViT-B (arm: F-p)

Outputs:
- analysis/out/table_defocus_did_k8.csv
- analysis/out/table_defocus_did_k8.tex
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from analysis.common import get_repo_root, get_out_dir
from knobs.stats import paired_did_ztest


MODELS = [
    ("deit_base", "standard", "DeiT-B/16"),
    ("efficientnet_b3", "standard", "EfficientNet-B3"),
    ("flexivit_base", "F-p", "FlexiViT-B (F-p)"),
]

CONDITIONS = [
    ("defocus_blur", 3, "Defocus Blur (s3)"),
    ("gaussian_noise", 3, "Gaussian Noise (s3)"),
]


def compute_k8_defocus_did(out_dir: str = None) -> Tuple[pd.DataFrame, str]:
    """Computes paired DiD for defocus and noise on K8 dataset and exports CSV and LaTeX."""
    if out_dir is None:
        out_dir = get_out_dir()

    repo_root = get_repo_root()
    k8_file = os.path.join(repo_root, "results", "raw", "k8-m7-v2", "shard_m7_v2.parquet")
    if not os.path.exists(k8_file):
        raise FileNotFoundError(f"K8 shard not found at {k8_file}")

    df_k8 = pd.read_parquet(k8_file)

    rows = []
    for model_key, arm_key, model_display in MODELS:
        sub = df_k8[(df_k8["model"] == model_key) & (df_k8["arm"] == arm_key)]

        # Clean 448 and 224
        clean_448 = sub[(sub["condition"] == "clean") & (sub["suite_condition"] == "A_orig448")].set_index("image_id")["correct"]
        clean_224 = sub[(sub["condition"] == "clean") & (sub["suite_condition"] == "C_down224")].set_index("image_id")["correct"]

        for cond_key, sev, cond_display in CONDITIONS:
            deg_448 = sub[(sub["condition"] == cond_key) & (sub["severity"] == sev) & (sub["suite_condition"] == "A_orig448")].set_index("image_id")["correct"]
            deg_224 = sub[(sub["condition"] == cond_key) & (sub["severity"] == sev) & (sub["suite_condition"] == "C_down224")].set_index("image_id")["correct"]

            # Intersect image IDs
            common_ids = clean_448.index.intersection(clean_224.index).intersection(deg_448.index).intersection(deg_224.index)
            c448_vals = clean_448.loc[common_ids].values.astype(int)
            c224_vals = clean_224.loc[common_ids].values.astype(int)
            d448_vals = deg_448.loc[common_ids].values.astype(int)
            d224_vals = deg_224.loc[common_ids].values.astype(int)

            z_res = paired_did_ztest(c448_vals, c224_vals, d448_vals, d224_vals)
            hw = 1.95996 * z_res.se_pp

            acc_c224 = float(c224_vals.mean() * 100.0)
            acc_c448 = float(c448_vals.mean() * 100.0)
            acc_d224 = float(d224_vals.mean() * 100.0)
            acc_d448 = float(d448_vals.mean() * 100.0)

            rows.append({
                "model": model_key,
                "arm": arm_key,
                "model_display": model_display,
                "condition": cond_key,
                "severity": sev,
                "condition_display": cond_display,
                "N": len(common_ids),
                "clean_224": round(acc_c224, 2),
                "clean_448": round(acc_c448, 2),
                "delta_clean": round(acc_c448 - acc_c224, 2),
                "deg_224": round(acc_d224, 2),
                "deg_448": round(acc_d448, 2),
                "delta_deg": round(acc_d448 - acc_d224, 2),
                "did_pp": round(z_res.did_pp, 2),
                "se_pp": round(z_res.se_pp, 2),
                "ci_half_width": round(hw, 2),
                "ci_95_low": round(z_res.ci_lo, 2),
                "ci_95_high": round(z_res.ci_hi, 2),
                "z": round(z_res.z, 2),
                "p_value": z_res.p_two_sided,
            })

    res_df = pd.DataFrame(rows)

    # Save CSV
    out_csv = os.path.join(out_dir, "table_defocus_did_k8.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"[K8 Defocus DiD] Wrote CSV to {out_csv}")

    # Generate LaTeX table
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired Difference-in-Differences on Verified K8 ImageNet-C Corruptions ($N=2{,}000$). DiD is $(\Delta_{\text{deg}} - \Delta_{\text{clean}})$, with analytic standard errors and 95\% CIs.}",
        r"\label{tab:defocus_did_k8}",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Model & Condition & Clean 224 & Clean 448 & Deg 224 & Deg 448 & DiD (pp) & 95\% CI \\",
        r"\midrule",
    ]

    for _, r in res_df.iterrows():
        p_str = f"{r['did_pp']:+.2f}"
        ci_str = f"$[{r['ci_95_low']:+.2f}, {r['ci_95_high']:+.2f}]$"
        tex_lines.append(
            f"{r['model_display']} & {r['condition_display']} & {r['clean_224']:.1f} & {r['clean_448']:.1f} & {r['deg_224']:.1f} & {r['deg_448']:.1f} & {p_str} & {ci_str} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(tex_lines)
    out_tex = os.path.join(out_dir, "table_defocus_did_k8.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex_str)
    print(f"[K8 Defocus DiD] Wrote TeX to {out_tex}")

    return res_df, tex_str


if __name__ == "__main__":
    compute_k8_defocus_did()
