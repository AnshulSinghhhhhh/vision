"""Oracle headroom analysis for adaptive resolution selection.

Demonstrates that a resolution-only selector has essentially zero headroom:
- Pool 1: clean + severity-3 pool (clean, gaussian_noise, defocus_blur, jpeg_compression, contrast)
  across N=50,000 images (250,000 evaluations per model).
- Pool 2: severity mixture pool on pilot images (clean, gaussian_noise s1/s3/s5, defocus_blur s1/s3/s5)
  across N=5,000 images (35,000 evaluations per model).

Outputs:
- analysis/out/table_oracle_headroom.csv
- analysis/out/table_oracle_headroom.tex
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from analysis.common import get_repo_root, get_out_dir, load_clean_and_corrupted


def compute_oracle_headroom(out_dir: str = None) -> Tuple[pd.DataFrame, str]:
    """Computes oracle headroom metrics across models and evaluation pools."""
    if out_dir is None:
        out_dir = get_out_dir()

    repo_root = get_repo_root()
    models = [
        ("deit_base", "DeiT-B/16"),
        ("efficientnet_b3", "EfficientNet-B3"),
    ]
    corrs_pool1 = ["gaussian_noise", "defocus_blur", "jpeg_compression", "contrast"]

    rows = []

    # 1. Pool 1: clean + s3 pool (all four corruptions, N=50k images)
    for model_key, model_display in models:
        dfs = []
        df_clean = load_clean_and_corrupted(corruption="gaussian_noise", severity=3, model=model_key)[
            ["image_id", "c_224", "c_320", "c_384", "c_448"]
        ].copy()
        df_clean["condition"] = "clean"
        df_clean = df_clean.rename(columns={"c_224": 224, "c_320": 320, "c_384": 384, "c_448": 448})
        dfs.append(df_clean)

        for c in corrs_pool1:
            df_c = load_clean_and_corrupted(corruption=c, severity=3, model=model_key)[
                ["image_id", "d_224", "d_320", "d_384", "d_448"]
            ].copy()
            df_c["condition"] = c
            df_c = df_c.rename(columns={"d_224": 224, "d_320": 320, "d_384": 384, "d_448": 448})
            dfs.append(df_c)

        pool1 = pd.concat(dfs, ignore_index=True)

        fixed_accs = {r: float(pool1[r].mean() * 100.0) for r in [224, 320, 384, 448]}
        best_fixed_r = max(fixed_accs, key=fixed_accs.get)
        best_fixed_acc = fixed_accs[best_fixed_r]

        cond_accs = []
        for cond, grp in pool1.groupby("condition"):
            c_accs = {r: float(grp[r].mean() * 100.0) for r in [224, 320, 384, 448]}
            best_r = max(c_accs, key=c_accs.get)
            cond_accs.append(c_accs[best_r])
        cond_oracle_acc = float(np.mean(cond_accs))

        per_image_oracle = float(pool1[[224, 320, 384, 448]].max(axis=1).mean() * 100.0)

        rows.append({
            "pool": "clean_plus_s3_50k",
            "pool_display": "Clean + Sev-3 (50k)",
            "model": model_key,
            "model_display": model_display,
            "N_evaluations": len(pool1),
            "best_fixed_res": best_fixed_r,
            "best_fixed_acc": round(best_fixed_acc, 2),
            "condition_oracle_acc": round(cond_oracle_acc, 2),
            "condition_oracle_gain_pp": round(cond_oracle_acc - best_fixed_acc, 2),
            "per_image_oracle_acc": round(per_image_oracle, 2),
            "per_image_oracle_gain_pp": round(per_image_oracle - best_fixed_acc, 2),
            "per_image_achievable": "Not achievable (best of 4 passes)",
        })

    # 2. Pool 2: Severity mixture pool on pilot images (clean, noise s1/s3/s5, defocus s1/s3/s5)
    pilot_path = os.path.join(repo_root, "results", "raw", "k2-pilot", "k2_pilot_results.parquet")
    if os.path.exists(pilot_path):
        df_pilot = pd.read_parquet(pilot_path)
        for model_key, model_display in models:
            sub = df_pilot[(df_pilot["model"] == model_key) & (df_pilot["arm"] == "standard")].copy()
            cond_mask = (
                (sub["condition"] == "clean")
                | ((sub["condition"] == "gaussian_noise") & (sub["severity"].isin([1, 3, 5])))
                | ((sub["condition"] == "defocus_blur") & (sub["severity"].isin([1, 3, 5])))
            )
            sub = sub[cond_mask]

            piv = sub.pivot(index=["image_id", "condition", "severity"], columns="resolution", values="correct").reset_index()

            fixed_accs = {r: float(piv[r].mean() * 100.0) for r in [224, 320, 384, 448]}
            best_fixed_r = max(fixed_accs, key=fixed_accs.get)
            best_fixed_acc = fixed_accs[best_fixed_r]

            cond_accs = []
            for (c, s), grp in piv.groupby(["condition", "severity"]):
                c_accs = {r: float(grp[r].mean() * 100.0) for r in [224, 320, 384, 448]}
                best_r = max(c_accs, key=c_accs.get)
                cond_accs.append(c_accs[best_r])
            cond_oracle_acc = float(np.mean(cond_accs))

            per_image_oracle = float(piv[[224, 320, 384, 448]].max(axis=1).mean() * 100.0)

            rows.append({
                "pool": "severity_mixture_pilot",
                "pool_display": "Severity Mixture (Pilot)",
                "model": model_key,
                "model_display": model_display,
                "N_evaluations": len(piv),
                "best_fixed_res": best_fixed_r,
                "best_fixed_acc": round(best_fixed_acc, 2),
                "condition_oracle_acc": round(cond_oracle_acc, 2),
                "condition_oracle_gain_pp": round(cond_oracle_acc - best_fixed_acc, 2),
                "per_image_oracle_acc": round(per_image_oracle, 2),
                "per_image_oracle_gain_pp": round(per_image_oracle - best_fixed_acc, 2),
                "per_image_achievable": "Not achievable (best of 4 passes)",
            })

    res_df = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, "table_oracle_headroom.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"[Oracle Headroom] Wrote CSV to {out_csv}")

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Resolution Selector Oracle Headroom. Condition-level oracle chooses the optimal fixed resolution per condition/severity. Per-image oracle is the theoretical maximum over all 4 forward passes (strictly not achievable with a pre-classification selector).}",
        r"\label{tab:oracle_headroom}",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Pool & Model & Best Fixed Res & Best Fixed Acc (\%) & Cond. Oracle Acc (\%) & Gain (pp) & Per-Image Oracle* (\%) & Gain* (pp) \\",
        r"\midrule",
    ]

    for _, r in res_df.iterrows():
        gain_cond = f"{r['condition_oracle_gain_pp']:+.2f}"
        gain_img = f"{r['per_image_oracle_gain_pp']:+.2f}"
        tex_lines.append(
            f"{r['pool_display']} & {r['model_display']} & {r['best_fixed_res']} & {r['best_fixed_acc']:.2f} & {r['condition_oracle_acc']:.2f} & {gain_cond} & {r['per_image_oracle_acc']:.2f}* & {gain_img}* \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\multicolumn{8}{l}{\footnotesize *Strictly not achievable; represents best of 4 forward passes.}",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(tex_lines)
    out_tex = os.path.join(out_dir, "table_oracle_headroom.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex_str)
    print(f"[Oracle Headroom] Wrote TeX to {out_tex}")

    return res_df, tex_str


if __name__ == "__main__":
    compute_oracle_headroom()
