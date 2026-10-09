"""Adds paired 95% confidence intervals to tables derived from K8, K9, K10, K12, and K13.

Methods:
- Proportions: Analytic Wilson/Wald 95% confidence intervals: p +/- 1.95996 * sqrt(p*(1-p)/N).
- Differences / DiD: Paired bootstrap with 2,000 resamples.

Outputs:
- analysis/out/m7_decomposition_ci.csv (K8)
- analysis/out/table_defocus_did_k8_ci.csv (K8)
- analysis/out/table_k9_freqnoise_ci.csv (K9)
- analysis/out/table_k10_resize_ci.csv (K10)
- analysis/out/table_tome_matched_ci.csv (K12)
- analysis/out/table_k13_bn_recal_ci.csv (K13)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from analysis.common import get_repo_root, get_out_dir


def proportion_ci(correct_arr: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, int]:
    """Computes percentage accuracy and analytic 95% CI."""
    n = len(correct_arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    p = float(np.mean(correct_arr))
    acc = p * 100.0
    se = np.sqrt(p * (1.0 - p) / n) * 100.0
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = max(0.0, acc - z_crit * se)
    ci_high = min(100.0, acc + z_crit * se)
    return round(acc, 2), round(ci_low, 2), round(ci_high, 2), n


def paired_difference_bootstrap_ci(
    diff_arr: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, int]:
    """Computes paired difference and 2,000 resample bootstrap 95% CI in percentage points."""
    n = len(diff_arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    point_est = float(np.mean(diff_arr) * 100.0)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n, size=(n_resamples, n))
    boot_means = np.mean(diff_arr[indices], axis=1) * 100.0
    ci_low = float(np.percentile(boot_means, (alpha / 2.0) * 100.0))
    ci_high = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100.0))
    return round(point_est, 2), round(ci_low, 2), round(ci_high, 2), n


def generate_all_ci_tables(out_dir: str = None) -> Dict[str, pd.DataFrame]:
    """Generates all CI tables for K8, K9, K10, K12, and K13."""
    if out_dir is None:
        out_dir = get_out_dir()

    repo = get_repo_root()
    results = {}

    # 1. K8: M7 Decomposition CI
    k8_m7_file = os.path.join(repo, "results", "raw", "k8-m7-v2", "shard_m7_v2.parquet")
    if os.path.exists(k8_m7_file):
        df_k8 = pd.read_parquet(k8_m7_file)
        rows_m7 = []
        for (m, arm, cond, sev, sc), grp in df_k8.groupby(["model", "arm", "condition", "severity", "suite_condition"]):
            acc, lo, hi, n = proportion_ci(grp["correct"].values)
            rows_m7.append({
                "model": m, "arm": arm, "condition": cond, "severity": sev,
                "suite_condition": sc, "accuracy": acc, "ci_low": lo, "ci_high": hi, "N": n
            })
        df_m7_ci = pd.DataFrame(rows_m7)
        out_m7_ci = os.path.join(out_dir, "m7_decomposition_ci.csv")
        df_m7_ci.to_csv(out_m7_ci, index=False)
        results["m7_decomposition_ci"] = df_m7_ci

        # K8 Defocus DiD CI (with 2,000 paired bootstrap resamples for differences)
        from analysis.k8_defocus_did import compute_k8_defocus_did
        df_did, _ = compute_k8_defocus_did(out_dir)
        # Add paired bootstrap CIs
        boot_los, boot_his = [], []
        for _, r in df_did.iterrows():
            sub = df_k8[(df_k8["model"] == r["model"]) & (df_k8["arm"] == r["arm"])]
            c448 = sub[(sub["condition"] == "clean") & (sub["suite_condition"] == "A_orig448")].set_index("image_id")["correct"]
            c224 = sub[(sub["condition"] == "clean") & (sub["suite_condition"] == "C_down224")].set_index("image_id")["correct"]
            d448 = sub[(sub["condition"] == r["condition"]) & (sub["severity"] == r["severity"]) & (sub["suite_condition"] == "A_orig448")].set_index("image_id")["correct"]
            d224 = sub[(sub["condition"] == r["condition"]) & (sub["severity"] == r["severity"]) & (sub["suite_condition"] == "C_down224")].set_index("image_id")["correct"]
            common = c448.index.intersection(c224.index).intersection(d448.index).intersection(d224.index)
            diff = (d448.loc[common].astype(float) - d224.loc[common].astype(float)) - (c448.loc[common].astype(float) - c224.loc[common].astype(float))
            _, b_lo, b_hi, _ = paired_difference_bootstrap_ci(diff.values, n_resamples=2000)
            boot_los.append(b_lo)
            boot_his.append(b_hi)

        df_did_ci = df_did.copy()
        df_did_ci["ci_low"] = boot_los
        df_did_ci["ci_high"] = boot_his
        out_did_ci = os.path.join(out_dir, "table_defocus_did_k8_ci.csv")
        df_did_ci.to_csv(out_did_ci, index=False)
        results["table_defocus_did_k8_ci"] = df_did_ci

    # 2. K9: Frequency-Controlled Noise CI
    k9_file = os.path.join(repo, "results", "raw", "k9-freqnoise-v2", "shard_freqnoise_v2.parquet")
    if os.path.exists(k9_file):
        df_k9 = pd.read_parquet(k9_file)
        rows_k9 = []
        for (m, cond, sev, band, res), grp in df_k9.groupby(["model", "condition", "severity", "band", "resolution"]):
            acc, lo, hi, n = proportion_ci(grp["correct"].values)
            rows_k9.append({
                "model": m, "condition": cond, "severity": sev,
                "band": band, "resolution": res, "accuracy": acc,
                "ci_low": lo, "ci_high": hi, "N": n
            })
        df_k9_ci = pd.DataFrame(rows_k9)
        out_k9_ci = os.path.join(out_dir, "table_k9_freqnoise_ci.csv")
        df_k9_ci.to_csv(out_k9_ci, index=False)
        df_k9_ci.to_csv(os.path.join(out_dir, "k9_freqnoise_ci.csv"), index=False)
        results["k9_freqnoise_ci"] = df_k9_ci

    # 3. K10: Resize Operator Ablation CI
    k10_file = os.path.join(repo, "results", "raw", "k10-resize-ablation", "shard_resize_ablation.parquet")
    if os.path.exists(k10_file):
        df_k10 = pd.read_parquet(k10_file)
        rows_k10 = []
        for (m, cond, sev, op, res), grp in df_k10.groupby(["model", "condition", "severity", "operator", "resolution"]):
            seff = float(grp["effective_sigma"].iloc[0])
            acc, lo, hi, n = proportion_ci(grp["correct"].values)
            rows_k10.append({
                "model": m, "condition": cond, "severity": sev,
                "operator": op, "resolution": res, "effective_sigma": round(seff, 4),
                "accuracy": acc, "ci_low": lo, "ci_high": hi, "N": n
            })
        df_k10_ci = pd.DataFrame(rows_k10)
        out_k10_ci = os.path.join(out_dir, "table_k10_resize_ci.csv")
        df_k10_ci.to_csv(out_k10_ci, index=False)
        df_k10_ci.to_csv(os.path.join(out_dir, "k10_resize_ablation_ci.csv"), index=False)
        results["k10_resize_ablation_ci"] = df_k10_ci

    # 4. K12: ToMe Matched Table CI
    k12_file = os.path.join(repo, "results", "raw", "k12-tome-matched", "shard_tome_matched.parquet")
    if os.path.exists(k12_file):
        df_k12 = pd.read_parquet(k12_file)
        summary_k12 = df_k12.groupby(["condition", "severity", "config_name", "resolution", "r_tome"]).agg(
            acc=("correct", lambda x: float(x.mean() * 100.0)),
            gflops=("gflops", "first"),
            fps=("throughput_img_per_s", "first"),
            n=("correct", "count")
        ).reset_index()

        ci_lows, ci_highs, ns = [], [], []
        for _, r in summary_k12.iterrows():
            sub = df_k12[(df_k12["condition"] == r["condition"]) &
                         (df_k12["severity"] == r["severity"]) &
                         (df_k12["config_name"] == r["config_name"]) &
                         (df_k12["resolution"] == r["resolution"])]
            _, lo, hi, n = proportion_ci(sub["correct"].values)
            ci_lows.append(lo)
            ci_highs.append(hi)
            ns.append(n)

        df_tome_ci = summary_k12.copy()
        df_tome_ci["ci_low"] = ci_lows
        df_tome_ci["ci_high"] = ci_highs
        df_tome_ci["N"] = ns
        out_tome_ci = os.path.join(out_dir, "table_tome_matched_ci.csv")
        df_tome_ci.to_csv(out_tome_ci, index=False)
        results["table_tome_matched_ci"] = df_tome_ci

    # 5. K13: BatchNorm Recalibration CI
    k13_file = os.path.join(repo, "results", "raw", "k13-g0a-and-bn", "shards", "shard_bn_recal_v2.parquet")
    if os.path.exists(k13_file):
        df_k13 = pd.read_parquet(k13_file)
        rows_k13 = []
        for (m, cond, sev, res, arm), grp in df_k13.groupby(["model", "condition", "severity", "resolution", "arm"]):
            acc, lo, hi, n = proportion_ci(grp["correct"].values)
            rows_k13.append({
                "model": m, "condition": cond, "severity": sev,
                "resolution": res, "arm": arm, "accuracy": acc,
                "ci_low": lo, "ci_high": hi, "N": n
            })
        df_k13_ci = pd.DataFrame(rows_k13)
        out_k13_ci = os.path.join(out_dir, "table_k13_bn_recal_ci.csv")
        df_k13_ci.to_csv(out_k13_ci, index=False)
        df_k13_ci.to_csv(os.path.join(out_dir, "k13_bn_recal_ci.csv"), index=False)
        results["k13_bn_recal_ci"] = df_k13_ci

    print(f"[CI Tables] Successfully generated {len(results)} CI tables in {out_dir}")
    return results


if __name__ == "__main__":
    generate_all_ci_tables()
