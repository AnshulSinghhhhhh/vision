"""Analysis for K16: BatchNorm (ResNet-50) vs LayerNorm (ConvNeXt-Base) under noise and resolution controls.

Reads shards from results/raw/k16-bn-vs-ln/, computes accuracy and paired differences / DiD
with 95% CIs.

Outputs:
- analysis/out/table_k16_bn_vs_ln.csv
- analysis/out/table_k16_bn_vs_ln.tex
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from analysis.common import get_repo_root, get_out_dir
from knobs.stats import paired_did_ztest


def compute_k16_table(parquet_path: Optional[str] = None, out_dir: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if out_dir is None:
        out_dir = get_out_dir()

    if parquet_path is None:
        raw_candidates = [
            os.path.join(get_repo_root(), "results", "raw", "k16-bn-vs-ln", "shard_bn_vs_ln.parquet"),
            os.path.join(get_repo_root(), "results", "raw", "k16-bn-vs-ln", "shards"),
        ]
        if os.path.exists(raw_candidates[0]):
            df = pd.read_parquet(raw_candidates[0])
        elif os.path.exists(raw_candidates[1]):
            shards = [os.path.join(raw_candidates[1], f) for f in os.listdir(raw_candidates[1]) if f.endswith(".parquet")]
            if shards:
                df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
            else:
                raise FileNotFoundError("No shards found in k16 folder")
        else:
            raise FileNotFoundError("K16 results parquet not found")
    else:
        df = pd.read_parquet(parquet_path)

    # 1. Compute accuracy per (model, norm_type, condition, severity, suite_condition)
    rows = []
    group_cols = ["model", "norm_type", "condition", "severity", "suite_condition"]
    for (m, norm, cond, sev, sc), grp in df.groupby(group_cols):
        n = len(grp)
        p = float(grp["correct"].mean())
        acc = p * 100.0
        se = np.sqrt(p * (1.0 - p) / n) * 100.0 if n > 0 else 0.0
        z_crit = 1.95996
        ci_lo = max(0.0, acc - z_crit * se)
        ci_hi = min(100.0, acc + z_crit * se)
        rows.append({
            "model": m,
            "norm_type": norm,
            "condition": cond,
            "severity": int(sev),
            "suite_condition": sc,
            "accuracy": round(acc, 2),
            "ci_low": round(ci_lo, 2),
            "ci_high": round(ci_hi, 2),
            "N": n,
        })

    res_df = pd.DataFrame(rows)
    res_df = res_df.sort_values(by=["condition", "severity", "suite_condition", "norm_type"]).reset_index(drop=True)
    out_csv = os.path.join(out_dir, "table_k16_bn_vs_ln.csv")
    res_df.to_csv(out_csv, index=False)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{K16: BatchNorm (ResNet-50) vs LayerNorm (ConvNeXt-Base) under noise and resolution controls ($N=5{,}000$).}",
        r"\label{tab:k16_bn_vs_ln}",
        r"\small",
        r"\begin{tabular}{llllccc}",
        r"\toprule",
        r"Model & Norm & Condition & Suite & Accuracy (\%) & 95\% CI & $N$ \\",
        r"\midrule",
    ]
    for _, r in res_df.iterrows():
        tex_lines.append(
            f"{r['model']} & {r['norm_type']} & {r['condition']} s{r['severity']} & {r['suite_condition']} & {r['accuracy']:.2f} & [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {r['N']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_str = "\n".join(tex_lines)
    with open(os.path.join(out_dir, "table_k16_bn_vs_ln.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)

    return res_df, tex_str


if __name__ == "__main__":
    compute_k16_table()
