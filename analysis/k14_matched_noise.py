"""Analysis for K14: Matched-noise and defocus verification on N=10,000 images.

Reads shards from results/raw/k14-matched-noise-10k/, computes accuracy and paired DiDs
with 95% CIs across DeiT-B, EfficientNet-B3, and FlexiViT-B (F-p).

Outputs:
- analysis/out/table_k14_matched_noise.csv
- analysis/out/table_k14_matched_noise.tex
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

from analysis.common import get_repo_root, get_out_dir
from knobs.stats import paired_did_ztest


def compute_k14_table(parquet_path: str = None, out_dir: str = None) -> Tuple[pd.DataFrame, str]:
    if out_dir is None:
        out_dir = get_out_dir()

    if parquet_path is None:
        raw_candidates = [
            os.path.join(get_repo_root(), "results", "raw", "k14-matched-noise-10k", "shard_matched_noise_10k.parquet"),
            os.path.join(get_repo_root(), "results", "raw", "k14-matched-noise-10k", "shards"),
        ]
        if os.path.exists(raw_candidates[0]):
            df = pd.read_parquet(raw_candidates[0])
        elif os.path.exists(raw_candidates[1]):
            shards = [os.path.join(raw_candidates[1], f) for f in os.listdir(raw_candidates[1]) if f.endswith(".parquet")]
            if shards:
                df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
            else:
                raise FileNotFoundError("No shards found in k14 folder")
        else:
            raise FileNotFoundError("K14 results parquet not found")
    else:
        df = pd.read_parquet(parquet_path)

    # Compute accuracy per (model, arm, condition, severity, suite_condition)
    rows = []
    group_cols = ["model", "arm", "condition", "severity", "suite_condition"]
    for (m, arm, cond, sev, sc), grp in df.groupby(group_cols):
        n = len(grp)
        p = float(grp["correct"].mean())
        acc = p * 100.0
        se = np.sqrt(p * (1.0 - p) / n) * 100.0 if n > 0 else 0.0
        z_crit = 1.95996
        ci_lo = max(0.0, acc - z_crit * se)
        ci_hi = min(100.0, acc + z_crit * se)
        rows.append({
            "model": m, "arm": arm, "condition": cond, "severity": sev,
            "suite_condition": sc, "accuracy": round(acc, 2),
            "ci_low": round(ci_lo, 2), "ci_high": round(ci_hi, 2), "N": n
        })

    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "table_k14_matched_noise.csv")
    res_df.to_csv(out_csv, index=False)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{K14: Matched-Noise and Defocus Verification ($N=10{,}000$).}",
        r"\label{tab:k14_matched_noise}",
        r"\small",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Model & Condition & Suite & Accuracy (\%) & 95\% CI & $N$ \\",
        r"\midrule",
    ]
    for _, r in res_df.iterrows():
        tex_lines.append(
            f"{r['model']} & {r['condition']} s{r['severity']} & {r['suite_condition']} & {r['accuracy']:.2f} & [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {r['N']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_str = "\n".join(tex_lines)
    with open(os.path.join(out_dir, "table_k14_matched_noise.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)

    return res_df, tex_str


if __name__ == "__main__":
    compute_k14_table()
