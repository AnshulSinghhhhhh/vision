"""Analysis for K15: FlexiViT resolution x patch size grid on N=3,000 images.

Reads shards from results/raw/k15-flexivit-grid/, computes accuracy, token counts, and 95% CIs.

Outputs:
- analysis/out/table_k15_flexivit_grid.csv
- analysis/out/table_k15_flexivit_grid.tex
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


def compute_k15_table(parquet_path: str = None, out_dir: str = None) -> Tuple[pd.DataFrame, str]:
    if out_dir is None:
        out_dir = get_out_dir()

    if parquet_path is None:
        raw_candidates = [
            os.path.join(get_repo_root(), "results", "raw", "k15-flexivit-grid", "shard_flexivit_grid.parquet"),
            os.path.join(get_repo_root(), "results", "raw", "k15-flexivit-grid", "shards"),
        ]
        if os.path.exists(raw_candidates[0]):
            df = pd.read_parquet(raw_candidates[0])
        elif os.path.exists(raw_candidates[1]):
            shards = [os.path.join(raw_candidates[1], f) for f in os.listdir(raw_candidates[1]) if f.endswith(".parquet")]
            if shards:
                df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
            else:
                raise FileNotFoundError("No shards found in k15 folder")
        else:
            raise FileNotFoundError("K15 results parquet not found")
    else:
        df = pd.read_parquet(parquet_path)

    rows = []
    group_cols = ["resolution", "patch_size", "tokens", "condition", "severity"]
    for (res, patch, tok, cond, sev), grp in df.groupby(group_cols):
        n = len(grp)
        p = float(grp["correct"].mean())
        acc = p * 100.0
        se = np.sqrt(p * (1.0 - p) / n) * 100.0 if n > 0 else 0.0
        z_crit = 1.95996
        ci_lo = max(0.0, acc - z_crit * se)
        ci_hi = min(100.0, acc + z_crit * se)
        rows.append({
            "model": "flexivit_base",
            "resolution": int(res),
            "patch_size": int(patch),
            "tokens": int(tok),
            "condition": cond,
            "severity": int(sev),
            "accuracy": round(acc, 2),
            "ci_low": round(ci_lo, 2),
            "ci_high": round(ci_hi, 2),
            "N": n,
        })

    res_df = pd.DataFrame(rows)
    res_df = res_df.sort_values(by=["resolution", "patch_size", "condition", "severity"]).reset_index(drop=True)
    out_csv = os.path.join(out_dir, "table_k15_flexivit_grid.csv")
    res_df.to_csv(out_csv, index=False)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{K15: FlexiViT Resolution $\times$ Patch Size Grid ($N=3{,}000$).}",
        r"\label{tab:k15_flexivit_grid}",
        r"\small",
        r"\begin{tabular}{ccccccr}",
        r"\toprule",
        r"Res & Patch & Tokens & Condition & Accuracy (\%) & 95\% CI & $N$ \\",
        r"\midrule",
    ]
    for _, r in res_df.iterrows():
        tex_lines.append(
            f"{r['resolution']} & {r['patch_size']} & {r['tokens']} & {r['condition']} s{r['severity']} & {r['accuracy']:.2f} & [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {r['N']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_str = "\n".join(tex_lines)
    with open(os.path.join(out_dir, "table_k15_flexivit_grid.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)

    return res_df, tex_str


if __name__ == "__main__":
    compute_k15_table()
