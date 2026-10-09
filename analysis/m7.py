"""M7 scale-vs-frequency breakdown analysis.

Decomposes spatial scale from high-frequency information via the M7 suite:
- A_orig448: Original 448x448 input
- B_filtered448: 448x448 with antialiasing lowpass filter (no decimation)
- B_g{sigma}: 448x448 with separable Gaussian lowpass filters
- C_down224: 224x224 downsampled
- D_up448: 224x224 downsampled and upsampled back to 448x448
- E_noise_matched: Noise-gain matched control

Loads measured K8-M7-v2 results from results/raw/k8-m7-v2/shard_m7_v2.parquet.
All mock-projection code paths have been deleted; all reported metrics represent
real empirical observations.

Outputs:
- analysis/out/m7_decomposition.csv
"""

import os
import numpy as np
import pandas as pd
from analysis.common import get_repo_root, get_out_dir


def load_m7_results(out_dir: str = None) -> pd.DataFrame:
    """Loads K8-M7-v2 results and generates the complete M7 decomposition table with 95% CIs."""
    if out_dir is None:
        out_dir = get_out_dir()

    repo_root = get_repo_root()
    m7_parquet = os.path.join(repo_root, "results", "raw", "k8-m7-v2", "shard_m7_v2.parquet")

    if not os.path.exists(m7_parquet):
        raise FileNotFoundError(f"K8-M7-v2 shard not found at {m7_parquet}")

    df_k8 = pd.read_parquet(m7_parquet)

    rows = []
    group_cols = ["model", "arm", "condition", "severity", "suite_condition"]
    for (m, arm, cond, sev, sc), grp in df_k8.groupby(group_cols):
        n = len(grp)
        p = float(grp["correct"].mean())
        acc = p * 100.0
        
        # Analytic standard error for proportion
        se = np.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0
        se_pp = se * 100.0
        z_crit = 1.95996
        ci_low = max(0.0, acc - z_crit * se_pp)
        ci_high = min(100.0, acc + z_crit * se_pp)

        rows.append({
            "model": m,
            "arm": arm,
            "condition": cond,
            "severity": sev,
            "suite_condition": sc,
            "accuracy": round(acc, 2),
            "N": n,
            "ci_low": round(ci_low, 2),
            "ci_high": round(ci_high, 2),
        })

    res_df = pd.DataFrame(rows)
    res_df = res_df.sort_values(by=["model", "arm", "condition", "severity", "suite_condition"]).reset_index(drop=True)

    out_csv = os.path.join(out_dir, "m7_decomposition.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"[M7 Analysis] Wrote {len(res_df)} real empirical decomposition records to {out_csv}")

    return res_df


if __name__ == "__main__":
    load_m7_results()
