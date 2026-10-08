"""Transition analysis of 224 -> 448 resolution scaling under clean vs degraded conditions.

Decomposes per-image accuracy transitions into four mutually exclusive categories:
- helped: 0 -> 1 (incorrect at 224, rescued at 448)
- hurt: 1 -> 0 (correct at 224, ruined at 448)
- invariant_correct: 1 -> 1 (correct at both)
- invariant_incorrect: 0 -> 0 (incorrect at both)

Proves algebraically and numerically that:
    DiD = 100 * [ (N_helped - N_hurt)_deg / N - (N_helped - N_hurt)_clean / N ]
matches stats.paired_did_ztest to within 1e-6.

Outputs:
- analysis/out/transitions.csv
"""

import os
import pandas as pd
import numpy as np
from knobs.stats import paired_did_ztest
from analysis.common import load_clean_and_corrupted, get_out_dir


PRIMARY_MODELS = ["deit_base", "efficientnet_b3", "flexivit_base"]
PRIMARY_CORRUPTIONS = ["gaussian_noise", "defocus_blur", "jpeg_compression", "contrast"]


def compute_transitions_table(out_dir: str = None) -> pd.DataFrame:
    """Computes transition breakdown and verifies identity with paired_did_ztest."""
    if out_dir is None:
        out_dir = get_out_dir()

    rows = []

    for model in PRIMARY_MODELS:
        for corr in PRIMARY_CORRUPTIONS:
            # Severity 3 primary headline plus severity 1/5 if available
            severities = [1, 3, 5] if corr in ("gaussian_noise", "defocus_blur") else [3]
            for sev in severities:
                try:
                    df = load_clean_and_corrupted(corruption=corr, severity=sev, model=model)
                except Exception as e:
                    print(f"Skipping {model} {corr} s{sev}: {e}")
                    continue

                c224 = df["c_224"].values
                c448 = df["c_448"].values
                d224 = df["d_224"].values
                d448 = df["d_448"].values
                n = len(df)

                # Clean transition counts
                c_helped = int(((c224 == 0) & (c448 == 1)).sum())
                c_hurt = int(((c224 == 1) & (c448 == 0)).sum())
                c_inv_corr = int(((c224 == 1) & (c448 == 1)).sum())
                c_inv_inc = int(((c224 == 0) & (c448 == 0)).sum())

                # Degraded transition counts
                d_helped = int(((d224 == 0) & (d448 == 1)).sum())
                d_hurt = int(((d224 == 1) & (d448 == 0)).sum())
                d_inv_corr = int(((d224 == 1) & (d448 == 1)).sum())
                d_inv_inc = int(((d224 == 0) & (d448 == 0)).sum())

                # Assert sums match total images
                assert c_helped + c_hurt + c_inv_corr + c_inv_inc == n
                assert d_helped + d_hurt + d_inv_corr + d_inv_inc == n

                delta_clean = 100.0 * (c_helped - c_hurt) / n
                delta_deg = 100.0 * (d_helped - d_hurt) / n
                did_trans = delta_deg - delta_clean

                z_res = paired_did_ztest(c448, c224, d448, d224)
                did_z = z_res.did_pp

                # Critical assertion: DiD transition matches paired_did_ztest point estimate
                diff_match = abs(did_trans - did_z)
                assert diff_match < 1e-6, f"DiD transition mismatch for {model} {corr} s{sev}: trans={did_trans}, z={did_z}, diff={diff_match}"

                rows.append({
                    "model": model,
                    "corruption": corr,
                    "severity": sev,
                    "N": n,
                    "clean_helped": c_helped,
                    "clean_hurt": c_hurt,
                    "clean_inv_corr": c_inv_corr,
                    "clean_inv_inc": c_inv_inc,
                    "deg_helped": d_helped,
                    "deg_hurt": d_hurt,
                    "deg_inv_corr": d_inv_corr,
                    "deg_inv_inc": d_inv_inc,
                    "delta_clean_pp": delta_clean,
                    "delta_deg_pp": delta_deg,
                    "did_transition_pp": did_trans,
                    "did_ztest_pp": did_z,
                    "diff_match": diff_match,
                })

    res_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "transitions.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Generated Transition Analysis -> {out_path} ({len(res_df)} conditions verified)")
    return res_df


if __name__ == "__main__":
    compute_transitions_table()
