"""M7 scale-vs-frequency breakdown analysis.

Decomposes spatial scale from high-frequency information via the M7 suite:
- A: Original 448x448 input
- B: 448x448 with antialiasing lowpass filter (no decimation)
- B_g{sigma}: 448x448 with separable Gaussian lowpass filters
- C: 224x224 downsampled
- D: 224x224 downsampled and upsampled back to 448x448
- E: Noise-gain matched control

Loads measured K6-MECH results (A, B, C, D) and marks newly added filters (B_g{sigma}, E)
as clearly labelled modeled/mocked projections pending K8-MECH-V2 GPU execution.

Outputs:
- analysis/out/m7_decomposition.csv
"""

import os
import glob
import pandas as pd
import numpy as np
from analysis.common import get_repo_root, get_out_dir


def load_m7_results(out_dir: str = None) -> pd.DataFrame:
    """Loads K6-MECH results and generates the complete M7 decomposition table."""
    if out_dir is None:
        out_dir = get_out_dir()

    repo_root = get_repo_root()
    m7_file = os.path.join(repo_root, "results", "raw", "k6-mech", "m7_information_control_results.parquet")

    rows = []
    if os.path.exists(m7_file):
        df_k6 = pd.read_parquet(m7_file)
        # Measured conditions in K6
        for (cond, suite_cond), grp in df_k6.groupby(["condition", "suite_condition"]):
            deit_acc = float(grp["deit_correct"].mean() * 100.0)
            eff_acc = float(grp["eff_correct"].mean() * 100.0)
            n = len(grp)

            rows.append({
                "condition": cond,
                "filter_suite": suite_cond,
                "deit_acc": deit_acc,
                "effnet_acc": eff_acc,
                "is_mock": False,
                "data_source": "measured_k6",
                "N": n,
            })
    else:
        print("[Warning] K6-MECH m7_information_control_results.parquet not found.")

    # Project / mock filters not evaluated in historical K6: B_g{0.5, 0.866, 1.5, 2.5} and E
    # These are clearly labelled as is_mock=True pending K8 execution.
    conditions = ["clean", "gaussian_noise", "defocus_blur"]
    gaussian_sigmas = [0.5, 0.866, 1.5, 2.5]

    for cond in conditions:
        # Get baseline A and B for this condition if present
        sub = [r for r in rows if r["condition"] == cond and not r["is_mock"]]
        a_row = next((r for r in sub if r["filter_suite"] == "A_orig448"), None)
        b_row = next((r for r in sub if r["filter_suite"] == "B_filtered448"), None)

        base_deit = a_row["deit_acc"] if a_row else 75.0
        base_eff = a_row["effnet_acc"] if a_row else 55.0
        b_eff = b_row["effnet_acc"] if b_row else (76.0 if cond == "gaussian_noise" else base_eff)

        for sigma in gaussian_sigmas:
            # Model accuracy progression with filter strength
            # sigma=0.866 px matches the variance (0.75) of the 448->224 filter (Filter B)
            if cond == "gaussian_noise":
                # Filtering rescues EfficientNet progressively as sigma increases
                ratio = min(1.0, sigma / 0.866)
                mock_eff = base_eff + (b_eff - base_eff) * ratio
                mock_deit = base_deit - (1.0 * ratio)
            else:
                mock_eff = base_eff - (0.5 * sigma)
                mock_deit = base_deit - (0.8 * sigma)

            rows.append({
                "condition": cond,
                "filter_suite": f"B_g{sigma}",
                "deit_acc": round(mock_deit, 2),
                "effnet_acc": round(mock_eff, 2),
                "is_mock": True,
                "data_source": "mock_projection (pending K8)",
                "N": 1000,
            })

        # Filter E: Noise-gain matched control (injecting noise at 0.3125 level directly at 448)
        if cond == "gaussian_noise":
            e_eff = 76.5
            e_deit = 77.8
        else:
            e_eff = base_eff
            e_deit = base_deit

        rows.append({
            "condition": cond,
            "filter_suite": "E_noise_matched",
            "deit_acc": round(e_deit, 2),
            "effnet_acc": round(e_eff, 2),
            "is_mock": True,
            "data_source": "mock_projection (pending K8)",
            "N": 1000,
        })

    res_df = pd.DataFrame(rows)
    # Sort logically
    res_df = res_df.sort_values(by=["condition", "is_mock", "filter_suite"]).reset_index(drop=True)

    out_path = os.path.join(out_dir, "m7_decomposition.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Generated M7 Scale vs Frequency Decomposition -> {out_path}")
    return res_df


if __name__ == "__main__":
    df_m7 = load_m7_results()
    print(df_m7.head(15).to_string())
