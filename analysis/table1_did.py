"""Table 1: Main Difference-in-Differences across architectures and core corruptions.

Reproduces Table 1 from raw shards with paired z-test and Holm-adjusted p-values.
Outputs analysis/out/table1.csv and analysis/out/table1.tex.
"""

import os
import pandas as pd
from typing import Tuple
from knobs.stats import paired_did_ztest, holm_adjust
from analysis.common import load_clean_and_corrupted, get_out_dir


PRIMARY_MODELS = ["deit_base", "efficientnet_b3", "flexivit_base"]
PRIMARY_CORRUPTIONS = ["gaussian_noise", "defocus_blur", "jpeg_compression", "contrast"]


def generate_table1(out_dir: str = None) -> Tuple[pd.DataFrame, str]:
    """Computes Table 1 from matched clean and degraded data."""
    if out_dir is None:
        out_dir = get_out_dir()

    rows = []
    for model in PRIMARY_MODELS:
        for corr in PRIMARY_CORRUPTIONS:
            df_match = load_clean_and_corrupted(corruption=corr, severity=3, model=model)
            
            c448_v = df_match["c_448"].values
            c224_v = df_match["c_224"].values
            d448_v = df_match["d_448"].values
            d224_v = df_match["d_224"].values
            
            acc_c224 = float(c224_v.mean() * 100.0)
            acc_c448 = float(c448_v.mean() * 100.0)
            acc_d224 = float(d224_v.mean() * 100.0)
            acc_d448 = float(d448_v.mean() * 100.0)
            
            delta_clean = acc_c448 - acc_c224
            delta_deg = acc_d448 - acc_d224
            
            z_res = paired_did_ztest(c448_v, c224_v, d448_v, d224_v)
            
            rows.append({
                "Model": model,
                "Corruption": corr,
                "N": len(df_match),
                "Clean_224": acc_c224,
                "Clean_448": acc_c448,
                "Delta_Clean": delta_clean,
                "Deg_224": acc_d224,
                "Deg_448": acc_d448,
                "Delta_Deg": delta_deg,
                "DiD_pp": z_res.did_pp,
                "SE_pp": z_res.se_pp,
                "CI_95_low": z_res.ci_lo,
                "CI_95_high": z_res.ci_hi,
                "z": z_res.z,
                "p_raw": z_res.p_two_sided,
            })
            
    df_t1 = pd.DataFrame(rows)
    p_holm_vals = holm_adjust(df_t1["p_raw"].tolist())
    df_t1["p_holm"] = p_holm_vals
    
    # Export CSV
    csv_path = os.path.join(out_dir, "table1.csv")
    df_t1.to_csv(csv_path, index=False)
    
    # Export LaTeX
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrrrrrrr}",
        r"\toprule",
        r"Model & Corruption & $N$ & Clean$_{224}$ & Clean$_{448}$ & $\Delta_{\text{clean}}$ & Deg$_{224}$ & Deg$_{448}$ & $\Delta_{\text{deg}}$ & DiD (pp) & 95\% CI & $z$ & $p_{\text{holm}}$ \\",
        r"\midrule",
    ]
    
    for _, row in df_t1.iterrows():
        m = row["Model"]
        c = row["Corruption"]
        model_disp = "DeiT-B/16" if m == "deit_base" else ("EfficientNet-B3" if m == "efficientnet_b3" else "FlexiViT-B")
        corr_disp = c.replace("_", " ").title()
        p_val = row["p_holm"]
        p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
        
        line = (
            f"{model_disp} & {corr_disp} & {row['N']:,} & "
            f"{row['Clean_224']:.2f}\\% & {row['Clean_448']:.2f}\\% & {row['Delta_Clean']:+.2f} & "
            f"{row['Deg_224']:.2f}\\% & {row['Deg_448']:.2f}\\% & {row['Delta_Deg']:+.2f} & "
            f"{row['DiD_pp']:+.2f} & [{row['CI_95_low']:+.2f}, {row['CI_95_high']:+.2f}] & "
            f"{row['z']:.2f} & {p_str} \\\\"
        )
        latex_lines.append(line)
        
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Difference-in-Differences (DiD) of ImageNet-1K accuracy under degradation vs clean across models. DiD is defined as $(Acc_{448} - Acc_{224})_{\text{deg}} - (Acc_{448} - Acc_{224})_{\text{clean}}$. $p$-values are Holm-adjusted across all 12 primary hypotheses.}",
        r"\label{tab:table1_did}",
        r"\end{table*}",
    ])
    
    latex_str = "\n".join(latex_lines) + "\n"
    tex_path = os.path.join(out_dir, "table1.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
        
    print(f"Saved Table 1 outputs to {csv_path} and {tex_path}")
    return df_t1, latex_str


if __name__ == "__main__":
    generate_table1()
