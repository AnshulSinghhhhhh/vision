"""Publication-grade figures generated from empirical evaluation shards.

Generates:
- analysis/out/fig1_dose_response.pdf
- analysis/out/fig2_effective_sigma.pdf
- analysis/out/fig3_tradeoff.pdf
- analysis/out/fig4_transitions.pdf

Outputs both PDF and PNG formats.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knobs.resize import effective_noise_gain
from analysis.common import load_all_raw_data, get_out_dir
from analysis.effective_sigma import compute_effective_sigma_table
from analysis.transitions import compute_transitions_table


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def generate_fig1_dose_response(df: pd.DataFrame, out_dir: str):
    """Figure 1: Accuracy vs severity for primary corruptions with 224 and 448 curves and CI bands."""
    corruptions = ["gaussian_noise", "defocus_blur", "jpeg_compression", "contrast"]
    models = [("deit_base", "DeiT-B", "#1f77b4"), ("efficientnet_b3", "EffNet-B3", "#ff7f0e"), ("flexivit_base", "FlexiViT-B", "#2ca02c")]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    
    for ax_idx, c in enumerate(corruptions):
        ax = axes[ax_idx]
        c_title = c.replace("_", " ").title()
        
        for m_key, m_name, color in models:
            sub = df[(df["model"] == m_key) & (df["arm"].isin(["standard", "F-p"]))]
            
            # Find available severities
            sevs = sorted([s for s in sub[sub["condition"] == c]["severity"].unique() if s > 0])
            # Include clean (s=0)
            all_sevs = [0] + sevs
            
            for res, ls, marker in [(224, "--", "o"), (448, "-", "s")]:
                accs = []
                ci_lows = []
                ci_highs = []
                valid_sevs = []
                
                for s in all_sevs:
                    cond_name = "clean" if s == 0 else c
                    s_sub = sub[(sub["condition"] == cond_name) & (sub["severity"] == s) & (sub["resolution"] == res)]
                    if s_sub.empty:
                        continue
                    n = len(s_sub)
                    p = float(s_sub["correct"].mean())
                    se = np.sqrt(p * (1.0 - p) / max(1, n))
                    
                    accs.append(p * 100.0)
                    ci_lows.append(max(0.0, (p - 1.96 * se) * 100.0))
                    ci_highs.append(min(100.0, (p + 1.96 * se) * 100.0))
                    valid_sevs.append(s)
                    
                if valid_sevs:
                    label = f"{m_name} @ {res}" if ax_idx == 0 else None
                    ax.plot(valid_sevs, accs, label=label, color=color, linestyle=ls, marker=marker, linewidth=1.8, markersize=5)
                    ax.fill_between(valid_sevs, ci_lows, ci_highs, color=color, alpha=0.15)
                    
        ax.set_title(c_title, fontweight="bold")
        ax.set_xlabel("Severity Level")
        ax.set_xticks([0, 1, 3, 5])
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax_idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower left")

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "fig1_dose_response.pdf")
    png_path = os.path.join(out_dir, "fig1_dose_response.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved Fig 1 -> {pdf_path}")


def generate_fig2_effective_sigma(df: pd.DataFrame, out_dir: str):
    """Figure 2: Accuracy vs effective sigma (sigma_eff) demonstrating collapse onto master curve."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    models = [("efficientnet_b3", "EfficientNet-B3"), ("deit_base", "DeiT-B/16"), ("flexivit_base", "FlexiViT-B")]
    sigmas = {0: 0.0, 1: 0.08, 3: 0.18, 5: 0.38}
    res_markers = {224: ("o", "#1f77b4"), 320: ("^", "#ff7f0e"), 384: ("s", "#2ca02c"), 448: ("D", "#d62728")}

    for ax_idx, (m_key, m_name) in enumerate(models):
        ax = axes[ax_idx]
        sub = df[(df["model"] == m_key) & (df["arm"].isin(["standard", "F-p"]))]
        
        # Plot points by resolution
        for r, (marker, color) in res_markers.items():
            r_seff = []
            r_acc = []
            for s, s_inj in sorted(sigmas.items()):
                cond = "clean" if s == 0 else "gaussian_noise"
                s_sub = sub[(sub["condition"] == cond) & (sub["severity"] == s) & (sub["resolution"] == r)]
                if s_sub.empty:
                    continue
                acc = float(s_sub["correct"].mean() * 100.0)
                g = effective_noise_gain(r, 448)
                seff = s_inj * g
                r_seff.append(seff)
                r_acc.append(acc)
                
            ax.plot(r_seff, r_acc, label=f"R={r} px", marker=marker, color=color, linestyle="--", linewidth=1.5, markersize=6)

        ax.set_title(m_name, fontweight="bold")
        ax.set_xlabel(r"Effective Noise $\sigma_{\mathrm{eff}} = \sigma_{\mathrm{inj}} \cdot \mathrm{gain}(R)$")
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax_idx == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "fig2_effective_sigma.pdf")
    png_path = os.path.join(out_dir, "fig2_effective_sigma.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved Fig 2 -> {pdf_path}")


def generate_fig3_tradeoff(df: pd.DataFrame, out_dir: str):
    """Figure 3: Accuracy vs Compute (GFLOPs) across resolutions and ToMe r under Gaussian Noise (s3)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # GFLOPs approximate benchmarks (2x MAC)
    gflops_map = {
        ("deit_base", 224): 35.1,
        ("deit_base", 320): 74.0,
        ("deit_base", 384): 108.5,
        ("deit_base", 448): 150.8,
        ("efficientnet_b3", 224): 1.8,
        ("efficientnet_b3", 320): 3.6,
        ("efficientnet_b3", 384): 5.3,
        ("efficientnet_b3", 448): 7.2,
        ("flexivit_base", 224): 35.1,
        ("flexivit_base", 320): 74.0,
        ("flexivit_base", 384): 108.5,
        ("flexivit_base", 448): 150.8,
        # ToMe r=4, r=8
        ("tome_r4", 224): 28.5,
        ("tome_r4", 448): 118.0,
        ("tome_r8", 224): 22.0,
        ("tome_r8", 448): 86.0,
    }

    # Plot DeiT-B curve
    deit_sub = df[(df["model"] == "deit_base") & (df["arm"] == "standard") & (df["condition"] == "gaussian_noise") & (df["severity"] == 3)]
    deit_flops = [gflops_map[("deit_base", r)] for r in [224, 320, 384, 448]]
    deit_accs = [float(deit_sub[deit_sub["resolution"] == r]["correct"].mean() * 100.0) for r in [224, 320, 384, 448]]
    ax.plot(deit_flops, deit_accs, "o-", label="DeiT-B/16 standard", color="#1f77b4", linewidth=2, markersize=7)

    # Plot FlexiViT-B curve
    flex_sub = df[(df["model"] == "flexivit_base") & (df["arm"] == "F-p") & (df["condition"] == "gaussian_noise") & (df["severity"] == 3)]
    flex_accs = [float(flex_sub[flex_sub["resolution"] == r]["correct"].mean() * 100.0) for r in [224, 320, 384, 448]]
    ax.plot(deit_flops, flex_accs, "s-", label="FlexiViT-B (F-p)", color="#2ca02c", linewidth=2, markersize=7)

    # Plot ToMe variants at 224 and 448
    tome_r4_sub = df[(df["model"] == "deit_base_tome_r4") & (df["condition"] == "gaussian_noise") & (df["severity"] == 3)]
    tome_r8_sub = df[(df["model"] == "deit_base_tome_r8") & (df["condition"] == "gaussian_noise") & (df["severity"] == 3)]
    
    if not tome_r4_sub.empty:
        r4_flops = [gflops_map[("tome_r4", 224)], gflops_map[("tome_r4", 448)]]
        r4_accs = [float(tome_r4_sub[tome_r4_sub["resolution"] == r]["correct"].mean() * 100.0) for r in [224, 448]]
        ax.scatter(r4_flops, r4_accs, color="#9467bd", marker="^", s=80, label="DeiT-B + ToMe r=4", zorder=5)

    if not tome_r8_sub.empty:
        r8_flops = [gflops_map[("tome_r8", 224)], gflops_map[("tome_r8", 448)]]
        r8_accs = [float(tome_r8_sub[tome_r8_sub["resolution"] == r]["correct"].mean() * 100.0) for r in [224, 448]]
        ax.scatter(r8_flops, r8_accs, color="#8c564b", marker="v", s=80, label="DeiT-B + ToMe r=8", zorder=5)

    ax.set_title("Figure 3: Accuracy vs Compute Trade-off (Gaussian Noise s3)", fontweight="bold")
    ax.set_xlabel("Compute (GFLOPs / image)")
    ax.set_ylabel("Top-1 Accuracy under Noise (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "fig3_tradeoff.pdf")
    png_path = os.path.join(out_dir, "fig3_tradeoff.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved Fig 3 -> {pdf_path}")


def generate_fig4_transitions(out_dir: str):
    """Figure 4: Stacked bar chart of accuracy transitions (Helped, Hurt, Invariant)."""
    trans_df = compute_transitions_table()
    sub = trans_df[trans_df["severity"] == 3].copy()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Compute proportions
    labels = []
    p_helped = []
    p_hurt = []
    p_inv_corr = []
    p_inv_inc = []
    
    for _, row in sub.iterrows():
        m_disp = "DeiT" if row["model"] == "deit_base" else ("EffNet" if row["model"] == "efficientnet_b3" else "FlexiViT")
        c_disp = row["corruption"].replace("_", " ").title()
        labels.append(f"{m_disp}\n{c_disp}")
        
        n = row["N"]
        p_helped.append(row["deg_helped"] / n * 100.0)
        p_hurt.append(row["deg_hurt"] / n * 100.0)
        p_inv_corr.append(row["deg_inv_corr"] / n * 100.0)
        p_inv_inc.append(row["deg_inv_inc"] / n * 100.0)

    x = np.arange(len(labels))
    width = 0.65

    # Stacked bars: Invariant Correct (bottom), Helped, Hurt, Invariant Incorrect
    b1 = ax.bar(x, p_inv_corr, width, label="Invariant Correct (1→1)", color="#2ca02c", alpha=0.85)
    b2 = ax.bar(x, p_helped, width, bottom=p_inv_corr, label="Helped (0→1)", color="#1f77b4", alpha=0.85)
    bottom_hurt = np.array(p_inv_corr) + np.array(p_helped)
    b3 = ax.bar(x, p_hurt, width, bottom=bottom_hurt, label="Hurt (1→0)", color="#d62728", alpha=0.85)
    bottom_inc = bottom_hurt + np.array(p_hurt)
    b4 = ax.bar(x, p_inv_inc, width, bottom=bottom_inc, label="Invariant Incorrect (0→0)", color="#7f7f7f", alpha=0.85)

    ax.set_ylabel("Share of Validation Set (%)")
    ax.set_title("Figure 4: Image-Level Transition Profiles (224 → 448 px Under Degradation)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "fig4_transitions.pdf")
    png_path = os.path.join(out_dir, "fig4_transitions.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved Fig 4 -> {pdf_path}")


def generate_all_figures():
    out_dir = get_out_dir()
    df = load_all_raw_data().copy()
    if "resolution" in df.columns:
        df = df.dropna(subset=["resolution"])
        df["resolution"] = df["resolution"].astype(int)

    generate_fig1_dose_response(df, out_dir)
    generate_fig2_effective_sigma(df, out_dir)
    generate_fig3_tradeoff(df, out_dir)
    generate_fig4_transitions(out_dir)


if __name__ == "__main__":
    generate_all_figures()
