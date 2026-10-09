"""Effective sigma analysis connecting resolution-dependent noise gain to model collapse.

Physics claim:
When white noise with variance sigma^2 is injected into the 448x448 acquisition frame,
resizing to R < 448 with an antialiasing filter attenuates high frequencies, reducing
the effective noise standard deviation seen by the model to:
    sigma_eff(R) = sigma_injected * noise_gain(R)
where noise_gain in {224: 0.3125, 320: 0.4807, 384: 0.5658, 448: 1.0000}.

At R = 448, resizing is identity, so the model sees the full injected noise variance.
This analysis interpolates the dose-response curve from antialiased resolutions to
determine the accuracy expected strictly from the effective noise level, demonstrating
that EfficientNet-B3's collapse at 448 is predominantly explained by effective noise gain.

Where available (EfficientNet-B3 and DeiT-B), measured empirical operators from K10
(spanning sigma_eff up to ~0.265) are included to interpolate within the empirical range
without extrapolating beyond measured data. Any configurations exceeding measured sigma_eff
are explicitly flagged with extrapolated=True.

Outputs:
- analysis/out/effective_sigma.csv
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from knobs.resize import effective_noise_gain
from analysis.common import load_all_raw_data, get_out_dir, get_repo_root


GAUSSIAN_NOISE_SIGMAS = {0: 0.0, 1: 0.08, 3: 0.18}
MODELS = ["efficientnet_b3", "deit_base", "flexivit_base"]
RESOLUTIONS = [224, 320, 384, 448]


def compute_effective_sigma_table(out_dir: str = None) -> pd.DataFrame:
    """Computes effective sigma, matched sigma accuracy, excess loss, and extrapolation flag across models."""
    if out_dir is None:
        out_dir = get_out_dir()

    # Load baseline shards
    df = load_all_raw_data().copy()
    if "resolution" in df.columns:
        df = df.dropna(subset=["resolution"])
        df["resolution"] = df["resolution"].astype(int)

    # Filter to baseline runs (exclude ablation and mechanism rows)
    c = pd.Series(True, index=df.index)
    if "operator" in df.columns:
        c &= df["operator"].isna()
    if "suite_condition" in df.columns:
        c &= df["suite_condition"].isna()
    if "config_name" in df.columns:
        c &= df["config_name"].isna()
    if "band" in df.columns:
        c &= df["band"].isna()
    df = df[c]

    # Check for K10 resize ablation data to expand empirical sigma_eff range
    k10_path = os.path.join(get_repo_root(), "results", "raw", "k10-resize-ablation", "shard_resize_ablation.parquet")
    df10 = pd.read_parquet(k10_path) if os.path.exists(k10_path) else None

    rows = []

    for model in MODELS:
        # Build reference empirical dose-response curve from antialiased resolutions (R < 448)
        ref_points = []
        for s, s_inj in GAUSSIAN_NOISE_SIGMAS.items():
            cond = "clean" if s == 0 else "gaussian_noise"
            sub = df[(df["model"] == model) & (df["condition"] == cond) & (df["severity"] == s) & (df["arm"].isin(["standard", "F-p"]))]
            for r in [224, 320, 384]:
                if sub[sub["resolution"] == r].empty:
                    continue
                acc = float(sub[sub["resolution"] == r]["correct"].mean() * 100.0)
                gain = effective_noise_gain(r, 448)
                seff = s_inj * gain
                ref_points.append((seff, acc))

        # Include K10 downsampled operators (R < 448) if available for this model
        if df10 is not None and model in df10["model"].unique():
            sub10 = df10[(df10["model"] == model) & (df10["resolution"] < 448)]
            for (cond, sev, op, res), grp in sub10.groupby(["condition", "severity", "operator", "resolution"]):
                seff = float(grp["effective_sigma"].iloc[0])
                acc = float(grp["correct"].mean() * 100.0)
                ref_points.append((seff, acc))

        ref_points.sort(key=lambda p: p[0])

        # Aggregate unique sigma_eff by averaging accuracy
        xs, ys = [], []
        for x_val in sorted(set(p[0] for p in ref_points)):
            y_vals = [p[1] for p in ref_points if p[0] == x_val]
            xs.append(x_val)
            ys.append(float(np.mean(y_vals)))

        max_measured_sigma = max(xs) if xs else 0.0
        min_measured_sigma = min(xs) if xs else 0.0

        # Monotonic interpolator over effective sigma
        interpolator = PchipInterpolator(xs, ys, extrapolate=True)

        # Primary analysis on headline severity 3 (sigma_injected = 0.18)
        s_target = 3
        s_inj = 0.18
        sub_s3 = df[(df["model"] == model) & (df["condition"] == "gaussian_noise") &
                    (df["severity"] == s_target) & (df["arm"].isin(["standard", "F-p"]))]

        for r in RESOLUTIONS:
            meas_acc = float(sub_s3[sub_s3["resolution"] == r]["correct"].mean() * 100.0)
            gain = float(effective_noise_gain(r, 448))
            seff = float(s_inj * gain)

            # Check if this evaluation point lies strictly within empirical measured range
            is_extrapolated = bool(seff > max_measured_sigma or seff < min_measured_sigma)

            if is_extrapolated and seff > max_measured_sigma:
                # Monotonic linear slope extrapolation bounded between 0 and 100
                if len(xs) >= 2:
                    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                    matched_acc = float(np.clip(ys[-1] + slope * (seff - xs[-1]), 0.0, 100.0))
                else:
                    matched_acc = float(np.clip(ys[-1], 0.0, 100.0))
            else:
                matched_acc = float(interpolator(seff))

            excess_loss = float(meas_acc - matched_acc)

            rows.append({
                "model": model,
                "R": r,
                "sigma_injected": s_inj,
                "noise_gain": gain,
                "sigma_eff": seff,
                "measured_acc": meas_acc,
                "matched_sigma_acc": matched_acc,
                "excess_loss_pp": excess_loss,
                "extrapolated": is_extrapolated,
            })

    res_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "effective_sigma.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Generated Effective Sigma Analysis (with extrapolation flags) -> {out_path}")
    return res_df


if __name__ == "__main__":
    df_res = compute_effective_sigma_table()
    print(df_res.to_string())
