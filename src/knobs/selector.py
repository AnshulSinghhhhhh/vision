"""Prototype. Resolution-only. No demonstrated gain: see analysis/out/table_oracle_headroom.csv."""

from typing import Dict, List, Any, Tuple
import numpy as np


# DEPRECATED: Resolution-only selector has no empirical headroom; see analysis/out/table_oracle_headroom.csv
class ZeroParamSpectralGate:
    """Input-dependent zero-parameter threshold selector based on noise sigma and Laplacian blur."""
    def __init__(self, noise_thresh: float = 0.05, blur_thresh: float = 0.005):
        self.noise_thresh = noise_thresh
        self.blur_thresh = blur_thresh

    def fit(self, cal_records: List[Dict[str, Any]]):
        """Sets noise threshold to the 60th percentile of noise_sigma and blur threshold to the 40th percentile of laplacian_var across calibration records (heuristic percentiles, not accuracy optimization)."""
        # Find median noise_sigma and laplacian_var on corrupted images in CAL-GATE
        noise_sigmas = [r["spectral"]["noise_sigma"] for r in cal_records if "spectral" in r]
        lap_vars = [r["spectral"]["laplacian_var"] for r in cal_records if "spectral" in r]
        
        if noise_sigmas:
            self.noise_thresh = float(np.percentile(noise_sigmas, 60))
        if lap_vars:
            self.blur_thresh = float(np.percentile(lap_vars, 40))

    def select_resolution(self, features: Dict[str, float]) -> int:
        """Selects resolution among {224, 320, 384, 448} based on spectral features."""
        noise_sigma = features.get("noise_sigma", 0.0)
        lap_var = features.get("laplacian_var", 0.01)
        
        # High broadband noise -> downsample to 224 for maximum low-pass filtering
        if noise_sigma > self.noise_thresh:
            return 224
        # Strong blur -> extra pixels convey no high frequencies, select 320
        elif lap_var < self.blur_thresh:
            return 320
        # Clean or moderate -> preserve resolution
        else:
            return 448


def compute_oracles(
    records: List[Dict[str, Any]],
    resolutions: List[int] = [224, 320, 384, 448],
) -> Dict[str, float]:
    """Calculates condition-level and per-image oracle accuracies.
    
    Expected record schema per image-condition:
    {
      "image_id": str,
      "condition": str,
      "correct_by_res": {224: bool, 320: bool, 384: bool, 448: bool}
    }
    """
    total_images = len(records)
    if total_images == 0:
        return {}

    # 1. Per-image oracle (upper bound: best resolution per image)
    per_image_correct = 0
    for r in records:
        correct_any = any(r["correct_by_res"].get(res, False) for res in resolutions)
        if correct_any:
            per_image_correct += 1
    per_image_oracle_acc = per_image_correct / total_images

    # 2. Condition-level oracle (best resolution for each condition)
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in records:
        by_condition[r.get("condition", "all")].append(r)
        
    cond_oracle_correct = 0
    for cond, cond_recs in by_condition.items():
        # Find which resolution has the highest accuracy for this condition
        best_res_acc = -1
        best_res = resolutions[0]
        for res in resolutions:
            acc = sum(1 for rec in cond_recs if rec["correct_by_res"].get(res, False)) / len(cond_recs)
            if acc > best_res_acc:
                best_res_acc = acc
                best_res = res
        # Accumulate correct count for that best resolution
        cond_oracle_correct += sum(1 for rec in cond_recs if rec["correct_by_res"].get(best_res, False))
        
    condition_oracle_acc = cond_oracle_correct / total_images

    # 3. Best fixed resolution
    fixed_accs = {
        res: sum(1 for r in records if r["correct_by_res"].get(res, False)) / total_images
        for res in resolutions
    }
    best_fixed_res = max(fixed_accs, key=fixed_accs.get)
    best_fixed_acc = fixed_accs[best_fixed_res]

    return {
        "best_fixed_resolution": best_fixed_res,
        "best_fixed_acc": best_fixed_acc,
        "condition_oracle_acc": condition_oracle_acc,
        "per_image_oracle_upper_bound": per_image_oracle_acc,
    }
