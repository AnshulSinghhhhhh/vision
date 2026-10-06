"""Mechanistic probes (M1-M7) for attention localization, feature drift, and causal controls.

Probes:
- M1: Area-normalized object attention mass in ground-truth bounding box + occlusion check.
- M2: Normalized attention entropy H / log(N) & common-grid JS divergence.
- M3: Within-resolution and coordinate-aligned representation perturbation.
- M4: Spectral / noise characteristics (via spectral.py).
- M5: FlexiViT token-count (F-p) vs. patch-density (F-t) contrast.
- M6: ToMe in-box vs. out-of-box token merging dynamics.
- M7: Filter-matched information control (via resize.py).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_area_normalized_attention_mass(
    attn_map_2d: np.ndarray,
    bbox_norm: Tuple[float, float, float, float],
) -> Dict[str, float]:
    """Computes area-normalized object attention mass for a spatial 2D attention map (H, W).
    
    Formula:
    NormMass = (Sum of attention in GT box / Total attention) / (GT box area fraction)
    
    Args:
        attn_map_2d: 2D numpy array of attention weights summing to 1
        bbox_norm: (xmin, ymin, xmax, ymax) normalized coordinates in [0, 1]
    """
    h, w = attn_map_2d.shape
    xmin, ymin, xmax, ymax = bbox_norm
    
    x1 = int(np.floor(xmin * w))
    y1 = int(np.floor(ymin * h))
    x2 = int(np.ceil(xmax * w))
    y2 = int(np.ceil(ymax * h))
    
    # Clip to bounds
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    
    box_area_frac = max(1e-6, (x2 - x1) * (y2 - y1) / (w * h))
    in_box_mass = float(np.sum(attn_map_2d[y1:y2, x1:x2]))
    total_mass = float(np.sum(attn_map_2d)) + 1e-12
    raw_fraction = in_box_mass / total_mass
    norm_mass = raw_fraction / box_area_frac
    
    return {
        "raw_in_box_mass": raw_fraction,
        "box_area_fraction": box_area_frac,
        "area_normalized_mass": norm_mass,
    }


def compute_normalized_attention_entropy(attn_weights: np.ndarray) -> float:
    """Computes normalized attention entropy H_norm = H / log(N)
    where N is the spatial token count, enabling valid comparisons across different token counts.
    
    Args:
        attn_weights: 1D or 2D array of attention weights over N tokens summing to 1.
    """
    p = attn_weights.flatten()
    p = p[p > 0]
    n = len(attn_weights.flatten())
    if n <= 1:
        return 0.0
    entropy = -np.sum(p * np.log(p + 1e-12))
    norm_entropy = float(entropy / np.log(n))
    return norm_entropy


def compute_resampled_js_divergence(
    attn_map_1: np.ndarray,
    attn_map_2: np.ndarray,
    common_grid_size: int = 28,
) -> float:
    """Resamples two spatial attention maps of different resolutions to a common grid
    and calculates their Jensen-Shannon divergence.
    """
    from PIL import Image
    from scipy.spatial.distance import jensenshannon
    
    # Resample to common grid
    im1 = Image.fromarray(attn_map_1.astype(np.float32), mode="F")
    im2 = Image.fromarray(attn_map_2.astype(np.float32), mode="F")
    
    r1 = np.array(im1.resize((common_grid_size, common_grid_size), resample=Image.Resampling.BILINEAR))
    r2 = np.array(im2.resize((common_grid_size, common_grid_size), resample=Image.Resampling.BILINEAR))
    
    p = (r1 / (np.sum(r1) + 1e-12)).flatten()
    q = (r2 / (np.sum(r2) + 1e-12)).flatten()
    
    js_div = float(jensenshannon(p, q) ** 2)
    return js_div


def compute_representation_drift(
    h_clean: torch.Tensor,
    h_corr: torch.Tensor,
) -> float:
    """Measures within-resolution normalized feature perturbation:
    drift = ||h_corr - h_clean||_2 / ||h_clean||_2
    """
    diff_norm = torch.norm(h_corr - h_clean, p=2).item()
    clean_norm = torch.norm(h_clean, p=2).item() + 1e-8
    return float(diff_norm / clean_norm)


def occlusion_sensitivity_check(
    model: nn.Module,
    input_tensor: torch.Tensor,
    bbox_norm: Tuple[float, float, float, float],
    target_class: int,
) -> Dict[str, float]:
    """Occlusion sanity check: masks inside-box vs outside-box region
    and records the logit/confidence drop to verify model reliance.
    """
    model.eval()
    with torch.no_grad():
        orig_out = model(input_tensor)
        orig_prob = F.softmax(orig_out, dim=-1)[0, target_class].item()
        
        b, c, h, w = input_tensor.shape
        xmin, ymin, xmax, ymax = bbox_norm
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        
        # Mask inside box (set to zero / mean)
        in_masked = input_tensor.clone()
        in_masked[:, :, y1:y2, x1:x2] = 0.0
        in_out = model(in_masked)
        in_prob = F.softmax(in_out, dim=-1)[0, target_class].item()
        
        # Mask outside box
        out_masked = torch.zeros_like(input_tensor)
        out_masked[:, :, y1:y2, x1:x2] = input_tensor[:, :, y1:y2, x1:x2]
        out_out = model(out_masked)
        out_prob = F.softmax(out_out, dim=-1)[0, target_class].item()
        
    return {
        "orig_prob": orig_prob,
        "inside_masked_prob": in_prob,
        "outside_masked_prob": out_prob,
        "drop_when_inside_masked": float(orig_prob - in_prob),
        "drop_when_outside_masked": float(orig_prob - out_prob),
    }
