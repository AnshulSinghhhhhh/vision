"""Spectral, noise, and blur statistics computed directly on input images.

Features:
- Immerkaer noise sigma estimation (fast 3x3 convolution).
- Variance of Laplacian for blur estimation.
- Radially averaged Power Spectral Density (PSD) slope and high-to-low energy ratio.
- Mean luminance and RMS contrast.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def immerkaer_noise_sigma(img_gray: torch.Tensor) -> torch.Tensor:
    """Estimates Gaussian noise standard deviation using Immerkaer's fast 3x3 difference operator.
    
    Operator kernel:
    [ 1, -2,  1]
    [-2,  4, -2]
    [ 1, -2,  1]
    Formula: sigma = sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))) * sum(|conv(img, kernel)|)
    """
    if img_gray.ndim == 2:
        img_gray = img_gray.unsqueeze(0).unsqueeze(0)
    elif img_gray.ndim == 3:
        img_gray = img_gray.unsqueeze(1)
        
    kernel = torch.tensor(
        [[1.0, -2.0, 1.0],
         [-2.0, 4.0, -2.0],
         [1.0, -2.0, 1.0]],
        dtype=img_gray.dtype,
        device=img_gray.device
    ).view(1, 1, 3, 3)
    
    b, c, h, w = img_gray.shape
    diff = F.conv2d(img_gray, kernel, padding=0)
    # Sum absolute values
    abs_sum = diff.abs().sum(dim=(-2, -1))
    factor = np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2))
    sigma = abs_sum * factor
    return sigma.squeeze()


def variance_of_laplacian(img_gray: torch.Tensor) -> torch.Tensor:
    """Measures blur severity using the variance of the Laplacian."""
    if img_gray.ndim == 2:
        img_gray = img_gray.unsqueeze(0).unsqueeze(0)
    elif img_gray.ndim == 3:
        img_gray = img_gray.unsqueeze(1)
        
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=img_gray.dtype,
        device=img_gray.device
    ).view(1, 1, 3, 3)
    
    lap = F.conv2d(img_gray, kernel, padding=1)
    var = lap.var(dim=(-2, -1))
    return var.squeeze()


def radial_psd_slope_and_energy(img_gray: np.ndarray) -> Tuple[float, float]:
    """Computes:
    1. Log-log slope of the radially averaged Power Spectral Density (PSD) over mid-frequencies.
    2. High-to-low frequency energy ratio.
    """
    h, w = img_gray.shape
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    psd = np.abs(fshift) ** 2
    
    # Coordinates from center
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    r = np.hypot(x, y).astype(np.int32)
    
    # Radial average
    r_max = min(cy, cx)
    radial_mean = np.zeros(r_max)
    for radius in range(r_max):
        mask = (r == radius)
        if np.any(mask):
            radial_mean[radius] = np.mean(psd[mask])
            
    # Mid-frequency range: [0.1 * r_max, 0.5 * r_max]
    r_start = max(1, int(0.1 * r_max))
    r_end = max(r_start + 2, int(0.5 * r_max))
    
    freqs = np.arange(r_start, r_end)
    vals = radial_mean[r_start:r_end] + 1e-12
    
    # Log-log slope
    log_freqs = np.log(freqs)
    log_vals = np.log(vals)
    slope = float(np.polyfit(log_freqs, log_vals, 1)[0])
    
    # High to low energy ratio
    low_mask = (r < 0.2 * r_max)
    high_mask = (r >= 0.2 * r_max) & (r < r_max)
    low_energy = np.sum(psd[low_mask]) + 1e-12
    high_energy = np.sum(psd[high_mask])
    hl_ratio = float(high_energy / low_energy)
    
    return slope, hl_ratio


def compute_spectral_features(img_u8: np.ndarray) -> Dict[str, float]:
    """Extracts the comprehensive ~8-feature descriptor from a uint8 RGB image."""
    # Convert to grayscale float in [0, 1]
    r, g, b = img_u8[..., 0], img_u8[..., 1], img_u8[..., 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32) / 255.0
    
    gray_t = torch.from_numpy(gray).float()
    noise_sigma = float(immerkaer_noise_sigma(gray_t).item())
    lap_var = float(variance_of_laplacian(gray_t).item())
    psd_slope, hl_energy = radial_psd_slope_and_energy(gray)
    
    mean_lum = float(np.mean(gray))
    rms_contrast = float(np.std(gray))
    
    return {
        "noise_sigma": noise_sigma,
        "laplacian_var": lap_var,
        "psd_slope": psd_slope,
        "high_low_energy_ratio": hl_energy,
        "mean_luminance": mean_lum,
        "rms_contrast": rms_contrast,
    }


def estimate_immerkaer_noise_sigma(img_u8: np.ndarray) -> float:
    feats = compute_spectral_features(img_u8)
    return feats["noise_sigma"]


def calculate_radial_psd_slope(img_u8: np.ndarray) -> float:
    feats = compute_spectral_features(img_u8)
    return feats["psd_slope"]
