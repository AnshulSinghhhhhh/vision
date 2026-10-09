"""Resolution resizing operators and M7 filter-matched information control.

Provides:
- Standard antialiased bilinear downsampling to target ladder resolutions {448, 384, 320, 224}.
- M7 Filter-Matched Information Control:
  - Condition A: Original 448 input (784 tokens)
  - Condition B: 448 input + exact antialiasing low-pass filter WITHOUT decimation (784 tokens)
  - Condition C: Actual 224 downsampled input (196 tokens)
  - Condition D: 224 downsampled input upsampled back to 448 (784 tokens)
"""

from typing import Tuple, Dict, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

VALID_RESOLUTIONS = (224, 320, 384, 448)


def resize_image_np(img: np.ndarray, target_size: int) -> np.ndarray:
    """Downsamples uint8 image of shape (H, W, 3) to (target_size, target_size)
    using antialiased bilinear interpolation.
    """
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img.copy()
    pil_img = Image.fromarray(img)
    # PIL bilinear resize with default antialias=True
    resized = pil_img.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def resize_tensor_torch(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """Downsamples a PyTorch tensor (B, C, H, W) to (B, C, target_size, target_size)
    using antialiased bilinear interpolation.
    """
    if tensor.shape[-2] == target_size and tensor.shape[-1] == target_size:
        return tensor
    return F.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def get_antialias_bilinear_2d_kernel(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Constructs a 3-tap discrete separable 2D FIR low-pass filter [0.25, 0.5, 0.25]
    that approximates the continuous antialiasing triangle filter for factor 0.5 (448 -> 224).
    
    Note: The actual PyTorch antialias downsampling filter centered across coordinates has taps
    [1/8, 3/8, 3/8, 1/8] (noise gain 0.313). This 3-tap filter has noise gain 0.375 and
    approximates antialiasing without decimation.
    """
    k1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32, device=device)
    k2d = torch.outer(k1d, k1d)
    return k2d


def effective_noise_gain(target_res: int, base_res: int = 448) -> float:
    """Computes the standard deviation of resized white noise relative to input noise,
    derived from the discrete PyTorch antialias triangle downsampling filter weights.
    
    Expected for base_res=448:
        224 -> 0.313 (exact 0.3125)
        320 -> 0.481 (approx 0.4807)
        384 -> 0.565 (approx 0.5658)
        448 -> 1.000 (identity)
    """
    if target_res >= base_res:
        return 1.0

    scale = base_res / target_res
    var_1d_list = []
    for y in range(target_res):
        center = (y + 0.5) * scale - 0.5
        left = int(np.floor(center - scale))
        right = int(np.ceil(center + scale))
        taps = []
        for i in range(left, right + 1):
            dist = abs(i - center)
            if dist < scale:
                taps.append(1.0 - dist / scale)
        weights = np.array(taps, dtype=np.float64)
        weights /= np.sum(weights)
        var_1d_list.append(np.sum(weights ** 2))

    # Bilinear interpolation is separable: 2D std = sqrt((var_1d)^2) = var_1d
    return float(np.mean(var_1d_list))


def apply_gaussian_lowpass(tensor: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Applies a separable Gaussian low-pass filter to a tensor with reflect padding.
    
    Args:
        tensor: (B, C, H, W) float tensor
        sigma_px: Gaussian filter standard deviation in pixels.
                  sigma_px = 0.866 px matches the variance (0.75) of the 448 -> 224 kernel.
    Returns:
        (B, C, H, W) filtered float tensor
    """
    if sigma_px <= 0.0:
        return tensor.clone()

    radius = max(1, int(np.ceil(3.0 * sigma_px)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=tensor.device)
    k1d = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k1d = k1d / k1d.sum()

    b, c, h, w = tensor.shape
    kernel_x = k1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kernel_y = k1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

    padded_x = F.pad(tensor, (radius, radius, 0, 0), mode="reflect")
    out_x = F.conv2d(padded_x, kernel_x, groups=c)
    padded_y = F.pad(out_x, (0, 0, radius, radius), mode="reflect")
    filtered = F.conv2d(padded_y, kernel_y, groups=c)
    return filtered


def apply_m7_filter_without_decimation(tensor_448: torch.Tensor) -> torch.Tensor:
    """Applies a 3-tap low-pass filter [0.25, 0.5, 0.25] approximating the antialiasing
    filter of 448->224 resize to a 448x448 tensor WITHOUT spatial decimation (keeps 448x448).
    
    Args:
        tensor_448: (B, C, 448, 448) float tensor
    Returns:
        (B, C, 448, 448) low-pass filtered float tensor
    """
    b, c, h, w = tensor_448.shape
    kernel = get_antialias_bilinear_2d_kernel(tensor_448.device)
    kernel_weights = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    padded = F.pad(tensor_448, (1, 1, 1, 1), mode="reflect")
    filtered = F.conv2d(padded, kernel_weights, groups=c)
    return filtered


def generate_m7_suite(
    tensor_448: torch.Tensor,
    filters: Optional[list] = None,
) -> Dict[str, torch.Tensor]:
    """Generates the M7 information control suite for a 448x448 input tensor:
    - Condition A_orig448: Original 448 input (784 tokens)
    - Condition B_filtered448: 448 input + 3-tap filter [1/4, 1/2, 1/4] (approximates downsampling filter)
    - Condition B_g{sigma}: Gaussian low-pass filtered at 448 with sigma (default [0.5, 0.866, 1.5, 2.5])
    - Condition C_down224: Actual 224 downsampled input (196 tokens)
    - Condition D_up448: 224 downsampled input upsampled back to 448 (784 tokens)
    """
    if filters is None:
        filters = [0.5, 0.866, 1.5, 2.5]

    cond_a = tensor_448.clone()
    cond_b = apply_m7_filter_without_decimation(tensor_448)
    cond_c = resize_tensor_torch(tensor_448, target_size=224)
    cond_d = F.interpolate(
        cond_c,
        size=(448, 448),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    
    suite = {
        "A_orig448": cond_a,
        "B_filtered448": cond_b,
    }
    for sigma in filters:
        suite[f"B_g{sigma}"] = apply_gaussian_lowpass(tensor_448, sigma_px=float(sigma))
        
    suite["C_down224"] = cond_c
    suite["D_up448"] = cond_d
    return suite


def build_m7_v2_suite(
    clean_tensor_448: torch.Tensor,
    corrupted_tensor_448: torch.Tensor,
    condition: str,
    severity: int,
    image_id: str,
    gaussian_sigmas: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Builds the expanded M7-v2 suite of representations for a single image.

    Suites:
    - A_orig448: Original 448 acquisition frame
    - B_filtered448: 3-tap lowpass filter [1/4, 1/2, 1/4] at 448
    - B_g{sigma}: Gaussian lowpass filters at 448
    - C_down224: True 224 downsampling (antialiased bilinear)
    - D_up448: 224 downsampled then upsampled back to 448
    - E_noise_matched: Noise injected at 448 scaled by effective noise gain (0.3125 * sigma_inj)

    Returns dictionary mapping condition name to (1, 3, H, W) tensor in [0, 1].
    """
    from knobs.corrupt import compute_seed, SeedContext
    if gaussian_sigmas is None:
        gaussian_sigmas = [0.5, 0.866, 1.5, 2.5]

    suite = generate_m7_suite(corrupted_tensor_448, filters=gaussian_sigmas)

    if condition == "gaussian_noise" and severity > 0:
        sigmas_map = {1: 0.08, 2: 0.12, 3: 0.18, 4: 0.26, 5: 0.38}
        sigma_inj = sigmas_map.get(severity, 0.18)
        sigma_matched = 0.3125 * sigma_inj
        seed = compute_seed(image_id, "gaussian_noise_matched", severity)
        clean_np = (clean_tensor_448.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        with SeedContext(seed):
            noise = np.random.normal(0, sigma_matched * 255.0, clean_np.shape)
        matched_np = np.clip(clean_np + noise, 0, 255).astype(np.uint8)
        e_tensor = torch.from_numpy(matched_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        suite["E_noise_matched"] = e_tensor

    return suite
