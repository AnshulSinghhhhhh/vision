"""Resolution resizing operators and M7 filter-matched information control.

Provides:
- Standard antialiased bilinear downsampling to target ladder resolutions {448, 384, 320, 224}.
- M7 Filter-Matched Information Control:
  - Condition A: Original 448 input (784 tokens)
  - Condition B: 448 input + exact antialiasing low-pass filter WITHOUT decimation (784 tokens)
  - Condition C: Actual 224 downsampled input (196 tokens)
  - Condition D: 224 downsampled input upsampled back to 448 (784 tokens)
"""

from typing import Tuple, Dict
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
    """Constructs the discrete separable 2D FIR low-pass filter corresponding
    to the continuous antialiasing triangle filter for factor 0.5 (448 -> 224).
    
    Continuous tent: f(x) = max(0, 1 - |0.5 * x|).
    Evaluated at integers x in {-1, 0, 1}: [0.5, 1.0, 0.5], normalized to [0.25, 0.5, 0.25].
    2D kernel: outer product [0.25, 0.5, 0.25]^T * [0.25, 0.5, 0.25].
    """
    k1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32, device=device)
    k2d = torch.outer(k1d, k1d)
    # Shape for depthwise conv2d: (C, 1, 3, 3) where C will be expanded as needed
    return k2d


def apply_m7_filter_without_decimation(tensor_448: torch.Tensor) -> torch.Tensor:
    """Applies the exact antialiasing low-pass filter of 448->224 resize to a 448x448 tensor
    WITHOUT spatial decimation (keeps 448x448 resolution).
    
    Args:
        tensor_448: (B, C, 448, 448) float tensor
    Returns:
        (B, C, 448, 448) low-pass filtered float tensor
    """
    b, c, h, w = tensor_448.shape
    kernel = get_antialias_bilinear_2d_kernel(tensor_448.device)
    kernel_weights = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    # Use reflection padding to avoid boundary darkening
    padded = F.pad(tensor_448, (1, 1, 1, 1), mode="reflect")
    filtered = F.conv2d(padded, kernel_weights, groups=c)
    return filtered


def generate_m7_suite(tensor_448: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Generates the complete four-condition M7 suite for a 448x448 input tensor:
    - Condition A: Original 448 input (784 tokens)
    - Condition B: 448 input + exact antialiasing low-pass filter without decimation (784 tokens)
    - Condition C: Actual 224 downsampled input (196 tokens)
    - Condition D: 224 downsampled input upsampled back to 448 (784 tokens)
    """
    cond_a = tensor_448.clone()
    cond_b = apply_m7_filter_without_decimation(tensor_448)
    cond_c = resize_tensor_torch(tensor_448, target_size=224)
    # Upsample Condition C back to 448 via bilinear interpolation
    cond_d = F.interpolate(
        cond_c,
        size=(448, 448),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    return {
        "A_orig448": cond_a,
        "B_filtered448": cond_b,
        "C_down224": cond_c,
        "D_up448": cond_d,
    }
