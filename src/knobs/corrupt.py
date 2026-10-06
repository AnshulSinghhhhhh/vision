"""Deterministic ImageNet-C style corruptions applied in 448x448 acquisition space.

Features:
- Deterministic seeding per (image_id, corruption, severity) via SHA-256 hash.
- Unscaled physical corruption directly in 448 coordinate space before resolution transformations.
- 4 primary frequency-representative corruptions:
  - gaussian_noise (broadband high-frequency)
  - defocus_blur (low-pass bandlimited)
  - jpeg_compression (high-frequency 8x8 block artifacts)
  - contrast (frequency-neutral control)
- Frequency-controlled noise experiment:
  - Broadband noise at 448
  - Bandlimited noise generated at 224 and upsampled to 448 with matched RMS power.
"""

import hashlib
import random
from typing import Tuple, Optional
import numpy as np
from PIL import Image

# Import imagecorruptions if available, otherwise fallback
try:
    from imagecorruptions import corrupt as ic_corrupt
except ImportError:
    ic_corrupt = None

PRIMARY_CORRUPTIONS = [
    "gaussian_noise",
    "defocus_blur",
    "jpeg_compression",
    "contrast",
]

CORRUPTION_GROUPS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def compute_seed(image_id: str, corruption: str, severity: int, salt: int = 0) -> int:
    """Generates a 32-bit deterministic integer seed from (image_id, corruption, severity, salt)."""
    key = f"{image_id}|{corruption}|{severity}|{salt}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Take first 8 hex characters -> 32-bit uint
    return int(digest[:8], 16)


class SeedContext:
    """Context manager to isolate global random seeds for deterministic corruptions."""
    def __init__(self, seed: int):
        self.seed = seed
        self.np_state = None
        self.py_state = None

    def __enter__(self):
        self.np_state = np.random.get_state()
        self.py_state = random.getstate()
        np.random.seed(self.seed % (2**32 - 1))
        random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.np_state is not None:
            np.random.set_state(self.np_state)
        if self.py_state is not None:
            random.setstate(self.py_state)


def apply_corruption(
    img_448: np.ndarray,
    image_id: str,
    corruption_name: str,
    severity: int,
    salt: int = 0,
) -> np.ndarray:
    """Applies a corruption deterministically to an image in 448x448 space.
    
    Args:
        img_448: uint8 RGB numpy array of shape (448, 448, 3)
        image_id: unique identifier of image
        corruption_name: corruption name (e.g. 'gaussian_noise', 'clean')
        severity: severity level 1..5 (ignored if 'clean')
        salt: integer salt for repeated observation experiments
        
    Returns:
        uint8 RGB numpy array of shape (448, 448, 3)
    """
    if corruption_name.lower() in ("clean", "none") or severity <= 0:
        return img_448.copy()
        
    seed = compute_seed(image_id, corruption_name, severity, salt=salt)
    
    with SeedContext(seed):
        if ic_corrupt is not None:
            # imagecorruptions takes uint8 array (H, W, 3)
            corrupted = ic_corrupt(img_448, corruption_name=corruption_name, severity=severity)
            return np.ascontiguousarray(corrupted, dtype=np.uint8)
        else:
            # Fallback simple deterministic implementations if imagecorruptions is missing
            return _fallback_corrupt(img_448, corruption_name, severity)


def _fallback_corrupt(img: np.ndarray, corruption_name: str, severity: int) -> np.ndarray:
    """Fallback basic implementations of primary corruptions for testing environments."""
    img_float = img.astype(np.float32) / 255.0
    if corruption_name == "gaussian_noise":
        sigmas = [0.08, 0.12, 0.18, 0.26, 0.38]
        sigma = sigmas[min(severity - 1, len(sigmas) - 1)]
        noise = np.random.normal(0, sigma, img.shape)
        out = np.clip(img_float + noise, 0.0, 1.0) * 255.0
        return out.astype(np.uint8)
    elif corruption_name == "contrast":
        factors = [0.4, 0.3, 0.2, 0.1, 0.05]
        factor = factors[min(severity - 1, len(factors) - 1)]
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        out = np.clip((img_float - mean) * factor + mean, 0.0, 1.0) * 255.0
        return out.astype(np.uint8)
    elif corruption_name == "defocus_blur":
        from scipy.ndimage import gaussian_filter
        sigmas = [1.0, 2.0, 3.0, 4.0, 6.0]
        sigma = sigmas[min(severity - 1, len(sigmas) - 1)]
        out = np.zeros_like(img_float)
        for c in range(3):
            out[..., c] = gaussian_filter(img_float[..., c], sigma=sigma)
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif corruption_name == "jpeg_compression":
        qualities = [25, 18, 15, 10, 7]
        quality = qualities[min(severity - 1, len(qualities) - 1)]
        pil_img = Image.fromarray(img)
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown corruption name: {corruption_name}")


def generate_frequency_controlled_noise(
    img_448: np.ndarray,
    image_id: str,
    noise_mode: str,
    target_rms: float = 30.0,
    salt: int = 0,
) -> np.ndarray:
    """Generates frequency-controlled noise on 448x448 image with matched RMS power.
    
    Modes:
    - 'broadband': white Gaussian noise injected directly at 448x448
    - 'bandlimited': white Gaussian noise generated at 224x224, upsampled to 448x448
      via bilinear interpolation, and calibrated to matched target RMS power.
    """
    seed = compute_seed(image_id, f"freq_noise_{noise_mode}", int(target_rms), salt=salt)
    with SeedContext(seed):
        h, w, c = img_448.shape
        if noise_mode == "broadband":
            noise = np.random.normal(0, 1.0, (h, w, c)).astype(np.float32)
            # Normalize to target RMS
            current_rms = np.sqrt(np.mean(noise ** 2))
            noise = noise * (target_rms / (current_rms + 1e-8))
        elif noise_mode == "bandlimited":
            # Generate at 224x224
            noise_224 = np.random.normal(0, 1.0, (224, 224, c)).astype(np.float32)
            # Upsample to 448x448 with bilinear interpolation
            noise_upsampled = np.zeros((h, w, c), dtype=np.float32)
            for ch in range(c):
                pil_n = Image.fromarray(noise_224[..., ch], mode="F")
                pil_up = pil_n.resize((w, h), resample=Image.Resampling.BILINEAR)
                noise_upsampled[..., ch] = np.array(pil_up)
            # Calibrate to matched target RMS power
            current_rms = np.sqrt(np.mean(noise_upsampled ** 2))
            noise = noise_upsampled * (target_rms / (current_rms + 1e-8))
        else:
            raise ValueError(f"Unknown noise_mode: {noise_mode}")
            
        corrupted = np.clip(img_448.astype(np.float32) + noise, 0.0, 255.0)
        return corrupted.astype(np.uint8)
