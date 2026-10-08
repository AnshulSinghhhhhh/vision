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

import os
import hashlib
import random
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image

# Import imagecorruptions if available, otherwise fallback
try:
    import imagecorruptions
    from imagecorruptions import corrupt as ic_corrupt
    CORRUPTION_BACKEND = "imagecorruptions"
except Exception:
    imagecorruptions = None
    ic_corrupt = None
    CORRUPTION_BACKEND = "fallback"

FALLBACK_GAUSSIAN_NOISE_SIGMAS = [0.08, 0.12, 0.18, 0.26, 0.38]
FALLBACK_CONTRAST_FACTORS = [0.4, 0.3, 0.2, 0.1, 0.05]
FALLBACK_DEFOCUS_BLUR_SIGMAS = [1.0, 2.0, 3.0, 4.0, 6.0]
FALLBACK_JPEG_QUALITIES = [25, 18, 15, 10, 7]

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
        if ic_corrupt is not None and CORRUPTION_BACKEND == "imagecorruptions":
            # imagecorruptions takes uint8 array (H, W, 3)
            corrupted = ic_corrupt(img_448, corruption_name=corruption_name, severity=severity)
            return np.ascontiguousarray(corrupted, dtype=np.uint8)
        else:
            # Fallback simple deterministic implementations if imagecorruptions is missing
            if os.environ.get("KNOBS_ALLOW_FALLBACK", "0") != "1":
                raise RuntimeError(
                    "Corruption backend is 'fallback' because imagecorruptions is unavailable. "
                    "Set environment variable KNOBS_ALLOW_FALLBACK=1 to explicitly allow fallback corruptions."
                )
            return _fallback_corrupt(img_448, corruption_name, severity)


def _fallback_corrupt(img: np.ndarray, corruption_name: str, severity: int) -> np.ndarray:
    """Fallback basic implementations of primary corruptions for testing environments."""
    img_float = img.astype(np.float32) / 255.0
    if corruption_name == "gaussian_noise":
        sigmas = FALLBACK_GAUSSIAN_NOISE_SIGMAS
        sigma = sigmas[min(severity - 1, len(sigmas) - 1)]
        noise = np.random.normal(0, sigma, img.shape)
        out = np.clip(img_float + noise, 0.0, 1.0) * 255.0
        return out.astype(np.uint8)
    elif corruption_name == "contrast":
        factors = FALLBACK_CONTRAST_FACTORS
        factor = factors[min(severity - 1, len(factors) - 1)]
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        out = np.clip((img_float - mean) * factor + mean, 0.0, 1.0) * 255.0
        return out.astype(np.uint8)
    elif corruption_name == "defocus_blur":
        from scipy.ndimage import gaussian_filter
        sigmas = FALLBACK_DEFOCUS_BLUR_SIGMAS
        sigma = sigmas[min(severity - 1, len(sigmas) - 1)]
        out = np.zeros_like(img_float)
        for c in range(3):
            out[..., c] = gaussian_filter(img_float[..., c], sigma=sigma)
        return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif corruption_name == "jpeg_compression":
        qualities = FALLBACK_JPEG_QUALITIES
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
    image_id: str = "default",
    noise_mode: Optional[str] = None,
    band: Optional[str] = None,
    target_rms: float = 30.0,
    sigma_total: Optional[float] = None,
    severity: int = 3,
    seed: Optional[int] = None,
    salt: int = 0,
) -> np.ndarray:
    """Generates frequency-controlled noise on 448x448 image with matched RMS power.
    
    Modes/Bands:
    - 'broadband' / 'high': white Gaussian noise injected directly at 448x448
    - 'bandlimited' / 'low': white Gaussian noise generated at 224x224, upsampled to 448x448
      via bilinear interpolation, and calibrated to matched target RMS power.
    - 'mid': bandpass noise calibrated to matched target RMS power.
    """
    mode = (band or noise_mode or "broadband").lower()
    rms = sigma_total if sigma_total is not None else target_rms
    
    if seed is None:
        seed = compute_seed(image_id, f"freq_noise_{mode}", severity, salt=salt)
        
    rng = np.random.RandomState(seed % (2**32 - 1))
    h, w, c = img_448.shape
    if mode in ("broadband", "high"):
        noise = rng.normal(0, 1.0, (h, w, c)).astype(np.float32)
        current_rms = np.sqrt(np.mean(noise ** 2))
        noise = noise * (rms / (current_rms + 1e-8))
    elif mode in ("bandlimited", "low"):
        noise_224 = rng.normal(0, 1.0, (224, 224, c)).astype(np.float32)
        noise_upsampled = np.zeros((h, w, c), dtype=np.float32)
        for ch in range(c):
            pil_n = Image.fromarray(noise_224[..., ch], mode="F")
            pil_up = pil_n.resize((w, h), resample=Image.Resampling.BILINEAR)
            noise_upsampled[..., ch] = np.array(pil_up)
        current_rms = np.sqrt(np.mean(noise_upsampled ** 2))
        noise = noise_upsampled * (rms / (current_rms + 1e-8))
    elif mode == "mid":
        n1 = rng.normal(0, 1.0, (336, 336, c)).astype(np.float32)
        up1 = np.zeros((h, w, c), dtype=np.float32)
        for ch in range(c):
            pil_n = Image.fromarray(n1[..., ch], mode="F")
            up1[..., ch] = np.array(pil_n.resize((w, h), resample=Image.Resampling.BILINEAR))
            
        n2 = rng.normal(0, 1.0, (168, 168, c)).astype(np.float32)
        up2 = np.zeros((h, w, c), dtype=np.float32)
        for ch in range(c):
            pil_n = Image.fromarray(n2[..., ch], mode="F")
            up2[..., ch] = np.array(pil_n.resize((w, h), resample=Image.Resampling.BILINEAR))
            
        noise_mid = up1 - up2
        current_rms = np.sqrt(np.mean(noise_mid ** 2))
        noise = noise_mid * (rms / (current_rms + 1e-8))
    else:
        raise ValueError(f"Unknown frequency mode/band: {mode}")
        
    corrupted = np.clip(img_448.astype(np.float32) + noise, 0.0, 255.0)
    return corrupted.astype(np.uint8)


FREQ_BANDS = {
    "low": (0.0, 56.0),
    "mid": (56.0, 112.0),
    "high": (112.0, 224.0),
    "broadband": (0.0, 224.0),
}


_FREQ_MASK_CACHE: Dict[Tuple[int, int, float, float], np.ndarray] = {}


def get_annular_freq_mask(h: int, w: int, r_min: float, r_max: float) -> np.ndarray:
    """Returns cached 2D annular frequency mask expanded to (H, W, 1)."""
    key = (h, w, float(r_min), float(r_max))
    if key not in _FREQ_MASK_CACHE:
        u = np.fft.fftfreq(h) * h
        v = np.fft.fftfreq(w) * w
        U, V = np.meshgrid(u, v, indexing="ij")
        R = np.sqrt(U**2 + V**2)
        if r_max < h // 2:
            mask = ((R >= r_min) & (R < r_max)).astype(np.float32)
        else:
            mask = ((R >= r_min) & (R <= r_max)).astype(np.float32)
        _FREQ_MASK_CACHE[key] = mask[:, :, None]
    return _FREQ_MASK_CACHE[key]


def generate_fft_bandlimited_noise(
    shape: Tuple[int, int, int] = (448, 448, 3),
    band: str = "broadband",
    target_rms: float = 45.9,
    seed: Optional[int] = None,
    custom_cutoff: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float]:
    """Generates 2D FFT bandpass filtered noise on specified frame shape (default 448x448x3).

    Annuli in cycles per image (for 448x448, Nyquist = 224 cyc/img):
    - 'low': [0, 56)
    - 'mid': [56, 112)
    - 'high': [112, 224]
    - 'broadband': [0, 224] (all frequencies)

    Args:
        shape: (H, W, C) shape of noise
        band: one of {'low', 'mid', 'high', 'broadband'}
        target_rms: desired pre-clip root-mean-square amplitude
        seed: optional 32-bit random seed for reproducibility
        custom_cutoff: optional (r_min, r_max) cutoff frequency tuple

    Returns:
        (noise_float32, pre_clip_rms)
    """
    h, w, c = shape
    if custom_cutoff is not None:
        r_min, r_max = custom_cutoff
    elif band.lower() in FREQ_BANDS:
        r_min, r_max = FREQ_BANDS[band.lower()]
    else:
        raise ValueError(f"Unknown frequency band: {band}. Choose from {list(FREQ_BANDS.keys())}")

    if seed is not None:
        rng = np.random.RandomState(seed % (2**32 - 1))
    else:
        rng = np.random.RandomState()

    w_noise = rng.normal(0, 1.0, shape).astype(np.float32)

    if band.lower() == "broadband" and custom_cutoff is None:
        noise = w_noise
    else:
        mask = get_annular_freq_mask(h, w, r_min, r_max)
        W = np.fft.fft2(w_noise, axes=(0, 1))
        W *= mask
        noise = np.fft.ifft2(W, axes=(0, 1)).real.astype(np.float32)
        del W, w_noise

    current_rms = float(np.sqrt(np.mean(noise**2)))
    if current_rms > 1e-12:
        noise = noise * (target_rms / current_rms)
        pre_clip_rms = float(target_rms)
    else:
        pre_clip_rms = 0.0

    return noise, pre_clip_rms


def apply_fft_frequency_noise(
    img_448: np.ndarray,
    image_id: str,
    band: str = "broadband",
    severity: int = 3,
    target_rms: Optional[float] = None,
    salt: int = 0,
) -> Tuple[np.ndarray, float, float]:
    """Applies FFT bandlimited noise to an image with per-image seeding.

    Pre-clip RMS levels:
    - severity 3: 0.18 * 255.0 = 45.9
    - severity 5: 0.38 * 255.0 = 96.9

    Returns:
        (corrupted_uint8, pre_clip_rms, post_clip_rms)
    """
    if target_rms is None:
        rms_map = {1: 0.08 * 255.0, 2: 0.12 * 255.0, 3: 0.18 * 255.0, 4: 0.26 * 255.0, 5: 0.38 * 255.0}
        target_rms = rms_map.get(severity, 0.18 * 255.0)

    seed = compute_seed(image_id, f"freq_{band}", severity, salt=salt)
    noise, pre_clip_rms = generate_fft_bandlimited_noise(
        shape=img_448.shape,
        band=band,
        target_rms=target_rms,
        seed=seed,
    )

    img_f = img_448.astype(np.float32)
    corrupted_f = np.clip(img_f + noise, 0.0, 255.0)
    corrupted_uint8 = corrupted_f.astype(np.uint8)

    effective_noise = corrupted_uint8.astype(np.float32) - img_f
    post_clip_rms = float(np.sqrt(np.mean(effective_noise**2)))
    del noise, img_f, corrupted_f, effective_noise

    return corrupted_uint8, pre_clip_rms, post_clip_rms

