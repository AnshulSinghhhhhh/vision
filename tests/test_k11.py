import sys
import importlib.util
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

k11_path = repo_root / "kaggle" / "kernels" / "k11-sensor-noise" / "run.py"
spec = importlib.util.spec_from_file_location("k11_run", k11_path)
k11_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k11_mod)


def test_poisson_gaussian_noise_psnr():
    dummy = np.random.randint(20, 235, size=(448, 448, 3), dtype=np.uint8)
    noisy, pre_rms, psnr_db = k11_mod.generate_poisson_gaussian_noise(
        dummy,
        target_sigma=0.18,
        shot_fraction=0.5,
        seed=42,
    )
    assert noisy.shape == (448, 448, 3)
    # Target PSNR for sigma=0.18 is 10 * log10(1 / 0.18^2) ~ 14.89 dB
    expected_psnr = 10.0 * np.log10(1.0 / (0.18**2))
    assert abs(psnr_db - expected_psnr) < 0.1, f"PSNR {psnr_db:.2f} deviates from expected {expected_psnr:.2f}"


def test_poisson_gaussian_determinism():
    dummy = np.random.randint(20, 235, size=(448, 448, 3), dtype=np.uint8)
    noisy1, _, _ = k11_mod.generate_poisson_gaussian_noise(dummy, target_sigma=0.18, seed=100)
    noisy2, _, _ = k11_mod.generate_poisson_gaussian_noise(dummy, target_sigma=0.18, seed=100)
    noisy3, _, _ = k11_mod.generate_poisson_gaussian_noise(dummy, target_sigma=0.18, seed=200)

    assert np.array_equal(noisy1, noisy2), "Same seed must produce identical sensor noise"
    assert not np.array_equal(noisy1, noisy3), "Different seeds must produce different sensor noise"
