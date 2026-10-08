import sys
import importlib.util
from pathlib import Path
import torch
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

k12_path = repo_root / "kaggle" / "kernels" / "k12-tome-matched" / "run.py"
spec = importlib.util.spec_from_file_location("k12_run", k12_path)
k12_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k12_mod)


def test_measure_throughput():
    device = torch.device("cpu")
    # Quick test on a tiny dummy module or DeiT
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(16, 1000))
    th = k12_mod.measure_throughput_img_per_s(m, resolution=224, batch_size=4, device=device, warmup_iters=1, timed_iters=2)
    assert th > 0.0, f"Throughput must be positive, got {th}"


def test_tome_matched_configs_monotonicity():
    device = torch.device("cpu")
    configs = k12_mod.build_tome_matched_configs(device=device, pretrained=False, profile_throughput=False)

    assert "res_448_base" in configs
    assert "tome_r32_448" in configs
    assert "tome_r64_448" in configs
    assert "res_224_base" in configs

    gf_448 = configs["res_448_base"]["gflops"]
    gf_t32 = configs["tome_r32_448"]["gflops"]
    gf_t64 = configs["tome_r64_448"]["gflops"]
    gf_224 = configs["res_224_base"]["gflops"]

    # Verify FLOP reductions
    assert gf_448 > gf_t32 > gf_t64 > gf_224, f"FLOPs ordering violated: {gf_448} > {gf_t32} > {gf_t64} > {gf_224}"
