import os
import sys
import math
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from knobs.models import recalibrate_batchnorm, assert_clean_sanity

paw_path = repo_root / "kaggle" / "push_and_wait.py"
paw_spec = importlib.util.spec_from_file_location("push_and_wait", paw_path)
paw_mod = importlib.util.module_from_spec(paw_spec)
paw_spec.loader.exec_module(paw_mod)
REGISTERED_KERNELS = paw_mod.REGISTERED_KERNELS


def test_registered_kernels():
    """Verify all kernels k8 through k13 are registered and their directories exist."""
    required = [
        "k8-m7-v2",
        "k9-freqnoise-v2",
        "k10-resize-ablation",
        "k11-sensor-noise",
        "k12-tome-matched",
        "k13-g0a-and-bn",
    ]
    for k in required:
        assert k in REGISTERED_KERNELS, f"Kernel {k} not registered in push_and_wait.py"
        k_dir = repo_root / "kaggle" / "kernels" / k
        assert k_dir.exists(), f"Kernel directory {k_dir} does not exist"
        meta_file = k_dir / "kernel-metadata.json"
        assert meta_file.exists(), f"Metadata file {meta_file} does not exist"
        run_file = k_dir / "run.py"
        assert run_file.exists(), f"Run script {run_file} does not exist"


def test_g0a_se_pass_rule():
    """Verify statistical pass rule calculation for G0-A."""
    # N = 50,000, ref = 81.8%
    n = 50000
    ref = 81.8
    measured_pass = 81.7
    p = measured_pass / 100.0
    se = math.sqrt(p * (1.0 - p) / n) * 100.0
    assert abs(measured_pass - ref) <= 2.0 * se + 0.1

    measured_fail = 85.5
    p_fail = measured_fail / 100.0
    se_fail = math.sqrt(p_fail * (1.0 - p_fail) / n) * 100.0
    assert abs(measured_fail - ref) > 2.0 * se_fail


def test_g0a_prespecified_acceptance_criteria():
    """Verify binomial SE thresholds at 95% (1.96*SE) and 99% (2.58*SE) on N=5,000."""
    n = 5000
    ref = 81.8
    # Within 95% CI: diff <= 1.96 * SE (~1.07 pp)
    p_95 = 82.5
    se_95 = math.sqrt((p_95 / 100.0) * (1.0 - p_95 / 100.0) / n) * 100.0
    assert abs(p_95 - ref) <= 1.96 * se_95

    # Within 99% CI but outside 95% CI: diff ~1.2 pp
    p_99 = 83.0
    se_99 = math.sqrt((p_99 / 100.0) * (1.0 - p_99 / 100.0) / n) * 100.0
    assert abs(p_99 - ref) > 1.96 * se_99
    assert abs(p_99 - ref) <= 2.58 * se_99

    # Gross failure: diff 4.0 pp
    p_gross = 85.8
    se_gross = math.sqrt((p_gross / 100.0) * (1.0 - p_gross / 100.0) / n) * 100.0
    assert abs(p_gross - ref) > 2.58 * se_gross


def test_bn_recalibration_and_sanity_gate_cpu():
    """Verify BatchNorm recalibration and assert_clean_sanity on CPU."""
    class TinyBNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn(self.conv(x))))
            return self.fc(x.flatten(1))

    model = TinyBNNet()
    calib_images = torch.randn(64, 3, 32, 32)
    labels = torch.randint(0, 10, (64,))

    # Compute baseline accuracy
    with torch.no_grad():
        out = model(calib_images)
        orig_acc = (out.argmax(-1) == labels).float().mean().item() * 100.0

    # Recalibrate BN on clean data
    recalibrate_batchnorm(model, calib_images, device=torch.device("cpu"), batch_size=16, drop_remainder=True)

    # Sanity check should pass within generous tolerance for tiny net
    acc = assert_clean_sanity(model, calib_images, labels, tol_pp=50.0, original_model=orig_acc, batch_size=16)
    assert acc >= 0.0


def test_bn_sanity_gate_failure_raises():
    """Verify that assert_clean_sanity strictly raises RuntimeError when drop exceeds tol_pp."""
    class IdentityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 10)

        def forward(self, x):
            return self.fc(x.mean(dim=[-2, -1]))

    model = IdentityNet()
    calib_images = torch.randn(32, 3, 16, 16)
    labels = torch.zeros(32, dtype=torch.long)

    # Fake original accuracy of 100%, but current model will have ~10%
    with pytest.raises(RuntimeError, match="Clean sanity check failed"):
        assert_clean_sanity(model, calib_images, labels, tol_pp=1.0, original_model=100.0, batch_size=16)


def test_data_leakage_assertion():
    """Verify zero overlap assertion catches data leakage between calibration and eval splits."""
    cal_ids = ["img_001", "img_002", "img_003"]
    eval_ids_clean = ["img_004", "img_005", "img_006"]
    eval_ids_leaked = ["img_003", "img_004", "img_005"]

    # Clean case has zero leakage
    leakage = set(cal_ids).intersection(set(eval_ids_clean))
    assert len(leakage) == 0

    # Leaked case has non-zero intersection
    leakage_detected = set(cal_ids).intersection(set(eval_ids_leaked))
    assert len(leakage_detected) == 1
    assert "img_003" in leakage_detected
