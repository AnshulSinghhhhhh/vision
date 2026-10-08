import os
import pytest
import torch
import torch.nn as nn
from knobs.models import recalibrate_batchnorm, assert_clean_sanity


class TinyBNNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 16 * 16, num_classes)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_recalibrate_batchnorm_running_stats():
    torch.manual_seed(42)
    net = TinyBNNet()
    orig_mean = net.bn.running_mean.clone()
    orig_var = net.bn.running_var.clone()
    
    # Generate calibration data with shifted distribution
    cal_data = torch.randn(100, 3, 16, 16) + 5.0
    
    # Recalibrate
    recalibrate_batchnorm(net, cal_data, batch_size=32, drop_remainder=True)
    
    # Verify running statistics updated away from initial (0, 1)
    assert not torch.allclose(net.bn.running_mean, orig_mean)
    assert not torch.allclose(net.bn.running_var, orig_var)
    assert net.training is False
    assert net.bn.training is False


def test_recalibrate_batchnorm_with_resolution():
    torch.manual_seed(42)
    net = TinyBNNet()
    # Input has resolution 32x32, target calibration_resolution is 16x16
    cal_data = torch.randn(64, 3, 32, 32)
    recalibrate_batchnorm(net, cal_data, batch_size=32, calibration_resolution=16)
    assert net.training is False


def test_assert_clean_sanity_pass():
    torch.manual_seed(42)
    net = TinyBNNet()
    images = torch.randn(50, 3, 16, 16)
    labels = torch.randint(0, 2, (50,))
    
    # Recalibrate on same distribution
    recalibrate_batchnorm(net, images, batch_size=25, drop_remainder=True)
    
    # Should pass sanity check when compared against itself or high reference
    acc = assert_clean_sanity(net, images, labels, original_model=net, tol_pp=1.0)
    assert 0.0 <= acc <= 100.0


def test_assert_clean_sanity_failure():
    torch.manual_seed(42)
    net = TinyBNNet()
    images = torch.randn(50, 3, 16, 16)
    labels = torch.randint(0, 2, (50,))
    
    # If reference accuracy is 100% and model achieves < 99%, should raise RuntimeError
    with pytest.raises(RuntimeError, match="Clean sanity check failed"):
        assert_clean_sanity(net, images, labels, original_model=100.0, tol_pp=0.01)
