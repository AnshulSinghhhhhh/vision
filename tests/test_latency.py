import pytest
import torch
import torch.nn as nn
from knobs.latency import benchmark_model_latency


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 16 * 16, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def test_benchmark_model_latency_cpu():
    net = TinyNet()
    bench = benchmark_model_latency(
        model=net,
        resolution=16,
        batch_size=2,
        n_warmup=2,
        n_timed=5,
        device=torch.device("cpu"),
        use_fp16=False,
    )
    
    assert bench["median_batch_latency_ms"] > 0
    assert bench["per_image_latency_ms"] > 0
    assert bench["throughput_img_per_sec"] > 0
    assert bench["gflops"] is not None and bench["gflops"] > 0
    assert bench["n_warmup"] == 2
    assert bench["n_timed"] == 5
    assert bench["channels_last"] is False


def test_benchmark_model_latency_channels_last():
    net = TinyNet()
    bench = benchmark_model_latency(
        model=net,
        resolution=16,
        batch_size=2,
        n_warmup=2,
        n_timed=5,
        device=torch.device("cpu"),
        use_fp16=False,
        channels_last=True,
    )
    assert bench["channels_last"] is True
    assert bench["median_batch_latency_ms"] > 0
    assert bench["gflops"] > 0
