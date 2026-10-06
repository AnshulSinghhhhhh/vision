"""CUDA latency and throughput benchmarking harness.

Standard protocol:
- CUDA events with torch.cuda.synchronize() around every measurement.
- 50 warmup runs, 200 timed runs.
- Batch sizes 1 and 64, FP16 autocast.
- Median and Interquartile Range (IQR).
- Executed strictly on Device 0 only; logs GPU device name.
"""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn


def benchmark_model_latency(
    model: nn.Module,
    resolution: int = 224,
    batch_size: int = 64,
    n_warmup: int = 50,
    n_timed: int = 200,
    device: torch.device = torch.device("cuda:0"),
    use_fp16: bool = True,
) -> Dict[str, Any]:
    """Measures model latency and throughput under standard protocol."""
    model.eval()
    model.to(device)
    
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    
    x = torch.randn(batch_size, 3, resolution, resolution, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            if is_cuda and use_fp16:
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
            
    # Timed runs
    timings_ms: List[float] = []
    
    if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for _ in range(n_timed):
                start_event.record()
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        _ = model(x)
                else:
                    _ = model(x)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                timings_ms.append(elapsed_ms)
    else:
        import time
        with torch.no_grad():
            for _ in range(n_timed):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                timings_ms.append((t1 - t0) * 1000.0)
                
    timings = np.array(timings_ms)
    median_batch_ms = float(np.median(timings))
    q25 = float(np.percentile(timings, 25))
    q75 = float(np.percentile(timings, 75))
    iqr_ms = float(q75 - q25)
    
    # Throughput (images per second)
    throughput_ips = float((batch_size / (median_batch_ms / 1000.0)))
    
    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "resolution": resolution,
        "batch_size": batch_size,
        "use_fp16": use_fp16,
        "median_batch_latency_ms": median_batch_ms,
        "per_image_latency_ms": median_batch_ms / batch_size,
        "iqr_ms": iqr_ms,
        "throughput_img_per_sec": throughput_ips,
    }
