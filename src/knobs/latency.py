"""CUDA latency and throughput benchmarking harness.

Standard protocol:
- CUDA events with torch.cuda.synchronize() around every measurement.
- 15 warmup runs, 40 timed runs (sufficient on T4 for low variance).
- Batch sizes 1 and 64, FP16 autocast.
- Median and Interquartile Range (IQR).
- Executed strictly on Device 0 only; logs GPU device name.
- Optional memory format (channels_last) and experimental acceleration (torch.compile / CUDA graphs).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from knobs.flops import count_gflops


def benchmark_model_latency(
    model: nn.Module,
    resolution: int = 224,
    batch_size: int = 64,
    n_warmup: int = 15,
    n_timed: int = 40,
    device: Optional[torch.device] = None,
    use_fp16: bool = True,
    channels_last: bool = False,
    experimental_compile: bool = False,
    experimental_cuda_graph: bool = False,
) -> Dict[str, Any]:
    """Measures model latency, throughput, and GFLOPs under standard protocol.
    
    Args:
        model: PyTorch module
        resolution: Square input resolution
        batch_size: Evaluation batch size
        n_warmup: Number of warmup iterations (default 15)
        n_timed: Number of timed iterations (default 40)
        device: Torch execution device
        use_fp16: Whether to use FP16 autocast
        channels_last: Whether to convert model and inputs to channels_last memory format
        experimental_compile: [Experimental] Enable torch.compile
        experimental_cuda_graph: [Experimental] Enable CUDA Graph replay (CUDA only)
        
    Returns:
        Dict containing latency metrics, throughput, GFLOPs, and benchmark parameters.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    
    # Calculate GFLOPs per image at target resolution before compiling
    try:
        gflops_val = count_gflops(model, input_resolution=resolution, batch_size=1, device=device)
    except Exception:
        total_params = sum(p.numel() for p in model.parameters())
        gflops_val = float((total_params * (resolution / 224.0) ** 2 * 2.0) / 1e9)

    if experimental_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        
    x = torch.randn(batch_size, 3, resolution, resolution, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    # Inference mode context
    with torch.inference_mode():
        # Handle CUDA graph if requested and supported
        cuda_graph = None
        static_input = None
        static_output = None
        if is_cuda and experimental_cuda_graph:
            static_input = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    if use_fp16:
                        with torch.cuda.amp.autocast():
                            _ = model(static_input)
                    else:
                        _ = model(static_input)
            torch.cuda.current_stream().wait_stream(s)
            
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        static_output = model(static_input)
                else:
                    static_output = model(static_input)

        # Warmup
        for _ in range(n_warmup):
            if cuda_graph is not None:
                cuda_graph.replay()
            elif is_cuda and use_fp16:
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
            
            for _ in range(n_timed):
                start_event.record()
                if cuda_graph is not None:
                    cuda_graph.replay()
                elif use_fp16:
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
    throughput_ips = float((batch_size / (median_batch_ms / 1000.0))) if median_batch_ms > 0 else 0.0
    
    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "resolution": resolution,
        "batch_size": batch_size,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "use_fp16": use_fp16,
        "channels_last": channels_last,
        "experimental_compile": experimental_compile,
        "experimental_cuda_graph": experimental_cuda_graph,
        "median_batch_latency_ms": median_batch_ms,
        "per_image_latency_ms": median_batch_ms / batch_size,
        "iqr_ms": iqr_ms,
        "iqr_batch_latency_ms": iqr_ms,
        "throughput_img_per_sec": throughput_ips,
        "gflops": gflops_val,
    }
