"""FLOP counting utility using torch.utils.flop_counter.FlopCounterMode.

Reports GFLOPs per image explicitly under the 2x Multiply-Accumulate (MAC) convention.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None


def count_gflops(
    model: nn.Module,
    input_resolution: int = 224,
    batch_size: int = 1,
    device: torch.device = None,
) -> float:
    """Counts floating-point operations (in GFLOPs, 2x MAC convention) for a model forward pass.
    
    Args:
        model: PyTorch module
        input_resolution: image size (H=W)
        batch_size: input batch size
        device: execution device
        
    Returns:
        float: GFLOPs per image (10^9 operations)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    model.eval()
    dummy_input = torch.randn(batch_size, 3, input_resolution, input_resolution, device=device)
    
    if FlopCounterMode is not None:
        with FlopCounterMode(display=False) as flop_counter:
            with torch.no_grad():
                _ = model(dummy_input)
        total_flops = flop_counter.get_total_flops()
        # Per image GFLOPs
        gflops_per_img = (total_flops / batch_size) / 1e9
        return float(gflops_per_img)
    else:
        # Fallback approximation based on parameter count and feature map sizes
        total_params = sum(p.numel() for p in model.parameters())
        # Heuristic order of magnitude estimate
        approx_gflops = (total_params * (input_resolution / 224.0) ** 2 * 2.0) / 1e9
        return float(approx_gflops)
