"""Tests for Token Merging (ToMe) wrapper and schedule validity."""

import torch
import pytest
import timm
from knobs.tokens import patch_vit_with_tome, solve_r_for_flop_ratio, get_tome_schedule_for_budget


def test_tome_r0_identity():
    """ToMe with r=0 produces logits identical to the unpatched model (diff < 1e-4 in fp32)."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        orig_logits = model(x)
        
    n_blocks = len(model.blocks)
    r_schedule = [0] * n_blocks
    patched_model = patch_vit_with_tome(model, r_schedule)
    
    with torch.no_grad():
        patched_logits = patched_model(x)
        
    max_diff = torch.max(torch.abs(orig_logits - patched_logits)).item()
    assert max_diff < 1e-4, f"ToMe r=0 differed from unpatched model: max diff = {max_diff}"


def test_tome_output_shape_and_token_count():
    """Verifies output shape and progressive token reduction under ToMe merging."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    n_blocks = len(model.blocks)
    r = 4
    patched_model = patch_vit_with_tome(model, [r] * n_blocks)

    b = 2
    x = torch.randn(b, 3, 224, 224)
    with torch.no_grad():
        out = patched_model(x)

    # 1. Output shape preserved
    assert out.shape == (b, 1000)

    # 2. Token count reduction: initial tokens = 14*14 + 1 = 197
    init_tokens = 197
    expected_final = init_tokens - n_blocks * r
    actual_final = patched_model._last_token_size.shape[1]
    assert actual_final == expected_final


def test_tome_size_conservation():
    """Verifies that token size sum equals original token count N across all layers."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    n_blocks = len(model.blocks)
    r = 6
    patched_model = patch_vit_with_tome(model, [r] * n_blocks)

    b = 3
    x = torch.randn(b, 3, 224, 224)
    with torch.no_grad():
        _ = patched_model(x)

    init_tokens = 197
    size_sums = patched_model._last_token_size.sum(dim=1).squeeze(-1)  # (B,)
    for b_idx in range(b):
        assert abs(size_sums[b_idx].item() - float(init_tokens)) < 1e-4


def test_solve_r_for_flop_ratio_monotonic():
    """Verifies solve_r_for_flop_ratio finds higher r for lower target FLOP ratios."""
    def factory():
        return timm.create_model("vit_tiny_patch16_224", pretrained=False)

    r_75, gflops_75, base_gflops = solve_r_for_flop_ratio(factory, resolution=224, ratio=0.75, max_r=16)
    r_50, gflops_50, _ = solve_r_for_flop_ratio(factory, resolution=224, ratio=0.50, max_r=16)

    assert r_50 >= r_75, f"r should be non-decreasing for lower target ratio: r_50={r_50}, r_75={r_75}"
    assert gflops_50 <= gflops_75, f"Measured GFLOPs should decrease: {gflops_50} vs {gflops_75}"
    assert gflops_75 < base_gflops
