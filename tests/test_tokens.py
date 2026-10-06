"""Tests for Token Merging (ToMe) wrapper and schedule validity."""

import torch
import pytest
import timm
from knobs.tokens import patch_vit_with_tome, get_tome_schedule_for_budget


def test_tome_r0_identity():
    """Test 6: ToMe with r=0 produces logits identical to the unpatched model (diff < 1e-3 in fp32)."""
    # Create small ViT model for CPU test
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        orig_logits = model(x)
        
    # Patch with r=0 schedule across all blocks
    n_blocks = len(model.blocks)
    r_schedule = [0] * n_blocks
    patched_model = patch_vit_with_tome(model, r_schedule)
    
    with torch.no_grad():
        patched_logits = patched_model(x)
        
    max_diff = torch.max(torch.abs(orig_logits - patched_logits)).item()
    assert max_diff < 1e-4, f"ToMe r=0 differed from unpatched model: max diff = {max_diff}"


def test_tome_schedule_token_reduction():
    """Verifies that validated ToMe schedules produce monotonic token reductions."""
    n_layers = 12
    initial_tokens = 785
    
    sched_75, info_75 = get_tome_schedule_for_budget(n_layers, initial_tokens, "75%")
    sched_50, info_50 = get_tome_schedule_for_budget(n_layers, initial_tokens, "50%")
    sched_25, info_25 = get_tome_schedule_for_budget(n_layers, initial_tokens, "25%")
    
    assert info_75["final_tokens"] > info_50["final_tokens"]
    assert info_50["final_tokens"] > info_25["final_tokens"]
    assert info_75["status"] == "validated_operating_regime"
    assert info_50["status"] == "validated_operating_regime"
    assert "exploratory" in info_25["status"]
