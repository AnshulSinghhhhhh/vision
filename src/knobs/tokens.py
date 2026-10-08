"""Token Merging (ToMe) implementation and matched-compute scheduling.

Features:
- Pure PyTorch bipartite soft matching for Vision Transformer blocks.
- Size-weighted token merging with proportional attention.
- Validated compute operating regimes (~50%, ~75% FLOP targets).
- Exploratory ~25% target with operating-range validation.
- Verifiable r=0 identity property (produces identical logits to unpatched model).
"""

from typing import Tuple, Callable, Optional, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from knobs.flops import count_gflops


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = True,
) -> Tuple[Callable, Callable]:
    """Computes bipartite soft matching between token subsets A and B.
    
    Args:
        metric: token features for similarity matching (B, N, D), e.g. mean of keys.
        r: number of tokens to merge
        class_token: whether first token is CLS
    """
    b, n, d = metric.shape
    if r <= 0:
        return lambda x, s: (x, s), lambda x: x

    offset = 1 if class_token else 0
    num_tokens = n - offset
    r = min(r, num_tokens // 2)
    if r <= 0:
        return lambda x, s: (x, s), lambda x: x

    tokens = metric[:, offset:, :]
    norm_tokens = F.normalize(tokens, p=2, dim=-1)

    # Partition into subsets A (even) and B (odd)
    a = norm_tokens[:, 0::2, :]
    b_toks = norm_tokens[:, 1::2, :]
    
    r = min(r, a.shape[1], b_toks.shape[1])
    if r <= 0:
        return lambda x, s: (x, s), lambda x: x
    
    # Cosine similarity matrix (B, len_a, len_b)
    sim = torch.bmm(a, b_toks.transpose(1, 2))
    
    # Best match in B for each token in A
    best_sim, best_idx = sim.max(dim=-1)  # (B, len_a)
    
    # Pick top-r matches
    _, topk_a = best_sim.topk(k=r, dim=-1, largest=True)  # (B, r)
    topk_b = torch.gather(best_idx, 1, topk_a)  # (B, r)

    def merge(x: torch.Tensor, size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merges tokens in x (B, N, C) and updates sizes (B, N, 1)."""
        cls_tok = x[:, :offset, :] if offset > 0 else None
        cls_sz = size[:, :offset, :] if offset > 0 else None
        
        x_patches = x[:, offset:, :]
        sz_patches = size[:, offset:, :]
        
        x_a = x_patches[:, 0::2, :]
        x_b = x_patches[:, 1::2, :].clone()
        sz_a = sz_patches[:, 0::2, :]
        sz_b = sz_patches[:, 1::2, :].clone()
        
        c = x.shape[-1]
        
        # Source tokens from A
        src_sz = sz_a.gather(1, topk_a.unsqueeze(-1))  # (B, r, 1)
        src_x = x_a.gather(1, topk_a.unsqueeze(-1).expand(-1, -1, c))  # (B, r, C)
        
        # Accumulate into B using scatter_add_ to handle any duplicate target indices in B
        dst_idx_sz = topk_b.unsqueeze(-1)  # (B, r, 1)
        dst_idx_x = topk_b.unsqueeze(-1).expand(-1, -1, c)  # (B, r, C)
        
        # Weighted accumulation: new_x_b * new_sz_b = x_b * sz_b + sum(src_x * src_sz)
        x_b_weighted = x_b * sz_b
        x_b_weighted.scatter_add_(1, dst_idx_x, src_x * src_sz)
        
        new_sz_b = sz_b.clone()
        new_sz_b.scatter_add_(1, dst_idx_sz, src_sz)
        
        new_x_b = x_b_weighted / (new_sz_b + 1e-8)
        
        # Remove merged tokens from A
        mask_a = torch.ones(b, x_a.shape[1], dtype=torch.bool, device=x.device)
        mask_a.scatter_(1, topk_a, False)
        
        rem_len = x_a.shape[1] - r
        if rem_len > 0:
            rem_a = x_a[mask_a].view(b, rem_len, c)
            rem_sz_a = sz_a[mask_a].view(b, rem_len, 1)
        else:
            rem_a = x_a.new_empty(b, 0, c)
            rem_sz_a = sz_a.new_empty(b, 0, 1)
        
        merged_x = torch.cat([rem_a, new_x_b], dim=1)
        merged_sz = torch.cat([rem_sz_a, new_sz_b], dim=1)
        
        if offset > 0:
            final_x = torch.cat([cls_tok, merged_x], dim=1)
            final_sz = torch.cat([cls_sz, merged_sz], dim=1)
        else:
            final_x = merged_x
            final_sz = merged_sz
            
        return final_x, final_sz

    return merge, None


class ToMeAttention(nn.Module):
    """Attention wrapper implementing proportional attention and key metric computation."""
    def __init__(self, original_attn: nn.Module):
        super().__init__()
        self.original_attn = original_attn

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = self.original_attn
        B, N, C = x.shape
        gate = attn.gate(x).sigmoid() if getattr(attn, "gate", None) is not None else None
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)

        # Published ToMe metric: average of keys across heads -> (B, N, head_dim)
        metric = k.mean(dim=1)

        # Proportional attention bias: log(size)
        attn_mask = None
        if size is not None and not torch.all(size == 1):
            log_size = torch.log(size.view(B, 1, 1, N).clamp(min=1e-8))
            attn_mask = log_size

        if attn.fused_attn:
            x_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=attn.attn_drop.p if attn.training else 0.,
            )
        else:
            q = q * attn.scale
            sim = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                sim = sim + attn_mask
            sim = sim.softmax(dim=-1)
            sim = attn.attn_drop(sim)
            x_out = sim @ v

        x_out = x_out.transpose(1, 2).reshape(B, N, attn.attn_dim)
        x_out = attn.norm(x_out)
        if gate is not None:
            x_out = x_out * gate
        x_out = attn.proj(x_out)
        x_out = attn.proj_drop(x_out)
        return x_out, metric


class ToMeBlock(nn.Module):
    """Wrapper around a timm ViT Block that performs ToMe token merging between attention and MLP."""
    def __init__(self, original_block: nn.Module, r: int = 0):
        super().__init__()
        self.original_block = original_block
        self.r = r
        self.tome_attn = ToMeAttention(original_block.attn)

    def forward(self, x: torch.Tensor, size: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if size is None:
            size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        # 1. Attention on normalized x
        x_attn, metric = self.tome_attn(self.original_block.norm1(x), size=size)
        
        if hasattr(self.original_block, "ls1"):
            x_attn = self.original_block.ls1(x_attn)
        if hasattr(self.original_block, "drop_path1"):
            x_attn = self.original_block.drop_path1(x_attn)
        elif hasattr(self.original_block, "drop_path"):
            x_attn = self.original_block.drop_path(x_attn)
            
        x = x + x_attn

        # 2. Token merging between attention and MLP on metric (mean of keys)
        if self.r > 0:
            merge_fn, _ = bipartite_soft_matching(metric, r=self.r, class_token=True)
            x, size = merge_fn(x, size)

        # 3. MLP on normalized x
        x_mlp = self.original_block.norm2(x)
        x_mlp = self.original_block.mlp(x_mlp)
        if hasattr(self.original_block, "ls2"):
            x_mlp = self.original_block.ls2(x_mlp)
        if hasattr(self.original_block, "drop_path2"):
            x_mlp = self.original_block.drop_path2(x_mlp)
        elif hasattr(self.original_block, "drop_path"):
            x_mlp = self.original_block.drop_path(x_mlp)
            
        x = x + x_mlp
        return x, size


def patch_vit_with_tome(
    model: nn.Module,
    r_schedule: Optional[List[int]] = None,
    schedule: Optional[List[int]] = None,
) -> nn.Module:
    """Patches blocks in a timm VisionTransformer with ToMe blocks using a specified per-layer r schedule.
    
    If r_schedule is all 0, forward pass is functionally and numerically identical to the unpatched model.
    """
    if r_schedule is None:
        r_schedule = schedule
    if r_schedule is None:
        raise ValueError("Must provide r_schedule or schedule")
        
    blocks = model.blocks
    if len(r_schedule) != len(blocks):
        raise ValueError(f"Schedule length ({len(r_schedule)}) must match block count ({len(blocks)})")
        
    wrapped_blocks = nn.ModuleList([
        ToMeBlock(block, r=r) for block, r in zip(blocks, r_schedule)
    ])
    
    def tome_forward_features(x, *args, **kwargs):
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        
        size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        for blk in wrapped_blocks:
            x, size = blk(x, size)
            
        x = model.norm(x)
        model._last_token_size = size
        return x
        
    model.forward_features = tome_forward_features
    model._tome_wrapped_blocks = wrapped_blocks
    model._tome_schedule = r_schedule
    return model


def solve_r_for_flop_ratio(
    model_factory: Callable[[], nn.Module],
    resolution: int,
    ratio: float,
    max_r: Optional[int] = None,
) -> Tuple[int, float, float]:
    """Finds constant r per block whose measured FLOPs is closest to ratio * baseline FLOPs.
    
    Measures GFLOPs using knobs.flops.count_gflops.
    
    Args:
        model_factory: Callable returning a fresh unpatched model instance.
        resolution: Input resolution (e.g. 224 or 448).
        ratio: Target FLOP ratio relative to baseline (e.g. 0.75 or 0.50).
        max_r: Optional maximum r to search.
        
    Returns:
        (best_r, measured_gflops, baseline_gflops)
    """
    baseline_model = model_factory()
    baseline_gflops = count_gflops(baseline_model, input_resolution=resolution)
    target_gflops = baseline_gflops * ratio
    n_blocks = len(baseline_model.blocks)

    if max_r is None:
        dummy_in = torch.randn(1, 3, resolution, resolution)
        with torch.no_grad():
            x_tok = baseline_model.patch_embed(dummy_in)
            init_tokens = x_tok.shape[1]
        max_r = min(64, max(1, init_tokens // 4))

    best_r = 0
    best_diff = float("inf")
    best_gflops = baseline_gflops

    step = 2 if max_r > 20 else 1
    for r_cand in range(0, max_r + 1, step):
        m = model_factory()
        patch_vit_with_tome(m, r_schedule=[r_cand] * n_blocks)
        gflops = count_gflops(m, input_resolution=resolution)
        diff = abs(gflops - target_gflops)
        if diff < best_diff:
            best_diff = diff
            best_r = r_cand
            best_gflops = gflops

    return best_r, best_gflops, baseline_gflops


def get_tome_schedule_for_budget(
    num_layers: int,
    initial_tokens: int,
    target_budget: str,
) -> Tuple[List[int], Dict[str, Any]]:
    """Legacy helper: computes a constant schedule heuristic for target compute budget."""
    if target_budget == "75%":
        r = 16
        schedule = [r] * num_layers
        status = "nominal_75pct_heuristic"
    elif target_budget == "50%":
        r = 32
        schedule = [r] * num_layers
        status = "nominal_50pct_heuristic"
    elif target_budget == "25%":
        r = 52
        schedule = [r] * num_layers
        status = "nominal_25pct_heuristic"
    else:
        schedule = [0] * num_layers
        status = "baseline_unreduced"
        
    info = {
        "budget": target_budget,
        "r_per_layer": schedule[0] if schedule else 0,
        "initial_tokens": initial_tokens,
        "final_tokens": max(1, initial_tokens - sum(schedule)),
        "status": status,
    }
    return schedule, info
