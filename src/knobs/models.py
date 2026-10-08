"""Model loading, official configuration lookup, dynamic image resolution adaptation,
and BatchNorm recalibration.

Supported models (Pretrained ImageNet-1K):
1. 'deit_base_patch16_224.fb_in1k' (DeiT-B/16, native 224x224)
2. 'deit_base_patch16_384.fb_in1k' (DeiT-B/16-384, native 384x384, control)
3. 'efficientnet_b3.ra2_in1k' (EfficientNet-B3, native test 320x320)
4. 'flexivit_base.1200ep_in1k' (FlexiViT-B, native 240x240, variable patch size)
   - Arm F-p: constant patch size 16 across resolutions {224, 320, 384, 448}
   - Arm F-t: constant grid 16x16 (256 tokens) with patch sizes {14, 20, 24, 28}
"""

import copy
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
import timm
from timm.data import create_transform

MODEL_TAGS = {
    "deit_base": "deit_base_patch16_224.fb_in1k",
    "deit_base_384": "deit_base_patch16_384.fb_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "flexivit_base": "flexivit_base.1200ep_in1k",
}

# Reference Top-1 from official literature / timm model cards (for G0-A dynamic validation)
DEFAULT_REFERENCES = {
    "deit_base_patch16_224.fb_in1k": 81.8,
    "deit_base_patch16_384.fb_in1k": 82.9,
    "efficientnet_b3.ra2_in1k": 81.5,
    "flexivit_base.1200ep_in1k": 82.5,
}


def get_model_cfg(model_tag: str) -> Dict[str, Any]:
    """Retrieves the pretrained configuration for a timm model."""
    cfg = timm.get_pretrained_cfg(model_tag)
    return cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)


def get_native_transform(model_tag: str):
    """Constructs the native G0-A evaluation transform as specified in model's pretrained_cfg."""
    cfg = get_model_cfg(model_tag)
    input_size = cfg.get("test_input_size") or cfg.get("input_size") or (3, 224, 224)
    crop_pct = cfg.get("test_crop_pct") or cfg.get("crop_pct") or 0.875
    interpolation = cfg.get("interpolation", "bicubic")
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    
    return create_transform(
        input_size=input_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        crop_mode=cfg.get("crop_mode", "center"),
    )


def normalize_tensor(tensor: torch.Tensor, model_tag: str) -> torch.Tensor:
    """Normalizes a float tensor in [0, 1] using model's mean and std."""
    cfg = get_model_cfg(model_tag)
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def create_model_instance(
    model_key: str,
    resolution: int = 224,
    arm: Optional[str] = None,
    pretrained: bool = True,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Creates and configures a model instance for a specific resolution and arm.
    
    Args:
        model_key: one of {'deit_base', 'deit_base_384', 'efficientnet_b3', 'flexivit_base'}
        resolution: target input resolution {224, 320, 384, 448}
        arm: For FlexiViT: 'F-p' (constant patch 16) or 'F-t' (constant grid 16x16, patch = res/16)
        pretrained: whether to load pretrained weights
        device: torch device
    """
    model_tag = MODEL_TAGS.get(model_key, model_key)
    
    if "deit" in model_key:
        # Enable dynamic_img_size for ViT models running at arbitrary resolutions
        model = timm.create_model(
            model_tag,
            pretrained=pretrained,
            dynamic_img_size=True,
        )
    elif "flexivit" in model_key:
        if arm == "F-t":
            # Arm F-t: constant grid 16x16 (256 tokens)
            # patch_size = resolution // 16: {224: 14, 320: 20, 384: 24, 448: 28}
            patch_size = resolution // 16
            model = timm.create_model(
                model_tag,
                pretrained=pretrained,
                img_size=resolution,
                patch_size=patch_size,
            )
        else:
            # Arm F-p: constant patch 16
            model = timm.create_model(
                model_tag,
                pretrained=pretrained,
                img_size=resolution,
                patch_size=16,
            )
    else:
        # EfficientNet-B3 (CNN supports variable input sizes natively)
        model = timm.create_model(
            model_tag,
            pretrained=pretrained,
        )
        
    model.eval()
    model.to(device)
    return model


def _apply_calibration_corruption(images: torch.Tensor, corruption: Union[str, tuple, list, object]) -> torch.Tensor:
    if callable(corruption):
        return corruption(images)
    if isinstance(corruption, str):
        c_name = corruption
        c_sev = 3
    elif isinstance(corruption, (tuple, list)):
        c_name, c_sev = corruption[0], int(corruption[1])
    else:
        return images

    import numpy as np
    from knobs.corrupt import apply_corruption

    corrupted = []
    for i in range(len(images)):
        img_t = images[i]
        if img_t.dtype in (torch.float32, torch.float64, torch.float16):
            arr = (img_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            c_arr = apply_corruption(arr, image_id=f"cal_{i}", corruption_name=c_name, severity=c_sev)
            t = torch.from_numpy(c_arr).permute(2, 0, 1).to(images.device).to(images.dtype) / 255.0
            corrupted.append(t)
        else:
            arr = img_t.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            c_arr = apply_corruption(arr, image_id=f"cal_{i}", corruption_name=c_name, severity=c_sev)
            t = torch.from_numpy(c_arr).permute(2, 0, 1).to(images.device)
            corrupted.append(t)
    return torch.stack(corrupted, dim=0)


def recalibrate_batchnorm(
    model: nn.Module,
    calib_loader: Union[torch.utils.data.DataLoader, torch.Tensor, list, tuple],
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    drop_remainder: bool = True,
    calibration_corruption: Optional[Union[str, tuple, list, object]] = None,
    calibration_resolution: Optional[int] = None,
    num_batches: Optional[int] = None,
) -> nn.Module:
    """Recomputes running mean and variance on unlabelled calibration images (CALIB)
    using cumulative averaging (momentum=None) with backbone weights frozen.
    
    Ensures all batches have equal weight by enforcing equal batch sizes and dropping
    any remainder batch. Supports adapting to target corruption and resolution.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
            
    model.eval()
    bn_layers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None  # Cumulative moving average
            m.train()
            bn_layers.append(m)
            
    if not bn_layers:
        return model

    with torch.no_grad():
        if isinstance(calib_loader, torch.Tensor):
            total_n = len(calib_loader)
            n_batches = total_n // batch_size if drop_remainder else (total_n + batch_size - 1) // batch_size
            if num_batches is not None:
                n_batches = min(n_batches, num_batches)
                
            for b_idx in range(n_batches):
                start_idx = b_idx * batch_size
                end_idx = start_idx + batch_size
                if end_idx > total_n:
                    if drop_remainder:
                        break
                    end_idx = total_n
                batch = calib_loader[start_idx:end_idx].to(device)
                
                if calibration_corruption is not None:
                    batch = _apply_calibration_corruption(batch, calibration_corruption)

                if calibration_resolution is not None and (batch.shape[-2] != calibration_resolution or batch.shape[-1] != calibration_resolution):
                    from knobs.resize import resize_tensor_torch
                    batch = resize_tensor_torch(batch, calibration_resolution)
                    
                _ = model(batch)
        else:
            processed = 0
            for i, batch in enumerate(calib_loader):
                if num_batches is not None and i >= num_batches:
                    break
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                if drop_remainder and len(images) != batch_size:
                    continue
                images = images.to(device)

                if calibration_corruption is not None:
                    images = _apply_calibration_corruption(images, calibration_corruption)

                if calibration_resolution is not None and (images.shape[-2] != calibration_resolution or images.shape[-1] != calibration_resolution):
                    from knobs.resize import resize_tensor_torch
                    images = resize_tensor_torch(images, calibration_resolution)

                _ = model(images)
                processed += 1

    model.eval()
    return model


def assert_clean_sanity(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    tol_pp: float = 1.0,
    original_model: Optional[Union[nn.Module, float]] = None,
    batch_size: int = 64,
) -> float:
    """Evaluates clean accuracy of recalibrated model on clean images and asserts
    it is not more than tol_pp below the original clean accuracy on >= 1,000 images.
    
    Args:
        model: Recalibrated model to test.
        images: Clean validation images tensor.
        labels: Ground-truth class labels.
        tol_pp: Tolerance in percentage points (default 1.0 pp).
        original_model: Uncalibrated reference model OR reference clean accuracy (float in percent).
        batch_size: Batch size for evaluation.
        
    Returns:
        float: Recalibrated clean accuracy in percent.
        
    Raises:
        RuntimeError if recalibrated accuracy drops by more than tol_pp below original.
    """
    model.eval()
    device = next(model.parameters()).device

    # Handle case where user passes original_model as 4th positional argument
    if isinstance(tol_pp, nn.Module) or (isinstance(tol_pp, (int, float)) and tol_pp > 10.0):
        original_model = tol_pp
        tol_pp = 1.0
    
    if isinstance(original_model, (int, float)):
        orig_acc = float(original_model)
    elif isinstance(original_model, nn.Module):
        original_model.eval()
        orig_correct = 0
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                b_imgs = images[i:i + batch_size].to(device)
                b_lbls = labels[i:i + batch_size].to(device)
                logits = original_model(b_imgs)
                orig_correct += int((logits.argmax(dim=-1) == b_lbls).sum().item())
        orig_acc = (orig_correct / max(1, len(labels))) * 100.0
    elif hasattr(model, "_original_clean_acc"):
        orig_acc = getattr(model, "_original_clean_acc")
    else:
        raise ValueError("original_model or reference accuracy must be provided to assert_clean_sanity.")

    recal_correct = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            b_imgs = images[i:i + batch_size].to(device)
            b_lbls = labels[i:i + batch_size].to(device)
            logits = model(b_imgs)
            recal_correct += int((logits.argmax(dim=-1) == b_lbls).sum().item())
    recal_acc = (recal_correct / max(1, len(labels))) * 100.0

    drop = orig_acc - recal_acc
    if drop > tol_pp:
        raise RuntimeError(
            f"Clean sanity check failed: recalibrated accuracy ({recal_acc:.2f}%) is "
            f"{drop:.2f} pp below original clean accuracy ({orig_acc:.2f}%), "
            f"exceeding tolerance of {tol_pp:.2f} pp."
        )
    return recal_acc


# Alias for backward-compatibility
recalibrate_bn_statistics = recalibrate_batchnorm

