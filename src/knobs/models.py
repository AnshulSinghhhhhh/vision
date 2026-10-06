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
from typing import Dict, Any, Tuple, Optional
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


def recalibrate_batchnorm(
    model: nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 40,
) -> nn.Module:
    """Recomputes running mean and variance on unlabelled calibration images (CALIB)
    using cumulative averaging (momentum=None) with backbone weights frozen.
    """
    model.eval()
    bn_layers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.reset_running_stats()
            m.momentum = None  # Cumulative moving average
            m.train()
            bn_layers.append(m)
            
    if not bn_layers:
        return model

    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if i >= num_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            _ = model(images)

    model.eval()
    return model
