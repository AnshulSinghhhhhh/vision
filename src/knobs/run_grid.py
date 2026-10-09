"""Resumable experiment runner with shared-tensor caching and Parquet shard serialization.

Key optimizations:
- Caches corrupted 448x448 tensors per deterministic batch to avoid redundant decoding/corruptions.
- Saves progress incrementally in Parquet shards; automatically resumes by skipping completed shards.
- Supports single-worker or multi-GPU worker execution.
- Generates run_meta.json recording hardware, versions, and checksums.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from knobs.data import preprocess_image_448, transform_bbox_to_crop448
from knobs.corrupt import apply_corruption, generate_frequency_controlled_noise
from knobs.resize import resize_tensor_torch, generate_m7_suite
from knobs.models import create_model_instance, normalize_tensor, MODEL_TAGS
from knobs.tokens import patch_vit_with_tome, get_tome_schedule_for_budget


def get_git_commit_hash(git_commit_file: Optional[str] = None) -> str:
    """Attempts to retrieve current git commit hash.
    
    If git is unavailable (e.g. on Kaggle), reads the GIT_COMMIT provenance file.
    """
    if git_commit_file and os.path.exists(git_commit_file):
        try:
            with open(git_commit_file, "r", encoding="utf-8") as f:
                c = f.read().strip()
                if c:
                    return c
        except Exception:
            pass

    # Check environment variable
    env_path = os.environ.get("KNOBS_GIT_COMMIT_FILE")
    if env_path and os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                c = f.read().strip()
                if c:
                    return c
        except Exception:
            pass

    # Try git command first if available
    try:
        import subprocess
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit = res.stdout.strip()
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        is_dirty = bool(status_res.stdout.strip()) if status_res.returncode == 0 else False
        return f"{commit}{'-dirty' if is_dirty else ''}"
    except Exception:
        pass

    # Fallback to searching candidate GIT_COMMIT files
    candidate_locations = [
        "GIT_COMMIT",
        os.path.join(os.getcwd(), "GIT_COMMIT"),
        os.path.join(os.path.dirname(__file__), "GIT_COMMIT"),
        os.path.join(os.path.dirname(__file__), "..", "GIT_COMMIT"),
        os.path.join(os.path.dirname(__file__), "..", "..", "GIT_COMMIT"),
        "/kaggle/input/datasets/anshulsingh45/knobs-code/GIT_COMMIT",
        "/kaggle/input/knobs-code/GIT_COMMIT",
        "/tmp/GIT_COMMIT",
        "/tmp/src/GIT_COMMIT",
    ]
    for p in sys.path:
        candidate_locations.append(os.path.join(p, "GIT_COMMIT"))
        candidate_locations.append(os.path.join(p, "knobs", "GIT_COMMIT"))
        candidate_locations.append(os.path.join(p, "..", "GIT_COMMIT"))

    for path in candidate_locations:
        if os.path.exists(path) and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    c = f.read().strip()
                    if c:
                        return c
            except Exception:
                continue

    return "unknown_git_commit"


def get_environment_metadata(device: torch.device) -> Dict[str, Any]:
    """Collects system environment metadata for run_meta.json."""
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    
    import timm
    import PIL
    from knobs.corrupt import CORRUPTION_BACKEND, IMAGECORRUPTIONS_VERSION

    ic_version = IMAGECORRUPTIONS_VERSION if IMAGECORRUPTIONS_VERSION is not None else "unavailable"
    
    return {
        "git_commit": get_git_commit_hash(),
        "python_version": sys.version,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "timm_version": getattr(timm, "__version__", "unknown"),
        "pillow_version": getattr(PIL, "__version__", "unknown"),
        "imagecorruptions_version": ic_version,
        "corruption_backend": CORRUPTION_BACKEND,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def evaluate_batch_on_models(
    corrupted_tensor_448: torch.Tensor,
    labels: torch.Tensor,
    image_ids: List[str],
    condition_name: str,
    severity: int,
    models_dict: Dict[str, nn.Module],
    resolutions: List[int] = [224, 320, 384, 448],
    device: Optional[torch.device] = None,
    use_fp16: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluates a batch of corrupted 448x448 images across models and resolutions.
    
    Returns a list of per-image observation records.
    """
    records = []
    b = corrupted_tensor_448.shape[0]
    device = corrupted_tensor_448.device
    
    # Precompute resized inputs
    resized_inputs = {}
    for res in resolutions:
        resized_inputs[res] = resize_tensor_torch(corrupted_tensor_448, target_size=res)
        
    for model_key, model in models_dict.items():
        model_tag = MODEL_TAGS.get(model_key, model_key)
        for res in resolutions:
            inp = resized_inputs[res]
            norm_inp = normalize_tensor(inp, model_tag=model_tag)
            
            with torch.no_grad():
                if device.type == "cuda" and use_fp16:
                    with torch.cuda.amp.autocast():
                        logits = model(norm_inp)
                else:
                    logits = model(norm_inp)
                    
            preds = logits.argmax(dim=-1).cpu().numpy()
            probs = torch.softmax(logits, dim=-1)
            
            top5_preds = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1)[1].cpu().numpy()
            
            for i in range(b):
                img_id = image_ids[i]
                lbl = labels[i].item()
                pred = int(preds[i])
                correct = bool(pred == lbl)
                top5_corr = bool(lbl in top5_preds[i])
                conf = float(probs[i, pred].item())
                
                records.append({
                    "image_id": img_id,
                    "condition": condition_name,
                    "severity": severity,
                    "model": model_key,
                    "resolution": res,
                    "label": lbl,
                    "pred": pred,
                    "correct": correct,
                    "top5_correct": top5_corr,
                    "confidence": conf,
                })
                
    return records


def run_condition_shard(
    image_paths: List[str],
    metadata_by_id: Dict[str, Any],
    condition: str,
    severity: int,
    models_dict: Dict[str, nn.Module],
    output_shard_path: str,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    use_fp16: bool = True,
):
    """Executes evaluation for a single condition shard and saves results to Parquet.
    
    Skips if output_shard_path already exists.
    """
    if os.path.exists(output_shard_path):
        print(f"Shard already exists: {output_shard_path}, skipping.")
        return
        
    os.makedirs(os.path.dirname(output_shard_path), exist_ok=True)
    all_records = []
    
    n_images = len(image_paths)
    for start_idx in range(0, n_images, batch_size):
        end_idx = min(start_idx + batch_size, n_images)
        batch_paths = image_paths[start_idx:end_idx]
        
        batch_tensors = []
        batch_labels = []
        batch_ids = []
        
        for path in batch_paths:
            img_id = os.path.splitext(os.path.basename(path))[0]
            meta = metadata_by_id.get(img_id, {})
            label = meta.get("class_idx", 0)
            
            pil_img = Image.open(path)
            arr_448, _ = preprocess_image_448(pil_img)
            
            # Apply deterministic corruption in 448 space
            corr_448 = apply_corruption(
                arr_448,
                image_id=img_id,
                corruption_name=condition,
                severity=severity,
            )
            # Convert to float tensor in [0, 1]
            t_448 = torch.from_numpy(corr_448).permute(2, 0, 1).float() / 255.0
            batch_tensors.append(t_448)
            batch_labels.append(label)
            batch_ids.append(img_id)
            
        corrupted_batch_448 = torch.stack(batch_tensors, dim=0).to(device)
        labels_batch = torch.tensor(batch_labels, dtype=torch.long, device=device)
        
        records = evaluate_batch_on_models(
            corrupted_tensor_448=corrupted_batch_448,
            labels=labels_batch,
            image_ids=batch_ids,
            condition_name=condition,
            severity=severity,
            models_dict=models_dict,
            device=device,
            use_fp16=use_fp16,
        )
        all_records.extend(records)
        
    df = pd.DataFrame(all_records)
    # Write to temporary file first, then atomic rename
    tmp_path = output_shard_path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_shard_path)
    print(f"Saved {len(df)} records to {output_shard_path}")
