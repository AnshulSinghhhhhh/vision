"""K1-sanity: GPU Sanity tests and G0-B experimental pipeline baseline evaluation.

Checks:
1. Pretrained ToMe r=0 GPU numerical identity (< 1e-3 logit difference in FP32).
2. FP16 vs FP32 precision agreement within 0.3 pp per model and resolution on 500 images.
3. Direct paths, 0 directory walks, batched inference.
"""

import os
import sys
import time
import json
import zipfile
import subprocess
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

start_time = time.time()
print("=== K1-Sanity Initializing ===")

# 1. Unpack zips from /kaggle/input/knobs-code if present (direct, no recursive walking)
if os.path.exists("/kaggle/input/knobs-code"):
    for f in os.listdir("/kaggle/input/knobs-code"):
        if f.endswith(".zip"):
            zip_path = os.path.join("/kaggle/input/knobs-code", f)
            dest = os.path.join("/tmp", f[:-4])
            if not os.path.exists(dest):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(dest)
                    print(f"Extracted {f} to {dest}")
                except Exception as e:
                    print(f"Note on {f}: {e}")

# 2. Add knobs to sys.path (direct paths)
KNOBS_CANDIDATES = [
    "/kaggle/input/datasets/anshulsingh45/knobs-code/src",
    "/kaggle/input/knobs-code/src",
    "/tmp/src",
    "/kaggle/input/datasets/anshulsingh45/knobs-code",
    "/kaggle/input/knobs-code",
]
for candidate in KNOBS_CANDIDATES:
    if os.path.exists(candidate) and (os.path.exists(os.path.join(candidate, "knobs")) or os.path.exists(os.path.join(candidate, "__init__.py"))):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        print(f"Added {candidate} to sys.path")
        break

import knobs
from knobs.models import create_model_instance, normalize_tensor, MODEL_TAGS
from knobs.data import preprocess_image_448
from knobs.resize import resize_tensor_torch
from knobs.tokens import patch_vit_with_tome
from knobs.run_grid import get_environment_metadata

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active device: {device}")

# 3. Check ToMe r=0 identity on GPU with pretrained DeiT-B
print("\n=== Testing Pretrained ToMe r=0 GPU Identity ===")
model_deit = create_model_instance("deit_base", resolution=224, pretrained=True, device=device)
x_test = torch.randn(2, 3, 224, 224, device=device)

with torch.no_grad():
    orig_logits = model_deit(x_test)
    
n_blocks = len(model_deit.blocks)
patched_deit = patch_vit_with_tome(model_deit, [0] * n_blocks)
with torch.no_grad():
    patched_logits = patched_deit(x_test)

max_diff = torch.max(torch.abs(orig_logits - patched_logits)).item()
tome_passed = bool(max_diff < 1e-3)
print(f"ToMe r=0 Max Logit Diff: {max_diff:.6f}, Passed (< 1e-3): {tome_passed}")

del model_deit, patched_deit
torch.cuda.empty_cache()

# 4. Locate PILOT split and metadata (direct paths)
SPLITS_CANDIDATES = [
    "/kaggle/input/datasets/anshulsingh45/knobs-code/splits",
    "/kaggle/input/knobs-code/splits",
    "/tmp/splits",
]
pilot_json_path = None
meta_json_path = None
for candidate_dir in SPLITS_CANDIDATES:
    p = os.path.join(candidate_dir, "PILOT.json")
    m = os.path.join(candidate_dir, "val_metadata.json")
    if os.path.exists(p) and os.path.exists(m):
        pilot_json_path = p
        meta_json_path = m
        print(f"Found splits in: {candidate_dir}")
        break

if not pilot_json_path or not meta_json_path:
    raise FileNotFoundError("Could not find PILOT.json or val_metadata.json")

with open(pilot_json_path, "r") as f:
    pilot_image_ids = json.load(f)[:500]  # First 500 images for precision check
with open(meta_json_path, "r") as f:
    val_metadata = json.load(f)

# Locate validation image directory
IMAGE_DIR_CANDIDATES = [
    "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
]
IMAGE_DIR = None
for c in IMAGE_DIR_CANDIDATES:
    if os.path.exists(c):
        IMAGE_DIR = c
        print(f"Found ImageNet validation images in: {IMAGE_DIR}")
        break

if not IMAGE_DIR:
    raise FileNotFoundError("Could not find ImageNet validation images")

print(f"ImageNet validation directory located at: {IMAGE_DIR}")

# 5. Evaluate FP16 vs FP32 agreement (batched)
print("\n=== Testing FP16 vs FP32 Precision Agreement (500 images) ===")
precision_results = {}
batch_size = 64

for model_key in ["deit_base", "efficientnet_b3"]:
    print(f"Checking precision agreement for {model_key}...")
    model = create_model_instance(model_key, resolution=224, pretrained=True, device=device)
    model_tag = MODEL_TAGS[model_key]
    
    fp32_correct = 0
    fp16_correct = 0
    total = 0
    
    for b_start in range(0, len(pilot_image_ids), batch_size):
        b_end = min(b_start + batch_size, len(pilot_image_ids))
        batch_ids = pilot_image_ids[b_start:b_end]
        
        batch_tensors = []
        batch_labels = []
        
        for img_id in batch_ids:
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
            if not os.path.exists(img_path):
                continue
            pil_img = Image.open(img_path)
            arr_448, _ = preprocess_image_448(pil_img)
            t_448 = torch.from_numpy(arr_448).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t_224 = resize_tensor_torch(t_448, 224)
            batch_tensors.append(t_224.squeeze(0))
            batch_labels.append(val_metadata[img_id]["class_idx"])
            
        if not batch_tensors:
            continue
            
        inp_tensor = torch.stack(batch_tensors, dim=0).to(device)
        norm_in = normalize_tensor(inp_tensor, model_tag=model_tag)
        labels_in = torch.tensor(batch_labels, dtype=torch.long, device=device)
        
        with torch.no_grad():
            # FP32
            out_fp32 = model(norm_in)
            preds_fp32 = out_fp32.argmax(dim=-1)
            fp32_correct += int((preds_fp32 == labels_in).sum().item())
            
            # FP16
            with torch.cuda.amp.autocast():
                out_fp16 = model(norm_in)
            preds_fp16 = out_fp16.argmax(dim=-1)
            fp16_correct += int((preds_fp16 == labels_in).sum().item())
            
            total += len(batch_labels)
            
    acc_fp32 = (fp32_correct / max(1, total)) * 100.0
    acc_fp16 = (fp16_correct / max(1, total)) * 100.0
    diff_pp = abs(acc_fp32 - acc_fp16)
    pass_prec = bool(diff_pp <= 0.3)
    
    print(f"{model_key}: FP32 Acc = {acc_fp32:.2f}%, FP16 Acc = {acc_fp16:.2f}%, Diff = {diff_pp:.2f} pp, Pass (<=0.3 pp): {pass_prec}")
    precision_results[model_key] = {
        "acc_fp32": acc_fp32,
        "acc_fp16": acc_fp16,
        "diff_pp": diff_pp,
        "passed": pass_prec,
    }
    
    del model
    torch.cuda.empty_cache()

# Save output
out_dir = "/kaggle/working"
with open(os.path.join(out_dir, "tome_identity.json"), "w") as f:
    json.dump({"tome_r0_max_diff": max_diff, "passed": tome_passed}, f, indent=2)

with open(os.path.join(out_dir, "precision_agreement.json"), "w") as f:
    json.dump(precision_results, f, indent=2)

with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
    json.dump(get_environment_metadata(device), f, indent=2)

elapsed = time.time() - start_time
print(f"\nK1-sanity checks complete in {elapsed:.1f}s!")
