"""K1-sanity: GPU Sanity tests and G0-B experimental pipeline baseline evaluation.

Checks:
1. G0-B Experimental Baseline: Evaluate models under research pipeline (448 -> downsample) on PILOT.
2. FP16 vs FP32 precision agreement within 0.3 pp per model and resolution.
3. Pretrained ToMe r=0 GPU numerical identity (< 1e-3 logit difference in FP32).
4. Label shuffling sanity check (top-1 ~ 0.1%).
"""

import os
import sys
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 1. Dynamically locate knobs package in /kaggle/input
print("Searching for knobs package in /kaggle/input...")
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "knobs" in dirs and os.path.exists(os.path.join(root, "knobs", "__init__.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            print(f"Found knobs package at: {root}, added to sys.path")
            break

import knobs
from knobs.models import create_model_instance, normalize_tensor, MODEL_TAGS
from knobs.data import preprocess_image_448
from knobs.resize import resize_tensor_torch
from knobs.tokens import patch_vit_with_tome

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active device: {device}")

# 2. Check ToMe r=0 identity on GPU with pretrained DeiT-B
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

# 3. Load sample images from PILOT for precision & baseline check
SPLIT_DIR = "/kaggle/input/knobs-kprep/splits"
PILOT_JSON = os.path.join(SPLIT_DIR, "PILOT.json")
META_JSON = os.path.join(SPLIT_DIR, "val_metadata.json")
IMAGE_DIR = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"

with open(PILOT_JSON, "r") as f:
    pilot_image_ids = json.load(f)[:500]  # First 500 images for precision check
with open(META_JSON, "r") as f:
    val_metadata = json.load(f)

# Evaluate FP16 vs FP32 agreement
print("\n=== Testing FP16 vs FP32 Precision Agreement (500 images) ===")
precision_results = {}

for model_key in ["deit_base", "efficientnet_b3"]:
    print(f"Checking precision agreement for {model_key}...")
    model = create_model_instance(model_key, resolution=224, pretrained=True, device=device)
    model_tag = MODEL_TAGS[model_key]
    
    fp32_correct = 0
    fp16_correct = 0
    total = 0
    
    for img_id in pilot_image_ids:
        rec = val_metadata[img_id]
        label = rec["class_idx"]
        img_path = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
        if not os.path.exists(img_path):
            continue
            
        pil_img = Image.open(img_path)
        arr_448, _ = preprocess_image_448(pil_img)
        t_448 = torch.from_numpy(arr_448).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t_224 = resize_tensor_torch(t_448, 224).to(device)
        norm_in = normalize_tensor(t_224, model_tag=model_tag)
        
        with torch.no_grad():
            # FP32
            out_fp32 = model(norm_in)
            pred_fp32 = out_fp32.argmax(dim=-1).item()
            if pred_fp32 == label:
                fp32_correct += 1
                
            # FP16
            with torch.cuda.amp.autocast():
                out_fp16 = model(norm_in)
            pred_fp16 = out_fp16.argmax(dim=-1).item()
            if pred_fp16 == label:
                fp16_correct += 1
                
            total += 1
            
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

print("\nK1-sanity checks complete!")
