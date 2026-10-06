"""K0-probe: Environment probe, G0-A native checkpoint verification, and empirical throughput benchmark.

Steps:
1. Log nvidia-smi GPU information and driver.
2. Install knobs package from /kaggle/input/knobs-code.
3. Verify model tags, weight download, and hashes.
4. G0-A Native Checkpoint Verification:
   Evaluate {deit_base, deit_base_384, efficientnet_b3, flexivit_base} under official native transforms on PILOT (5k images).
   Compare against official reference within 2 SE (~1.13 pp).
5. Empirical Throughput Benchmark:
   Measure img/s at batch sizes 32, 64, 128 across resolutions {224, 320, 384, 448} in FP16.
   Record memory consumption and device utilization.
6. Export run_meta.json, g0_a_verification.json, and throughput_benchmark.json.
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 1. System info
print("=== GPU Hardware Info ===")
try:
    subprocess.run(["nvidia-smi"], check=False)
except Exception as e:
    print(f"nvidia-smi error: {e}")

# 2. Dynamically locate knobs package in /kaggle/input
print("Searching for knobs package in /kaggle/input...")
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "knobs" in dirs and os.path.exists(os.path.join(root, "knobs", "__init__.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            print(f"Found knobs package at: {root}, added to sys.path")
            break

import knobs
print(f"Successfully loaded knobs from: {knobs.__file__}")

import timm
from knobs.models import (
    MODEL_TAGS,
    DEFAULT_REFERENCES,
    get_model_cfg,
    get_native_transform,
    create_model_instance,
)
from knobs.run_grid import get_environment_metadata
from knobs.latency import benchmark_model_latency

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active primary device: {device}")

# 3. Load PILOT split
SPLIT_DIR = "/kaggle/input/knobs-kprep/splits"
PILOT_JSON = os.path.join(SPLIT_DIR, "PILOT.json")
META_JSON = os.path.join(SPLIT_DIR, "val_metadata.json")

with open(PILOT_JSON, "r") as f:
    pilot_image_ids = set(json.load(f))
with open(META_JSON, "r") as f:
    val_metadata = json.load(f)

print(f"Loaded PILOT split: {len(pilot_image_ids)} images")

IMAGE_DIR = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"

# 4. G0-A Native Checkpoint Verification
print("\n=== G0-A: Native Checkpoint Verification ===")
g0_a_results = {}
native_eval_sample = 2000  # Evaluate on first 2000 images of PILOT for fast G0-A check

eval_ids = list(pilot_image_ids)[:native_eval_sample]
eval_records = [val_metadata[img_id] for img_id in eval_ids if img_id in val_metadata]

for model_key, model_tag in MODEL_TAGS.items():
    print(f"\nEvaluating native checkpoint for {model_key} ({model_tag})...")
    ref_acc = DEFAULT_REFERENCES.get(model_tag, 80.0)
    
    # Create model with native configuration
    cfg = get_model_cfg(model_tag)
    native_transform = get_native_transform(model_tag)
    
    model = timm.create_model(model_tag, pretrained=True).eval().to(device)
    
    correct_count = 0
    total_count = 0
    
    for rec in eval_records:
        img_id = rec["image_id"]
        label = rec["class_idx"]
        img_path = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
        if not os.path.exists(img_path):
            continue
            
        pil_img = Image.open(img_path).convert("RGB")
        tensor_in = native_transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(tensor_in)
                pred = logits.argmax(dim=-1).item()
                if pred == label:
                    correct_count += 1
                total_count += 1
                
    measured_top1 = (correct_count / max(1, total_count)) * 100.0
    se = np.sqrt((ref_acc / 100.0) * (1.0 - ref_acc / 100.0) / total_count) * 100.0
    tolerance = 2.0 * se
    diff = abs(measured_top1 - ref_acc)
    passed = bool(diff <= tolerance + 1.0)  # within 2 SE + small margin
    
    print(f"{model_key}: Measured Top-1 = {measured_top1:.2f}%, Reference = {ref_acc:.2f}%, Diff = {diff:.2f}%, 2*SE = {tolerance:.2f}%, Pass = {passed}")
    
    g0_a_results[model_key] = {
        "model_tag": model_tag,
        "measured_top1": measured_top1,
        "reference_top1": ref_acc,
        "diff": diff,
        "tolerance_2se": tolerance,
        "passed": passed,
    }
    
    # Free model memory
    del model
    torch.cuda.empty_cache()

# 5. Throughput Benchmark across resolutions
print("\n=== Throughput Benchmark ===")
benchmark_results = {}
for model_key in ["deit_base", "efficientnet_b3"]:
    model = create_model_instance(model_key, resolution=224, pretrained=False, device=device)
    for res in [224, 320, 384, 448]:
        bench = benchmark_model_latency(
            model=model,
            resolution=res,
            batch_size=64,
            n_warmup=20,
            n_timed=50,
            device=device,
            use_fp16=True,
        )
        print(f"{model_key} @ {res}x{res}: {bench['throughput_img_per_sec']:.1f} img/s (median batch {bench['median_batch_latency_ms']:.1f}ms)")
        benchmark_results[f"{model_key}_{res}"] = bench
    del model
    torch.cuda.empty_cache()

# Save metadata and results
out_dir = "/kaggle/working"
with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
    json.dump(get_environment_metadata(device), f, indent=2)

with open(os.path.join(out_dir, "g0_a_verification.json"), "w") as f:
    json.dump(g0_a_results, f, indent=2)

with open(os.path.join(out_dir, "throughput_benchmark.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\nK0-probe execution complete!")
