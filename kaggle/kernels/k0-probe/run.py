"""K0-probe: Environment probe, G0-A native checkpoint verification, and empirical throughput benchmark.

Key Features:
- Direct candidate paths matching Kaggle's /kaggle/input/datasets and /kaggle/input/competitions structure.
- Batched inference (batch size 64) for ultra-fast G0-A native verification.
- Empirical FP16 throughput benchmark across resolutions {224, 320, 384, 448}.
"""

import os
import sys
import time
import json
import zipfile
import subprocess
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

start_time = time.time()
print("=== GPU Hardware Info ===")
try:
    subprocess.run(["nvidia-smi"], check=False)
except Exception as e:
    print(f"nvidia-smi error: {e}")

# 1. Unpack any zips if present
for zip_dir in ["/kaggle/input/datasets/anshulsingh45/knobs-code", "/kaggle/input/knobs-code"]:
    if os.path.exists(zip_dir):
        for f in os.listdir(zip_dir):
            if f.endswith(".zip"):
                zpath = os.path.join(zip_dir, f)
                dest = os.path.join("/tmp", f[:-4])
                if not os.path.exists(dest):
                    try:
                        with zipfile.ZipFile(zpath, 'r') as zf:
                            zf.extractall(dest)
                        print(f"Extracted {f} to {dest}")
                    except Exception as e:
                        print(f"Note on {f}: {e}")

# 2. Add knobs to sys.path
KNOBS_CANDIDATES = [
    "/kaggle/input/datasets/anshulsingh45/knobs-code/src",
    "/kaggle/input/knobs-code/src",
    "/tmp/src",
    "/kaggle/input/datasets/anshulsingh45/knobs-code",
    "/kaggle/input/knobs-code",
]
knobs_found = False
for cand in KNOBS_CANDIDATES:
    if os.path.exists(cand) and (os.path.exists(os.path.join(cand, "knobs")) or os.path.exists(os.path.join(cand, "__init__.py"))):
        if cand not in sys.path:
            sys.path.insert(0, cand)
        print(f"Added knobs to sys.path from: {cand}")
        knobs_found = True
        break

if not knobs_found:
    # If not directly matched, find knobs package without entering competitions
    for root, dirs, files in os.walk("/kaggle/input"):
        if "competitions" in dirs:
            dirs.remove("competitions")
        if "imagenet-object-localization-challenge" in dirs:
            dirs.remove("imagenet-object-localization-challenge")
        if "knobs" in dirs and os.path.exists(os.path.join(root, "knobs", "__init__.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            print(f"Found knobs via pruned search at: {root}")
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

# 3. Locate PILOT split and metadata
SPLITS_CANDIDATES = [
    "/kaggle/input/datasets/anshulsingh45/knobs-code/splits",
    "/kaggle/input/knobs-code/splits",
    "/tmp/splits",
]
pilot_json_path = None
meta_json_path = None
for s_dir in SPLITS_CANDIDATES:
    p_path = os.path.join(s_dir, "PILOT.json")
    m_path = os.path.join(s_dir, "val_metadata.json")
    if os.path.exists(p_path) and os.path.exists(m_path):
        pilot_json_path = p_path
        meta_json_path = m_path
        print(f"Found splits in: {s_dir}")
        break

if not pilot_json_path or not meta_json_path:
    raise FileNotFoundError(f"Could not find PILOT.json or val_metadata.json in candidate dirs: {SPLITS_CANDIDATES}")

with open(pilot_json_path, "r") as f:
    pilot_image_ids = json.load(f)
with open(meta_json_path, "r") as f:
    val_metadata = json.load(f)

print(f"Loaded PILOT split: {len(pilot_image_ids)} images")

# 4. Locate validation images
IMAGE_DIR_CANDIDATES = [
    "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
]
IMAGE_DIR = None
for img_cand in IMAGE_DIR_CANDIDATES:
    if os.path.exists(img_cand):
        IMAGE_DIR = img_cand
        print(f"Found ImageNet validation images in: {IMAGE_DIR}")
        break

if not IMAGE_DIR:
    raise FileNotFoundError(f"Could not find ImageNet validation images in {IMAGE_DIR_CANDIDATES}")

# 5. G0-A Native Checkpoint Verification
print("\n=== G0-A: Native Checkpoint Verification ===")
g0_a_results = {}
# Evaluate the full PILOT (5,000 images, exactly 5 per class, class-balanced by construction)
eval_ids = list(pilot_image_ids)
eval_records = [val_metadata[img_id] for img_id in eval_ids if img_id in val_metadata]

for model_key, model_tag in MODEL_TAGS.items():
    print(f"\nEvaluating native checkpoint for {model_key} ({model_tag})...")
    ref_acc = DEFAULT_REFERENCES.get(model_tag, 80.0)
    
    cfg = get_model_cfg(model_tag)
    native_transform = get_native_transform(model_tag)
    model = timm.create_model(model_tag, pretrained=True).eval().to(device)
    
    correct_count = 0
    total_count = 0
    batch_size = 64
    
    for b_start in range(0, len(eval_records), batch_size):
        b_end = min(b_start + batch_size, len(eval_records))
        batch = eval_records[b_start:b_end]
        
        batch_tensors = []
        batch_labels = []
        
        for rec in batch:
            img_id = rec["image_id"]
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
            if not os.path.exists(img_path):
                continue
            pil_img = Image.open(img_path).convert("RGB")
            batch_tensors.append(native_transform(pil_img))
            batch_labels.append(rec["class_idx"])
            
        if not batch_tensors:
            continue
            
        tensor_in = torch.stack(batch_tensors, dim=0).to(device)
        labels_in = torch.tensor(batch_labels, dtype=torch.long, device=device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(tensor_in)
                preds = logits.argmax(dim=-1)
                correct_count += int((preds == labels_in).sum().item())
                total_count += len(batch_labels)
                
    measured_top1 = (correct_count / max(1, total_count)) * 100.0
    p_meas = measured_top1 / 100.0
    se_meas = np.sqrt(max(0.0, p_meas * (1.0 - p_meas) / max(1, total_count))) * 100.0
    tolerance = 2.0 * se_meas
    diff = abs(measured_top1 - ref_acc)
    passed = bool(diff <= tolerance)
    
    print(f"{model_key}: Measured Top-1 = {measured_top1:.2f}%, Reference = {ref_acc:.2f}%, Diff = {diff:.2f}%, 2*SE = {tolerance:.2f}%, Pass = {passed}")
    
    g0_a_results[model_key] = {
        "model_tag": model_tag,
        "measured_top1": measured_top1,
        "reference_top1": ref_acc,
        "diff": diff,
        "se_measured": se_meas,
        "tolerance_2se": tolerance,
        "passed": passed,
        "total_images": total_count,
        "subset_definition": "full_pilot_5000_class_balanced",
    }
    
    del model
    torch.cuda.empty_cache()

# 6. Throughput Benchmark across resolutions
print("\n=== Throughput Benchmark ===")
benchmark_results = {}
for model_key in ["deit_base", "efficientnet_b3"]:
    model = create_model_instance(model_key, resolution=224, pretrained=False, device=device)
    for res in [224, 320, 384, 448]:
        bench = benchmark_model_latency(
            model=model,
            resolution=res,
            batch_size=64,
            n_warmup=15,
            n_timed=40,
            device=device,
            use_fp16=True,
        )
        print(f"{model_key} @ {res}x{res}: {bench['throughput_img_per_sec']:.1f} img/s (median batch {bench['median_batch_latency_ms']:.1f}ms)")
        benchmark_results[f"{model_key}_{res}"] = bench
    del model
    torch.cuda.empty_cache()

# 7. Save metadata and results
out_dir = "/kaggle/working"
with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
    json.dump(get_environment_metadata(device), f, indent=2)

with open(os.path.join(out_dir, "g0_a_verification.json"), "w") as f:
    json.dump(g0_a_results, f, indent=2)

with open(os.path.join(out_dir, "throughput_benchmark.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

elapsed = time.time() - start_time
print(f"\nK0-probe execution complete in {elapsed:.1f}s ({elapsed/60.0:.2f} min)!")
