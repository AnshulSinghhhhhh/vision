"""K4-Headline-B: Tier 1 Headline Part B (JPEG s3, Contrast s3) on FULL (45k new images).

Scope:
- Models: DeiT-B, EfficientNet-B3, FlexiViT-B (F-p arm)
- Resolutions: 224, 320, 384, 448
- Conditions: JPEG compression (s3), Contrast loss (s3)
- Evaluates on FULL \ PILOT (45,000 images), skipping the 5,000 images already evaluated in K2 Pilot
- Outputs Parquet shards with per-image predictions
"""

import os
import sys
import time
import json
import zipfile
import numpy as np
import pandas as pd
import torch
from PIL import Image

print("=== K4-Headline-B Initializing ===")
start_wall_time = time.time()

# 1. Unzip packages if needed
# 1. Unpack any zips if present (direct check)
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

# Add knobs to sys.path
KNOBS_CANDIDATES = [
    "/kaggle/input/datasets/anshulsingh45/knobs-code/src",
    "/kaggle/input/knobs-code/src",
    "/tmp/src",
    "/kaggle/input/datasets/anshulsingh45/knobs-code",
    "/kaggle/input/knobs-code",
]
for cand in KNOBS_CANDIDATES:
    if os.path.exists(cand) and (os.path.exists(os.path.join(cand, "knobs")) or os.path.exists(os.path.join(cand, "__init__.py"))):
        if cand not in sys.path:
            sys.path.insert(0, cand)
        print(f"Added knobs to sys.path from: {cand}")
        break

import knobs
from knobs.models import MODEL_TAGS, create_model_instance, normalize_tensor
from knobs.data import preprocess_image_448
from knobs.corrupt import apply_corruption
from knobs.resize import resize_tensor_torch
from knobs.run_grid import get_environment_metadata

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Active device: {device}, Available GPUs: {n_gpus}")

# 2. Locate splits and metadata (direct paths)
SPLITS_CANDIDATES = [
    "/kaggle/input/datasets/anshulsingh45/knobs-code/splits",
    "/kaggle/input/knobs-code/splits",
    "/tmp/splits",
]
full_json_path = None
pilot_json_path = None
meta_json_path = None
for s_dir in SPLITS_CANDIDATES:
    f_p = os.path.join(s_dir, "FULL.json")
    p_p = os.path.join(s_dir, "PILOT.json")
    m_p = os.path.join(s_dir, "val_metadata.json")
    if os.path.exists(f_p) and os.path.exists(p_p) and os.path.exists(m_p):
        full_json_path = f_p
        pilot_json_path = p_p
        meta_json_path = m_p
        print(f"Found splits in: {s_dir}")
        break

if not full_json_path or not pilot_json_path or not meta_json_path:
    raise FileNotFoundError("Could not find FULL.json, PILOT.json, or val_metadata.json")

with open(full_json_path, "r") as f:
    full_image_ids = json.load(f)
with open(pilot_json_path, "r") as f:
    pilot_image_ids = set(json.load(f))
with open(meta_json_path, "r") as f:
    val_metadata = json.load(f)

eval_image_ids = [img_id for img_id in full_image_ids if img_id not in pilot_image_ids]
print(f"Total FULL images: {len(full_image_ids)}, Pilot images to reuse: {len(pilot_image_ids)}")
print(f"Remaining images to evaluate in K4: {len(eval_image_ids)}")

# 3. Locate validation image directory precisely
candidate_dirs = [
    "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
]
IMAGE_DIR = None
for c in candidate_dirs:
    if os.path.exists(c):
        IMAGE_DIR = c
        break

if not IMAGE_DIR:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs:
            dirs.remove("train")
        if "ILSVRC2012_val_00000001.JPEG" in files:
            IMAGE_DIR = root
            break

if not IMAGE_DIR:
    IMAGE_DIR = "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"

print(f"Validation images located at: {IMAGE_DIR}")

# 4. Load Models
print("\n=== Initializing Headline Models ===")
RESOLUTIONS = [224, 320, 384, 448]

models = {}
for m_key in ["deit_base", "efficientnet_b3"]:
    print(f"Loading {m_key}...")
    models[m_key] = create_model_instance(m_key, resolution=224, pretrained=True, device=device)

flex_models_fp = {}
for r in RESOLUTIONS:
    print(f"Loading flexivit_base F-p @ {r}...")
    flex_models_fp[r] = create_model_instance("flexivit_base", resolution=r, arm="F-p", pretrained=True, device=device)

SHARDS_DIR = "/kaggle/working/shards"
os.makedirs(SHARDS_DIR, exist_ok=True)

CONDITIONS = [
    ("jpeg_compression", 3),
    ("contrast", 3),
]
BATCH_SIZE = 64

print(f"\n=== Executing Tier 1 Part B Grid across {len(CONDITIONS)} Conditions ===")

for cond_name, severity in CONDITIONS:
    shard_file = os.path.join(SHARDS_DIR, f"shard_k4_{cond_name}_s{severity}.parquet")
    if os.path.exists(shard_file):
        print(f"Shard {shard_file} already exists, skipping.")
        continue
        
    print(f"\nProcessing Condition: {cond_name} (Severity {severity}) on {len(eval_image_ids)} images...")
    cond_start = time.time()
    records = []
    
    for b_start in range(0, len(eval_image_ids), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(eval_image_ids))
        batch_ids = eval_image_ids[b_start:b_end]
        
        batch_tensors_448 = []
        batch_labels = []
        
        for img_id in batch_ids:
            img_path = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
            if not os.path.exists(img_path):
                continue
            pil_img = Image.open(img_path)
            arr_448, _ = preprocess_image_448(pil_img)
            
            corr_arr = apply_corruption(arr_448, image_id=img_id, corruption_name=cond_name, severity=severity)
            t_448 = torch.from_numpy(corr_arr).permute(2, 0, 1).float() / 255.0
            batch_tensors_448.append(t_448)
            batch_labels.append(val_metadata[img_id]["class_idx"])
            
        if not batch_tensors_448:
            continue
            
        t_batch_448 = torch.stack(batch_tensors_448, dim=0).to(device)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        b_size = len(batch_tensors_448)
        
        res_tensors = {}
        for r in RESOLUTIONS:
            res_tensors[r] = resize_tensor_torch(t_batch_448, target_size=r)
            
        # Standard models: DeiT-B, EfficientNet-B3
        for m_key in ["deit_base", "efficientnet_b3"]:
            m = models[m_key]
            m_tag = MODEL_TAGS[m_key]
            for r in RESOLUTIONS:
                inp = normalize_tensor(res_tensors[r], model_tag=m_tag)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits = m(inp)
                preds = logits.argmax(dim=-1).cpu().numpy()
                probs = torch.softmax(logits, dim=-1)
                
                for idx in range(b_size):
                    p = int(preds[idx])
                    lbl = int(labels[idx].item())
                    records.append({
                        "image_id": batch_ids[idx],
                        "condition": cond_name,
                        "severity": severity,
                        "model": m_key,
                        "arm": "standard",
                        "resolution": r,
                        "label": lbl,
                        "pred": p,
                        "correct": bool(p == lbl),
                        "confidence": float(probs[idx, p].item()),
                    })
                    
        # FlexiViT-B F-p arm
        m_tag_flex = MODEL_TAGS["flexivit_base"]
        for r in RESOLUTIONS:
            inp = normalize_tensor(res_tensors[r], model_tag=m_tag_flex)
            m_fp = flex_models_fp[r]
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits_fp = m_fp(inp)
            preds_fp = logits_fp.argmax(dim=-1).cpu().numpy()
            probs_fp = torch.softmax(logits_fp, dim=-1)
            
            for idx in range(b_size):
                p_fp = int(preds_fp[idx])
                lbl = int(labels[idx].item())
                records.append({
                    "image_id": batch_ids[idx],
                    "condition": cond_name,
                    "severity": severity,
                    "model": "flexivit_base",
                    "arm": "F-p",
                    "resolution": r,
                    "label": lbl,
                    "pred": p_fp,
                    "correct": bool(p_fp == lbl),
                    "confidence": float(probs_fp[idx, p_fp].item()),
                })
                
        if b_end % 5000 < BATCH_SIZE and b_end > 0:
            print(f"Progress {cond_name}: {b_end}/{len(eval_image_ids)} images ({time.time() - cond_start:.1f}s)")
            
    df_cond = pd.DataFrame(records)
    tmp_path = shard_file + ".tmp"
    df_cond.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, shard_file)
    print(f"Completed {cond_name} s{severity}: saved {len(df_cond)} records to {shard_file}")

elapsed_wall = time.time() - start_wall_time
wall_h = elapsed_wall / 3600.0
device_h = wall_h * max(1, n_gpus)
print(f"\nK4-Headline-B Complete: Elapsed = {wall_h:.3f} Wall-Clock h, {device_h:.3f} Device-Hours")

with open("/kaggle/working/run_meta.json", "w") as f:
    json.dump(get_environment_metadata(device), f, indent=2)
