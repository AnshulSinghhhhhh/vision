"""K5-Controls: Tier 2 Dose-Response, Frequency Controls, and Mismatch Controls on SUB10K (10,000 images).

Scope:
- Target: SUB10K (10,000 images)
- Dose-Response: Noise (s1, s5) and Defocus (s1, s5) across models & resolutions
- Frequency-Controlled Noise: Low, Mid, High band noise with identical total variance
- ToMe Token Merging: r=4, r=8 under corruption
- Resolution Mismatch Controls: DeiT-384 native vs downsampled, EfficientNet-B3 BN-recalibration (E-bn)
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print("=== K5-Controls Initializing ===", flush=True)
start_wall_time = time.time()

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
from knobs.models import MODEL_TAGS, create_model_instance, normalize_tensor, recalibrate_bn_statistics
from knobs.data import preprocess_image_448
from knobs.corrupt import apply_corruption, generate_frequency_controlled_noise
from knobs.resize import resize_tensor_torch
from knobs.tokens import patch_vit_with_tome
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
sub10k_json_path = None
meta_json_path = None
for s_dir in SPLITS_CANDIDATES:
    s_p = os.path.join(s_dir, "SUB10K.json")
    m_p = os.path.join(s_dir, "val_metadata.json")
    if os.path.exists(s_p) and os.path.exists(m_p):
        sub10k_json_path = s_p
        meta_json_path = m_p
        print(f"Found splits in: {s_dir}")
        break

if not sub10k_json_path or not meta_json_path:
    raise FileNotFoundError("Could not find SUB10K.json or val_metadata.json in candidate dirs")

with open(sub10k_json_path, "r") as f:
    sub10k_image_ids = json.load(f)
with open(meta_json_path, "r") as f:
    val_metadata = json.load(f)

print(f"Loaded SUB10K split with {len(sub10k_image_ids)} images")

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
    IMAGE_DIR = "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"

print(f"Validation images located at: {IMAGE_DIR}")

# 4. Load Models and Controls
print("\n=== Initializing Models & Controls ===")
RESOLUTIONS = [224, 320, 384, 448]

models = {}
for m_key in ["deit_base", "efficientnet_b3"]:
    print(f"Loading {m_key}...")
    models[m_key] = create_model_instance(m_key, resolution=224, pretrained=True, device=device)

flex_models_fp = {}
flex_models_ft = {}
for r in RESOLUTIONS:
    print(f"Loading flexivit_base F-p @ {r}...")
    flex_models_fp[r] = create_model_instance("flexivit_base", resolution=r, arm="F-p", pretrained=True, device=device)
    print(f"Loading flexivit_base F-t @ {r}...")
    flex_models_ft[r] = create_model_instance("flexivit_base", resolution=r, arm="F-t", pretrained=True, device=device)

# Controls
models["deit_base_384"] = create_model_instance("deit_base_384", resolution=384, pretrained=True, device=device)

# ToMe variants
n_blocks = len(models["deit_base"].blocks)
models_tome_r4 = patch_vit_with_tome(
    create_model_instance("deit_base", resolution=224, pretrained=True, device=device),
    [4] * n_blocks
)
models_tome_r8 = patch_vit_with_tome(
    create_model_instance("deit_base", resolution=224, pretrained=True, device=device),
    [8] * n_blocks
)

# BN-recalibrated EfficientNet
print("Preparing BN-recalibrated EfficientNet...")
model_ebn = create_model_instance("efficientnet_b3", resolution=224, pretrained=True, device=device)
# Recalibrate on first 100 images
recal_imgs = []
for img_id in sub10k_image_ids[:100]:
    p = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
    if os.path.exists(p):
        arr, _ = preprocess_image_448(Image.open(p))
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        recal_imgs.append(resize_tensor_torch(t.unsqueeze(0), 224))
if recal_imgs:
    recal_tensor = torch.cat(recal_imgs, dim=0).to(device)
    norm_recal = normalize_tensor(recal_tensor, model_tag=MODEL_TAGS["efficientnet_b3"])
    recalibrate_bn_statistics(model_ebn, norm_recal, device=device)
models["efficientnet_b3_ebn"] = model_ebn

SHARDS_DIR = "/kaggle/working/shards"
os.makedirs(SHARDS_DIR, exist_ok=True)

# 5. Execute Evaluation
DOSE_CONDITIONS = [
    ("gaussian_noise", 1),
    ("gaussian_noise", 5),
    ("defocus_blur", 1),
    ("defocus_blur", 5),
]

RESOLUTIONS = [224, 320, 384, 448]
BATCH_SIZE = 64

print(f"\n=== Executing Dose-Response Grid on SUB10K ===")
all_records = []

for cond_name, severity in DOSE_CONDITIONS:
    shard_file = os.path.join(SHARDS_DIR, f"shard_k5_{cond_name}_s{severity}.parquet")
    if os.path.exists(shard_file):
        print(f"Shard {shard_file} already exists, skipping.")
        continue
        
    print(f"\nProcessing Dose Condition: {cond_name} (Severity {severity})...")
    cond_records = []
    
    for b_start in range(0, len(sub10k_image_ids), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(sub10k_image_ids))
        batch_ids = sub10k_image_ids[b_start:b_end]
        
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
                    cond_records.append({
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

        # Controls: DeiT-384 native
        inp_384 = normalize_tensor(res_tensors[384], model_tag=MODEL_TAGS["deit_base_384"])
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_384 = models["deit_base_384"](inp_384)
        preds_384 = logits_384.argmax(dim=-1).cpu().numpy()
        probs_384 = torch.softmax(logits_384, dim=-1)
        for idx in range(b_size):
            p = int(preds_384[idx])
            lbl = int(labels[idx].item())
            cond_records.append({
                "image_id": batch_ids[idx],
                "condition": cond_name,
                "severity": severity,
                "model": "deit_base_384",
                "arm": "native_384",
                "resolution": 384,
                "label": lbl,
                "pred": p,
                "correct": bool(p == lbl),
                "confidence": float(probs_384[idx, p].item()),
            })

        # Controls: ToMe r4, r8 at 224 and 448
        for tome_name, tome_m, tome_arm in [("deit_base_tome_r4", models_tome_r4, "tome_r4"), ("deit_base_tome_r8", models_tome_r8, "tome_r8")]:
            for r_tome in [224, 448]:
                inp_tome = normalize_tensor(res_tensors[r_tome], model_tag=MODEL_TAGS["deit_base"])
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits_tome = tome_m(inp_tome)
                preds_tome = logits_tome.argmax(dim=-1).cpu().numpy()
                probs_tome = torch.softmax(logits_tome, dim=-1)
                for idx in range(b_size):
                    p = int(preds_tome[idx])
                    lbl = int(labels[idx].item())
                    cond_records.append({
                        "image_id": batch_ids[idx],
                        "condition": cond_name,
                        "severity": severity,
                        "model": tome_name,
                        "arm": tome_arm,
                        "resolution": r_tome,
                        "label": lbl,
                        "pred": p,
                        "correct": bool(p == lbl),
                        "confidence": float(probs_tome[idx, p].item()),
                    })

        # Controls: EfficientNet BN-recalibrated at 224 and 448
        for r_ebn in [224, 448]:
            inp_ebn = normalize_tensor(res_tensors[r_ebn], model_tag=MODEL_TAGS["efficientnet_b3"])
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits_ebn = models["efficientnet_b3_ebn"](inp_ebn)
            preds_ebn = logits_ebn.argmax(dim=-1).cpu().numpy()
            probs_ebn = torch.softmax(logits_ebn, dim=-1)
            for idx in range(b_size):
                p = int(preds_ebn[idx])
                lbl = int(labels[idx].item())
                cond_records.append({
                    "image_id": batch_ids[idx],
                    "condition": cond_name,
                    "severity": severity,
                    "model": "efficientnet_b3_ebn",
                    "arm": "bn_recal",
                    "resolution": r_ebn,
                    "label": lbl,
                    "pred": p,
                    "correct": bool(p == lbl),
                    "confidence": float(probs_ebn[idx, p].item()),
                })
                    
        # FlexiViT F-p and F-t
        m_tag_flex = MODEL_TAGS["flexivit_base"]
        for r in RESOLUTIONS:
            inp = normalize_tensor(res_tensors[r], model_tag=m_tag_flex)
            m_fp = flex_models_fp[r]
            m_ft = flex_models_ft[r]
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits_fp = m_fp(inp)
                    logits_ft = m_ft(inp)
            preds_fp = logits_fp.argmax(dim=-1).cpu().numpy()
            probs_fp = torch.softmax(logits_fp, dim=-1)
            preds_ft = logits_ft.argmax(dim=-1).cpu().numpy()
            probs_ft = torch.softmax(logits_ft, dim=-1)
            for idx in range(b_size):
                lbl = int(labels[idx].item())
                p_fp = int(preds_fp[idx])
                cond_records.append({
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
                p_ft = int(preds_ft[idx])
                cond_records.append({
                    "image_id": batch_ids[idx],
                    "condition": cond_name,
                    "severity": severity,
                    "model": "flexivit_base",
                    "arm": "F-t",
                    "resolution": r,
                    "label": lbl,
                    "pred": p_ft,
                    "correct": bool(p_ft == lbl),
                    "confidence": float(probs_ft[idx, p_ft].item()),
                })
                
    df_cond = pd.DataFrame(cond_records)
    tmp_path = shard_file + ".tmp"
    df_cond.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, shard_file)
    print(f"Saved {len(df_cond)} records for {cond_name} s{severity}")
    all_records.extend(cond_records)

# 6. Frequency-Controlled Noise
print("\n=== Executing Frequency-Controlled Noise (Low, Mid, High Bands) ===")
freq_records = []
for band in ["low", "mid", "high"]:
    for b_start in range(0, min(2000, len(sub10k_image_ids)), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, min(2000, len(sub10k_image_ids)))
        batch_ids = sub10k_image_ids[b_start:b_end]
        batch_tensors = []
        batch_labels = []
        for img_id in batch_ids:
            p = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
            if not os.path.exists(p):
                continue
            arr, _ = preprocess_image_448(Image.open(p))
            corr = generate_frequency_controlled_noise(arr, band=band, sigma_total=30.0, seed=42)
            t = torch.from_numpy(corr).permute(2, 0, 1).float() / 255.0
            batch_tensors.append(t)
            batch_labels.append(val_metadata[img_id]["class_idx"])
            
        if not batch_tensors:
            continue
        t_batch = torch.stack(batch_tensors, dim=0).to(device)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        b_size = len(batch_tensors)
        
        for r in [224, 448]:
            r_t = resize_tensor_torch(t_batch, target_size=r)
            for m_key in ["deit_base", "efficientnet_b3"]:
                inp = normalize_tensor(r_t, model_tag=MODEL_TAGS[m_key])
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits = models[m_key](inp)
                preds = logits.argmax(dim=-1).cpu().numpy()
                probs = torch.softmax(logits, dim=-1)
                for idx in range(b_size):
                    p = int(preds[idx])
                    lbl = int(labels[idx].item())
                    freq_records.append({
                        "image_id": batch_ids[idx],
                        "condition": f"freq_noise_{band}",
                        "severity": 3,
                        "model": m_key,
                        "arm": f"band_{band}",
                        "resolution": r,
                        "label": lbl,
                        "pred": p,
                        "correct": bool(p == lbl),
                        "confidence": float(probs[idx, p].item()),
                    })

df_freq = pd.DataFrame(freq_records)
df_freq.to_parquet("/kaggle/working/shards/shard_k5_freq_noise.parquet", index=False)
print(f"Saved {len(df_freq)} records for frequency-controlled noise")

# Aggregate master controls dataframe
all_k5_dfs = [pd.read_parquet(os.path.join(SHARDS_DIR, f)) for f in os.listdir(SHARDS_DIR) if f.startswith("shard_k5_")]
df_master_k5 = pd.concat(all_k5_dfs, ignore_index=True)
master_k5_path = "/kaggle/working/k5_controls_results.parquet"
df_master_k5.to_parquet(master_k5_path, index=False)
print(f"Master K5 controls results saved to {master_k5_path} ({len(df_master_k5)} records)")

elapsed_wall = time.time() - start_wall_time
wall_h = elapsed_wall / 3600.0
device_h = wall_h * max(1, n_gpus)
print(f"\nK5-Controls Complete: Elapsed = {wall_h:.3f} Wall-Clock h, {device_h:.3f} Device-Hours")

with open("/kaggle/working/run_meta.json", "w") as f:
    json.dump(get_environment_metadata(device), f, indent=2)
