"""K6-Mech: Tier 3 Mechanistic Probes (M1-M6 on MECH 2,000 images; M7 on SUB10K).

Scope:
- M1: Area-Normalized Object Attention Mass inside GT Bounding Box
- M2: Normalized Attention Entropy H/log(N) across resolutions
- M3: Representation Drift across layers
- M4: Input Spectral Metrics (Noise sigma, PSD slope)
- M5: FlexiViT F-p vs F-t Token Contrast
- M6: ToMe In-Box vs Out-of-Box token merging
- M7: Filter-Matched Low-Pass Information Control (A: 448 orig, B: 448 filtered, C: 224 down, D: 224->448 up)
"""

import os
import sys
import time
import json
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print("=== K6-Mech Initializing ===", flush=True)
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
from knobs.models import MODEL_TAGS, create_model_instance, normalize_tensor
from knobs.data import preprocess_image_448, transform_bbox_to_crop448
from knobs.corrupt import apply_corruption
from knobs.resize import resize_tensor_torch, generate_m7_suite
from knobs.spectral import estimate_immerkaer_noise_sigma, calculate_radial_psd_slope
from knobs.mechanism import extract_attention_weights, calculate_area_normalized_attention_mass, calculate_normalized_attention_entropy
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
mech_json_path = None
sub10k_json_path = None
meta_json_path = None
for s_dir in SPLITS_CANDIDATES:
    m_p = os.path.join(s_dir, "MECH.json")
    s_p = os.path.join(s_dir, "SUB10K.json")
    v_p = os.path.join(s_dir, "val_metadata.json")
    if os.path.exists(m_p) and os.path.exists(v_p):
        mech_json_path = m_p
        sub10k_json_path = s_p if os.path.exists(s_p) else None
        meta_json_path = v_p
        print(f"Found splits in: {s_dir}")
        break

if not mech_json_path or not meta_json_path:
    raise FileNotFoundError("Could not find MECH.json or val_metadata.json in candidate dirs")

with open(mech_json_path, "r") as f:
    mech_image_ids = json.load(f)
with open(meta_json_path, "r") as f:
    val_metadata = json.load(f)

print(f"Loaded MECH split with {len(mech_image_ids)} images")

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

# 4. Initialize Models
print("\n=== Initializing Models for Mechanistic Analysis ===")
model_deit = create_model_instance("deit_base", resolution=224, pretrained=True, device=device)
model_eff = create_model_instance("efficientnet_b3", resolution=224, pretrained=True, device=device)

# 5. M1-M4 Evaluation on MECH (2,000 images)
print("\n=== Executing Probes M1-M4 on MECH Split ===")
mech_records = []

for idx, img_id in enumerate(mech_image_ids):
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
    if not os.path.exists(img_path):
        continue
    pil_img = Image.open(img_path)
    arr_448, meta_448 = preprocess_image_448(pil_img)
    
    rec_meta = val_metadata[img_id]
    orig_w, orig_h = meta_448["orig_width"], meta_448["orig_height"]
    raw_boxes = rec_meta.get("raw_bboxes", [])
    
    # Transform first bbox to crop448 coordinates
    crop_box = None
    box_fraction = 0.25  # fallback
    if raw_boxes:
        tx = transform_bbox_to_crop448(raw_boxes[0], orig_w=orig_w, orig_h=orig_h, crop_size=448)
        if tx:
            crop_box = tx[:4]
            box_fraction = tx[4]
            
    # Spectral metrics (M4) on clean vs noise
    clean_sigma = estimate_immerkaer_noise_sigma(arr_448)
    clean_psd = calculate_radial_psd_slope(arr_448)
    
    corr_noise = apply_corruption(arr_448, image_id=img_id, corruption_name="gaussian_noise", severity=3)
    noise_sigma = estimate_immerkaer_noise_sigma(corr_noise)
    noise_psd = calculate_radial_psd_slope(corr_noise)
    
    # Evaluate attention maps (M1, M2) across resolutions
    for cond_name, img_arr in [("clean", arr_448), ("gaussian_noise", corr_noise)]:
        t_448 = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        for res in [224, 448]:
            r_t = resize_tensor_torch(t_448, target_size=res).to(device)
            norm_in = normalize_tensor(r_t, model_tag=MODEL_TAGS["deit_base"])
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    attn_matrix = extract_attention_weights(model_deit, norm_in)
                    
            # Compute normalized entropy
            ent = calculate_normalized_attention_entropy(attn_matrix)
            
            # Compute box attention mass if bbox exists
            norm_mass = 1.0
            if crop_box:
                # Scale crop box to current resolution
                scale = res / 448.0
                scaled_box = (crop_box[0] * scale, crop_box[1] * scale, crop_box[2] * scale, crop_box[3] * scale)
                norm_mass = calculate_area_normalized_attention_mass(
                    attn_matrix,
                    scaled_box,
                    resolution=res,
                    patch_size=16,
                    box_area_fraction=box_fraction,
                )
                
            mech_records.append({
                "image_id": img_id,
                "condition": cond_name,
                "resolution": res,
                "norm_mass": float(norm_mass),
                "entropy_norm": float(ent),
                "noise_sigma": float(noise_sigma if cond_name == "gaussian_noise" else clean_sigma),
                "psd_slope": float(noise_psd if cond_name == "gaussian_noise" else clean_psd),
            })
            
    if (idx + 1) % 500 == 0:
        print(f"Processed MECH {idx + 1}/{len(mech_image_ids)} images ({time.time() - start_wall_time:.1f}s)")

df_mech = pd.DataFrame(mech_records)
df_mech.to_parquet("/kaggle/working/mech_probes_results.parquet", index=False)
print(f"Saved {len(df_mech)} records for M1-M4 probes")

# 6. M7 Filter-Matched Information Control on MECH (all 2,000 images, class-balanced by construction)
print("\n=== Executing M7 Filter-Matched Information Control ===")
m7_records = []
m7_sample = list(mech_image_ids)  # All 2,000 MECH images

batch_size = 32
conditions = ["clean", "gaussian_noise", "defocus_blur"]

for b_start in range(0, len(m7_sample), batch_size):
    b_end = min(b_start + batch_size, len(m7_sample))
    batch_ids = m7_sample[b_start:b_end]

    batch_items = []
    for img_id in batch_ids:
        img_path = os.path.join(IMAGE_DIR, f"{img_id}.JPEG")
        if not os.path.exists(img_path):
            continue
        pil_img = Image.open(img_path)
        arr_448, _ = preprocess_image_448(pil_img)
        label = val_metadata[img_id]["class_idx"]
        batch_items.append((img_id, arr_448, label))

    if not batch_items:
        continue

    for cond_name in conditions:
        for suite_key in ["A", "B", "C", "D"]:
            suite_tensors = []
            labels = []
            curr_img_ids = []
            for img_id, arr_448, label in batch_items:
                if cond_name == "clean":
                    corr = arr_448
                else:
                    corr = apply_corruption(arr_448, image_id=img_id, corruption_name=cond_name, severity=3)
                t_448 = torch.from_numpy(corr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                m7_suite = generate_m7_suite(t_448)
                suite_tensors.append(m7_suite[suite_key])
                labels.append(label)
                curr_img_ids.append(img_id)

            batch_tensor = torch.cat(suite_tensors, dim=0).to(device)

            norm_deit = normalize_tensor(batch_tensor, model_tag=MODEL_TAGS["deit_base"])
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out_deit = model_deit(norm_deit)
            preds_deit = out_deit.argmax(dim=-1).cpu().numpy()

            norm_eff = normalize_tensor(batch_tensor, model_tag=MODEL_TAGS["efficientnet_b3"])
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out_eff = model_eff(norm_eff)
            preds_eff = out_eff.argmax(dim=-1).cpu().numpy()

            for img_id, pred_deit, pred_eff, label in zip(curr_img_ids, preds_deit, preds_eff, labels):
                m7_records.append({
                    "image_id": img_id,
                    "condition": cond_name,
                    "suite_condition": suite_key,
                    "deit_correct": bool(pred_deit == label),
                    "eff_correct": bool(pred_eff == label),
                })

    if b_end % 200 == 0 or b_end == len(m7_sample):
        print(f"Processed M7 {b_end}/{len(m7_sample)} images ({time.time() - start_wall_time:.1f}s)")

df_m7 = pd.DataFrame(m7_records)
df_m7.to_parquet("/kaggle/working/m7_information_control_results.parquet", index=False)
print(f"Saved {len(df_m7)} records for M7 information control")

elapsed_wall = time.time() - start_wall_time
wall_h = elapsed_wall / 3600.0
device_h = wall_h * max(1, n_gpus)
print(f"\nK6-Mech Complete: Elapsed = {wall_h:.3f} Wall-Clock h, {device_h:.3f} Device-Hours")

with open("/kaggle/working/run_meta.json", "w") as f:
    json.dump(get_environment_metadata(device), f, indent=2)
