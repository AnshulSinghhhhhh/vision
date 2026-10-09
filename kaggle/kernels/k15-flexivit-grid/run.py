"""K15: FlexiViT resolution x patch size grid on N=3,000 balanced images.

Evaluates:
- flexivit_base across resolutions {224, 320, 448} and patch sizes {16, 24, 32}
- Checks divisibility: if (res % patch != 0), logs why and skips.
- Conditions: clean (s0), gaussian_noise (s3, s5), defocus_blur (s3).
- Records tokens per config: (res // patch) ** 2.
- N = 3,000 balanced images from FULL.json.
- Estimated GPU budget: ~0.6 hours on NVIDIA T4.
- Writes one parquet shard per (config, condition, severity) to shards/ and skips finished shards on resume.
"""

import os
import sys
import time
import json
import zipfile
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import subprocess
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "imagecorruptions", "--quiet"], check=False)
except Exception:
    pass

# 1. Unpack any zips if present on Kaggle
for zip_dir in ["/kaggle/input/datasets/anshulsingh45/knobs-code", "/kaggle/input/knobs-code", "/kaggle/input"]:
    if os.path.exists(zip_dir):
        for root, dirs, files in os.walk(zip_dir):
            for f in files:
                if f.endswith(".zip"):
                    zpath = os.path.join(root, f)
                    dest = os.path.join("/tmp", f[:-4])
                    if not os.path.exists(dest):
                        try:
                            with zipfile.ZipFile(zpath, "r") as zf:
                                zf.extractall(dest)
                            print(f"Unpacked {f} to {dest}", flush=True)
                        except Exception as e:
                            print(f"Error unpacking {f}: {e}", flush=True)

CANDIDATE_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
    "/tmp/knobs-code/src",
    "/tmp/knobs-code-staging/src",
    "/tmp/src",
    "/kaggle/input/datasets/anshulsingh45/knobs-code/src",
    "/kaggle/input/knobs-code/src",
]
for base in ["/tmp", "/kaggle/input"]:
    if os.path.exists(base):
        for root, dirs, _ in os.walk(base):
            if "knobs" in dirs and os.path.exists(os.path.join(root, "knobs", "__init__.py")):
                if root not in CANDIDATE_PATHS:
                    CANDIDATE_PATHS.append(root)

for p in CANDIDATE_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        print(f"Added knobs path to sys.path: {p}", flush=True)

import timm
from knobs.corrupt import apply_corruption, CORRUPTION_BACKEND
from knobs.data import preprocess_image_448, balanced_subset
from knobs.resize import resize_tensor_torch
from knobs.models import normalize_tensor, MODEL_TAGS
from knobs.run_grid import get_environment_metadata


# Verify corruption backend is real imagecorruptions
if CORRUPTION_BACKEND != "imagecorruptions":
    raise RuntimeError(
        f"K15 execution ABORTED: real 'imagecorruptions' backend is required, but current backend is '{CORRUPTION_BACKEND}'."
    )


RESOLUTIONS = [224, 320, 448]
PATCH_SIZES = [16, 24, 32]
CONDITIONS = [
    ("clean", 0),
    ("gaussian_noise", 3),
    ("gaussian_noise", 5),
    ("defocus_blur", 3),
]


def run_k15(
    image_dir: str,
    image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str = "/kaggle/working",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    model_tag = MODEL_TAGS["flexivit_base"]
    all_shards = []

    for res in RESOLUTIONS:
        for patch in PATCH_SIZES:
            # Check divisibility
            if res % patch != 0:
                print(f"[SKIP] Resolution {res} is NOT divisible by patch size {patch} (remainder {res % patch}). Skipping configuration.", flush=True)
                continue

            grid_dim = res // patch
            num_tokens = grid_dim * grid_dim
            cfg_name = f"flexivit_r{res}_p{patch}"
            print(f"\n=== Configuration: {cfg_name} (grid: {grid_dim}x{grid_dim}, {num_tokens} tokens) ===", flush=True)

            print(f"Instantiating model {model_tag} at img_size={res}, patch_size={patch}...", flush=True)
            model = timm.create_model(
                model_tag,
                pretrained=pretrained,
                img_size=res,
                patch_size=patch,
            )
            model.eval()
            model.to(device)

            for cond_name, severity in CONDITIONS:
                shard_file = shards_dir / f"shard_k15_{cfg_name}_{cond_name}_s{severity}.parquet"
                all_shards.append(shard_file)

                if shard_file.exists():
                    print(f"[RESUME] Skipping completed shard: {shard_file.name}", flush=True)
                    continue

                print(f"--- Running {cfg_name} on {cond_name} s{severity} ({len(image_ids)} images) ---", flush=True)
                cond_records = []
                t0 = time.time()

                for b_idx in range(0, len(image_ids), batch_size):
                    b_ids = image_ids[b_idx:b_idx + batch_size]
                    corrupted_tensors = []
                    valid_ids = []

                    for img_id in b_ids:
                        img_path = Path(image_dir) / f"{img_id}.JPEG"
                        if not img_path.exists():
                            continue
                        try:
                            clean_arr_448, _ = preprocess_image_448(str(img_path))
                            corr_arr_448 = apply_corruption(clean_arr_448, img_id, cond_name, severity)
                            corr_t_448 = torch.from_numpy(corr_arr_448).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                            
                            # Downsample to target resolution with antialiasing if res < 448
                            if res < 448:
                                corr_t_res = resize_tensor_torch(corr_t_448, target_size=res)
                            else:
                                corr_t_res = corr_t_448

                            corrupted_tensors.append(corr_t_res.squeeze(0))
                            valid_ids.append(img_id)
                        except Exception as ex:
                            print(f"Warning: Error processing {img_id}: {ex}", flush=True)
                            continue

                    if not valid_ids:
                        continue

                    batch_tensor = torch.stack(corrupted_tensors).to(device)
                    norm_batch = normalize_tensor(batch_tensor, model_tag)

                    with torch.inference_mode():
                        if device.type == "cuda":
                            with torch.amp.autocast("cuda"):
                                logits = model(norm_batch)
                        else:
                            logits = model(norm_batch)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()

                    for i, img_id in enumerate(valid_ids):
                        lbl = val_metadata[img_id]["class_idx"]
                        p = int(preds[i])
                        cond_records.append({
                            "image_id": img_id,
                            "model": "flexivit_base",
                            "resolution": res,
                            "patch_size": patch,
                            "tokens": num_tokens,
                            "condition": cond_name,
                            "severity": severity,
                            "label": lbl,
                            "pred": p,
                            "correct": (p == lbl),
                        })

                    if (b_idx // batch_size + 1) % 25 == 0:
                        elapsed = time.time() - t0
                        print(f"  Processed {min(b_idx + batch_size, len(image_ids))}/{len(image_ids)} images ({elapsed:.1f}s)", flush=True)

                df_cond = pd.DataFrame(cond_records)
                df_cond.to_parquet(shard_file, index=False)
                print(f"Saved {len(df_cond)} records to {shard_file.name}", flush=True)

                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Clean up model after completing all conditions for this config
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Consolidate all shards into final dataset
    print("\n=== Consolidating All K15 Shards ===", flush=True)
    all_dfs = [pd.read_parquet(s) for s in all_shards if s.exists()]
    if all_dfs:
        df_final = pd.concat(all_dfs, ignore_index=True)
        final_parquet = out_path / "shard_flexivit_grid.parquet"
        df_final.to_parquet(final_parquet, index=False)
        print(f"Final K15 dataset written to {final_parquet} ({len(df_final)} rows)", flush=True)

    # Save run metadata
    meta = get_environment_metadata(device)
    meta["kernel"] = "k15-flexivit-grid"
    with open(out_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {out_path / 'run_meta.json'}", flush=True)


def main():
    print("=== Launching K15-FlexiViT-Grid Kernel ===", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SPLITS_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "splits")),
        "/tmp/knobs-code-staging/splits",
        "/tmp/knobs-code/splits",
        "/tmp/splits",
        "/kaggle/input/datasets/anshulsingh45/knobs-code/splits",
        "/kaggle/input/knobs-code/splits",
    ]
    full_path, val_meta_path = None, None
    for sdir in SPLITS_DIRS:
        fp = Path(sdir) / "FULL.json"
        vp = Path(sdir) / "val_metadata.json"
        if fp.exists() and vp.exists():
            full_path, val_meta_path = fp, vp
            break

    if not full_path or not val_meta_path:
        for base in ["/tmp", "/kaggle/input"]:
            if os.path.exists(base):
                for root, dirs, _ in os.walk(base):
                    fp = Path(root) / "FULL.json"
                    vp = Path(root) / "val_metadata.json"
                    if fp.exists() and vp.exists():
                        full_path, val_meta_path = fp, vp
                        break
            if full_path:
                break

    if not full_path or not val_meta_path:
        raise FileNotFoundError("Could not find FULL.json or val_metadata.json in search paths")

    with open(full_path, "r", encoding="utf-8") as f:
        full_ids = json.load(f)
    with open(val_meta_path, "r", encoding="utf-8") as f:
        val_metadata = json.load(f)

    # Balanced subset of 3,000 images (3 per class)
    image_ids_3k = balanced_subset(full_ids, val_metadata, n=3000, seed=0)
    print(f"Extracted N={len(image_ids_3k)} class-balanced images (seed=0)", flush=True)

    IMG_DIRS = [
        "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
        "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "val")),
    ]
    img_dir = None
    for idir in IMG_DIRS:
        if os.path.exists(idir):
            img_dir = idir
            break

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    if not img_dir:
        print("[Notice] Image directory not found on host. Kernel prepared for Kaggle execution.", flush=True)
        return

    run_k15(
        image_dir=img_dir,
        image_ids=image_ids_3k,
        val_metadata=val_metadata,
        output_dir=out_dir,
        batch_size=32,
        device=device,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
