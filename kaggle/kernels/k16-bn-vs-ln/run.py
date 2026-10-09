"""K16: BatchNorm (ResNet-50) vs LayerNorm (ConvNeXt-Base) under noise and resolution controls.

Models evaluated:
1. resnet50 (BatchNorm CNN) - default tag: resnet50.a1_in1k
2. convnext_base (LayerNorm CNN) - default tag: convnext_base.fb_in22k_ft_in1k

Conditions & Suites:
- clean (s0): A_orig448, B_filtered448, C_down224
- gaussian_noise (s3, s5): A_orig448, B_filtered448, C_down224, E_noise_matched

N = 5,000 balanced images from FULL.json.
Estimated GPU budget: ~0.5 hours on NVIDIA T4.
Writes one parquet shard per (model, condition, severity) to shards/ and skips finished shards on resume.
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
from knobs.resize import build_m7_v2_suite
from knobs.models import normalize_tensor
from knobs.run_grid import get_environment_metadata


# Verify corruption backend is real imagecorruptions
if CORRUPTION_BACKEND != "imagecorruptions":
    raise RuntimeError(
        f"K16 execution ABORTED: real 'imagecorruptions' backend is required, but current backend is '{CORRUPTION_BACKEND}'."
    )


# Offline weight candidate search paths on Kaggle
WEIGHTS_CANDIDATE_DIRS = [
    "/kaggle/input/knobs-weights",
    "/kaggle/input/datasets/anshulsingh45/knobs-weights",
    "/kaggle/input/timm-pretrained-weights",
    "/kaggle/input/pretrained-models",
    os.path.expanduser("~/.cache/torch/hub/checkpoints"),
]
for wdir in WEIGHTS_CANDIDATE_DIRS:
    if os.path.exists(wdir):
        print(f"Found weights directory: {wdir}", flush=True)
        torch.hub.set_dir(wdir)
        break

MODEL_TAGS_K16 = {
    "resnet50": "resnet50.a1_in1k",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",
}

CONDITIONS_SUITES = [
    ("clean", 0, ["A_orig448", "B_filtered448", "C_down224"]),
    ("gaussian_noise", 3, ["A_orig448", "B_filtered448", "C_down224", "E_noise_matched"]),
    ("gaussian_noise", 5, ["A_orig448", "B_filtered448", "C_down224", "E_noise_matched"]),
]


def load_model_safely(model_key: str, tag: str, device: torch.device) -> Optional[nn.Module]:
    """Attempts to instantiate model with pretrained weights offline or online."""
    try:
        model = timm.create_model(tag, pretrained=True)
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        print(f"Warning: Failed to load pretrained {tag} ({e}). Attempting fallback architecture tag...", flush=True)
        try:
            model = timm.create_model(model_key, pretrained=True)
            model.eval()
            model.to(device)
            return model
        except Exception as e2:
            print(f"Error: Could not load pretrained weights for {model_key}: {e2}", flush=True)
            return None


def run_k16(
    image_dir: str,
    image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str = "/kaggle/working",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing K16 Models on {device} ===", flush=True)
    models_dict = {}
    actual_tags = {}

    for model_key, default_tag in MODEL_TAGS_K16.items():
        print(f"Loading {model_key} (tag: {default_tag})...", flush=True)
        m = load_model_safely(model_key, default_tag, device)
        if m is not None:
            models_dict[model_key] = m
            actual_tags[model_key] = default_tag
        else:
            print(f"Failed to load weights for {model_key}", flush=True)

    if not models_dict:
        print("\n[ABORT CLEAN EXIT] Neither convnext_base nor resnet50 could be loaded with pretrained weights offline.", flush=True)
        print("Please ensure the 'knobs-weights' dataset or internet connectivity is attached to this kernel.", flush=True)
        sys.exit(0)

    all_shards = []

    for model_key, model in models_dict.items():
        model_tag = actual_tags[model_key]

        for cond_name, severity, allowed_suites in CONDITIONS_SUITES:
            shard_file = shards_dir / f"shard_k16_{model_key}_{cond_name}_s{severity}.parquet"
            all_shards.append(shard_file)

            if shard_file.exists():
                print(f"[RESUME] Skipping completed shard: {shard_file.name}", flush=True)
                continue

            print(f"\n--- Running: {model_key} on {cond_name} s{severity} ({len(image_ids)} images) ---", flush=True)
            cond_records = []
            t0 = time.time()

            for b_idx in range(0, len(image_ids), batch_size):
                b_ids = image_ids[b_idx:b_idx + batch_size]
                
                clean_tensors_list = []
                corrupted_tensors_list = []
                valid_ids = []

                for img_id in b_ids:
                    img_path = Path(image_dir) / f"{img_id}.JPEG"
                    if not img_path.exists():
                        continue
                    try:
                        with Image.open(img_path) as pil_img:
                            clean_arr_448, _ = preprocess_image_448(pil_img)
                        corr_arr_448 = apply_corruption(clean_arr_448, img_id, cond_name, severity)
                        
                        clean_t = torch.from_numpy(clean_arr_448).permute(2, 0, 1).float() / 255.0
                        corr_t = torch.from_numpy(corr_arr_448).permute(2, 0, 1).float() / 255.0

                        clean_tensors_list.append(clean_t)
                        corrupted_tensors_list.append(corr_t)
                        valid_ids.append(img_id)
                    except Exception as ex:
                        print(f"Warning: Error processing {img_id}: {ex}", flush=True)
                        continue

                if not valid_ids:
                    continue

                clean_batch_448 = torch.stack(clean_tensors_list).to(device)
                corr_batch_448 = torch.stack(corrupted_tensors_list).to(device)

                for i, img_id in enumerate(valid_ids):
                    lbl = val_metadata[img_id]["class_idx"]
                    c_single = clean_batch_448[i:i + 1]
                    corr_single = corr_batch_448[i:i + 1]

                    suite = build_m7_v2_suite(c_single, corr_single, cond_name, severity, img_id)

                    for s_name in allowed_suites:
                        if s_name not in suite:
                            continue
                        in_t = suite[s_name].to(device)
                        in_norm = normalize_tensor(in_t, model_tag)

                        with torch.inference_mode():
                            if device.type == "cuda":
                                with torch.amp.autocast("cuda"):
                                    logits = model(in_norm)
                            else:
                                logits = model(in_norm)
                            pred = int(torch.argmax(logits, dim=1).item())

                        cond_records.append({
                            "image_id": img_id,
                            "condition": cond_name,
                            "severity": severity,
                            "suite_condition": s_name,
                            "model": model_key,
                            "model_tag": model_tag,
                            "norm_type": "BatchNorm" if "resnet" in model_key else "LayerNorm",
                            "label": lbl,
                            "pred": pred,
                            "correct": (pred == lbl),
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

    # Consolidate all shards into final dataset
    print("\n=== Consolidating All K16 Shards ===", flush=True)
    all_dfs = [pd.read_parquet(s) for s in all_shards if s.exists()]
    if all_dfs:
        df_final = pd.concat(all_dfs, ignore_index=True)
        final_parquet = out_path / "shard_bn_vs_ln.parquet"
        df_final.to_parquet(final_parquet, index=False)
        print(f"Final K16 dataset written to {final_parquet} ({len(df_final)} rows)", flush=True)

    # Save run metadata
    meta = get_environment_metadata(device)
    meta["kernel"] = "k16-bn-vs-ln"
    meta["model_tags"] = actual_tags
    with open(out_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {out_path / 'run_meta.json'}", flush=True)


def main():
    print("=== Launching K16-BN-vs-LN Kernel ===", flush=True)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f"Device set to TPU via PyTorch XLA: {device}", flush=True)
        except Exception:
            device = torch.device("cpu")
            print(f"Running on host CPU ({torch.get_num_threads()} threads)", flush=True)

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

    # Balanced subset of 5,000 images (5 per class, seed=0)
    image_ids_5k = balanced_subset(full_ids, val_metadata, n=5000, seed=0)
    print(f"Extracted N={len(image_ids_5k)} class-balanced images (seed=0)", flush=True)

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

    run_k16(
        image_dir=img_dir,
        image_ids=image_ids_5k,
        val_metadata=val_metadata,
        output_dir=out_dir,
        batch_size=32,
        device=device,
    )


if __name__ == "__main__":
    main()
