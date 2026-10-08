"""K10-Resize-Ablation: Antialiasing & Interpolation Operator Ablation on PILOT (5,000 images).

Compares downsampling operators:
1. Bilinear with antialiasing (PyTorch default in pipeline)
2. Bilinear without antialiasing
3. Bicubic with antialiasing
4. Area interpolation (box filter)
5. 448 with Gaussian lowpass pre-filter (sigma=0.866 px, matching 448->224 variance)
6. 448 Identity (baseline)

Models evaluated:
- EfficientNet-B3
- DeiT-B/16

Conditions:
- clean (severity 0)
- gaussian_noise (severity 3, severity 5)

Resolutions:
- 224, 320, 384 (and 448 controls)
"""

import os
import sys
import time
import json
import zipfile
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# 1. Unpack any zips if present
for zip_dir in ["/kaggle/input/datasets/anshulsingh45/knobs-code", "/kaggle/input/knobs-code"]:
    if os.path.exists(zip_dir):
        for f in os.listdir(zip_dir):
            if f.endswith(".zip"):
                zpath = os.path.join(zip_dir, f)
                dest = os.path.join("/tmp", f[:-4])
                if not os.path.exists(dest):
                    try:
                        with zipfile.ZipFile(zpath, "r") as zf:
                            zf.extractall(dest)
                        print(f"Extracted {f} to {dest}", flush=True)
                    except Exception as e:
                        print(f"Note on {f}: {e}", flush=True)

# Add knobs to sys.path
KNOBS_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
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
        print(f"Added knobs to sys.path from: {cand}", flush=True)
        break

from knobs.models import MODEL_TAGS, create_model_instance, normalize_tensor
from knobs.data import preprocess_image_448, balanced_subset
from knobs.corrupt import apply_corruption
from knobs.resize import apply_gaussian_lowpass
from knobs.run_grid import get_environment_metadata


def apply_resize_operator(tensor_448: torch.Tensor, operator: str, target_res: int) -> torch.Tensor:
    """Applies a specified resize operator to a batch tensor."""
    if operator == "bilinear_antialias":
        return F.interpolate(tensor_448, size=(target_res, target_res), mode="bilinear", align_corners=False, antialias=True)
    elif operator == "bilinear_no_antialias":
        return F.interpolate(tensor_448, size=(target_res, target_res), mode="bilinear", align_corners=False, antialias=False)
    elif operator == "bicubic_antialias":
        return F.interpolate(tensor_448, size=(target_res, target_res), mode="bicubic", align_corners=False, antialias=True)
    elif operator == "area":
        return F.interpolate(tensor_448, size=(target_res, target_res), mode="area")
    elif operator == "gaussian_prefilter_448":
        # 448 resolution with lowpass filter matching 224 antialiasing variance
        return apply_gaussian_lowpass(tensor_448, sigma_px=0.866)
    elif operator == "identity_448":
        return tensor_448
    else:
        raise ValueError(f"Unknown resize operator: {operator}")


def compute_operator_noise_gain(operator: str, target_res: int, device: torch.device) -> float:
    """Measures empirical noise gain of an operator on white noise."""
    if operator == "identity_448":
        return 1.0
    if operator == "gaussian_prefilter_448":
        # Gaussian filter with sigma=0.866 has gain = 1 / (2 * sqrt(pi) * sigma) ~ 0.326
        dummy = torch.randn(32, 3, 448, 448, device=device)
        filtered = apply_gaussian_lowpass(dummy, sigma_px=0.866)
        return float(filtered.std() / dummy.std())

    dummy = torch.randn(32, 3, 448, 448, device=device)
    out = apply_resize_operator(dummy, operator, target_res)
    return float(out.std() / dummy.std())


def run_resize_ablation(
    image_dir: str,
    pilot_image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str = "/kaggle/working",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
):
    """Runs resize ablation across operators and resolutions."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    print(f"=== Initializing Models on {device} ===", flush=True)
    model_eff = create_model_instance("efficientnet_b3", resolution=448, pretrained=pretrained, device=device)
    model_deit = create_model_instance("deit_base", resolution=448, pretrained=pretrained, device=device)

    models = [
        ("efficientnet_b3", model_eff),
        ("deit_base", model_deit),
    ]

    if torch.cuda.device_count() > 1:
        models = [(name, nn.DataParallel(m)) for name, m in models]

    operators = [
        ("bilinear_antialias", [224, 320, 384]),
        ("bilinear_no_antialias", [224, 320, 384]),
        ("bicubic_antialias", [224, 320, 384]),
        ("area", [224, 320, 384]),
        ("gaussian_prefilter_448", [448]),
        ("identity_448", [448]),
    ]

    # Precalculate noise gains
    print("Measuring operator noise gains...", flush=True)
    gain_table = {}
    for op_name, r_list in operators:
        for r in r_list:
            gain_table[(op_name, r)] = compute_operator_noise_gain(op_name, r, device)
            print(f"  {op_name} @ {r}: noise gain = {gain_table[(op_name, r)]:.4f}", flush=True)

    stages = [
        ("clean", 0),
        ("gaussian_noise", 3),
        ("gaussian_noise", 5),
    ]

    injected_sigmas = {0: 0.0, 3: 0.18, 5: 0.38}
    combined_shards = []

    for cond, sev in stages:
        shard_file = shards_dir / f"shard_resize_ablation_{cond}_{sev}.parquet"
        if shard_file.exists():
            print(f"Shard {shard_file.name} already exists. Skipping (resumable).", flush=True)
            combined_shards.append(shard_file)
            continue

        print(f"\n--- Running Stage: {cond} (severity={sev}) ---", flush=True)
        stage_records = []
        n_images = len(pilot_image_ids)
        inj_sigma = injected_sigmas.get(sev, 0.0)

        for b_start in range(0, n_images, batch_size):
            b_end = min(b_start + batch_size, n_images)
            b_ids = pilot_image_ids[b_start:b_end]

            batch_448 = []
            batch_labels = []
            valid_ids = []

            for img_id in b_ids:
                img_p = Path(image_dir) / f"{img_id}.JPEG"
                if not img_p.exists():
                    continue
                with Image.open(img_p) as pil_img:
                    arr_448, _ = preprocess_image_448(pil_img)

                if cond == "clean" or sev == 0:
                    corr_arr = arr_448
                else:
                    corr_arr = apply_corruption(arr_448, image_id=img_id, corruption_name=cond, severity=sev)

                t_448 = torch.from_numpy(corr_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                batch_448.append(t_448)
                batch_labels.append(val_metadata[img_id]["class_idx"])
                valid_ids.append(img_id)

            if not valid_ids:
                continue

            t_batch_448 = torch.cat(batch_448, dim=0).to(device)

            for op_name, r_list in operators:
                for r in r_list:
                    t_resized = apply_resize_operator(t_batch_448, op_name, r)
                    eff_sigma = gain_table[(op_name, r)] * inj_sigma

                    for m_name, model in models:
                        tag = MODEL_TAGS.get(m_name, m_name)
                        norm_in = normalize_tensor(t_resized, model_tag=tag)
                        with torch.no_grad():
                            if device.type == "cuda":
                                with torch.cuda.amp.autocast():
                                    logits = model(norm_in)
                            else:
                                logits = model(norm_in)
                        preds = logits.argmax(dim=-1).cpu().numpy()

                        for idx_img, iid in enumerate(valid_ids):
                            lbl = batch_labels[idx_img]
                            prd = int(preds[idx_img])
                            stage_records.append({
                                "image_id": iid,
                                "condition": cond,
                                "severity": sev,
                                "operator": op_name,
                                "resolution": r,
                                "model": m_name,
                                "label": lbl,
                                "pred": prd,
                                "correct": bool(prd == lbl),
                                "effective_sigma": eff_sigma,
                            })

            del batch_448, t_batch_448, t_resized, norm_in
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[{cond}-s{sev}] Processed {b_end}/{n_images} images", flush=True)

        df_stage = pd.DataFrame(stage_records)
        df_stage.to_parquet(shard_file, index=False)
        print(f"Saved stage shard {shard_file} ({len(df_stage)} rows)", flush=True)
        combined_shards.append(shard_file)
        del stage_records, df_stage
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Consolidate all shards
    print("\n=== Consolidating All Shards into Final Dataset ===", flush=True)
    all_dfs = [pd.read_parquet(s) for s in combined_shards if s.exists()]
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        final_parquet = out_path / "shard_resize_ablation.parquet"
        df_all.to_parquet(final_parquet, index=False)
        print(f"Final dataset saved to {final_parquet} ({len(df_all)} rows)", flush=True)

    # Save run metadata
    meta = get_environment_metadata(device)
    meta["kernel"] = "k10-resize-ablation"
    meta["gain_table"] = {f"{k[0]}@{k[1]}": float(v) for k, v in gain_table.items()}
    with open(out_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {out_path / 'run_meta.json'}", flush=True)


def main():
    print("=== Launching K10-Resize-Ablation Kernel ===", flush=True)
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Locate splits and metadata
    SPLITS_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "splits")),
        "/kaggle/input/datasets/anshulsingh45/knobs-code/splits",
        "/kaggle/input/knobs-code/splits",
        "/tmp/splits",
    ]
    pilot_path, val_meta_path = None, None
    for sdir in SPLITS_DIRS:
        pp = Path(sdir) / "PILOT.json"
        vp = Path(sdir) / "val_metadata.json"
        if pp.exists() and vp.exists():
            pilot_path, val_meta_path = pp, vp
            break

    if not pilot_path or not val_meta_path:
        raise FileNotFoundError("Could not find PILOT.json or val_metadata.json in search paths")

    with open(pilot_path, "r", encoding="utf-8") as f:
        pilot_ids = json.load(f)
    with open(val_meta_path, "r", encoding="utf-8") as f:
        val_metadata = json.load(f)

    # Ensure class-balanced subset of 5,000 images
    balanced_ids = balanced_subset(pilot_ids, val_metadata, n=5000)
    print(f"Selected {len(balanced_ids)} class-balanced PILOT image IDs", flush=True)

    # Locate validation image folder
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

    if not img_dir:
        img_dir = "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"

    out_dir = "/kaggle/working" if os.path.exists("/kaggle") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "derived"))

    run_resize_ablation(
        image_dir=img_dir,
        pilot_image_ids=balanced_ids,
        val_metadata=val_metadata,
        output_dir=out_dir,
        batch_size=32,
        device=device,
        pretrained=True,
    )
    print(f"K10-Resize-Ablation finished in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
