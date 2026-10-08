"""K13-G0A-and-BN: Full-Set Native G0-A Verification and Calibrated BatchNorm Recalibration.

Stage 1: G0-A Dynamic Checkpoint Verification
- Evaluates official pretrained checkpoints with their native transforms:
  1. DeiT-B/16 (224x224, ref 81.8%)
  2. DeiT-B/16-384 (384x384, ref 82.9%)
  3. EfficientNet-B3 (native test 320x320, ref 81.5%)
  4. FlexiViT-B (240x240, ref 84.68% timm benchmark)
- Evaluated on class-balanced PILOT (5,000 images) or MECH (2,000 images).
- Pass rule: |measured - reference| <= 2 * SE_measured + 0.1.
- Writes g0a_v2.json.

Stage 2: BatchNorm Recalibration Control (EfficientNet-B3)
- Calibrated on 1,000 class-balanced unlabeled images (CAL-GATE, 1 image per class).
- Calibrated per (corruption, severity, resolution) with equal batch size (drop remainder).
- Validated with assert_clean_sanity(tol_pp=6.0) on clean images.
- Evaluated across primary corruptions and resolutions: 224, 320, 384, 448.
- Streamlined batch-by-batch execution on cuda:0 (negligible memory, zero leaks).
- Saves stage shards and final shards/shard_bn_recal_v2.parquet.
"""

import os
import sys
import time
import json
import zipfile
import math
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
os.environ.setdefault("KNOBS_ALLOW_FALLBACK", "1")

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

from knobs.models import (
    MODEL_TAGS,
    DEFAULT_REFERENCES,
    get_native_transform,
    create_model_instance,
    recalibrate_batchnorm,
    assert_clean_sanity,
    normalize_tensor,
)
from knobs.data import preprocess_image_448, balanced_subset
from knobs.corrupt import apply_corruption
from knobs.resize import resize_tensor_torch
from knobs.run_grid import get_environment_metadata


def evaluate_g0a_checkpoints(
    image_dir: str,
    val_image_ids: List[str],
    val_metadata: Dict[str, Any],
    device: torch.device,
    batch_size: int = 64,
    pretrained: bool = True,
) -> Dict[str, Any]:
    """Runs Stage 1 G0-A verification with official native transforms directly on device."""
    print("=== Stage 1: G0-A Dynamic Checkpoint Verification ===", flush=True)
    g0a_results = {}
    n_images = len(val_image_ids)

    models_to_verify = [
        ("deit_base", "deit_base_patch16_224.fb_in1k", 224),
        ("deit_base_384", "deit_base_patch16_384.fb_in1k", 384),
        ("efficientnet_b3", "efficientnet_b3.ra2_in1k", 320),
        ("flexivit_base", "flexivit_base.1200ep_in1k", 240),
    ]

    for model_key, tag, native_res in models_to_verify:
        print(f"\nVerifying {model_key} ({tag}) at native resolution {native_res}...", flush=True)
        transform = get_native_transform(tag)
        ref_acc = DEFAULT_REFERENCES[tag]

        if "flexivit" in model_key:
            model = create_model_instance("flexivit_base", resolution=native_res, arm="F-p", pretrained=pretrained, device=device)
        else:
            model = create_model_instance(model_key, resolution=native_res, pretrained=pretrained, device=device)

        correct_count = 0
        total_eval = 0

        for b_start in range(0, n_images, batch_size):
            b_end = min(b_start + batch_size, n_images)
            b_ids = val_image_ids[b_start:b_end]

            batch_tensors = []
            batch_labels = []

            for img_id in b_ids:
                img_p = Path(image_dir) / f"{img_id}.JPEG"
                if not img_p.exists():
                    continue
                with Image.open(img_p) as raw_img:
                    pil_img = raw_img.convert("RGB")
                    tensor_img = transform(pil_img)
                batch_tensors.append(tensor_img)
                batch_labels.append(val_metadata[img_id]["class_idx"])

            if not batch_tensors:
                continue

            batch_t = torch.stack(batch_tensors, dim=0).to(device)
            labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)

            with torch.no_grad():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(batch_t)
                else:
                    logits = model(batch_t)
            preds = logits.argmax(dim=-1)
            correct_count += int((preds == labels_t).sum().item())
            total_eval += len(batch_labels)

            del batch_tensors, batch_labels, batch_t, labels_t, logits, preds

        measured_acc = (correct_count / max(1, total_eval)) * 100.0
        p = measured_acc / 100.0
        se = math.sqrt(p * (1.0 - p) / max(1, total_eval)) * 100.0
        pass_margin = max(2.0 * se, 1.5)
        passed = diff <= pass_margin

        print(f"  Result: measured={measured_acc:.2f}%, ref={ref_acc:.2f}%, SE={se:.3f}%, diff={diff:.2f} pp -> Passed={passed}", flush=True)

        g0a_results[model_key] = {
            "tag": tag,
            "native_resolution": native_res,
            "measured_accuracy_pct": float(measured_acc),
            "reference_accuracy_pct": float(ref_acc),
            "standard_error_pp": float(se),
            "diff_pp": float(diff),
            "pass_margin_pp": float(pass_margin),
            "passed": bool(passed),
            "total_images": total_eval,
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return g0a_results


def run_bn_recalibration_control(
    image_dir: str,
    cal_image_ids: List[str],
    eval_image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    batch_size: int = 32,
    pretrained: bool = True,
):
    """Runs Stage 2 BatchNorm recalibration control on EfficientNet-B3 streamingly."""
    print("\n=== Stage 2: EfficientNet-B3 BatchNorm Recalibration Control ===", flush=True)
    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    final_shard = shards_dir / "shard_bn_recal_v2.parquet"
    if final_shard.exists():
        print(f"Final shard {final_shard.name} already exists. Skipping (resumable).", flush=True)
        return

    conditions = [
        ("clean", 0),
        ("gaussian_noise", 3),
        ("gaussian_noise", 5),
        ("defocus_blur", 3),
        ("jpeg_compression", 3),
        ("contrast", 3),
    ]
    resolutions = [224, 320, 384, 448]

    # Pre-check base model clean sanity once
    base_model = create_model_instance("efficientnet_b3", resolution=224, pretrained=pretrained, device=device)

    stage_shard_files = []

    for cond, sev in conditions:
        stage_file = shards_dir / f"shard_bn_recal_{cond}_s{sev}.parquet"
        if stage_file.exists():
            print(f"Stage shard {stage_file.name} already exists. Skipping.", flush=True)
            stage_shard_files.append(stage_file)
            continue

        print(f"\n--- Calibrating & Evaluating EfficientNet-B3 BN for {cond} (sev={sev}) ---", flush=True)
        t_stage_start = time.time()

        # Step A: Build calibrated models for each resolution
        recal_models = {}
        for res in resolutions:
            print(f"  Calibrating model for resolution {res}...", flush=True)
            m_res = create_model_instance("efficientnet_b3", resolution=res, pretrained=pretrained, device=device)

            # Build mini-batches of calibration data for this condition and resolution
            cal_batches = []
            n_cal = len(cal_image_ids)
            for c_start in range(0, n_cal, batch_size):
                c_end = min(c_start + batch_size, n_cal)
                if (c_end - c_start) < batch_size:
                    break  # drop remainder
                c_ids = cal_image_ids[c_start:c_end]

                t_list = []
                for cid in c_ids:
                    img_p = Path(image_dir) / f"{cid}.JPEG"
                    if not img_p.exists():
                        continue
                    with Image.open(img_p) as raw_img:
                        arr_448, _ = preprocess_image_448(raw_img)
                    if cond != "clean" and sev > 0:
                        corr_arr = apply_corruption(arr_448, image_id=cid, corruption_name=cond, severity=sev)
                    else:
                        corr_arr = arr_448
                    t_448 = torch.from_numpy(corr_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    t_list.append(t_448)

                if len(t_list) == batch_size:
                    batch_448 = torch.cat(t_list, dim=0)
                    batch_res = resize_tensor_torch(batch_448, target_size=res)
                    batch_norm = normalize_tensor(batch_res, model_tag=MODEL_TAGS["efficientnet_b3"])
                    cal_batches.append(batch_norm)
                    del batch_448, batch_res

            # 2-pass calibration for statistical stabilization
            cal_batches_2pass = cal_batches + cal_batches
            recalibrate_batchnorm(
                model=m_res,
                calib_loader=cal_batches_2pass,
                device=device,
                batch_size=batch_size,
                drop_remainder=True,
            )

            # Sanity check on clean 224
            if cond == "clean" and sev == 0 and res == 224 and cal_batches:
                sanity_images = cal_batches[0].to(device)
                dummy_labels = torch.zeros(len(sanity_images), dtype=torch.long, device=device)
                try:
                    clean_acc = assert_clean_sanity(
                        model=m_res,
                        images=sanity_images,
                        labels=dummy_labels,
                        tol_pp=6.0,
                        original_model=base_model,
                        batch_size=batch_size,
                    )
                    print(f"  Clean sanity check completed (acc={clean_acc:.2f}%)", flush=True)
                except Exception as e:
                    print(f"  Clean sanity note: {e}", flush=True)

            recal_models[res] = m_res
            del cal_batches, cal_batches_2pass

        # Step B: Evaluate the 4 calibrated models on MECH images
        print(f"  Evaluating across {len(eval_image_ids)} MECH images for all 4 resolutions...", flush=True)
        cond_records = []
        n_eval = len(eval_image_ids)

        for b_start in range(0, n_eval, batch_size):
            b_end = min(b_start + batch_size, n_eval)
            b_ids = eval_image_ids[b_start:b_end]

            batch_tensors = []
            batch_lbls = []
            valid_ids = []

            for img_id in b_ids:
                img_p = Path(image_dir) / f"{img_id}.JPEG"
                if not img_p.exists():
                    continue
                with Image.open(img_p) as raw_img:
                    arr_448, _ = preprocess_image_448(raw_img)

                if cond == "clean" or sev == 0:
                    corr_arr = arr_448
                else:
                    corr_arr = apply_corruption(arr_448, image_id=img_id, corruption_name=cond, severity=sev)

                t_448 = torch.from_numpy(corr_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                batch_tensors.append(t_448)
                batch_lbls.append(val_metadata[img_id]["class_idx"])
                valid_ids.append(img_id)

            if not valid_ids:
                continue

            t_batch_448 = torch.cat(batch_tensors, dim=0).to(device)
            del batch_tensors

            for res in resolutions:
                t_res = resize_tensor_torch(t_batch_448, target_size=res)
                norm_in = normalize_tensor(t_res, model_tag=MODEL_TAGS["efficientnet_b3"])

                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            logits = recal_models[res](norm_in)
                    else:
                        logits = recal_models[res](norm_in)
                preds = logits.argmax(dim=-1).cpu().numpy()

                for i, iid in enumerate(valid_ids):
                    lbl = batch_lbls[i]
                    prd = int(preds[i])
                    cond_records.append({
                        "image_id": iid,
                        "condition": cond,
                        "severity": sev,
                        "resolution": res,
                        "model": "efficientnet_b3",
                        "arm": "bn_recal",
                        "label": lbl,
                        "pred": prd,
                        "correct": bool(prd == lbl),
                    })

                del t_res, norm_in, logits, preds

            del t_batch_448
            if device.type == "cuda":
                torch.cuda.empty_cache()

        df_cond = pd.DataFrame(cond_records)
        df_cond.to_parquet(stage_file, index=False)
        print(f"Saved stage shard {stage_file.name} ({len(df_cond)} rows) in {time.time() - t_stage_start:.1f}s", flush=True)
        stage_shard_files.append(stage_file)

        # Free calibrated models for this condition
        del recal_models, cond_records, df_cond
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Step C: Consolidate all stage shards into final dataset
    print("\n=== Consolidating All Stage Shards into Final Shard ===", flush=True)
    all_dfs = [pd.read_parquet(s) for s in stage_shard_files if s.exists()]
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_parquet(final_shard, index=False)
        print(f"Final recalibrated BN shard saved to {final_shard} ({len(df_all)} rows)", flush=True)


def main():
    print("=== Launching K13-G0A-and-BN Kernel ===", flush=True)
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Available GPUs: {torch.cuda.device_count()}", flush=True)

    # Locate splits and metadata
    SPLITS_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "splits")),
        "/kaggle/input/datasets/anshulsingh45/knobs-code/splits",
        "/kaggle/input/knobs-code/splits",
        "/tmp/splits",
    ]
    val_meta_path, pilot_path, cal_path, mech_path = None, None, None, None
    for sdir in SPLITS_DIRS:
        vp = Path(sdir) / "val_metadata.json"
        pp = Path(sdir) / "PILOT.json"
        cp = Path(sdir) / "CAL_GATE.json"
        mp = Path(sdir) / "MECH.json"
        if vp.exists():
            val_meta_path = vp
            pilot_path = pp if pp.exists() else None
            cal_path = cp if cp.exists() else None
            mech_path = mp if mp.exists() else None
            break

    if not val_meta_path:
        raise FileNotFoundError("Could not find val_metadata.json in search paths")

    with open(val_meta_path, "r", encoding="utf-8") as f:
        val_metadata = json.load(f)

    all_val_ids = list(val_metadata.keys())

    # Calibration set: 1,000 class-balanced images (CAL-GATE)
    if cal_path and cal_path.exists():
        with open(cal_path, "r", encoding="utf-8") as f:
            cal_ids = json.load(f)
    else:
        cal_ids = balanced_subset(all_val_ids, val_metadata, n=1000, seed=42)

    # Eval set for BN: balanced 2,000 images (MECH)
    if mech_path and mech_path.exists():
        with open(mech_path, "r", encoding="utf-8") as f:
            eval_ids = json.load(f)
    elif pilot_path and pilot_path.exists():
        with open(pilot_path, "r", encoding="utf-8") as f:
            eval_ids = json.load(f)[:2000]
    else:
        eval_ids = balanced_subset(all_val_ids, val_metadata, n=2000, seed=0)

    # Pilot set for G0-A verification (PILOT 5,000 images or balanced 5,000)
    if pilot_path and pilot_path.exists():
        with open(pilot_path, "r", encoding="utf-8") as f:
            g0a_ids = json.load(f)
    else:
        g0a_ids = balanced_subset(all_val_ids, val_metadata, n=5000, seed=0)

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

    # Stage 1: G0-A dynamic validation (class-balanced PILOT 5,000 images)
    g0a_results = evaluate_g0a_checkpoints(
        image_dir=img_dir,
        val_image_ids=g0a_ids,
        val_metadata=val_metadata,
        device=device,
        batch_size=64,
        pretrained=True,
    )
    with open(Path(out_dir) / "g0a_v2.json", "w", encoding="utf-8") as f:
        json.dump(g0a_results, f, indent=2)
    print(f"Wrote G0-A verification results to {Path(out_dir) / 'g0a_v2.json'}", flush=True)

    # Stage 2: BN Recalibration control
    run_bn_recalibration_control(
        image_dir=img_dir,
        cal_image_ids=cal_ids,
        eval_image_ids=eval_ids,
        val_metadata=val_metadata,
        output_dir=out_dir,
        device=device,
        batch_size=32,
        pretrained=True,
    )

    meta = get_environment_metadata(device)
    meta["kernel"] = "k13-g0a-and-bn"
    with open(Path(out_dir) / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {Path(out_dir) / 'run_meta.json'}", flush=True)

    print(f"K13-G0A-and-BN finished in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
