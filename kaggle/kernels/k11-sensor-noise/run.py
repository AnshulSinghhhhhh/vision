"""K11-Sensor-Noise: Physical Poisson-Gaussian Sensor Noise Model vs ImageNet-C Pixel Noise.

Generates Poisson-Gaussian heteroscedastic noise in the 448 acquisition frame:
  var(x) = a * x + b
calibrated to the exact same PSNR as ImageNet-C gaussian_noise s3 (PSNR = 14.89 dB, sigma=0.18).

Also compares:
- Arm 'acquisition_noise': Noise injected in 448 frame, then downsampled (attenuated by antialiasing filter)
- Arm 'sensor_matched': Physical sensor noise where downsampling binning keeps noise variance proportional to pixel area (sigma_R = sigma_448 * (448 / R))
- Arm 'clean': Clean baseline

Models evaluated:
- DeiT-B/16
- EfficientNet-B3
- FlexiViT-B (Arms F-p and F-t)

Resolutions: 224, 320, 384, 448
Images: 5,000 class-balanced PILOT images
"""

import os
import sys
import time
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
from knobs.corrupt import compute_seed, SeedContext
from knobs.resize import resize_tensor_torch
from knobs.run_grid import get_environment_metadata


def generate_poisson_gaussian_noise(
    img_448: np.ndarray,
    target_sigma: float = 0.18,
    shot_fraction: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, float]:
    """Generates heteroscedastic Poisson-Gaussian sensor noise calibrated to target_sigma.

    Args:
        img_448: uint8 array (448, 448, 3)
        target_sigma: target RMS noise on [0, 1] scale (0.18 matches ImageNet-C s3)
        shot_fraction: fraction of variance allocated to Poisson shot noise (vs Gaussian read noise)
        seed: random seed

    Returns:
        (corrupted_uint8, pre_clip_rms, psnr_db)
    """
    img_norm = img_448.astype(np.float64) / 255.0
    target_mse = float(target_sigma ** 2)
    mean_val = float(np.mean(img_norm)) + 1e-6

    # Variance model: var(x) = a * x + b
    a = shot_fraction * target_mse / mean_val
    b = (1.0 - shot_fraction) * target_mse

    local_var = np.maximum(a * img_norm + b, 1e-8)
    std_map = np.sqrt(local_var)

    if seed is not None:
        rng = np.random.RandomState(seed % (2**32 - 1))
    else:
        rng = np.random.RandomState()

    raw_noise = rng.normal(0, 1.0, img_448.shape).astype(np.float64) * std_map
    current_rms = float(np.sqrt(np.mean(raw_noise ** 2)))

    # Rescale to match target_sigma
    if current_rms > 1e-8:
        calibrated_noise = raw_noise * (target_sigma / current_rms)
    else:
        calibrated_noise = raw_noise

    pre_clip_rms = float(target_sigma)
    psnr_db = float(10.0 * np.log10(1.0 / (pre_clip_rms ** 2 + 1e-12)))

    noisy_norm = np.clip(img_norm + calibrated_noise, 0.0, 1.0)
    noisy_uint8 = np.clip(np.round(noisy_norm * 255.0), 0.0, 255.0).astype(np.uint8)

    return noisy_uint8, pre_clip_rms, psnr_db


def run_sensor_noise_evaluation(
    image_dir: str,
    pilot_image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str = "/kaggle/working",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
):
    """Executes sensor noise evaluation on PILOT images."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [224, 320, 384, 448]

    # Initialize models
    print(f"=== Initializing Models on {device} ===", flush=True)
    m_deit = create_model_instance("deit_base", resolution=448, pretrained=pretrained, device=device)
    m_eff = create_model_instance("efficientnet_b3", resolution=448, pretrained=pretrained, device=device)

    flex_models = {}
    for r in resolutions:
        flex_models[("F-p", r)] = create_model_instance("flexivit_base", resolution=r, arm="F-p", pretrained=pretrained, device=device)
        flex_models[("F-t", r)] = create_model_instance("flexivit_base", resolution=r, arm="F-t", pretrained=pretrained, device=device)

    noise_modes = ["clean", "poisson_gaussian_s3", "sensor_matched_s3"]
    combined_shards = []

    for mode in noise_modes:
        shard_file = shards_dir / f"shard_sensor_noise_{mode}.parquet"
        if shard_file.exists():
            print(f"Shard {shard_file.name} already exists. Skipping (resumable).", flush=True)
            combined_shards.append(shard_file)
            continue

        print(f"\n--- Running Sensor Noise Mode: {mode} ---", flush=True)
        stage_records = []
        n_images = len(pilot_image_ids)

        for b_start in range(0, n_images, batch_size):
            b_end = min(b_start + batch_size, n_images)
            b_ids = pilot_image_ids[b_start:b_end]

            batch_clean_448 = []
            batch_labels = []
            valid_ids = []

            for img_id in b_ids:
                img_p = Path(image_dir) / f"{img_id}.JPEG"
                if not img_p.exists():
                    continue
                pil_img = Image.open(img_p)
                arr_448, _ = preprocess_image_448(pil_img)

                t_clean = torch.from_numpy(arr_448).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                batch_clean_448.append(t_clean)
                batch_labels.append(val_metadata[img_id]["class_idx"])
                valid_ids.append(img_id)

            if not valid_ids:
                continue

            for res in resolutions:
                if mode == "clean":
                    # Clean baseline resized to target resolution
                    t_in_448 = torch.cat(batch_clean_448, dim=0)
                    t_in = resize_tensor_torch(t_in_448, target_size=res).to(device)
                    eff_sigma = 0.0
                elif mode == "poisson_gaussian_s3":
                    # Noise injected at 448 frame, then resized to res
                    noisy_batch = []
                    for i, iid in enumerate(valid_ids):
                        seed = compute_seed(iid, "poisson_gaussian", 3)
                        arr = (batch_clean_448[i].squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                        noisy_arr, _, _ = generate_poisson_gaussian_noise(arr, target_sigma=0.18, seed=seed)
                        noisy_batch.append(torch.from_numpy(noisy_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
                    t_in_448 = torch.cat(noisy_batch, dim=0)
                    t_in = resize_tensor_torch(t_in_448, target_size=res).to(device)
                    eff_sigma = 0.18
                elif mode == "sensor_matched_s3":
                    # Sensor noise model: noise scales inversely with resolution binning
                    # sigma(R) = 0.18 * (448 / R)
                    sigma_res = 0.18 * (448.0 / float(res))
                    noisy_res = []
                    t_clean_448 = torch.cat(batch_clean_448, dim=0)
                    t_clean_res = resize_tensor_torch(t_clean_448, target_size=res)
                    for i, iid in enumerate(valid_ids):
                        seed = compute_seed(iid, f"sensor_matched_{res}", 3)
                        arr_res = (t_clean_res[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                        noisy_arr, _, _ = generate_poisson_gaussian_noise(arr_res, target_sigma=sigma_res, seed=seed)
                        noisy_res.append(torch.from_numpy(noisy_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
                    t_in = torch.cat(noisy_res, dim=0).to(device)
                    eff_sigma = sigma_res

                eval_configs = [
                    ("deit_base", "standard", m_deit),
                    ("efficientnet_b3", "standard", m_eff),
                    ("flexivit_base", "F-p", flex_models[("F-p", res)]),
                    ("flexivit_base", "F-t", flex_models[("F-t", res)]),
                ]

                for m_name, arm_name, model in eval_configs:
                    tag = MODEL_TAGS.get(m_name, m_name)
                    norm_in = normalize_tensor(t_in, model_tag=tag)
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
                            "mode": mode,
                            "resolution": res,
                            "model": m_name,
                            "arm": arm_name,
                            "label": lbl,
                            "pred": prd,
                            "correct": bool(prd == lbl),
                            "target_sigma": eff_sigma,
                        })

            print(f"[{mode}] Processed {b_end}/{n_images} images", flush=True)

        df_stage = pd.DataFrame(stage_records)
        df_stage.to_parquet(shard_file, index=False)
        print(f"Saved shard {shard_file} ({len(df_stage)} rows)", flush=True)
        combined_shards.append(shard_file)

    # Consolidate all shards
    print("\n=== Consolidating All Shards into Final Dataset ===", flush=True)
    all_dfs = [pd.read_parquet(s) for s in combined_shards if s.exists()]
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        final_parquet = out_path / "shard_sensor_noise.parquet"
        df_all.to_parquet(final_parquet, index=False)
        print(f"Final dataset saved to {final_parquet} ({len(df_all)} rows)", flush=True)

    # Save run metadata
    meta = get_environment_metadata(device)
    meta["kernel"] = "k11-sensor-noise"
    meta["psnr_matched_target"] = 14.89
    with open(out_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {out_path / 'run_meta.json'}", flush=True)


def main():
    print("=== Launching K11-Sensor-Noise Kernel ===", flush=True)
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

    # Class balanced 5,000 images
    balanced_ids = balanced_subset(pilot_ids, val_metadata, n=5000)
    print(f"Loaded {len(balanced_ids)} balanced PILOT images", flush=True)

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

    run_sensor_noise_evaluation(
        image_dir=img_dir,
        pilot_image_ids=balanced_ids,
        val_metadata=val_metadata,
        output_dir=out_dir,
        batch_size=32,
        device=device,
        pretrained=True,
    )
    print(f"K11-Sensor-Noise finished in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
