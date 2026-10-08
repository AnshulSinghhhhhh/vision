"""K8-M7-v2: Mechanistic Information Control M7 on MECH 2,000 images across 4 architectures.

Architectures evaluated:
1. DeiT-B/16
2. EfficientNet-B3
3. FlexiViT-B Arm F-p (constant patch size 16)
4. FlexiViT-B Arm F-t (constant token count 256)

Suite conditions:
- A_orig448: Original 448 acquisition frame
- B_filtered448: 3-tap lowpass filter [1/4, 1/2, 1/4] at 448
- B_g0.5, B_g0.866, B_g1.5, B_g2.5: Gaussian lowpass filters at 448
- C_down224: True 224 downsampling (antialiased bilinear)
- D_up448: 224 downsampled then upsampled back to 448
- E_noise_matched: Noise injected at 448 scaled by effective noise gain (0.3125 * sigma_inj)

Conditions evaluated:
- clean (severity 0)
- gaussian_noise (severity 3, severity 5)
- defocus_blur (severity 3)

Saves per-condition shards to `shards/shard_m7_v2_{cond}_{sev}.parquet` (resumable),
and combines all shards into `shard_m7_v2.parquet`.
"""

import os
import sys
import time
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
                        with zipfile.ZipFile(zpath, 'r') as zf:
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
from knobs.corrupt import apply_corruption, compute_seed, SeedContext
from knobs.resize import resize_tensor_torch, generate_m7_suite
from knobs.spectral import estimate_immerkaer_noise_sigma
from knobs.run_grid import get_environment_metadata


def build_m7_v2_suite(
    clean_tensor_448: torch.Tensor,
    corrupted_tensor_448: torch.Tensor,
    condition: str,
    severity: int,
    image_id: str,
    gaussian_sigmas: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Builds the complete M7-v2 suite of input representations for a single image.
    
    Returns dictionary mapping condition name to (1, 3, H, W) tensor in [0, 1].
    """
    if gaussian_sigmas is None:
        gaussian_sigmas = [0.5, 0.866, 1.5, 2.5]

    # Standard M7 suite on corrupted 448 image
    suite = generate_m7_suite(corrupted_tensor_448, filters=gaussian_sigmas)

    # Condition E: noise-gain matched injection at 448
    if condition == "gaussian_noise" and severity > 0:
        sigmas_map = {1: 0.08, 2: 0.12, 3: 0.18, 4: 0.26, 5: 0.38}
        sigma_inj = sigmas_map.get(severity, 0.18)
        # 0.3125 noise gain from 448->224 bilinear filter
        sigma_matched = 0.3125 * sigma_inj
        seed = compute_seed(image_id, "gaussian_noise_matched", severity)
        clean_np = (clean_tensor_448.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        with SeedContext(seed):
            noise = np.random.normal(0, sigma_matched * 255.0, clean_np.shape)
        matched_np = np.clip(clean_np + noise, 0, 255).astype(np.uint8)
        e_tensor = torch.from_numpy(matched_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        suite["E_noise_matched"] = e_tensor

    return suite


def run_m7_v2(
    image_dir: str,
    mech_image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str = "/kaggle/working",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
):
    """Executes M7-v2 information control on the MECH dataset across 4 models."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing Models on {device} ===", flush=True)
    # DeiT-B handles 224 and 448 dynamically
    m_deit = create_model_instance("deit_base", resolution=448, pretrained=pretrained, device=device)
    # EfficientNet-B3 handles 224 and 448
    m_eff = create_model_instance("efficientnet_b3", resolution=448, pretrained=pretrained, device=device)
    # FlexiViT F-p models
    m_flex_fp_448 = create_model_instance("flexivit_base", resolution=448, arm="F-p", pretrained=pretrained, device=device)
    m_flex_fp_224 = create_model_instance("flexivit_base", resolution=224, arm="F-p", pretrained=pretrained, device=device)
    # FlexiViT F-t models
    m_flex_ft_448 = create_model_instance("flexivit_base", resolution=448, arm="F-t", pretrained=pretrained, device=device)
    m_flex_ft_224 = create_model_instance("flexivit_base", resolution=224, arm="F-t", pretrained=pretrained, device=device)

    stages = [
        ("clean", 0),
        ("gaussian_noise", 3),
        ("gaussian_noise", 5),
        ("defocus_blur", 3),
    ]

    combined_shards = []

    for cond, sev in stages:
        shard_file = shards_dir / f"shard_m7_v2_{cond}_{sev}.parquet"
        if shard_file.exists():
            print(f"Shard {shard_file.name} already exists. Skipping (resumable).", flush=True)
            combined_shards.append(shard_file)
            continue

        print(f"\n--- Running M7-v2 Stage: {cond} (severity={sev}) ---", flush=True)
        stage_records = []
        n_images = len(mech_image_ids)

        for b_start in range(0, n_images, batch_size):
            b_end = min(b_start + batch_size, n_images)
            b_ids = mech_image_ids[b_start:b_end]

            batch_clean_t = []
            batch_corr_t = []
            batch_labels = []
            valid_ids = []

            for img_id in b_ids:
                img_p = Path(image_dir) / f"{img_id}.JPEG"
                if not img_p.exists():
                    continue
                pil_img = Image.open(img_p)
                arr_clean, _ = preprocess_image_448(pil_img)
                if cond == "clean" or sev == 0:
                    arr_corr = arr_clean
                else:
                    arr_corr = apply_corruption(arr_clean, image_id=img_id, corruption_name=cond, severity=sev)

                t_clean = torch.from_numpy(arr_clean).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t_corr = torch.from_numpy(arr_corr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                label = val_metadata[img_id]["class_idx"]

                batch_clean_t.append(t_clean)
                batch_corr_t.append(t_corr)
                batch_labels.append(label)
                valid_ids.append(img_id)

            if not valid_ids:
                continue

            # Process each suite arm for this batch
            first_suite = build_m7_v2_suite(batch_clean_t[0], batch_corr_t[0], cond, sev, valid_ids[0])
            suite_keys = list(first_suite.keys())

            for skey in suite_keys:
                suite_tensors = []
                res_sigmas = []

                for i, iid in enumerate(valid_ids):
                    s_dict = build_m7_v2_suite(batch_clean_t[i], batch_corr_t[i], cond, sev, iid)
                    s_t = s_dict[skey]
                    suite_tensors.append(s_t)

                    # Estimate residual noise sigma on the filtered tensor
                    np_img = (s_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    res_sigmas.append(float(estimate_immerkaer_noise_sigma(np_img)))

                batch_in = torch.cat(suite_tensors, dim=0).to(device)
                curr_res = batch_in.shape[-1]  # 448 or 224

                # Select active models based on resolution
                if curr_res == 224:
                    models_to_run = [
                        ("deit_base", "standard", m_deit),
                        ("efficientnet_b3", "standard", m_eff),
                        ("flexivit_base", "F-p", m_flex_fp_224),
                        ("flexivit_base", "F-t", m_flex_ft_224),
                    ]
                else:
                    models_to_run = [
                        ("deit_base", "standard", m_deit),
                        ("efficientnet_b3", "standard", m_eff),
                        ("flexivit_base", "F-p", m_flex_fp_448),
                        ("flexivit_base", "F-t", m_flex_ft_448),
                    ]

                for m_name, arm_name, model in models_to_run:
                    tag = MODEL_TAGS.get(m_name, m_name)
                    norm_in = normalize_tensor(batch_in, model_tag=tag)
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
                            "suite_condition": skey,
                            "resolution": curr_res,
                            "model": m_name,
                            "arm": arm_name,
                            "label": lbl,
                            "pred": prd,
                            "correct": bool(prd == lbl),
                            "residual_noise_sigma": res_sigmas[idx_img],
                        })

            print(f"[{cond}-s{sev}] Processed {b_end}/{n_images} images", flush=True)

        df_stage = pd.DataFrame(stage_records)
        df_stage.to_parquet(shard_file, index=False)
        print(f"Saved stage shard: {shard_file} ({len(df_stage)} rows)", flush=True)
        combined_shards.append(shard_file)

    # Combine all shards
    print("\n=== Consolidating All Shards into Final Dataset ===", flush=True)
    all_dfs = [pd.read_parquet(s) for s in combined_shards if s.exists()]
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        final_parquet = out_path / "shard_m7_v2.parquet"
        df_all.to_parquet(final_parquet, index=False)
        print(f"Final M7-v2 dataset written to {final_parquet} ({len(df_all)} rows)", flush=True)

    # Save run metadata
    meta = get_environment_metadata(device)
    meta["kernel"] = "k8-m7-v2"
    with open(out_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {out_path / 'run_meta.json'}", flush=True)


def main():
    print("=== Launching K8-M7-v2 Kernel ===", flush=True)
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Locate splits and metadata
    SPLITS_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "splits")),
        "/kaggle/input/datasets/anshulsingh45/knobs-code/splits",
        "/kaggle/input/knobs-code/splits",
        "/tmp/splits",
    ]
    mech_path, val_meta_path = None, None
    for sdir in SPLITS_DIRS:
        mp = Path(sdir) / "MECH.json"
        vp = Path(sdir) / "val_metadata.json"
        if mp.exists() and vp.exists():
            mech_path, val_meta_path = mp, vp
            break

    if not mech_path or not val_meta_path:
        raise FileNotFoundError("Could not find MECH.json or val_metadata.json in search paths")

    with open(mech_path, "r", encoding="utf-8") as f:
        mech_ids = json.load(f)
    with open(val_meta_path, "r", encoding="utf-8") as f:
        val_metadata = json.load(f)

    print(f"Loaded {len(mech_ids)} MECH image IDs", flush=True)

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

    run_m7_v2(
        image_dir=img_dir,
        mech_image_ids=mech_ids,
        val_metadata=val_metadata,
        output_dir=out_dir,
        batch_size=32,
        device=device,
        pretrained=True,
    )
    print(f"K8-M7-v2 finished in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
