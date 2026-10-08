"""K9-FreqNoise-v2: Frequency-Controlled Annular Noise on MECH (2,000 images).

Decomposes white noise into spatial frequency annuli in the 448 acquisition frame:
- low: [0, 56) cycles/image
- mid: [56, 112) cycles/image
- high: [112, 224] cycles/image
- broadband: [0, 224] (unfiltered baseline)

Evaluated across:
- Severities: s3 (pre-clip RMS = 45.9 = 0.18 * 255), s5 (pre-clip RMS = 96.9 = 0.38 * 255)
- Clean baseline (RMS = 0)
- Resolutions: 224, 320, 384, 448
- Models: EfficientNet-B3, DeiT-B, FlexiViT F-p, FlexiViT F-t

Hypothesis recorded prior to execution:
"If the spectral story holds, low-band noise produces little 448-vs-384 drop for
EfficientNet-B3, while high-band noise accounts for most of the high-resolution collapse."
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
from knobs.corrupt import apply_fft_frequency_noise, compute_seed
from knobs.resize import resize_tensor_torch
from knobs.run_grid import get_environment_metadata


def run_freqnoise_v2(
    image_dir: str,
    mech_image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str = "/kaggle/working",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
):
    """Executes K9 annular frequency noise across models, bands, and resolutions."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [224, 320, 384, 448]
    bands = ["low", "mid", "high", "broadband"]
    severities = [3, 5]

    stages = [("clean", 0)]
    for band in bands:
        for sev in severities:
            stages.append((f"freq_{band}", sev))

    print(f"=== Initializing Model Instances on {device} ===", flush=True)
    # DeiT and EfficientNet instantiate once for all resolutions
    model_deit = create_model_instance("deit_base", resolution=448, pretrained=pretrained, device=device)
    model_eff = create_model_instance("efficientnet_b3", resolution=448, pretrained=pretrained, device=device)

    # FlexiViT models per resolution and arm
    flex_models = {}
    for r in resolutions:
        flex_models[("F-p", r)] = create_model_instance("flexivit_base", resolution=r, arm="F-p", pretrained=pretrained, device=device)
        flex_models[("F-t", r)] = create_model_instance("flexivit_base", resolution=r, arm="F-t", pretrained=pretrained, device=device)

    combined_shards = []

    for stage_name, sev in stages:
        shard_file = shards_dir / f"shard_freqnoise_v2_{stage_name}_{sev}.parquet"
        if shard_file.exists():
            print(f"Shard {shard_file.name} already exists. Skipping (resumable).", flush=True)
            combined_shards.append(shard_file)
            continue

        print(f"\n--- Running Stage: {stage_name} (sev={sev}) ---", flush=True)
        stage_records = []
        n_images = len(mech_image_ids)

        band_name = stage_name.replace("freq_", "") if stage_name.startswith("freq_") else "none"

        for b_start in range(0, n_images, batch_size):
            b_end = min(b_start + batch_size, n_images)
            b_ids = mech_image_ids[b_start:b_end]

            batch_448 = []
            batch_labels = []
            pre_rms_list = []
            post_rms_list = []
            valid_ids = []

            for img_id in b_ids:
                img_p = Path(image_dir) / f"{img_id}.JPEG"
                if not img_p.exists():
                    continue
                with Image.open(img_p) as pil_img:
                    arr_448, _ = preprocess_image_448(pil_img)

                if stage_name == "clean":
                    corr_arr = arr_448
                    pre_rms = 0.0
                    post_rms = 0.0
                else:
                    corr_arr, pre_rms, post_rms = apply_fft_frequency_noise(
                        img_448=arr_448,
                        image_id=img_id,
                        band=band_name,
                        severity=sev,
                    )

                t_448 = torch.from_numpy(corr_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                batch_448.append(t_448)
                batch_labels.append(val_metadata[img_id]["class_idx"])
                pre_rms_list.append(pre_rms)
                post_rms_list.append(post_rms)
                valid_ids.append(img_id)
                del arr_448, corr_arr

            if not valid_ids:
                continue

            t_batch_448 = torch.cat(batch_448, dim=0).to(device)
            del batch_448

            for res in resolutions:
                t_res = resize_tensor_torch(t_batch_448, target_size=res)

                # Models to run at this resolution
                eval_configs = [
                    ("deit_base", "standard", model_deit),
                    ("efficientnet_b3", "standard", model_eff),
                    ("flexivit_base", "F-p", flex_models[("F-p", res)]),
                    ("flexivit_base", "F-t", flex_models[("F-t", res)]),
                ]

                for m_name, arm_name, m_inst in eval_configs:
                    tag = MODEL_TAGS.get(m_name, m_name)
                    norm_in = normalize_tensor(t_res, model_tag=tag)
                    with torch.no_grad():
                        if device.type == "cuda":
                            with torch.amp.autocast("cuda"):
                                logits = m_inst(norm_in)
                        else:
                            logits = m_inst(norm_in)
                    preds = logits.argmax(dim=-1).cpu().numpy()

                    for idx_img, iid in enumerate(valid_ids):
                        lbl = batch_labels[idx_img]
                        prd = int(preds[idx_img])
                        stage_records.append({
                            "image_id": iid,
                            "condition": stage_name,
                            "severity": sev,
                            "band": band_name,
                            "resolution": res,
                            "model": m_name,
                            "arm": arm_name,
                            "label": lbl,
                            "pred": prd,
                            "correct": bool(prd == lbl),
                            "pre_clip_rms": pre_rms_list[idx_img],
                            "post_clip_rms": post_rms_list[idx_img],
                        })
                    del norm_in, logits

                del t_res

            del t_batch_448
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[{stage_name}-s{sev}] Processed {b_end}/{n_images} images", flush=True)

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
        final_parquet = out_path / "shard_freqnoise_v2.parquet"
        df_all.to_parquet(final_parquet, index=False)
        print(f"Final dataset saved to {final_parquet} ({len(df_all)} rows)", flush=True)

    # Save run metadata
    meta = get_environment_metadata(device)
    meta["kernel"] = "k9-freqnoise-v2"
    meta["hypothesis_prediction"] = (
        "If the spectral story holds, low-band noise gives little 448-vs-384 drop for "
        "EfficientNet and high-band noise gives most of it."
    )
    with open(out_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {out_path / 'run_meta.json'}", flush=True)


def main():
    print("=== Launching K9-FreqNoise-v2 Kernel ===", flush=True)
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

    run_freqnoise_v2(
        image_dir=img_dir,
        mech_image_ids=mech_ids,
        val_metadata=val_metadata,
        output_dir=out_dir,
        batch_size=32,
        device=device,
        pretrained=True,
    )
    print(f"K9-FreqNoise-v2 finished in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
