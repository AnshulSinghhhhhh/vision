"""K12-ToMe-Matched: FLOP-Matched Token Merging vs Spatial Resolution Reduction on DeiT-B.

Compares compute-matched arms on DeiT-B/16 across 5,000 class-balanced PILOT images:
1. Standard 448 px baseline (r=0, ~134.3 GFLOPs, 100% compute)
2. ToMe r=32 at 448 px (~100.9 GFLOPs, ~75% compute)
3. Plain resolution reduction 384 px (~98.6 GFLOPs, ~73% compute, FLOP-matched to r=32)
4. ToMe r=64 at 448 px (~67.3 GFLOPs, ~50% compute)
5. Plain resolution reduction 320 px (~68.5 GFLOPs, ~51% compute, FLOP-matched to r=64)
6. Standard 224 px baseline (~33.5 GFLOPs, ~25% compute)

Evaluated under:
- clean (severity 0)
- gaussian_noise (severity 3, severity 5)
- defocus_blur (severity 3)

Measures and records per-arm GFLOPs and empirical throughput (img/s).
Saves per-(condition, severity) shards to `shards/shard_tome_matched_{cond}_{sev}.parquet`,
and combines into `shard_tome_matched.parquet`.
"""

import os
import sys
import time
import json
import zipfile
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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

from knobs.models import MODEL_TAGS, create_model_instance, normalize_tensor
from knobs.data import preprocess_image_448, balanced_subset
from knobs.corrupt import apply_corruption
from knobs.resize import resize_tensor_torch
from knobs.tokens import patch_vit_with_tome
from knobs.flops import count_gflops
from knobs.run_grid import get_environment_metadata


def measure_throughput_img_per_s(
    model: torch.nn.Module,
    resolution: int,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    warmup_iters: int = 5,
    timed_iters: int = 15,
) -> float:
    """Measures steady-state inference throughput (images/sec)."""
    model.eval()
    dummy = torch.randn(batch_size, 3, resolution, resolution, device=device)
    tag = MODEL_TAGS["deit_base"]
    dummy_norm = normalize_tensor(dummy, model_tag=tag)

    with torch.no_grad():
        for _ in range(warmup_iters):
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(dummy_norm)
            else:
                _ = model(dummy_norm)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(timed_iters):
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(dummy_norm)
            else:
                _ = model(dummy_norm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - start

    total_images = timed_iters * batch_size
    return float(total_images / max(total_time, 1e-6))


def build_tome_matched_configs(
    device: torch.device,
    pretrained: bool = True,
    profile_throughput: bool = True,
    warmup_iters: Optional[int] = None,
    timed_iters: Optional[int] = None,
    batch_size: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """Builds the 6 evaluation model configurations with GFLOPs and throughput."""
    print("Building model instances and patching ToMe...", flush=True)

    configs = {}

    # 1. 448 standard baseline
    m_448 = create_model_instance("deit_base", resolution=448, pretrained=pretrained, device=device)
    configs["res_448_base"] = {
        "model": m_448,
        "resolution": 448,
        "r_tome": 0,
        "arm": "standard",
    }

    # 2. ToMe r=32 at 448 (nominal ~75% compute)
    m_tome32 = create_model_instance("deit_base", resolution=448, pretrained=pretrained, device=device)
    patch_vit_with_tome(m_tome32, r_schedule=[32] * len(m_tome32.blocks))
    configs["tome_r32_448"] = {
        "model": m_tome32,
        "resolution": 448,
        "r_tome": 32,
        "arm": "tome_r32",
    }

    # 3. Plain 384 px baseline (FLOP matched to r=32)
    m_384 = create_model_instance("deit_base", resolution=384, pretrained=pretrained, device=device)
    configs["res_384_base"] = {
        "model": m_384,
        "resolution": 384,
        "r_tome": 0,
        "arm": "standard",
    }

    # 4. ToMe r=64 at 448 (nominal ~50% compute)
    m_tome64 = create_model_instance("deit_base", resolution=448, pretrained=pretrained, device=device)
    patch_vit_with_tome(m_tome64, r_schedule=[64] * len(m_tome64.blocks))
    configs["tome_r64_448"] = {
        "model": m_tome64,
        "resolution": 448,
        "r_tome": 64,
        "arm": "tome_r64",
    }

    # 5. Plain 320 px baseline (FLOP matched to r=64)
    m_320 = create_model_instance("deit_base", resolution=320, pretrained=pretrained, device=device)
    configs["res_320_base"] = {
        "model": m_320,
        "resolution": 320,
        "r_tome": 0,
        "arm": "standard",
    }

    # 6. Standard 224 px baseline
    m_224 = create_model_instance("deit_base", resolution=224, pretrained=pretrained, device=device)
    configs["res_224_base"] = {
        "model": m_224,
        "resolution": 224,
        "r_tome": 0,
        "arm": "standard",
    }

    # Profile GFLOPs and throughput
    print("Profiling GFLOPs and throughput per config...", flush=True)
    w_iters = warmup_iters if warmup_iters is not None else (3 if device.type == "cuda" else 1)
    t_iters = timed_iters if timed_iters is not None else (10 if device.type == "cuda" else 2)
    b_size = batch_size if device.type == "cuda" else min(batch_size, 4)

    for cname, cinfo in configs.items():
        res = cinfo["resolution"]
        model = cinfo["model"]
        gf = count_gflops(model, input_resolution=res, device=device)
        cinfo["gflops"] = float(gf)

        if profile_throughput:
            th = measure_throughput_img_per_s(
                model,
                resolution=res,
                batch_size=b_size,
                device=device,
                warmup_iters=w_iters,
                timed_iters=t_iters,
            )
            cinfo["throughput"] = float(th)
        else:
            cinfo["throughput"] = 0.0

        print(f"  [{cname}] Res={res}, r={cinfo['r_tome']}: GFLOPs={gf:.2f}, Throughput={cinfo['throughput']:.1f} img/s", flush=True)

    return configs


def run_tome_matched(
    image_dir: str,
    pilot_image_ids: List[str],
    val_metadata: Dict[str, Any],
    output_dir: str = "/kaggle/working",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
):
    """Executes ToMe-matched evaluation on PILOT images."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    shards_dir = out_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    configs = build_tome_matched_configs(device=device, pretrained=pretrained)

    stages = [
        ("clean", 0),
        ("gaussian_noise", 3),
        ("gaussian_noise", 5),
        ("defocus_blur", 3),
    ]

    combined_shards = []
    n_images = len(pilot_image_ids)

    for cond, sev in stages:
        shard_file = shards_dir / f"shard_tome_matched_{cond}_{sev}.parquet"
        if shard_file.exists():
            print(f"Shard {shard_file.name} already exists. Skipping (resumable).", flush=True)
            combined_shards.append(shard_file)
            continue

        print(f"\n--- Running Stage: {cond} (severity={sev}) ---", flush=True)
        stage_records = []

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

            t_batch_448 = torch.cat(batch_448, dim=0)

            # Evaluate each configuration
            for cname, cinfo in configs.items():
                res = cinfo["resolution"]
                model = cinfo["model"]
                arm = cinfo["arm"]
                r_tome = cinfo["r_tome"]
                gf = cinfo["gflops"]
                th = cinfo["throughput"]

                t_in = resize_tensor_torch(t_batch_448, target_size=res).to(device)
                norm_in = normalize_tensor(t_in, model_tag=MODEL_TAGS["deit_base"])

                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda"):
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
                        "config_name": cname,
                        "arm": arm,
                        "resolution": res,
                        "r_tome": r_tome,
                        "gflops": gf,
                        "throughput_img_per_s": th,
                        "model": "deit_base",
                        "label": lbl,
                        "pred": prd,
                        "correct": bool(prd == lbl),
                    })

            del batch_448, t_batch_448, t_in, norm_in
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
        final_parquet = out_path / "shard_tome_matched.parquet"
        df_all.to_parquet(final_parquet, index=False)
        print(f"Final dataset saved to {final_parquet} ({len(df_all)} rows)", flush=True)

    # Save run metadata
    meta = get_environment_metadata(device)
    meta["kernel"] = "k12-tome-matched"
    meta["configs_profile"] = {
        cname: {
            "resolution": cinfo["resolution"],
            "r_tome": cinfo["r_tome"],
            "gflops": cinfo["gflops"],
            "throughput_img_per_s": cinfo["throughput"],
        }
        for cname, cinfo in configs.items()
    }
    with open(out_path / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata to {out_path / 'run_meta.json'}", flush=True)


def main():
    print("=== Launching K12-ToMe-Matched Kernel ===", flush=True)
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

    # Class-balanced 5,000 images
    balanced_ids = balanced_subset(pilot_ids, val_metadata, n=5000)
    print(f"Selected {len(balanced_ids)} class-balanced PILOT images", flush=True)

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

    run_tome_matched(
        image_dir=img_dir,
        pilot_image_ids=balanced_ids,
        val_metadata=val_metadata,
        output_dir=out_dir,
        batch_size=32,
        device=device,
        pretrained=True,
    )
    print(f"K12-ToMe-Matched finished in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
