"""K7-Latency: Formal Latency and Throughput Benchmarking on single T4 GPU.

Scope:
- Measures batch 1 (interactive latency) and batch 64 (throughput) across models and resolutions
- Precision: FP16 with CUDA events
- Evaluates: DeiT-B, EfficientNet-B3, FlexiViT-B, ToMe variants
- Generates Table 8 hardware metrics
"""

import os
import sys
import time
import json
import zipfile
import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print("=== K7-Latency Initializing ===", flush=True)
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
from knobs.models import MODEL_TAGS, create_model_instance
from knobs.tokens import patch_vit_with_tome
from knobs.latency import benchmark_model_latency
from knobs.run_grid import get_environment_metadata

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active timing device: {device} ({torch.cuda.get_device_name(device)})")

# 2. Benchmark Suite Configurations
MODELS = ["deit_base", "efficientnet_b3", "flexivit_base"]
RESOLUTIONS = [224, 320, 384, 448]
BATCH_SIZES = [1, 64]

results = []

print("\n=== Running Latency & Throughput Benchmark Suite ===")

for m_key in MODELS:
    print(f"\nBenchmarking model: {m_key}...")
    for res in RESOLUTIONS:
        model = create_model_instance(m_key, resolution=res, arm="F-p", pretrained=False, device=device)
        for b_size in BATCH_SIZES:
            n_warm = 30 if b_size == 1 else 15
            n_timed = 100 if b_size == 1 else 50
            
            bench = benchmark_model_latency(
                model=model,
                resolution=res,
                batch_size=b_size,
                n_warmup=n_warm,
                n_timed=n_timed,
                device=device,
                use_fp16=True,
            )
            
            row = {
                "model": m_key,
                "variant": "standard",
                "resolution": res,
                "batch_size": b_size,
                "median_latency_ms": bench["median_batch_latency_ms"],
                "iqr_latency_ms": bench.get("iqr_ms", bench.get("iqr_batch_latency_ms", 0.0)),
                "throughput_img_per_sec": bench["throughput_img_per_sec"],
            }
            results.append(row)
            print(f"  {m_key} @ {res}x{res} (b={b_size}): Latency={bench['median_batch_latency_ms']:.2f}ms, Throughput={bench['throughput_img_per_sec']:.1f} img/s")
        del model
        torch.cuda.empty_cache()

# ToMe variants
print("\nBenchmarking ToMe variants on DeiT-B...")
deit_base_model = create_model_instance("deit_base", resolution=224, pretrained=False, device=device)
n_blocks = len(deit_base_model.blocks)

for r_tome, tome_label in [(4, "tome_r4"), (8, "tome_r8")]:
    patched_model = patch_vit_with_tome(deit_base_model, [r_tome] * n_blocks)
    for res in [224, 448]:
        for b_size in BATCH_SIZES:
            n_warm = 30 if b_size == 1 else 15
            n_timed = 100 if b_size == 1 else 50
            
            bench = benchmark_model_latency(
                model=patched_model,
                resolution=res,
                batch_size=b_size,
                n_warmup=n_warm,
                n_timed=n_timed,
                device=device,
                use_fp16=True,
            )
            
            row = {
                "model": "deit_base",
                "variant": tome_label,
                "resolution": res,
                "batch_size": b_size,
                "median_latency_ms": bench["median_batch_latency_ms"],
                "iqr_latency_ms": bench.get("iqr_ms", bench.get("iqr_batch_latency_ms", 0.0)),
                "throughput_img_per_sec": bench["throughput_img_per_sec"],
            }
            results.append(row)
            print(f"  DeiT-B {tome_label} @ {res}x{res} (b={b_size}): Latency={bench['median_batch_latency_ms']:.2f}ms, Throughput={bench['throughput_img_per_sec']:.1f} img/s")

# Export results
df_results = pd.DataFrame(results)
csv_path = "/kaggle/working/Table8_hardware_efficiency.csv"
df_results.to_csv(csv_path, index=False)

json_path = "/kaggle/working/k7_hardware_benchmark.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved benchmark outputs to {csv_path} and {json_path}")

elapsed_wall = time.time() - start_wall_time
wall_h = elapsed_wall / 3600.0
print(f"K7-Latency Complete: Elapsed = {wall_h:.3f} Wall-Clock h ({elapsed_wall:.1f}s)")

with open("/kaggle/working/run_meta.json", "w") as f:
    json.dump(get_environment_metadata(device), f, indent=2)
