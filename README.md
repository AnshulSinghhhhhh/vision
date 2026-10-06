# Knobs Under Corruption

> **"Which efficiency knob should you turn when the input is degraded?"**  
> Empirical study on input resolution, token reduction, and model architecture under ImageNet degradation.

## 1. Quickstart & Local Setup

```bash
# Clone and install in editable mode
git clone <repo-url> knobs-under-corruption
cd knobs-under-corruption
pip install -e .

# Run local unit tests (17 CPU tests)
pytest tests -v
```

## 2. Experimental Execution Protocol (Kaggle T4×2)

Execution runs sequentially through pre-specified stages via the Kaggle CLI:

```bash
# Phase 0: Data Preparation & Partitioning (CPU Kernel, 0 GPU quota)
python kaggle/push_and_wait.py k-prep

# Phase 0.5: Native Checkpoint Verification & Throughput Benchmark (GPU T4x2)
python kaggle/push_and_wait.py k0-probe

# Phase 1: Pipeline Sanity (GPU T4x2)
python kaggle/push_and_wait.py k1-sanity

# Phase 2: Pilot Study & Decision Gates Evaluation (5k images)
python kaggle/push_and_wait.py k2-pilot
```

## 3. Repository Architecture

```
knobs-under-corruption/
  README.md                  # Reproduction instructions & overview
  pyproject.toml             # Package specification
  requirements.txt           # Pinned dependencies
  src/knobs/
    data.py                  # ImageNet val loader, bbox transform, partition builder
    corrupt.py               # Deterministic ImageNet-C corruptions & frequency-noise control
    models.py                # Model loader (DeiT-B, DeiT-384, EffNet-B3, FlexiViT-B)
    resize.py                # Antialiased bilinear downsampling & M7 filter control
    tokens.py                # Token Merging (ToMe) wrapper & schedule builder
    flops.py                 # FlopCounterMode GFLOPs accounting (2x MACs)
    latency.py               # CUDA event latency & throughput benchmarking
    spectral.py              # Immerkaer noise sigma, variance of Laplacian, radial PSD
    selector.py              # Zero-parameter spectral gate & oracle upper bounds
    mechanism.py             # Area-normalized attention mass, entropy, drift, occlusion
    stats.py                 # Paired bootstrap CIs, DiD, Holm correction, TOST
    run_grid.py              # Resumable Parquet shard runner with tensor caching
  kaggle/
    push_and_wait.py         # Automated code dataset packaging, pushing & polling
    kernels/
      k-prep/                # CPU split generator
      k0-probe/              # GPU native verification & benchmark
  tests/                     # Complete pytest suite (17 tests)
  results/                   # Parquet shards and merged results
  analysis/                  # Analysis scripts for Tables 1-8 and Figures 1-6
  report/                    # Pre-specified gates, pilot findings, and paper report
```

## 4. Security & Credentials
- No credentials or API tokens are stored in notebooks, datasets, or code files.
- Authenticated pushes are managed via the host Kaggle CLI configuration (`~/.kaggle/`).
