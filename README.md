# Knobs Under Corruption

> **"Which efficiency knob should you turn when the input is degraded?"**  
> An empirical study of inference resolution scaling, token pruning, and model architectures under ImageNet-1K degradation.

---

## 1. Verified Pipeline Architecture & Physical Facts

- **Standardized Acquisition Frame (`data.preprocess_image_448`)**: Every validation image is converted to a fixed 448×448 acquisition frame (short side resized to 512 with bilinear interpolation, followed by a 448×448 center crop with crop ratio 0.875).
- **Deterministic In-Frame Corruption (`corrupt.apply_corruption`)**: Corruptions are injected strictly within this 448×448 acquisition frame using isolated 32-bit seeds derived from SHA-256 hashes (`compute_seed(image_id, corruption, severity)`).
- **Resampling to Inference Resolution (`resize.resize_tensor_torch`)**: The corrupted frame is subsequently resized to target inference resolutions $R \in \{224, 320, 384, 448\}$ using PyTorch's antialiased bilinear interpolation (`F.interpolate(mode="bilinear", antialias=True)`). At $R = 448$, the operation is the exact mathematical identity.
- **Physical Antialiasing Noise Gain (`resize.effective_noise_gain`)**: Downsampling applies an area-weighted continuous triangle convolution. When white noise is injected at 448 px, the residual standard deviation seen by the model is:
  - **224 px**: $\sigma_{\text{eff}} = \mathbf{0.313} \times \sigma_{\text{inj}}$ (variance gain = $0.098$)
  - **320 px**: $\sigma_{\text{eff}} = \mathbf{0.481} \times \sigma_{\text{inj}}$ (variance gain = $0.231$)
  - **384 px**: $\sigma_{\text{eff}} = \mathbf{0.565} \times \sigma_{\text{inj}}$ (variance gain = $0.320$)
  - **448 px**: $\sigma_{\text{eff}} = \mathbf{1.000} \times \sigma_{\text{inj}}$ (identity, no filtering)  
  *This explains the abrupt collapse of EfficientNet-B3 between 384 and 448 px as an input noise variance doubling.*
- **Class-Balanced Sampling**: Partition files (`PILOT.json`, `MECH.json`) are class-ordered lists. Taking `[:N]` gives only the first classes (mostly animals). Deterministic balanced sampling must always use `balanced_subset(ids, meta, n)`.
- **Project Sign Convention**:
  $$\text{DiD} = (\text{Acc}_{448} - \text{Acc}_{224})_{\text{degraded}} - (\text{Acc}_{448} - \text{Acc}_{224})_{\text{clean}}$$
  *A negative value means 448 px hurts more under degradation than on clean images.*

---

## 2. Quickstart & Testing

```bash
# Clone repository
git clone https://github.com/AnshulSinghhhhhh/vision knobs-under-corruption
cd knobs-under-corruption

# Install in editable mode with development dependencies
pip install -e .[dev]

# Run full test suite (61 tests, CPU-only, completes in ~1 minute)
pytest tests -q
```

---

## 3. Reproducing Every Paper Table & Figure

All analysis scripts read raw per-image evaluation shards (`results/raw/*/shards/*.parquet`, containing 4,565,000 forward passes) and output reproducible tables and figures into `analysis/out/`:

```bash
# Table 1: Primary Difference-in-Differences with analytic paired z-tests and Holm correction
python -m analysis.table1_did

# Tables 2-5: Matched controls (FlexiViT F-p vs F-t, DeiT-384 native, ToMe, BatchNorm recalibration)
python -m analysis.matched_tables

# Table of Effective Noise Sigma & Excess Loss Decomposition
python -m analysis.effective_sigma

# Per-Image Accuracy Transition Matrix & Invariance Proof
python -m analysis.transitions

# M7 Information Control Multi-Cutoff Decomposition
python -m analysis.m7

# Figures 1-4: Publication vector figures (PDF and 300 DPI PNG)
python -m analysis.figures

# Model Configurations and Positional Embedding Inspection (writes docs/model_configs.md)
python -m analysis.flexivit_config
```

---

## 4. Kaggle Execution Protocol & Registered Kernels

Kaggle kernels are managed and dispatched using the unified runner `kaggle/push_and_wait.py`:

```bash
# List all registered kernels
python kaggle/push_and_wait.py --list

# Dispatch and monitor a specific kernel:
python kaggle/push_and_wait.py k8-m7-v2           # Mechanistic M7 information control across 4 models
python kaggle/push_and_wait.py k9-freqnoise-v2    # Annular FFT bandlimited noise (low/mid/high/broadband)
python kaggle/push_and_wait.py k10-resize-ablation# Interpolation operator noise gain ablation
python kaggle/push_and_wait.py k11-sensor-noise   # Poisson-Gaussian realistic sensor noise model
python kaggle/push_and_wait.py k12-tome-matched   # FLOP-matched ToMe vs resolution reduction on DeiT
python kaggle/push_and_wait.py k13-g0a-and-bn     # Full 50k native G0-A check & calibrated EfficientNet BN
```

---

## 5. Repository Structure

```
knobs-under-corruption/
├── README.md                      # Pipeline facts, reproduction instructions, and quickstart
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Pinned dependencies for numerical reproducibility
├── .github/workflows/ci.yml       # GitHub Actions CI workflow (CPU PyTorch + pytest)
├── docs/
│   ├── CHANGELOG_FIXES.md         # Detailed audit log of A1-A9 hardening fixes
│   └── model_configs.md           # Pretrained architecture and positional embedding audit
├── src/knobs/
│   ├── data.py                    # 448 acquisition frame, bbox coordinate transforms, balanced_subset
│   ├── corrupt.py                 # Deterministic ImageNet-C corruptions & 2D FFT annular bandpass noise
│   ├── models.py                  # Pretrained model loader, dynamic resolution adaptation, calibrated BN
│   ├── resize.py                  # Bilinear antialiased resize, effective noise gain, Gaussian filters
│   ├── tokens.py                  # Faithful Token Merging (ToMe) with proportional attention & r solver
│   ├── flops.py                   # FlopCounterMode GFLOPs accounting (2x MAC convention)
│   ├── latency.py                 # Latency & throughput harness (channels_last, inference_mode)
│   ├── stats.py                   # Analytic paired DiD z-test, Holm-Bonferroni correction, bootstrap CIs
│   └── run_grid.py                # Resumable Parquet shard runner with git commit provenance
├── analysis/                      # Standalone reproduction scripts for all paper artifacts
│   ├── common.py                  # Raw shard loader, validation, image matching, and caching
│   ├── table1_did.py              # Headline Table 1 reproduction
│   ├── matched_tables.py          # Tables 2-5 reproduction
│   ├── effective_sigma.py         # Sigma_eff table and excess drop analysis
│   ├── transitions.py             # Accuracy transitions and DiD invariance proof
│   ├── m7.py                      # M7 information control breakdown
│   ├── figures.py                 # Vector figures 1-4 generation
│   └── flexivit_config.py         # FlexiViT & DeiT architectural inspection
├── kaggle/
│   ├── push_and_wait.py           # Automated packaging, git provenance embedding, and kernel runner
│   └── kernels/                   # Registered reproducible Kaggle kernels (k0 through k13)
├── tests/                         # Pytest test suite (61 tests covering all modules)
└── report/                        # Manuscript source (paper.tex), tables, and findings
```
