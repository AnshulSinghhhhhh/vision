# Spectral Pass-Through or Grid Mismatch?
### Decomposing Why Higher Test Resolution Amplifies Corruption Damage in CNNs and Vision Transformers

[![CI](https://github.com/AnshulSinghhhhhh/vision/actions/workflows/ci.yml/badge.svg)](https://github.com/AnshulSinghhhhhh/vision/actions/workflows/ci.yml)
[![Tests: 81 Passed](https://img.shields.io/badge/tests-81%20passed-brightgreen.svg)](tests/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![ImageNet-1K: 50,000 Paired Images](https://img.shields.io/badge/ImageNet--1K-50k%20paired-orange.svg)](splits/FULL.json)
[![Evaluations: 5.7M Forward Passes](https://img.shields.io/badge/evaluations-5.7M%20passes-blueviolet.svg)](results/raw/)
[![Paper Draft: IEEE Conference](https://img.shields.io/badge/paper-IEEE%20conference%20draft-red.svg)](paper/main.tex)

---

## Publication Manuscript & Audit Navigation

This repository contains the complete empirical data shards, experimental code, and reproduction test suite for the IEEE conference paper:
> **"Spectral Pass-Through or Grid Mismatch? Decomposing Why Higher Test Resolution Amplifies Corruption Damage in CNNs and Vision Transformers"**

- 📄 **Conference Paper Manuscript:** [`paper/main.tex`](paper/main.tex) (Official IEEEtran layout, self-contained)
- 📊 **Publication Figures & Empirical Datasets:** [`paper/figures/`](paper/figures/) (Figures 1–4 in vector PDF, 300 DPI PNG, CSV/JSON numerical data)
  - `fig1_resolution_curves.pdf` / `.png` & `headline_curves.csv`: Full multi-resolution accuracy trajectories
  - `fig2_decomposition.pdf` / `.png` & `decomposition_summary.csv`: Exact spectral vs grid decomposition
  - `fig3_band_did.pdf` / `.png` & `k9_band_did.csv`: Annular Fourier band-limited noise interaction
  - `fig4_operator_gain.pdf` / `.png` & `k10_operator_gain.csv`: Accuracy vs measured white-noise gain
  - `paper_numbers.json`: Source-of-truth machine-readable dictionary for all manuscript metrics
- 📑 **Comprehensive Audit Suite:**
  - [`00_FINAL_REPORT.md`](00_FINAL_REPORT.md) — Executive audit summary, referee evaluation & venue recommendations (IEEE ICIP/ICME/IJCNN)
  - [`01_experiment_inventory.md`](01_experiment_inventory.md) — Comprehensive inventory of all 16 execution kernels (`k0`–`k16`)
  - [`02_experiment_dependency_timeline.md`](02_experiment_dependency_timeline.md) — Execution dependency timeline and provenance graph
  - [`03_cross_experiment_results_matrix.md`](03_cross_experiment_results_matrix.md) — Unified cross-kernel results matrix and hidden patterns
  - [`04_claim_verification_audit.md`](04_claim_verification_audit.md) — Claim-by-claim verification, corrections, and pre-registration audit
  - [`05_novelty_and_related_work_audit.md`](05_novelty_and_related_work_audit.md) — Verified literature citations and novelty matrix
  - [`06_ranked_paper_opportunities.md`](06_ranked_paper_opportunities.md) — Evidence-ranked research opportunities and proposed experiments (`E1`–`E7`)

---

## Executive Summary

Running a pretrained classifier at a higher test resolution usually buys clean accuracy (e.g., Touvron et al., 2019). However, when images suffer from capture corruptions (noise, blur, compression), **the extra pixels can cost far more than they return**.

Prior literature has shown that smaller-resolution checkpoints are more robust to Gaussian noise due to anti-aliased downsampling (Kim et al., 2025). But within a single fixed set of weights, raising test resolution changes two physical factors simultaneously:
1. **Spectral Pass-Through**: Frequencies above the lower resolution's Nyquist limit—which continuous anti-aliased downsampling would have removed—now reach the network.
2. **Grid Mismatch**: The same semantic information is processed on a finer pixel grid; for Vision Transformers, token count quadruples and position embeddings are interpolated.

```
[448x448 Acquisition Frame] ──► Deterministic SHA-256 Corruption Injection
         │
         ├── Rendering A: 448 px (Identity, σ_eff = 1.000 × σ_inj, 10.22× Noise Variance)
         ├── Rendering C: 224 px Downsampled (Continuous AA Triangle Filter, σ_eff = 0.313 × σ_inj)
         ├── Rendering D: 224 px Downsampled & Upsampled back to 448 px (Information-Matched Control)
         └── Rendering B: 448 px with Low-Pass Prefiltering (3-tap binomial / Gaussian)
```

Through **5,700,000+ forward passes** across five vision models (**DeiT-B/16**, **EfficientNet-B3**, **FlexiViT-B**, **ResNet-50**, and **ConvNeXt-Base**) on ImageNet-1K with corruptions fixed in a 448 px acquisition frame:

1. **Higher Test Resolution Amplifies Corruption Damage**:
   Under Gaussian noise, the paired difference-in-differences (DiD) between 448 px and 224 px is **$-21.92\text{ pp}$** for EfficientNet-B3, **$-7.21\text{ pp}$** for FlexiViT-B, and **$-4.08\text{ pp}$** for DeiT-B/16 on $N=50{,}000$ paired images (all $|z| > 22$, Holm-adjusted $p < 10^{-15}$). Under severe noise (s5), EfficientNet-B3 collapses to **$3.72\%$ accuracy** at 448 px (vs $59.04\%$ at 224 px; DiD **$-59.85\text{ pp}$**).

2. **The Exact Spectral/Grid Decomposition**:
   We decouple the two factors using an information-matched control: **Rendering D** ($U(S_{224}\tilde{x})$), the 224 px image upsampled back to 448 px. Rendering D carries **strictly the information of C on the grid of A**. The per-image identity:
   $$\underbrace{\Delta(\mathrm{A}, \mathrm{C})}_{\text{Total DiD } \Delta} = \underbrace{\Delta(\mathrm{A}, \mathrm{D})}_{\text{Spectral Term } \Delta_{\mathrm{spec}}} + \underbrace{\Delta(\mathrm{D}, \mathrm{C})}_{\text{Grid Term } \Delta_{\mathrm{grid}}}$$
   holds exactly with no modeling assumptions.

3. **The Mechanisms Dissociate by Architecture**:
   - **CNN (EfficientNet-B3)**: Overwhelmingly **spectral pass-through**. At noise s5 ($N=2{,}000$), the spectral term carries **89%** of the penalty ($\Delta_{\mathrm{spec}} = -52.30 \pm 2.43\text{ pp}$), while the grid term is only $-6.40 \pm 2.21\text{ pp}$.
   - **Vision Transformer (DeiT-B/16)**: Dominated by **grid mismatch**. At noise s5, the grid term carries **87%** of the penalty ($\Delta_{\mathrm{grid}} = -17.85 \pm 2.39\text{ pp}$), while the spectral term is negligible ($-2.65 \pm 2.20\text{ pp}$). At noise s3, high frequencies actually *help* DeiT ($\Delta_{\mathrm{spec}} = +3.75 \pm 1.94\text{ pp}$) while the grid costs $-7.50 \pm 1.98\text{ pp}$.
   - **Flexible ViT (FlexiViT-B)**: When run at a constant 256-token grid (Arm F-t), the **grid term completely disappears** ($\Delta_{\mathrm{grid}} = +0.55 \pm 1.16\text{ pp}$ at s5).

4. **Two Failure Modes $\longrightarrow$ Two Distinct Prescriptive Remedies**:
   - **For CNNs**: Low-pass prefiltering at 448 px (Rendering B) removes 85–93% of EfficientNet's penalty and lifts accuracy by $+51.3\text{ pp}$ at s5.
   - **For ViTs**: Low-pass prefiltering fails to repair DeiT (at s3 it makes DeiT *worse*, and across Gaussian widths 0.5–2.5 px DeiT remains $>16\text{ pp}$ below 224 px baseline). Instead, **coarsening the token grid** restores performance: switching from patch 16 (784 tokens) to patch 32 (196 tokens) at 448 px rescues FlexiViT by **$+39.13\text{ pp}$** under severe noise ($30.97\% \rightarrow 70.10\%$).

5. **Frequency Localization (Band-Limited Noise)**:
   - CNN vulnerability is concentrated in the **high band** above the 224 px Nyquist limit ($-54.95 \pm 2.51\text{ pp}$ at s5); low-band noise hurts equally across resolutions ($40.5\% \rightarrow 40.3\%$).
   - ViT vulnerability is concentrated in the **mid band** ($[56, 112)$ cycles/frame, corresponding to periods of 4–8 px, or 2–4 cycles per 16 px patch): DeiT DiD is **$-47.35 \pm 2.59\text{ pp}$** in mid-band vs $-13.05\text{ pp}$ in high-band.

6. **Noise Gain Explains CNNs, Not ViTs**:
   Across 14 resize operator $\times$ resolution combinations, EfficientNet-B3 accuracy closely tracks measured white-noise gain (Spearman $\rho = -0.92, p=4\times 10^{-6}$), whereas DeiT does not ($\rho = -0.41, p=0.14$). `area@224` and `bilinear_no_antialias@224` produce **100% identical predictions** (pure $2\times 2$ box decimation), showing that the 20-point deficit relative to antialiased bilinear resizing is driven by noise gain ($0.500$ vs $0.313$), not aliasing.

---

## Table of Contents

- [1. Standardized Acquisition & Physical Principles](#1-standardized-acquisition--physical-principles)
- [2. Primary Empirical Results (Table 1, N=50,000 Paired Images)](#2-primary-empirical-results-table-1-n50000-paired-images)
- [3. The Exact Spectral/Grid Decomposition (Table 2)](#3-the-exact-spectralgrid-decomposition-table-2)
- [4. Low-Pass Filtering Replications at 448 px (Table 3)](#4-low-pass-filtering-replications-at-448-px-table-3)
- [5. Frequency Localization: Annular Fourier Bands](#5-frequency-localization-annular-fourier-bands)
- [6. Operator Noise Gain vs Aliasing](#6-operator-noise-gain-vs-aliasing)
- [7. Token Grid Scaling & Structural Controls in ViTs](#7-token-grid-scaling--structural-controls-in-vits)
- [8. Checkpoint Benchmark Verification (G0-A Benchmark)](#8-checkpoint-benchmark-verification-g0-a-benchmark)
- [9. Audit Grounding & Corrections of Prior Claims](#9-audit-grounding--corrections-of-prior-claims)
- [10. Kaggle Kernel Registry & Provenance](#10-kaggle-kernel-registry--provenance)
- [11. Repository Architecture](#11-repository-architecture)
- [12. Quickstart & Complete Reproduction](#12-quickstart--complete-reproduction)
- [13. Citation](#13-citation)

---

## 1. Standardized Acquisition & Physical Principles

### Standardized Acquisition Pipeline
Every ImageNet-1K validation image undergoes a deterministic acquisition process:
1. Short side resized to $512$ px using bilinear interpolation (`Image.Resampling.BILINEAR`).
2. Center crop of $448 \times 448$ extracted (crop ratio $0.875$).
3. Corruption $c$ at severity $s$ injected **strictly inside this standardized $448 \times 448$ acquisition frame** using seed $\xi_i = \text{SHA256}(\text{image\_id} \mathbin{\Vert} c \mathbin{\Vert} s) \pmod{2^{32}}$.
4. The corrupted frame is subsequently resized to target inference resolution $R \in \{224, 320, 384, 448\}$ using PyTorch's continuous antialiased bilinear interpolation (`F.interpolate(mode="bilinear", antialias=True)`).

### Continuous Antialiasing Filter Noise Gain
When an image with independent Gaussian noise $\mathcal{N}(0, \sigma_{\text{inj}}^2)$ is downsampled with scale factor $s = R / 448 < 1$, the continuous reconstruction kernel acts as a low-pass filter with spatial support $w = 1/s$. The effective noise standard deviation is $g_R \sigma_{\text{inj}}$:

$$g_R = \frac{1}{R} \sum_{m} \sum_{k} w_{m, k}^2$$

| Target Resolution $R$ | Scale Factor $s$ | Theoretical Gain $g_R$ | Measured Empirical Gain | Relative Noise Variance vs 224 px ($g_R^2 / g_{224}^2$) |
| :---: | :---: | :---: | :---: | :---: |
| **224 px** | $0.500$ | **$0.3125$** | **$0.313$** | **$1.00\times$** (Baseline) |
| **320 px** | $0.714$ | **$0.4807$** | **$0.481$** | **$2.37\times$** |
| **384 px** | $0.857$ | **$0.5658$** | **$0.565$** | **$3.28\times$** |
| **448 px** | $1.000$ | **$1.0000$** | **$1.000$** | **$10.22\times$** |

*Notice: Downsampling to 224 px reduces noise standard deviation by $69\%$ ($g_{224} = 0.3125$), which suppresses noise variance by $\sim 90\%$ (factor of $10.22\times$). At 448 px, downsampling is the identity.*

### Difference-in-Differences (DiD) Sign Convention
$$\Delta = (\text{Acc}_{448} - \text{Acc}_{224})_{\text{corrupted}} - (\text{Acc}_{448} - \text{Acc}_{224})_{\text{clean}}$$
- **Negative $\Delta$**: Increasing resolution from 224 to 448 px is **less beneficial (or more harmful)** under corruption than on clean images.
- **Positive $\Delta$**: Increasing resolution from 224 to 448 px provides **greater benefit** under corruption than on clean images.

---

## 2. Primary Empirical Results (Table 1, N=50,000 Paired Images)

Evaluated on the full 50,000-image ImageNet-1K validation set (or 15,000 class-balanced images for s1/s5) with corruptions fixed in the 448 px frame (`analysis/out/table1.csv`):

| Model | Corruption | $N$ | $\text{Clean}_{224}$ | $\text{Clean}_{448}$ | $\Delta_{\text{clean}}$ | $\text{Deg}_{224}$ | $\text{Deg}_{448}$ | $\Delta_{\text{deg}}$ | **DiD $\Delta$ (pp)** | **95% CI (pp)** | **$z$-score** | **$p_{\text{holm}}$** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B3** | Clean | 50,000 | 78.29% | 82.71% | **+4.42** | — | — | — | — | — | — | — |
| | Gaussian Noise (s1)$^\dagger$ | 15,000 | 78.29% | 82.71% | +4.42 | 76.05% | 78.95% | +2.90 | **-1.64** | $\pm 0.56$ | -5.72 | $< 10^{-7}$ |
| | Gaussian Noise (s3) | 50,000 | 78.29% | 82.71% | +4.42 | 71.37% | 53.87% | -17.50 | **-21.92** | $\pm 0.48$ | -90.30 | $< 10^{-15}$ |
| | Gaussian Noise (s5)$^\dagger$ | 15,000 | 78.29% | 82.71% | +4.42 | 59.04% | 3.72% | -55.32 | **-59.85** | $\pm 0.90$ | -130.6 | $< 10^{-15}$ |
| | Defocus Blur (s3)$^\ddagger$ | 10,000 | 78.29% | 82.71% | +4.42 | 67.28% | 69.94% | +2.66 | **-1.71** | $\pm 0.87$ | -3.85 | $1.2 \times 10^{-3}$ |
| | JPEG Compress (s3) | 50,000 | 78.29% | 82.71% | +4.42 | 74.32% | 78.13% | +3.82 | **-0.60** | $\pm 0.33$ | -3.60 | $3.1 \times 10^{-4}$ |
| | Contrast (s3) | 50,000 | 78.29% | 82.71% | +4.42 | 75.86% | 81.44% | +5.58 | **+1.16** | $\pm 0.27$ | +8.59 | $< 10^{-15}$ |
| **FlexiViT-B (F-p)** | Clean | 50,000 | 83.55% | 84.27% | **+0.72** | — | — | — | — | — | — | — |
| | Gaussian Noise (s1)$^\dagger$ | 15,000 | 83.55% | 84.27% | +0.72 | 82.87% | 81.72% | -1.15 | **-1.91** | $\pm 0.45$ | -8.29 | $< 10^{-15}$ |
| | Gaussian Noise (s3) | 50,000 | 83.55% | 84.27% | +0.72 | 80.31% | 73.82% | -6.49 | **-7.21** | $\pm 0.34$ | -42.04 | $< 10^{-15}$ |
| | Gaussian Noise (s5)$^\dagger$ | 15,000 | 83.55% | 84.27% | +0.72 | 74.53% | 31.57% | -42.96 | **-43.72** | $\pm 0.89$ | -96.38 | $< 10^{-15}$ |
| | Defocus Blur (s3)$^\ddagger$ | 10,000 | 83.55% | 84.27% | +0.72 | 76.25% | 76.40% | +0.15 | **-0.24** | $\pm 0.60$ | -0.78 | $0.43$ (n.s.) |
| | JPEG Compress (s3) | 50,000 | 83.55% | 84.27% | +0.72 | 80.46% | 78.72% | -1.73 | **-2.45** | $\pm 0.27$ | -18.01 | $< 10^{-15}$ |
| | Contrast (s3) | 50,000 | 83.55% | 84.27% | +0.72 | 83.20% | 84.62% | +1.42 | **+0.70** | $\pm 0.22$ | +6.20 | $1.1 \times 10^{-9}$ |
| **DeiT-B/16** | Clean | 50,000 | 81.54% | 79.94% | **-1.60** | — | — | — | — | — | — | — |
| | Gaussian Noise (s1)$^\dagger$ | 15,000 | 81.54% | 79.94% | -1.60 | 80.05% | 77.79% | -2.26 | **-0.77** | $\pm 0.54$ | -2.78 | $5.4 \times 10^{-3}$ |
| | Gaussian Noise (s3) | 50,000 | 81.54% | 79.94% | -1.60 | 78.02% | 72.33% | -5.69 | **-4.08** | $\pm 0.36$ | -22.25 | $< 10^{-15}$ |
| | Gaussian Noise (s5)$^\dagger$ | 15,000 | 81.54% | 79.94% | -1.60 | 72.85% | 51.65% | -21.20 | **-19.71** | $\pm 0.85$ | -45.54 | $< 10^{-15}$ |
| | Defocus Blur (s3)$^\ddagger$ | 10,000 | 81.54% | 79.94% | -1.60 | 72.96% | 65.63% | -7.33 | **-6.04** | $\pm 0.84$ | -14.10 | $< 10^{-15}$ |
| | JPEG Compress (s3) | 50,000 | 81.54% | 79.94% | -1.60 | 77.38% | 70.37% | -7.01 | **-5.41** | $\pm 0.36$ | -29.74 | $< 10^{-15}$ |
| | Contrast (s3) | 50,000 | 81.54% | 79.94% | -1.60 | 79.91% | 77.04% | -2.88 | **-1.27** | $\pm 0.29$ | -8.64 | $< 10^{-15}$ |

$^\dagger$: Evaluated on $N=15{,}000$ class-balanced images (PILOT + SUB10K).  
$^\ddagger$: Evaluated on $N=10{,}000$ images with verified ImageNet-C optical disk defocus kernel (K14). Headline k0–k7 runs used a Gaussian-blur fallback (86–95% agreement; reported in Section 10).

---

## 3. The Exact Spectral/Grid Decomposition (Table 2)

Using four standardized renderings on $N=2{,}000$ paired class-balanced images:
- **$\mathrm{A} = \tilde{x}$**: Original corrupted $448 \times 448$ frame.
- **$\mathrm{C} = S_{224}\tilde{x}$**: Antialiased downsampled $224 \times 224$ frame.
- **$\mathrm{D} = U(S_{224}\tilde{x})$**: $224 \text{ px}$ image upsampled back to $448 \text{ px}$ with $2\times$ bilinear upsampling.
- **$\mathrm{B} = h * \tilde{x}$**: $448 \text{ px}$ image prefiltered with 3-tap binomial kernel $[1/4, 1/2, 1/4]$.

Because D carries **strictly the information of C on the grid of A**, the per-image identity:
$$\underbrace{\Delta(\mathrm{A}, \mathrm{C})}_{\text{Total DiD } \Delta} = \underbrace{\Delta(\mathrm{A}, \mathrm{D})}_{\text{Spectral Term } \Delta_{\mathrm{spec}}} + \underbrace{\Delta(\mathrm{D}, \mathrm{C})}_{\text{Grid Term } \Delta_{\mathrm{grid}}}$$
holds exactly with zero modeling assumptions (`analysis/out/decomposition_summary.csv`):

| Model | Corruption | Total DiD $\Delta$ | Spectral Term $\Delta_{\mathrm{spec}}$ (A $-$ D) | Grid Term $\Delta_{\mathrm{grid}}$ (D $-$ C) | Spectral Share $\varphi = \Delta_{\mathrm{spec}} / \Delta$ | Primary Mechanism |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B3** | Noise s3 | $-21.80 \pm 2.41$ | **$-20.35 \pm 2.27$** | $-1.45 \pm 1.84$ | **93%** | Spectral pass-through |
| | Noise s5 | $-58.70 \pm 2.49$ | **$-52.30 \pm 2.43$** | $-6.40 \pm 2.21$ | **89%** | Spectral pass-through |
| | Defocus s3 | $-2.70 \pm 2.12$ | $+2.15 \pm 1.49$ | $-4.85 \pm 2.10$ | — | Grid mismatch (HF helps) |
| **FlexiViT-B (F-p)** | Noise s3 | $-7.50 \pm 1.72$ | **$-5.40 \pm 1.57$** | $-2.10 \pm 1.43$ | **72%** | Mixed |
| | Noise s5 | $-42.70 \pm 2.41$ | **$-33.65 \pm 2.33$** | $-9.05 \pm 1.82$ | **79%** | Mixed |
| | Defocus s3 | $-0.70 \pm 1.39$ | $+1.35 \pm 1.33$ | $-2.05 \pm 1.41$ | — | Grid mismatch (HF helps) |
| **FlexiViT-B (F-t, 256 tok)**| Noise s3 | $-1.50 \pm 1.22$ | $-1.30 \pm 1.35$ | **$-0.20 \pm 0.96$** | 87% | Grid term eliminated |
| | Noise s5 | $-7.05 \pm 1.65$ | $-7.60 \pm 1.80$ | **$+0.55 \pm 1.16$** | 108% | **Grid term zero ($+0.55$)** |
| | Defocus s3 | $+0.50 \pm 1.02$ | $+0.70 \pm 1.35$ | **$-0.20 \pm 0.98$** | — | **Grid term zero ($-0.20$)** |
| **DeiT-B/16** | Noise s3 | $-3.75 \pm 1.88$ | **$+3.75 \pm 1.94$** | **$-7.50 \pm 1.98$** | — | **Grid mismatch (HF helps!)** |
| | Noise s5 | $-20.50 \pm 2.33$ | $-2.65 \pm 2.20$ | **$-17.85 \pm 2.39$** | **13%** | **Grid mismatch (87% grid)** |
| | Defocus s3 | $-6.75 \pm 1.96$ | $+1.35 \pm 1.59$ | **$-8.10 \pm 2.08$** | — | **Grid mismatch (100% grid)** |

---

## 4. Low-Pass Filtering Replications at 448 px (Table 3)

Applying low-pass filtering directly at 448 px isolates whether removing high frequencies restores performance:

### 3-Tap Binomial Filter $[1/4, 1/2, 1/4]$ (K14, $N=10{,}000$; K16, $N=5{,}000$)

| Model ($N$) | Total DiD $\Delta$ (A vs C) | Residual DiD $\Delta(\mathrm{B}, \mathrm{C})$ | Share Removed | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **EfficientNet-B3** (10k) | $-58.95$ [$-22.18$] | $-8.89$ [$-1.57$] | **85% [93%]** | Filter removes almost entire penalty |
| **FlexiViT-B F-p** (10k) | $-43.22$ [$-7.26$] | $-14.37$ [$-4.51$] | **67% [38%]** | Moderate recovery |
| **DeiT-B/16** (10k) | $-19.54$ [$-4.90$] | $-15.24$ [$-6.44$] | **22% [$-31\%$]** | **Filter fails; makes s3 worse!** |
| **ConvNeXt-B** (5k) | $-35.84$ [$-2.94$] | $-10.22$ [$-3.18$] | **71% [$-8\%$]** | Filter removes 71% of s5 penalty |
| **ResNet-50** (5k) | $-40.40$ [$-20.90$] | $-26.98$ [$-10.14$] | **33% [51%]** | Partial recovery |

*Bracketed numbers indicate Severity 3 values; unbracketed indicate Severity 5.*

### Gaussian Prefiltering Sweep ($\sigma = 0.866\text{ px}$, K10, $N=5{,}000$)
- **EfficientNet-B3**: Prefiltering at 448 px lifts accuracy by **$+51.3 \pm 1.4\text{ pp}$** under noise s5 (from $3.64\%$ to $54.90\%$), bringing it within $4.6\text{ pp}$ of the 224 px baseline.
- **DeiT-B/16**: Prefiltering lifts accuracy by only **$+4.2 \pm 1.1\text{ pp}$** at s5, leaving it $17.2\text{ pp}$ below 224 px baseline. At s3, prefiltering **lowers** DeiT's accuracy ($-2.5 \pm 0.9\text{ pp}$). Across a parameter sweep ($\sigma \in [0.5, 2.5]\text{ px}$), no Gaussian prefilter brings DeiT within 16 pp of its 224 px accuracy.

---

## 5. Frequency Localization: Annular Fourier Bands (K9, N=2,000 per arm)

Noise was partitioned into orthogonal Fourier annuli with equal pre-clipping RMS ($r_3 = 45.9$, $r_5 = 96.9$ on $[0, 255]$ scale). DiD (448 vs 224 px, pp $\pm$ SE):

| Model | Low Band $[0, 56)$ | Mid Band $[56, 112)$ | High Band $[112, 224]$ | Broadband | Sensitive Band |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B3** | $-5.25\text{ (s3)} \;/\; -10.75\text{ (s5)}$ | $-13.00 \;/\; -15.90$ | **$-23.50 \;/\; -54.95$** | $-21.90 \;/\; -59.20$ | **High Band** (above 224 Nyquist) |
| **DeiT-B/16** | $-13.95 \;/\; -11.05$ | **$-15.95 \;/\; -47.35$** | $-5.35 \;/\; -13.05$ | $-4.05 \;/\; -18.95$ | **Mid Band** (2–4 cycles/patch) |
| **FlexiViT-B (F-p)** | $-16.30 \;/\; -1.20^*$ | **$-41.15 \;/\; -73.60$** | $-6.25 \;/\; -37.00$ | $-7.40 \;/\; -42.45$ | **Mid Band** |
| **FlexiViT-B (F-t)** | $-4.60 \;/\; -0.60^*$ | **$-5.95 \;/\; -53.85$** | $-2.30 \;/\; -6.35$ | $-1.50 \;/\; -6.75$ | **Mid Band** (reduced at s3) |

*\*Floor effect: accuracy is $< 1.5\%$ across all resolutions for low-band s5.*  
The 224 px Nyquist limit corresponds to 112 cycles per 448 px frame. At 448 px, mid-band periods (4–8 px) place noise directly into the passband of the $16 \times 16$ patch embedding (2–4 cycles per patch), where self-attention is maximally vulnerable.

---

## 6. Operator Noise Gain vs Aliasing (K10, N=5,000)

Across 14 resize operator $\times$ resolution cells (`results/raw/k10-resize-ablation/`), we evaluate how interpolation operators interact with corruption:

```
EfficientNet-B3 Operator Ablation Under Gaussian Noise (N=5,000):
-------------------------------------------------------------------------
Operator / Filter at Target Resolution         Noise s3       Noise s5
-------------------------------------------------------------------------
448 px Identity (Unfiltered standard)          53.40%          3.64%
448 px Gaussian Prefilter (σ=0.866 px)         74.54% (+21.1) 54.90% (+51.3) <-- Full Recovery
224 px Bilinear with Antialiasing (gain 0.313) 72.16%         58.79%
224 px Bicubic with Antialiasing  (gain 0.320) 71.94%         58.20%
224 px Area Downsampling          (gain 0.500) 65.54%         38.84%
224 px Bilinear WITHOUT Antialiasing (gain 0.500) 65.58%      38.90%
-------------------------------------------------------------------------
```

- **Identity of Area & Non-Antialiased Bilinear**: At exact $2\times$ downsampling ($448 \rightarrow 224$), both operators compute an identical $2\times 2$ box average. In our empirical shards, they produce **100% identical predictions on DeiT** and **99.9% on EfficientNet**.
- **Noise Gain Explains the Deficit**: The ~20 pp deficit of these operators relative to antialiased bilinear resizing reflects their higher white-noise gain ($0.500$ vs $0.313$), not aliasing.
- **Rank Correlation**: EfficientNet-B3 accuracy follows measured operator noise gain (Spearman $\rho = -0.92, p = 4\times 10^{-6}$), while DeiT does not ($\rho = -0.41, p = 0.14$).

---

## 7. Token Grid Scaling & Structural Controls in ViTs

### FlexiViT Resolution $\times$ Patch Size Grid (K15, $N=3{,}000$)
Switching from patch 16 (784 tokens) to patch 32 (196 tokens) at 448 px:

```
FlexiViT Top-1 Accuracy (%):
--------------------------------------------------------------------------------------
Resolution   Patch Size   Tokens   Clean (s0)   Defocus (s3)   Noise (s3)   Noise (s5)
--------------------------------------------------------------------------------------
224 px       Patch 16     196      83.13%       76.83%         80.47%       74.77%
224 px       Patch 32      49      79.13%       73.17%         75.70%       69.73%
320 px       Patch 16     400      84.03%       77.13%         80.07%       71.63%
320 px       Patch 32     100      83.57%       77.33%         79.53%       73.47%
448 px       Patch 16     784      84.13%       76.57%         73.43%       30.97%  <-- Severe Collapse
448 px       Patch 32     196      83.77%       78.47%         80.10%       70.10%  <-- +39.13 pp Rescue!
--------------------------------------------------------------------------------------
```
Coarsening the token grid rescues **$+39.13 \pm 1.82\text{ pp}$** at 448 px under severe noise, at zero clean cost ($-0.37\text{ pp}$).

### The Grid Term Is Not a Margin Artefact
Standardizing for clean confidence (ten bins) among correctly classified clean images:
- DeiT-B/16 mean max-softmax confidence is $0.806$ at 448 px vs $0.796$ at 224 px.
- Corruption-induced flip rates at 448 px exceed expectations at 224 px by:
  - Gaussian Noise: **$+6.1\text{ pp}$**
  - JPEG Compression: **$+7.7\text{ pp}$**
  - Defocus Blur: **$+8.2\text{ pp}$**
  - Contrast: **$+2.0\text{ pp}$**

The grid penalty is a true structural property of the representations, not an artefact of confidence margins.

---

## 8. Checkpoint Benchmark Verification (G0-A Benchmark)

To establish that observed behaviors reflect genuine architectural properties rather than checkpoint corruption, sub-optimal weights, or preprocessing errors, all models were evaluated on the **complete 50,000-image ImageNet-1K validation set** at native test resolutions using their official published configurations (`results/raw/k13-g0a-and-bn/g0a_v2.json`):

| Model Key | Checkpoint Identifier | Native Resolution | Official Published Top-1 | Measured Top-1 ($N=50{,}000$) | Absolute Diff | Binomial $SE_{50k}$ | 95% Pass Margin ($1.96 \cdot SE$) | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `deit_base` | `deit_base_patch16_224.fb_in1k` | $224 \times 224$ | $81.80\%$ | **$81.98\%$** (40,988 / 50k) | $+0.18$ pp | $0.172\%$ | $0.34$ pp | **PASSED** (95% CI) |
| `deit_base_384` | `deit_base_patch16_384.fb_in1k` | $384 \times 384$ | $82.90\%$ | **$83.11\%$** (41,554 / 50k) | $+0.21$ pp | $0.168\%$ | $0.33$ pp | **PASSED** (95% CI) |
| `efficientnet_b3` | `efficientnet_b3.ra2_in1k` | $320 \times 320$ | $82.25\%$ | **$82.25\%$** (41,126 / 50k) | $+0.00$ pp | $0.171\%$ | $0.33$ pp | **PASSED** (95% CI) |
| `flexivit_base` | `flexivit_base.1200ep_in1k` | $240 \times 240$ | $84.68\%$ | **$84.67\%$** (42,336 / 50k) | $-0.01$ pp | $0.161\%$ | $0.32$ pp | **PASSED** (95% CI) |

*Verdict: 100% of native checkpoints pass within their 95% binomial standard error confidence interval ($\le 0.21\text{ pp}$ deviation across all architectures).*

---

## 9. Audit Grounding & Corrections of Prior Claims

As verified in [`04_claim_verification_audit.md`](04_claim_verification_audit.md):

1. **Filter Specifications Corrected**: K10 evaluated Gaussian prefiltering with **$\sigma = 0.866\text{ px}$** (not 1.2), while K14/K16 evaluated the 3-tap binomial filter $[1/4, 1/2, 1/4]$ (`B_filtered448`).
2. **Noise Gain vs Aliasing**: `area@224` and `bilinear_no_antialias@224` produce **100% identical predictions** because both compute an identical $2\times 2$ box average. The deficit relative to antialiased bilinear resizing tracks white-noise gain ($0.500$ vs $0.313$), not aliasing.
3. **Contrast Is Not Uniformly Positive**: DeiT-B contrast DiD is **$-1.27\text{ pp}$** ($z = -8.64$). EfficientNet's $+1.16\text{ pp}$ vanishes when measured from its native 320 px baseline ($-0.00 \pm 0.11\text{ pp}$).
4. **Condition E (Nominal-Gain-Matched Noise)**: Condition E scales white noise standard deviation to $\sigma = 0.3125\sigma_{\text{inj}}$ at 448 px. Because white noise at 448 px spreads power across a wider frequency band, it injects less noise power into shared frequency bands than 224 px receives. The rigorous information-matched control is **Rendering D** (upsample-back).
5. **BatchNorm Is Not Required for Collapse**: ConvNeXt-Base (LayerNorm CNN) collapses at 448 px under noise s5 (to 38.18%), proving BatchNorm is not required. Test-time BatchNorm recalibration is unreliable and degrades clean accuracy across resolutions.

---

## 10. Kaggle Kernel Registry & Provenance

All heavy GPU evaluations run through isolated, headless Kaggle kernels orchestrated via `kaggle/push_and_wait.py`. Raw Parquet shards and logs are preserved in `results/raw/`:

| Kernel ID | Description | Sample Size | Primary Output Shard / Summary | Provenance / Backend Status |
| :--- | :--- | :---: | :--- | :--- |
| `k0-probe` | Model forward pass & GPU probe | $N=100$ | Environment & CUDA verification | Baseline probe |
| `k1-sanity` | Label mapping & deterministic seeding | $N=1{,}000$ | Deterministic SHA-256 seeding check | Seed validation |
| `k2-pilot` | Exploratory pilot across resolutions | $N=5{,}000$ | Pilot sweep shards | `unknown_git_commit`, fallback backend |
| `k3-headline-a` | Headline DiD: DeiT, EfficientNet clean + noise | $N=50{,}000$ | `results/raw/k3-headline-a/` | `unknown_git_commit`, fallback backend |
| `k4-headline-b` | Headline DiD: FlexiViT, blur, JPEG, contrast | $N=50{,}000$ | `results/raw/k4-headline-b/` | `unknown_git_commit` (Gaussian fallback defocus) |
| `k5-controls` | Dose-response s1/s5; ToMe; BN recal | $N=10{,}000$ | Exploratory control runs | `unknown_git_commit` |
| `k6-mech` | Initial M7 mechanistic scale decomposition | $N=2{,}000$ | Exploratory M7 runs | `unknown_git_commit` |
| `k7-latency` | Standardized GPU latency harness | -- | 15 warmup, 40 timed iterations | Benchmark harness |
| `k8-m7-v2` | Information-matched scale decomposition (D) | $N=2{,}000$ | `shard_m7_v2.parquet` (272k rows) | Git tracked, `imagecorruptions` |
| `k9-freqnoise-v2`| Annular Fourier bandlimited noise evaluation | $N=2{,}000$ | `shard_freqnoise_v2.parquet` (288k rows) | Git tracked, Parseval energy conservation |
| `k10-resize-ablation`| Interpolation operator & Gaussian prefilter | $N=5{,}000$ | `shard_resize_ablation.parquet` (420k rows)| Git tracked, operator suite |
| `k11-sensor-noise`| Realistic Poisson-Gaussian sensor noise | -- | *Documented unrun / scrubbed* | Transparently documented in `PROVENANCE.md` |
| `k12-tome-matched`| FLOP-matched ToMe vs resolution scaling | $N=5{,}000$ | `shard_tome_matched.parquet` (32k rows) | Git tracked, proportional attention ToMe |
| `k13-g0a-and-bn` | Full 50k native G0-A check & calibrated BN | $N=50{,}000$ | `g0a_v2.json`, calibrated BN shards | Git tracked, clean sanity gate |
| `k14-matched-noise-10k`| Nominal-gain-matched 448 & verified defocus | $N=10{,}000$ | `shard_matched_noise_10k.parquet` (390k rows)| Git tracked, verified ImageNet-C disk blur |
| `k15-flexivit-grid`| FlexiViT resolution $\times$ patch size grid | $N=3{,}000$ | `shard_flexivit_grid.parquet` (24 shards) | Git tracked, 196 vs 784 token control |
| `k16-bn-vs-ln` | ResNet-50 (BN) vs ConvNeXt-Base (LN) | $N=5{,}000$ | `shard_bn_vs_ln.parquet` (6 shards) | Git tracked, norm-type ablation |

---

## 11. Repository Architecture

```
knobs-under-corruption/
├── 00_FINAL_REPORT.md             # Executive audit summary, referee evaluation & roadmap
├── 01_experiment_inventory.md     # Full inventory of all 16 execution kernels (k0–k16)
├── 02_experiment_dependency_timeline.md # Execution timeline and dependency graph
├── 03_cross_experiment_results_matrix.md # Cross-kernel results matrix and hidden patterns
├── 04_claim_verification_audit.md # Claim-by-claim verification and correction audit
├── 05_novelty_and_related_work_audit.md # Verified literature citations & novelty analysis
├── 06_ranked_paper_opportunities.md # Evidence-ranked research opportunities & proposed E1–E7
├── paper/                         # Publication manuscript and figures
│   ├── IEEEtran.cls               # Official IEEE conference template
│   ├── main.tex                   # Complete LaTeX manuscript
│   └── figures/                   # Figures 1–4 (vector PDF, PNG, CSV/JSON numbers)
├── analysis/                      # Downstream statistical analysis & table generators
│   ├── ci_tables.py               # Paired analytic and bootstrap confidence intervals
│   ├── common.py                  # Core DiD statistical estimator & loader
│   ├── effective_sigma.py         # Noise gain and excess loss analysis
│   ├── figures.py                 # Publication figure generators
│   ├── flexivit_config.py         # FlexiViT patch size & token mapping
│   ├── k8_defocus_did.py          # Verified ImageNet-C defocus blur DiD analysis
│   ├── k14_matched_noise.py       # Nominal-gain-matched verification (N=10,000)
│   ├── k15_flexivit_grid.py       # FlexiViT resolution × patch size grid analysis
│   ├── k16_bn_vs_ln.py            # BatchNorm vs LayerNorm CNN control analysis
│   ├── m7.py                      # Exact scale vs spectral decomposition (D control)
│   ├── matched_tables.py          # Architecture controls (Tables 2–5)
│   ├── oracle_headroom.py         # Adaptive resolution selection oracle headroom
│   ├── table1_did.py              # Headline DiD generator (Table 1, N=50,000)
│   ├── table_tome_matched.py      # Compute-matched ToMe vs resolution generator
│   ├── token_budget.py            # Comprehensive ViT token budget contrasts
│   ├── transitions.py             # Per-image accuracy transition matrices
│   └── out/                       # Generated publication tables (.csv, .tex)
├── kaggle/                        # Headless Kaggle GPU execution harness
│   ├── kernels/                   # Isolated execution kernels
│   ├── push_and_wait.py           # Automated kernel packaging and execution
│   └── wait_and_download.py       # Standalone artifact downloader
├── results/
│   ├── raw/                       # Immutable raw evaluation Parquet shards & run metadata
│   └── PROVENANCE.md              # Experimental provenance & backend audit documentation
├── splits/                        # Class-balanced ImageNet-1K index partitions
│   ├── CAL-GATE.json              # Calibration partition (1,000 images, 1/class)
│   ├── MECH.json                  # Mechanism evaluation partition (2,000 images, 2/class)
│   ├── PILOT.json                 # Pilot evaluation partition (5,000 images, 5/class)
│   ├── FULL.json                  # Complete ImageNet validation set (50,000 images)
│   └── val_metadata.json          # ImageNet validation metadata & class mappings
├── src/knobs/                     # Core Python package
│   ├── corrupt.py                 # Deterministic corruptions with backend safety guard
│   ├── data.py                    # Standardized acquisition pipeline & dataset loaders
│   ├── models.py                  # Unified model factory (timm & torchvision wrappers)
│   ├── resize.py                  # Antialiased downsampling operators & Gaussian prefilters
│   ├── run_grid.py                # Deterministic grid execution engine
│   ├── stats.py                   # Difference-in-Differences statistical estimators
│   ├── tome.py                    # Faithful Token Merging implementation
│   └── tokens.py                  # ViT token budget calculation utilities
├── tests/                         # 81 passing unit, regression, and integrity tests
└── pyproject.toml                 # Build configuration and dependency specification
```

---

## 12. Quickstart & Complete Reproduction

### Environment Setup
```bash
git clone https://github.com/AnshulSinghhhhhh/vision knobs-under-corruption
cd knobs-under-corruption
pip install -e .[dev]
```

### Running the 81-Test Suite
```bash
pytest tests/ -v
# 81 passed in ~140 seconds
```

### Regenerating All Publication Tables & Figures
```bash
# Table 1: Primary Difference-in-Differences (N=50,000 paired images)
python -m analysis.table1_did

# Table 2: Exact Information-Matched Decomposition (N=2,000)
python -m analysis.m7

# Table 3: 3-Tap Prefilter & 10k Replications (K14, K16)
python -m analysis.k14_matched_noise
python -m analysis.k16_bn_vs_ln

# Frequency Bands (K9) and Operator Noise Gains (K10)
python -m analysis.effective_sigma

# FlexiViT Token Grid (K15) and Token Budget Matrix
python -m analysis.k15_flexivit_grid
python -m analysis.token_budget

# Publication Figures (Figures 1-4)
python -m analysis.figures
```

---

## 13. Citation

```bibtex
@inproceedings{singh2027spectral,
  title={Spectral Pass-Through or Grid Mismatch? Decomposing Why Higher Test Resolution Amplifies Corruption Damage in CNNs and Vision Transformers},
  author={Singh, Anshul},
  booktitle={Proceedings of the IEEE International Conference on Image Processing (ICIP)},
  year={2027}
}
```
