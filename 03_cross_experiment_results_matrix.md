# 03 — Cross-experiment results matrix and hidden patterns

Machine-readable versions: `derived/results_matrix_long.csv` (every kernel × model × arm × condition ×
severity × configuration with N and binomial 95 % CI half-width), plus the per-analysis CSVs in `derived/`
and `paper/figures/`. All values are top-1 accuracy (%) or paired difference-in-differences (pp) recomputed
from the raw shards. "DiD" means (Acc₄₄₈ − Acc₂₂₄)_corrupted − (Acc₄₄₈ − Acc₂₂₄)_clean, computed per image.
Uncertainties are ±1 SE unless marked "95 % CI".

## 1. Headline accuracy surface (K2+K3+K4+K5; 50,000 images for clean/s3, 15,000 for s1/s5)

| Model (arm) | Condition | 224 | 320 | 384 | 448 |
|---|---|---|---|---|---|
| EfficientNet-B3 | clean | 78.29 | 81.86 | 82.57 | 82.71 |
| | noise s1 / s3 / s5 | 76.05 / 71.37 / 59.04 | 79.61 / 74.52 / 58.45 | 80.27 / 74.04 / 52.11 | 78.95 / **53.87** / **3.72** |
| | defocus s3 (fallback blur) | 69.47 | 73.07 | 72.75 | 71.95 |
| | JPEG s3 / contrast s3 | 74.32 / 75.86 | 78.11 / 80.60 | 78.49 / 81.43 | 78.13 / 81.44 |
| DeiT-B/16 | clean | 81.54 | 81.77 | 81.00 | 79.94 |
| | noise s1 / s3 / s5 | 80.05 / 78.02 / 72.85 | 80.21 / 77.47 / 69.33 | 79.13 / 75.39 / 63.89 | 77.79 / 72.33 / 51.65 |
| | defocus s3 / s5 (fallback) | 75.44 / 58.90 | 73.04 / 52.10 | 70.27 / 46.75 | 68.28 / 41.96 |
| | JPEG s3 / contrast s3 | 77.38 / 79.91 | 76.81 / 79.83 | 74.80 / 78.57 | 70.37 / 77.04 |
| FlexiViT-B (F-p, patch 16) | clean | 83.55 | 84.56 | 84.54 | 84.27 |
| | noise s1 / s3 / s5 | 82.87 / 80.31 / 74.53 | 83.61 / 80.36 / 71.61 | 83.13 / 78.71 / 64.63 | 81.72 / 73.82 / **31.57** |
| | JPEG s3 / contrast s3 | 80.46 / 83.20 | 80.90 / 84.38 | 80.43 / 84.56 | 78.72 / 84.62 |
| FlexiViT-B (F-t, 256 tokens; PILOT 5k + SUB10K) | clean | 84.98 | 85.36 | 85.34 | 85.42 |
| | noise s3 / s5 | 81.18 / 74.53 | 81.16 / 73.76 | 80.96 / 72.80 | 80.76 / 68.00 |

## 2. Headline DiD (448 vs 224), N = 50,000, 95 % CI — reproduces `analysis/out/table1.csv`

| | Gaussian noise s3 | Defocus s3 (fallback) | Defocus s3 **verified** (K14, N=10k) | JPEG s3 | Contrast s3 |
|---|---|---|---|---|---|
| EfficientNet-B3 | −21.92 [−22.39, −21.44] | −1.94 [−2.32, −1.56] | −1.71 ± 0.87 | −0.60 [−0.93, −0.27] | +1.16 [+0.90, +1.43] |
| DeiT-B/16 | −4.08 [−4.44, −3.72] | −5.56 [−5.92, −5.19] | −6.04 ± 0.84 | −5.41 [−5.76, −5.05] | −1.27 [−1.56, −0.99] |
| FlexiViT-B (F-p) | −7.21 [−7.54, −6.87] | −1.26 [−1.53, −1.00] | −0.24 ± 0.60 | −2.45 [−2.72, −2.19] | +0.70 [+0.48, +0.92] |

Severity dose-response (noise; N = 15,000 at s1/s5): EfficientNet −1.64 / −21.92 / −59.85; DeiT −0.77 / −4.08 / −19.71;
FlexiViT-p −1.91 / −7.21 / −43.72 pp. Log-odds DiD for EfficientNet: −0.12 / −1.04 / −3.91 (super-linear on both scales).

## 3. Mechanism matrix — what removes the 448 penalty? (Gaussian noise; DiD vs C_down224, pp ± SE)

| Control (kernel, N) | EfficientNet-B3 | FlexiViT-B F-p | FlexiViT-B F-t | DeiT-B/16 | ResNet-50 | ConvNeXt-B |
|---|---|---|---|---|---|---|
| Raw 448 (A), s3 · K8 2k | −21.80 ± 1.23 | −7.50 ± 0.88 | −1.50 ± 0.62 | −3.75 ± 0.96 | – | – |
| Raw 448 (A), s5 · K8 2k | −58.70 ± 1.27 | −42.70 ± 1.23 | −7.05 ± 0.84 | −20.50 ± 1.19 | – | – |
| **Same information on the 448 grid (D − C), s5 · K8** | −6.40 ± 1.13 | −9.05 ± 0.93 | **+0.55 ± 0.59** | **−17.85 ± 1.22** | – | – |
| **High-frequency pass-through (A − D), s5 · K8** | **−52.30 ± 1.24** | −33.65 ± 1.19 | −7.60 ± 0.92 | −2.65 ± 1.12 | – | – |
| Same, s3 · K8: (D − C) / (A − D) | −1.45 / −20.35 | −2.10 / −5.40 | −0.20 / −1.30 | −7.50 / **+3.75** | – | – |
| 3-tap low-pass at 448 (B − C), s5 · K14 10k / K16 5k | −8.89 ± 0.51 | −14.37 ± 0.45 | – | −15.24 ± 0.50 | −26.98 ± 0.83 | −10.22 ± 0.65 |
| Share of penalty removed by B, s5 | 85 % | 67 % | – | 22 % | 33 % | 71 % |
| Share removed by B, s3 | 93 % | 38 % | – | **−31 %** (B worse) | 51 % | −8 % |
| Gaussian prefilter σ=0.866 at 448 vs raw 448, s5 · K10 5k (acc. difference) | +51.26 ± 0.72 | – | – | +4.20 ± 0.58 | – | – |
| …vs 224 bilinear-AA, s5 | −4.62 ± 0.62 | – | – | **−17.20 ± 0.63** | – | – |
| Best Gaussian σ at 448 (K8 sweep), s5 | σ 1.5: 54.25 % (C 57.60) | σ 1.5: 65.10 % (C 74.50) | σ 0.866: 74.55 % (C 73.75) | σ 0.5: 55.50 % (C 72.00) | – | – |
| Larger patch at 448 (p32 − p16), s5 · K15 3k | – | +39.13 ± 0.93 | – | – | – | – |
| Native-384 DeiT vs 224-ckpt evaluated at 384 (accuracy difference), s5 · K2+K5 15k | – | – | – | +6.35 ± 0.32 (clean +1.20 ± 0.42, 5k) | – | – |
| ToMe r64@448 vs plain 448, s5 · K12 5k | – | – | – | −4.74 ± 0.49 | – | – |

Reading: for EfficientNet almost the whole penalty disappears once the high frequencies are removed,
whatever removes them. For DeiT the penalty persists even when the 448 input carries exactly the 224
information (D), and filtering does not help at s3. FlexiViT at a fixed 256-token grid has no grid
component at all.

## 4. Band-limited noise (K9, N = 2,000 per arm): DiD 448 vs 224 (pp, SE ≈ 0.4–1.3)

| | low [0,56) s3 / s5 | mid [56,112) s3 / s5 | high [112,224] s3 / s5 | broadband s3 / s5 |
|---|---|---|---|---|
| EfficientNet-B3 | −5.25 / −10.75 | −13.00 / −15.90 | **−23.50 / −54.95** | −21.90 / −59.20 |
| FlexiViT-B F-p | −16.30 / −1.20* | **−41.15 / −73.60** | −6.25 / −37.00 | −7.40 / −42.45 |
| FlexiViT-B F-t | −4.60 / −0.60* | −5.95 / **−53.85** | −2.30 / −6.35 | −1.50 / −6.75 |
| DeiT-B/16 | −13.95 / −11.05 | **−15.95 / −47.35** | −5.35 / −13.05 | −4.05 / −18.95 |

\* floor effect: low-band s5 accuracy is < 1.5 % at every resolution for FlexiViT.
Bands are in cycles per 448-px frame; the 224-px Nyquist is 112.

## 5. Other cross-cutting measurements

| Measurement | Result | Source |
|---|---|---|
| Operator noise gain predicts accuracy (14 operator×resolution cells, s5) | EfficientNet Spearman −0.92 (p = 4e-6); DeiT −0.41 (p = 0.14) | `a05_k10_operators.py` |
| area@224 ≡ bilinear-noAA@224 | 100 % identical predictions (DeiT), 99.9 % (EffNet) | same |
| Confidence-standardised excess flip rate at 448 vs 224 (clean-correct images) | DeiT +6.1 (noise) / +8.2 (defocus) / +7.7 (JPEG) / +2.0 (contrast) pp; FlexiViT +9.9 / +4.4 / +4.8 / +0.2 pp | `a08_margin.py` |
| Clean max-softmax confidence 224 → 448 | DeiT 0.796 → 0.806; FlexiViT 0.823 → 0.854; EffNet 0.816 → 0.641 (not comparable across R) | same |
| Contrast DiD from EffNet's native 320 px (320 → 448) | −0.00 ± 0.11 pp | `a06_headline_stats.py` |
| Noise DiD from EffNet's native 320 px | −21.49 ± 0.23 pp (s3) | same |
| DeiT attention entropy (K6, normalised) | clean 0.691 → 0.769, noise 0.754 → 0.834 (224 → 448); interaction +0.002 | K6 shard |
| Condition-level oracle headroom over best fixed resolution | +0.05 pp (DeiT), +0.19 pp (EffNet) | recomputed |
| Throughput (T4, batch 64) | DeiT 465 → 89 img/s; EffNet 792 → 222 img/s (224 → 448) | K7 |

## 6. Hidden patterns (ranked by evidence strength)

| # | Pattern | Evidence | Strength |
|---|---|---|---|
| H1 | **Architecture dissociation.** The 448 penalty is spectral pass-through for EfficientNet-B3 (89–93 % of the total) but grid/scale-driven for DeiT-B (≤ 13 %; at s3 the high frequencies even *help*, +3.75 pp). FlexiViT-p is mixed; FlexiViT-t has no grid component. | K8 exact decomposition (2k), replicated qualitatively by K14 B (10k) and K10 prefilter (5k) | Strong (3 independent kernels, all z > 5 for the key cells) |
| H2 | **ViTs are hit hardest by mid-band noise at 448**, the CNN by high-band noise. README states the opposite for ViTs. | K9 (2k per arm) | Strong |
| H3 | **DeiT's extra corruption sensitivity at 448 is not a confidence-margin artefact** (its clean confidence is slightly *higher* at 448). | K2–K4 confidence column, ~40k images per cell | Moderate (max-softmax is a crude margin) |
| H4 | **Noise gain, not aliasing, governs the CNN.** Across operators the CNN follows the measured gain; the DeiT does not. | K10 (5k) | Strong for EffNet |
| H5 | **"Contrast benefits from resolution" is a sub-native artefact** (EffNet 320 → 448: 0.00) and is false for DeiT (−1.27). | Headline 50k | Strong |
| H6 | **Blur penalty is a grid effect for every model** (defocus: D − C = −8.10 DeiT, −4.85 EffNet, −2.05 FlexiViT-p; high frequencies help, A − D > 0). | K8 verified defocus (2k) | Moderate (one severity, 2k) |
| H7 | **Fixed token grids neutralise the resolution interaction for all non-noise corruptions** (F-t DiD within ±0.4 pp for contrast, JPEG and defocus s1/s3/s5). | K2/K5 F-t (5k) | Moderate |
| H8 | **Super-linear severity scaling** of the interaction (EffNet log-OR −0.12 → −1.04 → −3.91). | Headline + K5 | Strong |
| H9 | **The B (3-tap) filter is not enough for ResNet-50** (removes only 33 % at s5) while it removes 71 % for ConvNeXt-B. Without a D control, ResNet-50's residual cannot be attributed. | K16 (5k) | Weak, ambiguous |
| H10 | **ToMe makes 448 slightly worse under noise** (−4.74 pp at s5) and cannot substitute for input-side filtering. | K12 (5k) | Moderate, limited scope |

## 7. Contradictions found between experiments

- Headline FlexiViT defocus DiD −1.26 [−1.53, −1.00] (fallback blur) vs verified K14 −0.24 [−0.84, +0.36]: the backend changes the conclusion for that cell.
- K13 BN recal "rescues" 448 noise s5 (3.95 → 32.35 %) while lowering clean accuracy everywhere; K5 `ebn` collapses at 224. The two BN runs disagree, and both are procedurally suspect.
- README K9 low-band "scale invariance" (EffNet) vs DeiT's 13.10 → 0.65 % drop.

## 8. Unexplored combinations (zero-cost to identify, need new GPU runs)

| Missing cell | Why it matters | Rough cost (T4) |
|---|---|---|
| D_up448 for ResNet-50, ConvNeXt-B, Swin/hierarchical ViT, DeiT-III, RoPE-ViT at N = 10k | Generalises H1 beyond 3 architectures; resolves H9 | ~0.5–1 GPU-h per model |
| D for JPEG, contrast, defocus at 10k | Is H6/H7 a general grid law for every corruption? | ~1 GPU-h |
| Band noise (K9) + D | Ties H2 to H1 | ~0.5 GPU-h |
| Multiple noise draws per image (salt) | Separates image from noise-draw variance | ~0.3 GPU-h |
| Natively ≥ 512-px images only (stratify by JPEG header size) | Removes the up-sampled-frame confound | CPU only for stratification; ~0 GPU if re-using shards |
| Sensor-grid noise (K11, Poisson–Gaussian before up-sampling) | Physical realism | ~1 GPU-h |
| 320 and 384 decomposition | Resolution dose-response of each component | ~1 GPU-h |
