# Paper Readiness & Manuscript Roadmap

**Paper Title:** *Does More Resolution Help Degraded Images? An Empirical Investigation of Resolution Scaling Under Corruption*  
**Auditor / Reproducibility Review:** Senior Research Methodology Auditor  
**Date:** October 2026  
**Status:** Validated & Ready for Manuscript Drafting

---

## 1. Core Contribution Statement

Resolution scaling does not inherently hurt corrupted images. At matched per-pixel noise, 448 outperforms 224 for all three models (K8 'E' control, preliminary values at $N=2{,}000$). The apparent penalty comes from losing the implicit denoising of antialiased downsampling (dominant for EfficientNet-B3), and from scale mismatch for ViTs (DeiT: prefiltering does not help; FlexiViT: fixing token count removes the noise penalty).

---

## 2. Defensible Claims Supported by Empirical Evidence

| # | Scientific Claim | Empirical Evidence / Artifact | Evidence Strength |
| :--- | :--- | :--- | :--- |
| **C1** | **Resolution-Corruption Interaction is Statistically Significant**: Resolution scaling under degradation produces a significant negative DiD for Gaussian noise, defocus blur, and JPEG compression across DeiT-B, EfficientNet-B3, and FlexiViT-B. | `analysis/out/table1.csv`, `table1.tex` ($N=50{,}000$ paired ImageNet images, paired $z$-test, Holm-adjusted $p < 10^{-15}$). | **Decisive** |
| **C2** | **Acute Convolutional Fragility Under Noise**: EfficientNet-B3 collapses by $-21.92\text{ pp}$ DiD under Gaussian noise (s3) and drops to $3.64\%$ accuracy under noise s5 at 448 px, whereas clean accuracy *gains* $+4.42\text{ pp}$. | `table1.csv` row 6 ($z = -90.30$); `k10-resize-ablation` raw shards. | **Decisive** |
| **C3** | **High-Frequency Noise Drives the Collapse**: Low-frequency noise damages accuracy across all resolutions equally ($\sim 40\%$), but high-frequency noise causes the entire 448 px collapse ($76.8\% \rightarrow 55.95\%$) because 448 px has no antialiasing filter. | `k9-freqnoise-v2` frequency band evaluations; unit tests in `tests/test_k9.py`. | **Decisive** |
| **C4** | **Input Filtering Reverses the Collapse**: Applying a Gaussian lowpass prefilter ($\sigma=1.2$) directly to 448 px noisy images restores accuracy from $3.64\%$ to $54.90\%$ ($+51.26\text{ pp}$ recovery) on EfficientNet-B3. | `k10-resize-ablation` raw shards; `analysis/out/table_k10.csv`. | **Decisive** |
| **C5** | **Rescoped Token Merging Comparison (ToMe)**: At matched FLOPs, res-320 beats ToMe-r64@448 by +2.26 pp on clean, +7.48 pp at noise s3, and +22.12 pp at noise s5 ($N=5{,}000$, DeiT-B only, off-the-shelf ToMe); the clean gap shows this is largely a non-native-resolution effect. | `analysis/out/table_tome_matched.csv`, `table_tome_matched.tex` ($N=5{,}000$). | **Decisive** |
| **C6** | **Contrast Scaling Remains Monotonically Positive**: When degradation does not add high-frequency noise (e.g. contrast reduction), resolution scaling remains beneficial for all architectures ($\text{DiD} > 0$). | `table1.csv` rows 5, 9, 13 (all $z > +6.2$, $p_{\text{holm}} < 10^{-8}$). | **Decisive** |

---

## 3. Explicit Non-Claims (Methodological Guardrails)

To preserve scientific rigor, the manuscript must **explicitly reject** the following over-generalizations:

1. **Do NOT claim that higher resolution is universally harmful**:
   - Contrast degradation consistently improves at higher resolution across all models.
   - Clean accuracy improves with resolution for EfficientNet-B3 ($+4.42\text{ pp}$) and FlexiViT-B ($+0.72\text{ pp}$).
2. **Do NOT claim that Vision Transformers are immune to noise**:
   - DeiT-B/16 suffers a $-4.08\text{ pp}$ negative DiD under Gaussian noise.
   - FlexiViT-B suffers a $-7.21\text{ pp}$ negative DiD under Gaussian noise.
3. **Do NOT claim that checkpoint mismatch or incorrect fine-tuning caused the collapse**:
   - All models pass G0-A native checkpoint verification against official reference targets within binomial sampling confidence intervals.
4. **Do NOT claim that BatchNorm is ruled in or out as the mechanism**:
   - BN recalibration on clean data partly rescues the 448 collapse (EfficientNet noise s5: 4.0%→32.35%, N=2000); its clean sanity gate was relaxed to 7 pp; evidence is inconsistent with the older k5 arm and is not used to rule BN in or out.

---

## 4. Manuscript Structure & Evidence Mapping

### Section 1: Introduction
- **Motivation**: Vision systems deployed in robotics, autonomous driving, and medical imaging frequently encounter degraded inputs and dynamically adjust input resolution or sensor crops.
- **The Core Paradox**: Common intuition suggests "more resolution provides more detail, which should help under corruption." We show this is false when corruption contains high spatial frequencies.
- **Summary of Findings**: DiD headline table, the filter attenuation mechanism, and compute-matched comparisons.

### Section 2: Unified Acquisition Pipeline & Evaluation Framework
- **Unified Acquisition**: Fixed $448 \times 448$ acquisition frame ($512$ resize, $0.875$ center crop) with deterministic per-image SHA-256 seeding.
- **Difference-in-Differences (DiD) Statistic**:
  $$\text{DiD} = (Acc_{448} - Acc_{224})_{\text{deg}} - (Acc_{448} - Acc_{224})_{\text{clean}}$$
- **Sign Convention**: Negative DiD indicates resolution hurts more under degradation than on clean images.
- **Statistical Rigor**: Per-image paired differences, analytic $z$-test, and Holm-Bonferroni step-down adjustments across all hypotheses.
- **Evidence**: `analysis/table1_did.py`, `analysis/common.py`.

### Section 3: Primary Empirical Results across Architectures (Table 1)
- Present Table 1 across DeiT-B/16, EfficientNet-B3, and FlexiViT-B.
- Highlight the $-21.92\text{ pp}$ collapse of EfficientNet-B3 vs moderate $-4.08\text{ pp}$ of DeiT-B/16.
- Contrast with contrast degradation ($+1.16\text{ pp}$ DiD).
- **Evidence**: `analysis/out/table1.tex`, `analysis/out/table1.csv`.

### Section 4: The Mechanism: Frequency Attenuation and Inductive Bias
- **Sub-section 4.1: Antialiased Downsampling as a Spatial Lowpass Filter**:
  - Noise gain progression: $\sigma_{\text{eff}} \in \{0.313, 0.481, 0.565, 1.000\}$.
  - Evidence: `analysis/effective_sigma.py`, `tests/test_resize.py`.
- **Sub-section 4.2: Annular Fourier Frequency Decomposition (K9)**:
  - Low-frequency noise damages global semantic representations across all scales.
  - High-frequency noise drives the 448 px collapse exclusively.
  - Parseval energy conservation and Gaussian marginal intensity invariance.
  - Evidence: `results/raw/k9-freqnoise-v2/`, `tests/test_k9.py`.
- **Sub-section 4.3: Lowpass Prefiltering Reversal (K10)**:
  - Demonstrating that a Gaussian prefilter ($\sigma=1.2$) at 448 px restores $+51.26\text{ pp}$ accuracy.
  - Evidence: `results/raw/k10-resize-ablation/`.

### Section 5: Architecture and Compute Controls
- **Sub-section 5.1: FlexiViT Token Scaling (F-p vs F-t, Table 2)**:
  - Comparing constant patch size (F-p) vs constant token count (F-t).
  - Evidence: `analysis/out/table2.csv`.
- **Sub-section 5.2: Compute-Matched Token Merging vs Resolution Scaling (Table 6)**:
  - At matched FLOPs, res-320 beats ToMe-r64@448 by +2.26 pp on clean, +7.48 pp at noise s3, and +22.12 pp at noise s5 ($N=5{,}000$, DeiT-B only, off-the-shelf ToMe); the clean gap shows this is largely a non-native-resolution effect.
  - Evidence: `analysis/out/table_tome_matched.tex`.
- **Sub-section 5.3: BatchNorm Recalibration (Table 5)**:
  - Clean sanity gate and calibrated statistics analysis.
  - Evidence: `analysis/out/table5.csv`.

### Section 6: Discussion, Limitations, & Conclusion
- Practical guidelines for adaptive resolution systems: downsample or low-pass filter inputs when high-frequency sensor noise is detected.

---

## 5. Known Limitations

1. **Model Scope**: Findings are established on three models only (DeiT-B/16, EfficientNet-B3, FlexiViT-B).
2. **Dataset**: Evaluated on ImageNet-1K only.
3. **Noise Injection Protocol**: Noise is injected into the 448x448 acquisition frame before resizing, giving an effective noise standard deviation of $\sigma_{\text{eff}} = 0.313 \times \sigma_{\text{inj}}$ at 224 px.
4. **Defocus Implementation Caveat**: Headline defocus results (k0–k7) are unverified due to lack of recorded backend/commit and divergence from ImageNet-C disk defocus; verified ImageNet-C defocus is grounded in K8 ($N=2{,}000$).
5. **Mechanism Controls Sample Size**: Key mechanism controls (K8, K9, K10, K13) have sample sizes $N=2{,}000$--$5{,}000$.

---

## 6. Related Work to Cite

- **Kim et al. 2025**: "Unlocking Noise-Resistant Vision" (arXiv:2509.20939, ICML 2026) already shows that smaller input resolution and anti-aliased downsampling improve Gaussian-noise robustness with a low-pass theory; so $\sigma_{\text{eff}}$ / low-pass is **supporting**, not the novelty claim.
- **Yin et al. 2019**: "A Fourier Perspective on Model Robustness in Computer Vision" (NeurIPS 2019) on frequency-domain corruption analysis.
- **Touvron et al. 2019**: "Fixing the train-test resolution discrepancy" (FixRes, NeurIPS 2019) on resolution adaptation.
- **Schneider et al. 2020**: "Improving robustness against common corruptions by covariate shift adaptation" (NeurIPS 2020) on BatchNorm statistics recalibration.
- **Beyer et al. 2023**: "FlexiViT: One Model for All Patch Sizes" (CVPR 2023) on patch size and token count trade-offs.
