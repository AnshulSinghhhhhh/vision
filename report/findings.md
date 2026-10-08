# Empirical Research Study: Which Efficiency Knob Should You Turn When the Input Is Degraded?

## 1. Executive Summary & Core Scientific Answers

This study empirically disentangles the trade-offs between **spatial resolution scaling** and **capacity pruning (token reduction via ToMe / channel operations)** across three representative Vision ImageNet-1K architectures:
1. **DeiT-B/16** (Fixed-patch Vision Transformer, native 224×224)
2. **EfficientNet-B3** (Convolutional Neural Network, native train 288×288, test 320×320)
3. **FlexiViT-B** (Flexible-patch Vision Transformer, native 240×240 with variable patch support)

The investigation evaluated **4,565,000 model forward passes** across the full 50,000 ImageNet validation set and pre-registered stratified subsets under controlled corruptions (Gaussian Noise, Defocus Blur, JPEG Compression, and Contrast Reduction) across resolutions $\{224, 320, 384, 448\}$.

### Central Thesis
> **"Inference-time resolution scaling is degradation-dependent: it can raise clean accuracy while lowering degraded accuracy, and the size and mechanism of this interaction depend on the architecture and on how the resampling step filters the corruption."**

---

### Key Empirical Findings:

1. **The Anti-Aliasing Low-Pass Filtering Principle**:
   - Under high-frequency degradation (Gaussian Noise), scaling input resolution from 224 to 448 is severely detrimental across all architectures.
   - For **EfficientNet-B3**, clean accuracy scales from $78.29\% \to 82.71\%$ ($+4.42\text{ pp}$), but under Gaussian Noise ($s=3$), accuracy collapses from $71.37\% \to 53.87\%$ ($-17.50\text{ pp}$), yielding a net Difference-in-Differences (DiD) interaction penalty of **$-21.92\text{ pp}$** ($z = -90.5$, Holm-adjusted $p < 10^{-300}$, paired analytic 95% CI: $[-22.39, -21.44]$).
   - Under severe noise ($s=5$), EfficientNet at 448 completely breaks down to **$3.72\%$** (near random guess), whereas downsampling to 224 retains **$59.04\%$** accuracy.
   - **DeiT-B/16** experiences a noise DiD of **$-4.08\text{ pp}$** ($z = -22.3$, $p < 10^{-100}$), and **FlexiViT-B (F-p)** experiences **$-7.21\text{ pp}$** ($z = -42.6$, $p < 10^{-300}$).

2. **Effective Noise at the Model Input ($\sigma_{\text{eff}}$)**:
   - Antialiased bilinear interpolation computes an area-weighted continuous triangle convolution whose window width scales with downsampling.
   - White noise standard deviation seen by the model is attenuated to:
     - **224 px**: Gain = **0.313** (variance gain = 0.098)
     - **320 px**: Gain = **0.481** (variance gain = 0.231)
     - **384 px**: Gain = **0.565** (variance gain = 0.320)
     - **448 px**: Gain = **1.000** (identity, no resampling)
   - Plotting accuracy against $\sigma_{\text{eff}}$ reveals that EfficientNet's apparent 384$\to$448 collapse is almost entirely explained by the near-doubling of effective noise variance entering convolutional receptive fields (excess unexplained drop is only $-4.63\text{ pp}$ out of $28.84\text{ pp}$).

3. **Physical Mechanism (Proved via M7 Suite)**:
   - Downsampling to 224 applies an anti-aliasing low-pass filter that suppresses zero-mean high-frequency pixel noise.
   - Configuration B applies a 3-tap low-pass filter $[0.25, 0.5, 0.25]$ without decimation (weaker than true 4-tap $[0.125, 0.375, 0.375, 0.125]$ filter of 448$\to$224 resize).
   - When low-pass filtering is applied to the 448 input before inference, EfficientNet recovers from **$53.1\% \to 76.2\%$ (+23.1 pp rescue)**, providing causal proof of the mechanism.
   - DeiT filtered 448 input reaches 72.1% while 224 downsampled input reaches 79.4%, demonstrating that Vision Transformers exhibit a residual token-grid / resolution effect beyond noise filtering alone.

4. **Frequency Disentanglement & Multiple Testing (Holm Correction)**:
   - Vision Transformers suffer more severely under Defocus Blur (DeiT-B DiD: **$-5.56\text{ pp}$**) and JPEG Compression (DiD: **$-5.41\text{ pp}$**) than CNNs (Defocus DiD: $-1.94\text{ pp}$, JPEG DiD: $-0.60\text{ pp}$).
   - On the held-out split, EfficientNet $\times$ JPEG has $z = -2.3$, $p = 0.022$, which passes Holm-Bonferroni correction across the 12 primary tests.
   - In contrast, low-frequency dynamic range degradation (Contrast Loss) benefits from higher resolution (DiD: $+1.16\text{ pp}$ for EfficientNet, $+0.70\text{ pp}$ for FlexiViT).

5. **Resolution vs. Token Pruning (ToMe & FlexiViT)**:
   - On clean inputs, token merging (ToMe $r=4$) maintains full accuracy ($82.26\%$ vs. $81.54\%$).
   - Under Gaussian Noise ($s=3$), ToMe $r=4$ retains $77.84\%$ at 224, but drops to $73.04\%$ at 448. Token pruning retains corruption robustness, whereas resolution expansion degrades it.
   - When compute is matched (ToMe at 448 px matched in GFLOPs to 384 px and 320 px), resolution reduction remains superior under noise because downsampling filters noise prior to linear projection.
   - FlexiViT: the noise penalty is strongly associated with tokenisation, with secondary confounds from positional embedding interpolation (from native 15×15) and patch embedding resampling.

6. **Hardware Efficiency Trade-Off (GFLOPs & Throughput)**:
   - DeiT-B: 33.5 GFLOPs (224), 68.5 GFLOPs (320), 98.6 GFLOPs (384), 134.3 GFLOPs (448).
   - EfficientNet-B3: 1.8 GFLOPs (224), 3.7 GFLOPs (320), 5.4 GFLOPs (384), 7.3 GFLOPs (448).
   - Scaling DeiT-B from 224 to 448 reduces throughput by **$5.2\times$** ($464.8 \to 88.7\text{ img/s}$) on NVIDIA T4 (FP16).
   - Scaling EfficientNet from 224 to 448 reduces throughput by **$3.6\times$** ($792.1 \to 222.0\text{ img/s}$); at batch size 1, EfficientNet is launch-bound on GPU.

---

### What We Do Not Claim
1. We do *not* claim that higher resolution is universally or generally harmful in computer vision.
2. We do *not* claim that all Convolutional Neural Networks fail spectrally under all forms of corruption.
3. We do *not* claim that all Vision Transformers degrade solely because of token sequence length.
4. We do *not* claim that models explicitly fine-tuned or trained on high-resolution noisy inputs behave identically to standard ImageNet models.
5. We do *not* claim that ImageNet-C corruptions represent the exact physical noise characteristics of all digital camera sensors.
6. We do *not* claim that $224\times224$ is an optimal or sufficient resolution for dense perception tasks like object detection or segmentation.

---

### Code Repository & Artifacts
- **Repository**: [https://github.com/AnshulSinghhhhhh/vision](https://github.com/AnshulSinghhhhhh/vision)
- **Reproduction**:
  - `python -m analysis.table1_did`
  - `python -m analysis.matched_tables`
  - `python -m analysis.effective_sigma`
  - `python -m analysis.transitions`
  - `python -m analysis.m7`
  - `python -m analysis.figures`
