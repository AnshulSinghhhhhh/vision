# Empirical Research Study: Which Efficiency Knob Should You Turn When the Input Is Degraded?

## 1. Executive Summary & Core Scientific Answers

This study empirically disentangles the trade-offs between **spatial resolution scaling** and **capacity pruning (token reduction via ToMe / channel operations)** across three representative Vision ImageNet-1K architectures:
1. **DeiT-B/16** (Fixed-patch Vision Transformer, native 224×224)
2. **EfficientNet-B3** (Convolutional Neural Network, native 320×320)
3. **FlexiViT-B** (Flexible-patch Vision Transformer, native 240×240 with variable patch support)

The investigation evaluated **4,576,000 model forward passes** across the full 50,000 ImageNet validation set and pre-registered stratified subsets under controlled corruptions (Gaussian Noise, Defocus Blur, JPEG Compression, and Contrast Reduction) across resolutions $\{224, 320, 384, 448\}$.

### Key Empirical Findings:
1. **The Anti-Aliasing Low-Pass Filtering Principle**:
   - Under high-frequency degradation (Gaussian Noise), scaling input resolution from 224 to 448 is severely detrimental across all architectures.
   - For **EfficientNet-B3**, clean accuracy scales from $78.29\% \to 82.71\%$ ($+4.42\text{ pp}$), but under Gaussian Noise ($s=3$), accuracy collapses from $71.37\% \to 53.87\%$ ($-17.50\text{ pp}$), yielding a net Difference-in-Differences (DiD) interaction penalty of **$-21.92\text{ pp}$** ($p < 10^{-300}$, 95% CI: $[21.43, 22.39]$).
   - Under severe noise ($s=5$), EfficientNet at 448 completely breaks down to **$3.72\%$** (near random guess), whereas downsampling to 224 retains **$59.04\%$** accuracy.
   - **DeiT-B/16** experiences a noise DiD of **$-4.08\text{ pp}$** ($p < 10^{-280}$), and **FlexiViT-B** experiences **$-7.21\text{ pp}$** ($p < 10^{-300}$).
   - **Physical Mechanism (Proved via M7 Suite)**: Downsampling to 224 applies an anti-aliasing low-pass filter that suppresses zero-mean high-frequency pixel noise. When low-pass anti-aliasing filtering is applied to the 448 input before inference (`B_filtered448`), EfficientNet recovers from **$53.1\% \to 76.2\%$ (+23.1 pp rescue)**, providing causal proof of the mechanism.

2. **Frequency Disentanglement (Blur & JPEG vs. Contrast)**:
   - Vision Transformers suffer more severely under Defocus Blur (DeiT-B DiD: **$-5.56\text{ pp}$**) and JPEG Compression (DiD: **$-5.41\text{ pp}$**) than CNNs (Defocus DiD: $-1.94\text{ pp}$, JPEG DiD: $-0.60\text{ pp}$).
   - In contrast, low-frequency dynamic range degradation (Contrast Loss) benefits from higher resolution (DiD: $+1.16\text{ pp}$ for EfficientNet, $+0.70\text{ pp}$ for FlexiViT), because resolving spatial boundaries requires high-frequency edge localization despite low contrast.

3. **Resolution vs. Token Pruning (ToMe)**:
   - On clean inputs, token merging (ToMe $r=4$) maintains or exceeds full accuracy ($82.26\%$ vs. $81.54\%$).
   - Under Gaussian Noise ($s=3$), ToMe $r=4$ retains $77.84\%$ at 224, but drops to $73.04\%$ at 448. Token pruning retains corruption robustness, whereas resolution expansion degrades it.

4. **Hardware Efficiency Trade-Off (Table 8)**:
   - Scaling DeiT-B from 224 to 448 reduces throughput by **$5.2\times$** ($464.8 \to 88.7\text{ img/s}$) on NVIDIA T4 (FP16).
   - Scaling EfficientNet from 224 to 448 reduces throughput by **$3.6\times$** ($792.1 \to 222.0\text{ img/s}$).
   - Therefore, under high-frequency degradation, operating at 224 is simultaneously **5× faster** and up to **55 pp more accurate**.

---

## 2. Table Summary

All tables are generated in both LaTeX (`report/tables/*.tex`) and CSV (`report/tables/*.csv`):

- **Table 1 (`tab:did_headline`)**: Primary Difference-in-Differences across all 50,000 ImageNet validation images with paired McNemar exact tests and 95% bootstrap confidence intervals.
- **Table 2 (`tab:res_ladder`)**: Accuracy progression across the compact resolution ladder $\{224, 320, 384, 448\}$.
- **Table 3 (`tab:flexivit_disentangle`)**: Disentanglement of spatial resolution vs. token count via FlexiViT-B Arm F-p (variable tokens) vs. Arm F-t (constant 256 tokens).
- **Table 4 (`tab:dose_response`)**: Dose-response scaling trajectories across severity levels $s \in \{1, 3, 5\}$.
- **Table 5 (`tab:m7_control`)**: M7 filter-matched causal information control (A: 448 native, B: 448 low-pass filtered, C: 224 downsampled, D: 224 $\to$ 448 upsampled).
- **Table 6 (`tab:tome_comparison`)**: Token merging (ToMe $r=4, 8$) versus native resolution scaling across clean and corrupted inputs.
- **Table 8 (`tab:hardware_benchmarks`)**: Empirical NVIDIA T4 hardware benchmarks (interactive batch 1 latency, batch 64 throughput in img/s).

## 3. Figures Generated

All figures are compiled in vector PDF (`report/figures/*.pdf`) and 300 DPI publication PNG (`report/figures/*.png`):
- **Figure 1**: Accuracy trajectories across resolutions under clean, noise, and blur.
- **Figure 2**: Forest plot of pre-registered Difference-in-Differences effect sizes with 95% bootstrap CIs.
- **Figure 3**: FlexiViT token vs. resolution disentanglement curves.
- **Figure 4**: Causal M7 anti-aliasing filter rescue (+23.1 pp recovery) under high-frequency noise.
