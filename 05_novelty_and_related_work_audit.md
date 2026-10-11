# 05 — Novelty and related-work audit

Searches were run on 11 Oct 2026 (web search over arXiv, CVF open access, PMLR, OpenReview mirrors, IEEE
listings). **Verified-online** means the bibliographic record and the specific content cited were read this
session. **Standard reference** means a well-known paper whose bibliographic details are given from
established knowledge; re-check page numbers and DOIs against IEEE Xplore / CVF before submission.
No citation in this audit was invented. Where nothing was found, the result is stated as "no prior work
found", which is weaker than "no prior work exists".

## 1. The two citations the repository depends on

| Repo citation | Status | What it actually says | Consequence |
|---|---|---|---|
| Kim, Kawano, Iwasawa, Matsuo, *Unlocking Noise-Resistant Vision: Key Architectural Secrets for Robust Models*, arXiv:2509.20939, 25 Sep 2025 | **Verified-online** ([abs](https://arxiv.org/abs/2509.20939), [html](https://arxiv.org/html/2509.20939)). The README's "ICML 2026" venue is **not confirmed** by the arXiv page. | Across 1,174 timm models, **smaller input resolution, larger stem kernels / patch sizes, average pooling and supervised (non-CLIP) ViTs** predict Gaussian-noise robustness. A theorem shows anti-aliased downsampling cuts noise energy roughly with the square of the downsampling factor. Resolution is compared **across separately trained checkpoints** (e.g. ViT-S/16 224 vs 384). No same-weights multi-resolution sweep, no noise injected before resizing, no band-limited noise, no prefilter or upsample-back control, and no ViT-vs-CNN mechanism split. | The repo's core "downsampling denoises → lower resolution is more noise-robust" claim and "bigger patches help" are **prior art**. Novelty has to come from what Kim et al. did not do. |
| Cha, Yoon, Lee, *Later Is Better: Token Reduction for ViTs Under Distribution Shift*, arXiv:2610.07758, 6 Oct 2026 | **Verified-online** ([abs](https://arxiv.org/abs/2610.07758)) | Flat token-reduction schedules lose accuracy under ImageNet-C; a late-concentrated schedule recovers 83–99 % of the gap. | The repo's ToMe-vs-resolution comparison (flat schedule, ToMe only at 448) is not a contribution and should be dropped. |

## 2. Novelty assessment per finding

| Finding (from `03_*`) | Closest prior work | What prior work shows | What is new here | Novelty |
|---|---|---|---|---|
| F1. Same weights, higher test resolution ⇒ large negative corruption interaction under noise (EffNet −21.9 pp DiD) | Kim et al. 2025; Hendrycks & Dietterich 2019 | Lower-resolution *checkpoints* are more noise-robust; ImageNet-C defines corruptions at fixed 224 px | Within-model, paired 50k-image measurement with corruption fixed in the acquisition frame | Low on its own (an expected consequence of Kim et al.'s theorem) |
| F2. **Exact decomposition** of the interaction into spectral pass-through (A − D) and grid/scale (D − C) using an information-matched upsample-back control; the components **dissociate by architecture** (EffNet ≈ 90 % spectral; DeiT's grid term is ≥ 87 % of its total and exceeds it at s3, where the spectral term is positive; FlexiViT at constant 256 tokens: no grid term) | FixRes (Touvron et al. 2019: train/test apparent-object-size mismatch); ResFormer, ViTAR, NaViT (ViT resolution generalisation on clean data); Kim et al. 2025 | Resolution mismatch hurts clean accuracy; antialiasing reduces noise | **No prior work found** that separates "more noise passes through" from "same information, finer grid" under corruption, or shows the split depends on architecture | **Moderate–high** (the main contribution) |
| F3. Band-limited noise: ViTs' resolution penalty is concentrated in the **mid band** (periods 4–8 px at 448), the CNN's in the high band | Park & Kim 2022 (ViTs vulnerable to low-frequency, ResNets to high-frequency noise); Yin et al. 2019 (Fourier heat maps); Shao et al. 2022; Kim et al. WACV 2024 | Architecture-specific frequency sensitivity at a fixed resolution | How the **test resolution moves a fixed physical noise band into or out of each architecture's sensitive band**, measured with paired DiDs | Moderate (extends known frequency biases with a resolution axis) |
| F4. Across resize operators, the measured white-noise gain predicts the CNN's accuracy; "aliasing" is not the active factor | Kim et al. 2025 (theory); Zhang 2019 (anti-aliased CNNs); Vasconcelos et al. 2021 (aliasing and generalisation) | Antialiasing helps shift-invariance and robustness; the noise-gain theorem | Empirical operator-level check, and the contrast that the ViT does *not* follow noise gain | Low–moderate (supporting evidence) |
| F5. Larger FlexiViT patches at 448 recover +39 pp under s5 noise | Beyer et al. 2023 (FlexiViT, PI-resize); Kim et al. 2025 (patch-32 checkpoints more noise-robust) | Patch size as compute knob; larger-patch *checkpoints* more robust | Test-time patch switching in one model as a robustness knob; quantified gap to the 224 baseline (−4.67 pp) | Low as a standalone paper; good supporting section |
| F6. ToMe at 448 vs resolution at matched FLOPs | Bolya et al. 2023; Cha et al. 2026 | Token-reduction behaviour under shift is schedule-dependent | Confounded with F2 (ToMe only at 448) | None |
| F7. BN statistics do not explain the collapse | Schneider et al. 2020 | BN adaptation improves corruption robustness | Repo's BN control is unreliable | Not supported |
| F8. Adaptive per-condition resolution choice has ≤ 0.2 pp headroom | Dynamic-resolution networks (e.g. Zhu et al., NeurIPS 2021, *Dynamic Resolution Network*) | Input-adaptive resolution saves compute on clean data | Negative result under corruption | Low (workshop note at most) |
| F9. Confidence-standardised flips: DeiT flips more at 448 at equal clean confidence | – | – | Supports F2 (grid effect is not a margin artefact) | Supporting |

## 3. Verified and standard references used by the manuscript

| # | Reference | Status |
|---|---|---|
| 1 | D. Hendrycks, T. Dietterich, "Benchmarking neural network robustness to common corruptions and perturbations," ICLR 2019 | Standard; search hit confirms ([arXiv 1903.12261](https://www.alphaxiv.org/overview/1903.12261v1)) |
| 2 | B. J. Kim, M. Kawano, Y. Iwasawa, Y. Matsuo, "Unlocking noise-resistant vision: Key architectural secrets for robust models," arXiv:2509.20939, 2025 | Verified-online |
| 3 | N. Park, S. Kim, "How do vision transformers work?," ICLR 2022 | Verified-online ([arXiv 2202.06709](https://arxiv.org/pdf/2202.06709)) |
| 4 | D. Yin, R. G. Lopes, J. Shlens, E. D. Cubuk, J. Gilmer, "A Fourier perspective on model robustness in computer vision," NeurIPS 2019 | Standard |
| 5 | R. Shao, Z. Shi, J. Yi, P.-Y. Chen, C.-J. Hsieh, "On the adversarial robustness of vision transformers," TMLR 2022 (arXiv:2103.15670) | Search hit confirms existence |
| 6 | G. Kim et al., "Exploring adversarial robustness of vision transformers in the spectral perspective," WACV 2024 | Verified-online ([CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Exploring_Adversarial_Robustness_of_Vision_Transformers_in_the_Spectral_Perspective_WACV_2024_paper.pdf)) |
| 7 | R. Zhang, "Making convolutional networks shift-invariant again," ICML 2019 | Standard |
| 8 | C. Vasconcelos, H. Larochelle, V. Dumoulin, R. Romijnders, N. Le Roux, R. Goroshin, "Impact of aliasing on generalization in deep convolutional networks," ICCV 2021 | Standard |
| 9 | H. Touvron, A. Vedaldi, M. Douze, H. Jégou, "Fixing the train-test resolution discrepancy," NeurIPS 2019 | Verified-online ([arXiv 1906.06423](https://arxiv.org/pdf/1906.06423)) |
| 10 | L. Beyer et al., "FlexiViT: One model for all patch sizes," CVPR 2023 | Verified-online ([CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Beyer_FlexiViT_One_Model_for_All_Patch_Sizes_CVPR_2023_paper.html)) |
| 11 | R. Tian et al., "ResFormer: Scaling ViTs with multi-resolution training," CVPR 2023 | Standard (search hit) |
| 12 | Q. Fan et al., "ViTAR: Vision transformer with any resolution," arXiv:2403.18361, 2024 | Search hit confirms existence |
| 13 | M. Dehghani et al., "Patch n' Pack: NaViT, a vision transformer for any aspect ratio and resolution," NeurIPS 2023 | Standard |
| 14 | D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, J. Hoffman, "Token merging: Your ViT but faster," ICLR 2023 | Standard |
| 15 | H. Cha, H. Yoon, S.-J. Lee, "Later is better: Token reduction for ViTs under distribution shift," arXiv:2610.07758, 2026 | Verified-online |
| 16 | S. Schneider et al., "Improving robustness against common corruptions by covariate shift adaptation," NeurIPS 2020 | Standard |
| 17 | H. Touvron et al., "Training data-efficient image transformers & distillation through attention," ICML 2021 | Standard |
| 18 | M. Tan, Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," ICML 2019 | Standard |
| 19 | Z. Liu et al., "A ConvNet for the 2020s," CVPR 2022 | Standard |
| 20 | R. Wightman, H. Touvron, H. Jégou, "ResNet strikes back: An improved training procedure in timm," arXiv:2110.00476, 2021 | Standard |
| 21 | A. Dosovitskiy et al., "An image is worth 16x16 words," ICLR 2021 | Standard |
| 22 | O. Russakovsky et al., "ImageNet large scale visual recognition challenge," IJCV 2015 | Standard |
| 23 | J. Immerkær, "Fast noise variance estimation," CVIU 64(2), 1996 | Standard |
| 24 | S. Holm, "A simple sequentially rejective multiple test procedure," Scand. J. Statist. 6(2), 1979 | Standard |
| 25 | J. Djolonga et al., "On robustness and transferability of convolutional neural networks," CVPR 2021 | Verified-online ([CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Djolonga_On_Robustness_and_Transferability_of_Convolutional_Neural_Networks_CVPR_2021_paper.pdf)) |
| 26 | S. Dodge, L. Karam, "Understanding how image quality affects deep neural networks," QoMEX 2016 | Standard |
| 27 | R. Wightman, "PyTorch image models (timm)," GitHub, 2019 | Standard |

## 4. Searches that returned nothing directly relevant (documented to bound the novelty claim)

- Same pretrained weights evaluated at several test resolutions under noise injected *before* resizing.
- ImageNet-C evaluated across test resolutions for ViTs with interpolated position embeddings.
- Compound object-scale × corruption interaction in classifiers. Closest: SI-Score (Djolonga et al. 2021) varies scale alone.
- Mid-frequency noise vulnerability specific to ViTs (closest: high-frequency Fourier heat maps for ViT-B; Park & Kim's low- vs high-frequency split).
- Test-time FlexiViT patch-size switching as a robustness knob.

A reviewer may know work that these searches missed. Before submission, run Semantic Scholar / IEEE Xplore
queries for "test resolution" + "corruption robustness" + "vision transformer" restricted to 2024–2026,
and check the CVPR 2026 / ICCV 2025 / NeurIPS 2025 robustness workshops.
