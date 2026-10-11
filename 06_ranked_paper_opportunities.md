# 06 — Ranked paper opportunities

Ranking criteria (1–5 each): **N**ovelty against verified prior work (`05_*`), **E**vidence strength in the
existing shards, **F**easibility of closing the gaps without large compute, **P**ublication potential at
IEEE venues. Proposals were built from the evidence, not from the repository's title. One weak result was
not split into several papers. Experiments that are not yet run are marked **[PROPOSED]**.

| Rank | Working title | N | E | F | P | Verdict |
|---|---|---|---|---|---|---|
| **1** | *Spectral Pass-Through or Grid Mismatch? Decomposing Why Higher Test Resolution Amplifies Corruption Damage in CNNs and Vision Transformers* | 4 | 4 | 5 | 4 | **Write now** (manuscript in `paper/`) |
| 2 | *Patch Size as a Test-Time Noise-Robustness Knob for Flexible Vision Transformers* | 2 | 3 | 4 | 2 | Fold into #1 as a section; standalone only after new experiments |
| 3 | *Resolution vs. Token Merging at Matched FLOPs Under Corruption* | 1 | 2 | 3 | 1 | Do not pursue (confounded; superseded by Cha et al. 2026) |
| 4 | *Antialiasing Noise Gain Predicts the Resolution Penalty* | 2 | 3 | 5 | 2 | Supporting evidence for #1; prior theory in Kim et al. 2025 |
| 5 | *No Headroom for Adaptive Resolution Under Corruption* | 2 | 3 | 5 | 1 | Negative result; workshop note at most |
| 6 | *BatchNorm Recalibration Rescues High-Resolution Collapse* | 1 | 1 | 2 | 1 | Not supported; the control is broken |

---

## Rank 1 — the recommended paper

**Research question.** When a corruption is fixed at acquisition and the same pretrained classifier is run at a
higher test resolution, *why* does accuracy fall more under corruption than on clean images, and does the
cause depend on the architecture?

**Title.** Spectral Pass-Through or Grid Mismatch? Decomposing Why Higher Test Resolution Amplifies Corruption
Damage in CNNs and Vision Transformers

**Abstract (draft).** See `paper/main.tex`. One sentence: an exact per-image decomposition with an
information-matched upsample-back control shows the 448-vs-224 penalty is ≈90 % spectral pass-through for
EfficientNet-B3 but almost entirely a grid/scale effect for DeiT-B/16, so the correct fix differs by architecture.

**Contributions.**
1. A paired difference-in-differences protocol that fixes the corruption in a 448-px acquisition frame and
   evaluates the same weights at 224–448 px (50k images, 3 architectures, 4 corruptions, 3 severities), robust
   to the pre-registered split and to the log-odds scale.
2. An **exact additive decomposition** Δ = Δ_spec + Δ_grid via an upsample-back control (D). Spectral share at
   s5: EfficientNet 89 %, FlexiViT-p 79 %, DeiT 13 %. FlexiViT at a constant 256-token grid: Δ_grid = +0.55 ± 0.59.
3. Frequency localisation with band-limited noise: CNN penalty in the high band (−54.95 pp), ViT penalty in the
   mid band (DeiT −47.35, FlexiViT-p −73.60 pp at s5).
4. Prescriptive consequence, confirmed by three independent controls (3-tap filter at 10k, Gaussian prefilter
   at 5k, larger patches at 3k): input low-pass filtering repairs the CNN but not DeiT; for ViTs a native token
   grid (or larger test-time patches) is the remedy.

**Supporting experiments.** K2–K5 (headline), K8 (decomposition), K9 (bands), K10 (operators), K14 (10k
replication + verified defocus), K15 (patch grid), K16 (two extra CNNs, B-filter only).

**Key quantitative results.** See `paper/figures/paper_numbers.json`; tables in `paper/main.tex` are
generated from it.

**Closest related work.** Kim et al. 2025 (resolution and noise across checkpoints, antialiasing theory);
Park & Kim 2022 (ViT/CNN frequency bias); FixRes; FlexiViT; Hendrycks & Dietterich 2019.

**Novelty gap.** Prior work shows *that* lower resolution helps under noise across checkpoints. No work found
separates the two mechanisms inside one model, or shows they dissociate by architecture with a prescriptive
consequence.

**Limitations.**
- One checkpoint per family; three architectures in the decomposition.
- D control only at N = 2,000.
- The 448 frame is usually an upsampled image.
- Synthetic corruptions; one noise draw per image; ImageNet only.
- B-filter replication is not a pure information match.
- K16 lacks D.

**Missing experiments [PROPOSED], with cost on a Kaggle T4 at the measured throughput.**

| ID | Experiment | Why | Cost |
|---|---|---|---|
| E1 | D_up448 control at N = 10k for DeiT, EfficientNet, FlexiViT-p/t, ResNet-50, ConvNeXt-B, Swin-T/B, DeiT-III-B (abs PE), one RoPE ViT; noise s3/s5, defocus s3, JPEG s3 | Generalise the dissociation; resolve ResNet-50 | ≈ 4–6 GPU-h (inference only; ~0.5 h/model) |
| E2 | Three noise draws per image (salt 0/1/2) for the decomposition cells | Variance decomposition, honest CIs | ≈ 1 GPU-h |
| E3 | Read JPEG header sizes (CPU) and stratify all existing DiDs by native short side (< 448, 448–512, ≥ 512) | Removes the "upsampled frame" confound at **zero GPU cost** | CPU minutes |
| E4 | K9 band noise with the D control | Link frequency band to grid component | ≈ 0.5 GPU-h |
| E5 | Decomposition at 320 and 384 | Resolution dose-response of each component | ≈ 1 GPU-h |
| E6 | Sensor-grid Poisson–Gaussian noise (finish K11) | Physical realism | ≈ 1 GPU-h |
| E7 | D built with nearest / bicubic / Lanczos upsampling | Rules out "the grid term is just DeiT disliking bilinear-blurred 448 inputs" (DeiT D is 1.95 pp below A on clean images). This is the most likely reviewer objection. | ≈ 0.5 GPU-h |

The pre-registration budget allowed 10 GPU-h, and Kaggle provides ~30 GPU-h per week, so E1–E6 fit inside
one week.

**Acceptance assessment.** Moderate as-is at mid-tier IEEE venues (IEEE ICIP, ICME, IJCNN): estimated
30–40 %, because three architectures and N = 2k for the key decomposition will draw "limited scope" reviews.
With E1 + E3 done: 45–55 %. CVPR/ICCV/WACV main track: below 15 % without more datasets and a training-side
intervention. Main risks are in `00_FINAL_REPORT.md`.

---

## Rank 2 — Test-time patch size as a robustness knob (fold into #1)

- **RQ:** Can a flexible-patch ViT trade tokens for noise robustness at a fixed input resolution without retraining?
- **Evidence:** K15 (3k): p32 vs p16 at 448 → +39.13 ± 0.93 pp (s5), +6.67 ± 0.58 (s3), −0.37 ± 0.35 (clean), +1.90 ± 0.40 (defocus); p32@448 vs p16@224 → −4.67 ± 0.58 (s5), +0.63 ± 0.40 (clean). F-t arm (K2/K5): 448 noise s5 68.00 % vs F-p 31.57 %.
- **Prior work:** Kim et al. 2025 (patch-32 checkpoints more robust); FlexiViT PI-resize, under which a larger PI-resized patch acts as a resampling filter inside the embedding.
- **Gap/limits:** One model, 3k images, two patch sizes. The mechanism is mostly in-embedding downsampling, so as a standalone claim it is close to "downsample first".
- **[PROPOSED]:** full patch-size sweep {16, 20, 24, 28, 32} × 4 resolutions × 6 corruptions; compare with explicit downsampling at equal token count; measure the PI-resize kernel's noise gain analytically.
- **Acceptance standalone:** low (≈ 15–20 % at a mid-tier venue).

## Rank 3 — ToMe vs resolution (not viable)

ToMe was applied only at 448, where noise passes unfiltered, so "res-320 beats ToMe-r64@448 by 22 pp" measures
F2, not token reduction. Cha et al. 2026 already show that flat schedules fail under shift. The idea would need
ToMe at 224/320, late schedules and matched-noise inputs, which amounts to a different study.

## Rank 4 — Noise gain predicts the penalty (supporting)

EfficientNet accuracy across 14 operator×resolution cells follows measured gain (Spearman −0.92, s5);
area ≡ no-antialias at 224. Kim et al. 2025 already prove the gain theorem. Useful as Fig. 4 of #1.

## Rank 5 — Adaptive resolution headroom (negative)

Condition oracle +0.05 / +0.19 pp over the best fixed resolution. Correct, but a negative result with a narrow
scope; workshop note at most.

## Rank 6 — BN recalibration (not supported)

The recalibration procedure lowers clean accuracy by 1.4–5.3 pp at every resolution, its tolerance was relaxed
after the fact, and the two BN runs disagree. Re-running with a standard, validated adaptation (e.g. Schneider
et al. 2020 with N ≥ 10k and a clean-accuracy check) would be required before any claim.
