# 01 — Repository-wide experiment inventory

Repository: `https://github.com/AnshulSinghhhhhh/vision` (package `knobs`), audited at commit `bd36b83`.
All heavy computation ran as headless Kaggle kernels on NVIDIA T4 GPUs (fp16 autocast). Status was
determined from the presence of per-image parquet shards, `run_meta.json`, and the kernel logs, never
from the existence of code alone.

Common pipeline (all kernels): ImageNet-1k validation image → bilinear resize of the short side to 512 →
centre crop 448×448 ("acquisition frame") → corruption applied in that frame with a SHA-256 per-image
seed → antialiased bilinear resize (`F.interpolate(..., antialias=True)`) to R ∈ {224, 320, 384, 448} →
pretrained timm classifier. Corruption parameters follow `imagecorruptions` (Gaussian σ = 0.08/0.12/0.18/0.26/0.38
for s1–s5, in [0,1] units).

Models: DeiT-B/16 `deit_base_patch16_224.fb_in1k` (dynamic position-embedding interpolation);
EfficientNet-B3 `efficientnet_b3.ra2_in1k` (native test 320 px); FlexiViT-B `flexivit_base.1200ep_in1k`
in two arms — **F-p** (patch 16, tokens grow 196→784) and **F-t** (16×16 grid, patch = R/16);
DeiT-B/16-384 `deit_base_patch16_384.fb_in1k`; ResNet-50 `resnet50.a1_in1k`; ConvNeXt-B
`convnext_base.fb_in22k_ft_in1k`.

Splits (`splits/`, seed 42, class-stratified, disjoint): PILOT 5k · CAL-GATE 1k · CONFIRM_POOL 44k
(= PRIMARY_CONFIRM 34k + SUB10K 10k ⊃ MECH 2k) · FULL 50k. Later kernels (K8–K16) draw their own
class-balanced samples of FULL (`balanced_subset`, seed 0) or use MECH; every one of their images also
appears in the 50k headline shards, which is what makes the cross-kernel agreement check possible.

| ID | Purpose | Data / N (images) | Models · arms | Conditions × resolutions | Status | Provenance | Outputs (rows) | Audit notes |
|---|---|---|---|---|---|---|---|---|
| **K-prep** | Index val set, build splits, map boxes | 50,000 | – | – | Ran (CPU) | no commit | `splits/*.json`, `val_metadata.json` | Native image sizes **not** recorded (plan required `src_short_side`). |
| **K0-probe** | GPU probe, G0-A native check, throughput | first ~100 imgs of class-ordered list | 4 | native transform | Ran | `unknown_git_commit` | `g0_a_verification.json`, `throughput_benchmark.json` | G0-A **failed** (+5 to +6.6 pp) because the slice was class-biased; superseded by K13 50k check. |
| **K1-sanity** | fp32 vs fp16; ToMe r=0 identity | 500 | DeiT, EffNet | clean | Ran | unknown | `precision_agreement.json` (0.0 pp diff), `tome_identity.json` | OK. |
| **K2-pilot** | Pilot grid + pre-registered gates | PILOT 5,000 | DeiT, EffNet, FlexiViT F-p/F-t, DeiT-384, DeiT+ToMe r4/r8 | clean; noise/defocus s1,s3,s5; JPEG, contrast s3 × 224–448 | Ran | unknown, **fallback backend** | 945,000 | Gate G1 failed for EffNet/noise (power 0.48), study proceeded. ToMe = old implementation. Defocus = Gaussian-blur fallback. |
| **K3-headline-a** | Headline: clean, noise s3, defocus s3 | 45,000 (FULL − PILOT) | DeiT, EffNet, FlexiViT F-p | × 224/320/384/448 | Ran | unknown, fallback | 1,620,000 | Merged with K2 to give N = 50,000 per cell. |
| **K4-headline-b** | Headline: JPEG s3, contrast s3 | 45,000 | same | × 4 res | Ran | unknown, fallback | 1,080,000 | JPEG/contrast fallback formulas equal ImageNet-C. |
| **K5-controls** | Dose-response s1/s5; ToMe; BN recal; DeiT-384; freq-noise | SUB10K 10,000 | DeiT, EffNet, FlexiViT F-p/F-t, ToMe r4/r8, EffNet-ebn, DeiT-384 | noise/defocus s1,s5 × res | **Partial**: dose shards saved; **crashed** at freq-noise stage (`TypeError`) | no run_meta | 920,000 | `ebn` arm broken (56 % at noise s1). |
| **K6-mech** | M1–M4 attention probes (DeiT); M7-v1 info control | MECH 2,000 / 1,000 | DeiT (+EffNet for M7) | clean, noise s3 × 224/448; A/B/C/D | Ran | unknown | 8,000 + 12,000 | M1–M4 **never reported**; no resolution×noise interaction in attention entropy. M7-v1 superseded by K8. |
| **K7-latency** | T4 latency/throughput | – | DeiT, EffNet, FlexiViT, ToMe | batch 1 and 64 × 4 res | Ran | unknown | `Table8_hardware_efficiency.csv` | 15 warm-up / 40 timed; EffNet batch-1 is launch-bound. |
| **K8-m7-v2** | Information-matched decomposition | 2,000 | DeiT, EffNet, FlexiViT F-p and F-t | clean, defocus s3, noise s3/s5 × {A_orig448, B 3-tap, B Gaussian σ ∈ {0.5, 0.866, 1.5, 2.5}, C_down224, **D_up448**, E_noise_matched} | Ran | `c1a8cb8-dirty`, imagecorruptions | 272,000 | **Key experiment for the recommended paper** (D condition). Records Immerkaer residual σ. |
| **K9-freqnoise-v2** | FFT annular band-limited noise | 2,000 | DeiT, EffNet, FlexiViT F-p and F-t | bands low [0,56), mid [56,112), high [112,224], broadband × s3/s5 (RMS 45.9/96.9) × 4 res | Ran | `2668a8b-dirty`, **fallback** backend recorded | 288,000 | Pre-clip RMS matched exactly; post-clip RMS equal across bands. |
| **K10-resize-ablation** | Resize-operator ablation + 448 prefilter | 5,000 | DeiT, EffNet | clean, noise s3/s5 × {bilinear-AA, bilinear-noAA, bicubic-AA, area} @ 224/320/384 + {Gaussian σ = 0.866, identity} @ 448 | Ran | `c1a8cb8-dirty` | 420,000 | Records measured white-noise gain per operator. |
| **K11-sensor-noise** | Poisson–Gaussian sensor noise | – | – | – | **Never run** | – | none | Correctly scrubbed from claims. |
| **K12-tome-matched** | ToMe vs resolution at matched FLOPs | 5,000 | DeiT | clean, defocus s3, noise s3/s5 × {224,320,384,448, ToMe r32@448, r64@448} | Ran | `c1a8cb8-dirty` | 120,000 | ToMe only at 448. GFLOPs and throughput logged. |
| **K13-g0a-and-bn** | 50k G0-A; EffNet BN recalibration | 50,000 (G0-A); cal 1,000 / eval 2,000 (BN) | 4 (G0-A); EffNet (BN) | 6 conditions × 4 res | Ran (final + earlier "pilot-5k-backup" run) | `a75e5df-dirty` | 48,000 ×2 | G0-A passes. BN control **unreliable** (lowers clean accuracy at every resolution; tolerance relaxed 1→7 pp post hoc). |
| **K14-matched-noise-10k** | Scale-up of A/B/C/E + verified defocus | 10,000 | DeiT, EffNet, FlexiViT F-p | clean; noise s3/s5 (A,B 3-tap,C,E); defocus s3 (A,C) | Ran in two passes | `52606f6-dirty` | 390,000 | Committed `run.py` now lists only FlexiViT (edited between passes). |
| **K15-flexivit-grid** | Resolution × patch-size grid | 3,000 | FlexiViT | {224,320,448} × patch {16,32} × clean, defocus s3, noise s3/s5 | Ran | `52606f6-dirty` | 72,000 | Patch 24 skipped (divisibility). |
| **K16-bn-vs-ln** | BN CNN vs LN CNN | 5,000 | ResNet-50, ConvNeXt-B | clean; noise s3/s5 × A, B (3-tap), C, E | Ran | `52606f6-dirty` | 110,000 | No D condition; pretraining confound. |

## Analysis layer (`analysis/*.py`, CPU only)

| Script | Produces | Audit status |
|---|---|---|
| `common.py` | merged loader with de-duplication | Fixed on 9 Oct after the old key silently dropped 677k distinct rows. Current key omits `patch_size`, and both K13 runs load together (the earlier run sorts first); K15/K13 scripts read files directly, so published tables are unaffected. |
| `table1_did.py` | Table 1 | Reproduced exactly. |
| `matched_tables.py` | Tables 2–5 | Table 5 (K5 ebn) built on a broken arm. |
| `k8_defocus_did.py`, `k14_*`, `k15_*`, `k16_*`, `table_tome_matched.py`, `token_budget.py`, `oracle_headroom.py`, `transitions.py` | per-kernel tables | Numbers reproduce; interpretations audited in `04_*`. |
| `ci_tables.py` | CI tables | K9 table pools FlexiViT F-p with F-t. |
| `effective_sigma.py` | σ_eff "excess loss" | Interpolation model; not used by this audit. |
| `figures.py` | Fig. 1–4 | Not re-rendered; this audit generates its own figures. |
| `src/knobs/selector.py` | spectral gate selector | Marked deprecated; never evaluated. |

## Independent audit scripts (this folder, `scripts/`)

`load_raw.py` (shared loader) · `a01_integrity.py` · `a02_crosskernel.py` · `a03_k8_dissociation.py` ·
`a04_k9_bands.py` · `a05_k10_operators.py` · `a06_headline_stats.py` · `a07_controls.py` · `a08_margin.py` ·
`a09_confirm_pool.py` · `a10_paper_assets.py` (all manuscript numbers, tables, figures) · `a11_results_matrix.py`.
Run each as `python -I scripts/<name>.py <path-to-cloned-repo> <out-dir>`.
