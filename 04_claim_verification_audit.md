# 04 — Claim-by-claim verification audit

Repository audited: `AnshulSinghhhhhh/vision` @ `bd36b83` (11 Oct 2026, 70 commits since 7 Oct 2026).
Method: every number below was recomputed independently from the per-image parquet shards in
`results/raw/**` with the scripts in `scripts/` (no model was run, no shard was modified).
Verdicts: **VERIFIED** (reproduced exactly), **VERIFIED-WITH-CAVEAT**, **MISREPORTED** (the data exist
but the document describes them wrongly), **NOT SUPPORTED** (the data contradict the claim),
**UNRELIABLE** (the experiment itself is compromised).

## A. Integrity checks that passed

| Check | Result | Script |
|---|---|---|
| Duplicate (image, condition, severity, model, arm, resolution) keys across K2–K5 | 0 duplicates in 4,565,000 rows | `a01_integrity.py` |
| PILOT (K2) leakage into K3/K4/K5 | 0 shared images | `a01_integrity.py` |
| Headline cell sizes | 50,000 paired images per clean/s3 cell; 15,000 per s1/s5 cell | `a01_integrity.py` |
| Cross-kernel determinism (same image, same config, different kernel) | 99.69–100 % per-image prediction agreement for clean and Gaussian noise across K8, K10, K12, K14, K15 vs headline (fp16 autocast jitter only) | `a02_crosskernel.py` |
| Table 1 (`analysis/out/table1.csv`) | Reproduced to every printed digit | `a06_headline_stats.py` |
| Table 1 on the pre-registered CONFIRM_POOL (44k, excludes PILOT) | Same signs and magnitudes (e.g. EffNet noise −21.77 ± 0.26 pp vs −21.92 on 50k); all 12 Holm p ≤ 0.0016 | `a09_confirm_pool.py` |
| Table 1 on the log-odds scale (scale-dependence of pp DiDs) | No sign changes; e.g. DeiT contrast log-OR DiD −0.067 [−0.084, −0.050] | `a06_headline_stats.py` |
| G0-A native checkpoint check (50k) | 81.98 / 83.11 / 82.25 / 84.67 % vs refs 81.8 / 82.9 / 82.25 / 84.68 | `results/raw/k13-g0a-and-bn/g0a_v2.json` |
| Condition-level oracle headroom | +0.05 pp DeiT (224 fixed), +0.19 pp EffNet (384 fixed) on clean+s3 pool | recomputed from curves |

## B. Claims in README / docs and their status

| # | Claim (where) | Verdict | Evidence |
|---|---|---|---|
| 1 | "Gaussian lowpass prefilter **σ = 1.2**" restores +49.88 / +51.26 pp (README C4, K10, K14; `paper_readiness.md`) | **MISREPORTED** | σ = 1.2 appears nowhere in the code. K10 uses a Gaussian with **σ = 0.866 px** (`k10 run.py:99`); K14/K16 "B_filtered448" is the **3-tap binomial [¼,½,¼]** filter (`resize.apply_m7_filter_without_decimation`). The accuracy numbers themselves are correct for those filters. |
| 2 | K9 table: EffNet high band 72.25→38.08, low band "~24 %", "scale-invariant" (README Control 4) | **MISREPORTED** | Every README K9 value is the **unweighted mean of severity 3 and severity 5**, which is never stated (e.g. high@448 = (55.95+20.20)/2). `paper_readiness.md` quotes the s3 values instead (76.8→55.95, "~40 %"), so the two documents disagree. The repo's own `table_k9_freqnoise_ci.csv` additionally **pools FlexiViT F-p with F-t** (N = 4,000 = two arms). |
| 3 | "Low-frequency noise damages global semantic representation equally across resolutions" (README C3) | **NOT SUPPORTED for ViTs** | Holds for EffNet (low band 40.45 → 40.25 %, s3). For DeiT low-band accuracy falls 53.60 → 38.25 % (s3) and 13.10 → 0.65 % (s5); FlexiViT-p 52.40 → 36.75 %. |
| 4 | "High-frequency noise drives the entire 448 px collapse" (README C3) | **VERIFIED for EffNet only** | For DeiT and FlexiViT the largest 448-vs-224 DiD is in the **mid** band [56,112) cyc/frame: DeiT −47.35 ± 1.32 pp vs high −13.05 ± 1.10 (s5); FlexiViT-p −73.60 vs −37.00. |
| 5 | K10 operator table, "Noise s3" column 28.52 / 64.72 / 65.84 … (README Control 5) | **MISREPORTED** | That column is mean(s3, s5). True s3 values: identity@448 53.40, prefilter@448 74.54, bilinear-AA@224 72.16. README s5 values for area (42.10) / no-AA (41.90) are not reproducible (data: 38.84 / 38.90). |
| 6 | "Disabling antialiasing incurs an 11 pp **aliasing** penalty" (README Control 5) | **NOT SUPPORTED** | `area@224` and `bilinear_no_antialias@224` are the same 2×2 box average at exact 2× decimation: **100 % identical predictions** (DeiT) and 99.9 % (EffNet). The loss tracks white-noise gain (0.500 vs 0.313), not aliasing. Across 14 operator×resolution cells EffNet accuracy follows noise gain (Spearman −0.92, s5) while DeiT does not (−0.41, n.s.). |
| 7 | "Contrast: resolution scaling remains beneficial for **all** architectures (DiD > 0)" (README C6, guardrail 1) | **NOT SUPPORTED** | DeiT contrast DiD is **−1.27 pp** [−1.56, −0.99] (Table 1 row 4 itself). EffNet's +1.16 pp vanishes when measured from its native 320 px: 320→448 contrast DiD = −0.00 ± 0.11 pp. |
| 8 | "Resolution scaling does not inherently hurt degraded images — 448 beats 224 in every architecture at nominal-gain-matched noise" (README key finding 2) | **NOT SUPPORTED as stated** | Condition E adds *white* noise of std 0.3125·σ to the clean 448 image. In the frequency band shared with the 224 image it carries ≈¼ of the noise power the 224 input sees, so E is a much weaker corruption, not a matched one (the repo itself calls it "nominal", and K8's Immerkaer estimates show E ≠ C). The information-matched control that *does* exist (D_up448, K8) shows the opposite for DeiT: same information on the 448 grid loses −17.85 ± 1.22 pp (s5, DiD). |
| 9 | "Token count is the dominant driver of ViT vulnerability" (README Control 2/8/11) | **VERIFIED-WITH-CAVEAT** | The numbers reproduce (p32@448 − p16@448 = +39.13 ± 0.93 pp, s5). But a larger patch with PI-resized embedding weights is itself a resampling filter inside the patch embedding, so "token count" is confounded with in-embedding low-pass filtering; p32@448 still trails p16@224 by −4.67 ± 0.58 pp at s5. |
| 10 | "BatchNorm recalibration on **clean** data partly rescues" (README guardrail 4) vs "recalibrating BN directly on **corrupted** test images" (README Control 10) | **MISREPORTED + UNRELIABLE** | Code (`k13 run.py`) calibrates on 1,000 CAL-GATE images *corrupted with the same corruption*, per resolution. The recalibrated model loses clean accuracy at **every** resolution on MECH (224: 71.10 vs 76.40 %; 448: 80.05 vs 81.45 %). The pre-registered 1.0 pp sanity tolerance was raised post hoc to 7.0 pp in code, while `bn_recal_meta.json` still records 1.0. The older K5 `ebn` arm is catastrophically broken (56.2 % at noise s1, 224 px). No BN conclusion should be drawn. |
| 11 | "BatchNorm is not required for the collapse" (K16) | **VERIFIED-WITH-CAVEAT** | ConvNeXt-B (LayerNorm) DiD −35.84 ± 0.81 pp at s5, but only −2.94 ± 0.53 at s3; the comparison with ResNet-50 is confounded by IN-22k pretraining and architecture. "Proving" is too strong; "not sufficient to explain" is defensible. |
| 12 | "Resolution scaling beats deep token pruning by +22.12 pp at matched FLOPs" (README C5, "Decisive") | **VERIFIED number, over-claimed** | res-320 vs ToMe-r64@448: +22.12 ± 0.65 pp (s5), but ToMe was only applied at 448, where noise passes unfiltered; the comparison measures the spectral/grid mechanism, not token reduction. Clean gap +2.26 pp. K2/K5 ToMe r4/r8 used the pre-A5 implementation (no proportional attention). Cha et al. (arXiv:2610.07758) show schedule matters. |
| 13 | "Antialiased downsampling suppresses noise **variance** by −70 %" (README exec. summary) | **MISREPORTED** | σ_eff = 0.313·σ is a 69 % reduction in **standard deviation**, i.e. a 90 % reduction in variance (factor 10.2). |
| 14 | K9 registry "N = 5,000" | **MISREPORTED** | K9 used 2,000 MECH images per arm (288,000 rows). |
| 15 | "Kernels k8+ ran with the verified imagecorruptions backend" (PROVENANCE.md) | **MISREPORTED (harmless)** | K9 `run_meta.json` records `corruption_backend: fallback`. K9 uses its own FFT band noise, so results are unaffected, but the provenance statement is inaccurate. |
| 16 | Headline defocus (Table 1) | **Correctly flagged by authors** | K2–K4 used a Gaussian-blur fallback (σ = 3 px at s3), not ImageNet-C disk defocus: 86–95 % per-image agreement with the verified K8/K14 runs. Verified K14 DiDs (N = 10,000): DeiT −6.04 ± 0.84, EffNet −1.71 ± 0.87, FlexiViT −0.24 ± 0.60 (95 % CI half-widths). |
| 17 | K11 sensor-noise results | **Correctly scrubbed** | No K11 output exists; the kernel was never run. |
| 18 | "Reproduce with the committed code" (README §9) | **PARTIAL** | K0–K7 record `unknown_git_commit`; K8–K16 record `…-dirty` commits (uncommitted edits at run time). `k14 run.py` now lists only FlexiViT in `MODELS` (edited between passes), so the committed file cannot regenerate the DeiT/EffNet K14 shards without editing. |
| 19 | 81 passing tests | **Not executed in this audit** | Requires torch/timm, which are not installed here. Most tests run on synthetic shards and do not test the claims above. |

## C. Pre-registration deviations (from `docs/implementation_plan.md`)

1. Confirmatory analysis was planned on CONFIRM_POOL (44k) but reported on FULL (50k, includes PILOT used for gates). *Impact: none (re-run above).*
2. Gate G1 **failed** for EfficientNet/noise (projected power 0.48, `k2-pilot/gates_summary.json`); the study proceeded unchanged. *Impact: none; the observed effect is far larger than planned.*
3. Planned GEE/logit sensitivity, TOST equivalence and per-image bootstrap CIs were replaced by analytic paired z-tests. *Logit sensitivity now done (section A).*
4. Planned stratification by native source resolution (`src_short_side`) was never recorded. This matters: most ImageNet validation images have a short side below 512 px, so the "448 acquisition frame" is usually an **upsampled** image. *Impact: unquantified (no image dimensions in repo).*
5. M1–M4 attention/representation probes (K6) were run but never reported. DeiT normalised attention entropy rises with resolution (0.691 → 0.769 clean) and with noise (0.754 → 0.834), with no interaction (+0.002).
6. The spectral gate selector (plan §9) was never evaluated; the oracle analysis replaced it.
7. K5's frequency-noise stage crashed (`TypeError` in `knobs-k5-controls-v2.log`); K9 replaced it a day later.

## D. What survives

- The resolution × corruption interaction is real, large, reproducible across kernels, robust to the scale of measurement and to the pre-registered split.
- The *mechanism* story in the README (antialias denoising explains everything; ViTs = token count; contrast always benefits) is only true for EfficientNet. The repo's own data contain a stronger, architecture-dependent result that the README misses (see `03_*` and `06_*`).
