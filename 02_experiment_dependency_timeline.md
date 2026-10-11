# 02 — Experiment dependency timeline

Sources: `git log` (commit times in IST, +05:30), `run_meta.json` timestamps (UTC, written at the end of
each kernel), and kernel logs. Run times marked "≈" are inferred from logs or ordering and are uncertain.
The whole study ran in ~5 days (6–11 Oct 2026), largely through an AI agent ("Antigravity Research" commits).

## Chronology

| When (UTC) | Event | What it tested / changed | What it led to |
|---|---|---|---|
| 6 Oct 20:37–20:45 | Initial `knobs` package + `push_and_wait` + K-prep/K0 kernels | Original brief: "which efficiency knob (resolution, token merging, exit depth) to turn under corruption" (`docs/ANTIGRAVITY_WORKFLOW.md`) | Pre-registered plan v2.2 (`docs/implementation_plan.md`) |
| 6 Oct 23:35 | **K0** probe | G0-A on a class-ordered slice → all 4 models "fail" by 5–7 pp | Bug noted later (A2); real G0-A only in K13 |
| 6 Oct 23:40 | **K1** sanity | fp16 = fp32 on 500 images; ToMe r=0 identity | K2 allowed |
| 7 Oct 01:59 | **K2** pilot (5k) | Full grid on PILOT, gates G0-B/G1/G2 | G1 fails for EffNet/noise; study proceeds. DiD sign later found inverted (A1) |
| 7 Oct 07:36 | **K3** headline A (45k) | clean, noise s3, defocus s3 (fallback blur) | Table 1 |
| 7 Oct 10:20 | **K4** headline B (45k) | JPEG, contrast | Table 1 |
| ≈7 Oct 10:30–12:30 | **K5** controls (SUB10K) | s1/s5 dose-response; ToMe r4/r8; BN `ebn`; DeiT-384; freq noise | Crashed at the freq-noise stage; `ebn` broken → motivated K9, K12, K13 |
| 7 Oct 13:59 | **K6** mech | M1–M4 attention probes (never reported); M7-v1 A/B/C/D on 1k | M7-v2 (K8) |
| 7 Oct 14:11 | **K7** latency | T4 timing | Table 8 |
| 8 Oct 02:41 | Results committed (`c35084e`), first paper draft (in a `report/` folder, deleted on 11 Oct) | – | Self-audit |
| 8 Oct 06:32–07:18 | **Hardening A1–A9** | A1 DiD sign + Holm; A2 class-biased slices; A3 backend guard; A4 noise-gain function; A5 ToMe re-implementation; A6 BN control rewrite; A7 latency; A8 git provenance; A9 freq-noise seeding | All later kernels use the fixed code |
| 8 Oct 07:32–07:56 | **Analysis B1–B8** | Loader with image_id matching, Table 1 reproduction, M7, transitions, figures | Reproducible tables |
| 8 Oct 08:09–15:06 | **Kernels C1–C6 prepared** | K8 (M7-v2 incl. D and E), K9 (FFT bands), K10 (operators), K11 (sensor noise), K12 (ToMe matched), K13 (G0-A + BN) | Runs below |
| 8 Oct 23:35 | **K9** freq-noise-v2 | Band-limited noise × resolution × model | README mechanism claim (misreported, see `04_*`) |
| 8 Oct 23:58 | **K13** first run | BN recal (later kept as "pilot-5k-backup") | Re-run after gate tweak |
| 9 Oct 00:24 | **K10** resize ablation | 6 operators, measured noise gains | "Prefilter recovers +51 pp" (σ mislabelled 1.2) |
| 9 Oct 00:58 | **K12** ToMe matched | res-320 vs ToMe-r64@448 | C5 claim |
| 9 Oct 01:44 | **K8** M7-v2 | A/B/B_g/C/D/E, 4 configs, 2k images | Basis of this audit's decomposition |
| 9 Oct 05:40 | Loader de-duplication bug found (677k rows lost) and fixed (`d48fd16`) | – | All tables regenerated |
| 9 Oct 07:28 | **K13** final | 50k G0-A passes; BN tolerance raised to 7 pp | BN "rescue" claim (unreliable) |
| 9 Oct 12:15–12:54 | **Tasks A–E** "publication-honest audit" | Backend guards, defocus flagged unverified, K11 scrubbed, paired CIs, oracle headroom, K14–K16 prepared | K14, K15, K16 |
| 9 Oct (DeiT + EffNet pass, done before 21:05) → 10 Oct 09:45 (FlexiViT pass ends) | **K14** (two passes; commit `52606f6` at 9 Oct 21:05 notes "DeiT and EfficientNet complete") | A/B/C/E at 10k; verified ImageNet-C defocus | "448 beats 224 at matched noise" (not supported; see `04_*` #8) |
| 10 Oct 19:56 | **K15** FlexiViT res × patch | Tests the token-count hypothesis from F-p vs F-t | "+39 pp rescue" |
| 10 Oct 23:56 | **K16** ResNet-50 vs ConvNeXt-B | Tests the BN hypothesis left open by K13 | "BN not required" |
| 11 Oct 00:01–04:30 | K14–K16 results committed; `report/` deleted; README rewritten | – | Current state |

## Dependency graph

```
K-prep ──► splits ──► K0 ──► K1 ──► K2 (pilot, gates) ──► K3 + K4 (headline, Table 1)
                                    │                       │
                                    ├──► K5 (dose, controls)┼──► crash (freq noise) ──► K9 (bands)
                                    │        └── ebn broken ─┴──► K13 (BN v2) ──► K16 (BN vs LN)
                                    │        └── ToMe old ───────► K12 (ToMe matched)
                                    ├──► K6 (M1–M4, M7-v1) ──────► K8 (M7-v2: B, B_g, C, D, E)
                                    │                                 └──► K14 (A/B/C/E @10k, verified defocus)
                                    ├──► F-p vs F-t (K2/K5) ─────────► K15 (patch grid)
                                    └──► K7 (latency)
Self-audit A1–A10 (8 Oct) ─► fixes used by K8–K16; defocus in K2–K4 re-labelled unverified ─► K14 defocus
```

## What was learned, in order

1. **K2–K4:** large negative DiD under noise for every model; EfficientNet-B3 collapses (−21.9 pp at s3).
2. **K5:** the effect grows super-linearly with severity (EffNet −1.6 → −21.9 → −59.9 pp for s1/s3/s5).
3. **K6/K8:** removing high frequencies at 448 (B) or re-using the 224 information on the 448 grid (D)
   repairs EfficientNet but **not DeiT**. The repo recorded this ("DeiT: prefiltering does not help") but did
   not develop it.
4. **K9:** band-limited noise shows the resolution penalty sits in different bands for the CNN (high)
   and the ViTs (mid).
5. **K10:** the operator's white-noise gain predicts EfficientNet's accuracy across operators, and
   "aliasing" is not the active factor.
6. **K12:** ToMe at 448 cannot recover what downsampling recovers (ToMe never removes the input noise).
7. **K13/K16:** BN statistics are not the explanation (and the BN control itself is unreliable).
8. **K14:** the B-filter results replicate at 10k. The verified defocus DiDs differ from the fallback
   headline for FlexiViT (−0.24 vs −1.26 pp).
9. **K15:** larger patches at 448 recover most of FlexiViT's loss.

## Gaps in history and metadata

- K0–K7 have no git commit or backend metadata, and K5 has no `run_meta.json`.
- The deleted `report/paper.tex` (removed in `cd99167`) is the only record of the first manuscript.
  It is recoverable from git history but was not audited here.
- The K14 DeiT/EfficientNet pass ran under an earlier, unrecorded edit of `run.py`.
- No native image dimensions were stored, so the share of upsampled 448 frames is unknown.
