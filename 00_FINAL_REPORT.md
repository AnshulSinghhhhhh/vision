# Final report — audit of `AnshulSinghhhhhh/vision` and paper recommendation

*Audit date: 11 Oct 2026 · repository commit `bd36b83` · reviewer stance: IEEE conference referee.*

## Bottom line

The repository's central measurement is sound. Raising test resolution from 224 to 448 px amplifies
corruption damage for all three models (EfficientNet-B3 Gaussian-noise DiD −21.92 pp, 95 % CI [−22.39, −21.44],
N = 50,000), and the result reproduces exactly from the raw shards, on the pre-registered 44k split and on the
log-odds scale.

The repository's explanation of that result is mostly wrong or already published. "Downsampling denoises,
so lower resolution is more noise-robust" is prior art (Kim et al., arXiv:2509.20939, 2025). Several README
mechanism claims are mislabelled or contradicted by the repo's own data.

The repository does, however, contain evidence for a different and stronger result it never wrote up. An
information-matched control already in K8 (D = the 224-px image upsampled back to 448 px) gives an exact
decomposition of the penalty into **spectral pass-through** and **grid mismatch**, and the two dissociate by
architecture:

| Gaussian noise s5 (N = 2,000) | Total DiD | Spectral (A − D) | Grid (D − C) |
|---|---|---|---|
| EfficientNet-B3 | −58.70 ± 2.49 | **−52.30** | −6.40 |
| FlexiViT-B (patch 16) | −42.70 ± 2.41 | −33.65 | −9.05 |
| FlexiViT-B (fixed 256 tokens) | −7.05 ± 1.65 | −7.60 | **+0.55** |
| DeiT-B/16 | −20.50 ± 2.33 | −2.65 | **−17.85** |

(pp, 95 % CI half-widths.) Low-pass prefiltering therefore repairs the CNN but not DeiT. This is replicated
by three independent kernels: a 3-tap filter at N = 10k, a Gaussian prefilter at N = 5k, and a patch-size grid
at N = 3k. Band-limited noise localises the damage to the high band for the CNN and the mid band for the ViTs.

**Recommendation: write one paper, on this decomposition.** The draft manuscript is in `paper/main.tex`.
There is not enough independent evidence for a second paper. The other candidates are either prior art,
confounded, or supporting sections of the first (see `06_ranked_paper_opportunities.md`).

## What was and was not done

- **Done:** full read of code, kernels, logs, metadata, docs and git history. Independent re-computation of
  every reported table from the 5.7M per-image predictions. Cross-kernel reproducibility check. Statistical
  re-analysis (paired SEs, Holm, log-odds bootstrap, pre-registered split, confidence-standardised flips).
  Literature verification. Figures and manuscript.
- **Not done (by instruction):** no model was run and no new experiment was performed. Every proposed
  experiment is marked [PROPOSED] with a cost estimate.
- **Not done (environment):** the repo's pytest suite was not executed (torch/timm not installed here), and
  `main.tex` was not compiled (no TeX engine on this machine). It passes a structural lint (balanced
  environments, every citation and label resolved, all figure files present); compile it on Overleaf.

## Most important audit findings (details: `04_claim_verification_audit.md`)

1. **Prefilter mislabelled.** The "σ = 1.2 Gaussian prefilter" appears nowhere in the code. K10 used σ = 0.866 px and K14/K16 used a 3-tap binomial filter.
2. **K9 table mislabelled.** The README's K9 frequency table silently averages severities 3 and 5. The repo's K9 CSV pools two different FlexiViT arms.
3. **K10 table mislabelled.** The README's K10 "s3" column is mean(s3, s5). The claimed "aliasing penalty" is a noise-gain effect: area and no-antialias resizing give identical predictions.
4. **Contrast claim false.** "Contrast always benefits from resolution" is false for DeiT (−1.27 pp). For EfficientNet it disappears when measured from its native 320 px.
5. **"448 beats 224 at matched noise" is not supported.** Condition E is a much weaker corruption, not a matched one.
6. **The BatchNorm control is unreliable.** It lowers clean accuracy at every resolution, and its tolerance was relaxed from 1 to 7 pp after the fact.
7. **Weak provenance.**
   - K0–K7 have no git commit, and K8–K16 ran from uncommitted ("dirty") code.
   - The committed K14 script cannot regenerate its own DeiT/EfficientNet shards.
   - The headline defocus used a Gaussian-blur stand-in.
   - Native image sizes were never recorded.

## Paper to prioritise

**Title:** *Spectral Pass-Through or Grid Mismatch? Decomposing Why Higher Test Resolution Amplifies Corruption
Damage in CNNs and Vision Transformers.*

**Why it is defensible.**
- The decomposition is an exact per-image identity. It needs no model of the network.
- Its key cells are far from zero (DeiT grid term −17.85 ± 2.39 pp; EfficientNet spectral term −52.30 ± 2.43 pp).
- The qualitative split replicates in three separately run kernels with different filters and sample sizes.
- The confound "DeiT is just less confident at 448" is ruled out by the stored confidences.
- The prescriptive claim (filter vs. native token grid) follows directly from the data.

**Remaining work, in priority order** (all inference-only; total about 8 T4 GPU-hours, within one week of Kaggle quota):

1. **E1:** D control at N = 10k for ResNet-50, ConvNeXt-B, a Swin, DeiT-III and a RoPE ViT, plus JPEG and contrast. Turns "three architectures" into a general claim.
2. **E7:** D with nearest, bicubic and Lanczos upsamplers. Pre-empts the strongest reviewer objection, that the grid term is DeiT's dislike of bilinear-blurred inputs.
3. **E3:** stratify all existing DiDs by native image size, read from JPEG headers. CPU only. Removes the upsampled-frame confound.
4. **E2:** three noise draws per image, for honest variance.
5. **E4–E6:** bands combined with D; 320/384 decomposition; sensor-grid noise.

Then re-run the headline and decomposition kernels from a **clean, tagged commit** with the real ImageNet-C
backend, and publish the code with the paper.

**Suitable IEEE targets.** Dates are unannounced as of 11 Oct 2026; verify on the official sites.

| Venue | Status / dates | Fit |
|---|---|---|
| IEEE ICIP 2027 | Singapore, listed for 29 Nov–3 Dec 2027; paper deadline TBA | Best fit: image-processing mechanism story |
| IEEE ICME 2027 | Xiamen, listed for 13–17 Jul 2027; deadline TBA, historically early December | Good fit |
| IEEE IJCNN 2027 | Deadline usually in January | Acceptable fit |
| IEEE/CVF CVPR 2027 robustness/efficiency workshops | Deadline usually March | Workshop route |
| IEEE/CVF WACV 2028 | – | Main-track stretch; needs E1 + E3 + E7 |

**Estimated acceptance.**
- Mid-tier IEEE venue, as is: ~30–40 %.
- Same venues with E1 + E3 + E7: ~45–55 %.
- CVPR/ICCV main track: < 15 % without more datasets and a training-side intervention.

**Main risks to acceptance.**
1. *Scope:* one checkpoint per family; the decomposition covers three architectures at N = 2,000.
2. *Prior-art pushback:* reviewers who know Kim et al. 2025 may say "just downsample". The paper must lead with the dissociation and the DeiT result, not the noise gain.
3. *Physical realism:* noise is added after up-sampling into a 448 frame, not at a sensor grid.
4. *D-control interpretation:* bilinear upsampling adds blur (E7 fixes this).
5. *Reproducibility optics:* agent-written code, dirty commits and the README errors above. None of the README's mislabelled numbers may be copied into the paper.
6. *AI-use disclosure:* IEEE requires disclosure of AI-generated text, figures and code in the acknowledgments. A placeholder is included; adapt it to the venue's policy.

## Deliverables (this folder)

| File | Content |
|---|---|
| `00_FINAL_REPORT.md` | This summary and recommendation |
| `01_experiment_inventory.md` | All 18 kernels, status, N, provenance, outputs, issues |
| `02_experiment_dependency_timeline.md` | Chronology, dependency graph, what was learned when |
| `03_cross_experiment_results_matrix.md` | Accuracy surfaces, DiDs, mechanism matrix, 10 hidden patterns, contradictions, unexplored cells |
| `04_claim_verification_audit.md` | Claim-by-claim verdicts, integrity checks, pre-registration deviations |
| `05_novelty_and_related_work_audit.md` | Verified citations, closest prior work, novelty per finding |
| `06_ranked_paper_opportunities.md` | Six candidates ranked; full proposal for #1; proposed experiments with costs |
| `paper/main.tex` + `IEEEtran.cls` + `figures/` | IEEE conference manuscript (official template), 4 column-width figures, 3 tables, all numbers from `figures/paper_numbers.json` |
| `scripts/` | Audit scripts `a01`–`a11` and `lint_tex.py` |
| `derived/` | Recomputed CSVs, including `results_matrix_long.csv` (every configuration cell with N and CI) |

**Reproduce everything:** clone the repo, then for each script run
`python -I scripts/aXX_*.py <clone-path> derived/` (`a10` writes to `paper/figures/`).
