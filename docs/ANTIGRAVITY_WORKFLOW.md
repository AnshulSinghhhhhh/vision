# Agent Brief: "Which efficiency knob should you turn when the input is degraded?"

You are an autonomous research-engineering agent. Execute this brief end to end: build the repo, run every experiment on Kaggle T4 GPUs through the Kaggle CLI, collect results, and produce tables, figures and a findings report. Work phase by phase. Do not skip a phase's gate. Ask the human only for the items in section 1.3.

## 0. Study in one paragraph

Pretrained ImageNet classifiers are evaluated on corrupted images (ImageNet-C style) while three inference "compute knobs" are varied at matched FLOPs: (A) **input resolution**, (B) **token reduction** in Vision Transformers (training-free ToMe merging vs training-free attention-based pruning), and (C, stretch) **exit depth**. Questions:

1. Does lower resolution ever beat native resolution under noise or blur, once train-test resolution mismatch is controlled for?
2. Does token merging stay more accurate than token pruning under corruption at equal FLOPs, and why?
3. Does the best knob depend on corruption type, and can a cheap spectral selector pick it per image better than any fixed setting?

Everything is inference-only on pretrained checkpoints, except tiny selector or exit heads. No model is trained from scratch.

## 1. Environment and rules

### 1.1 Credentials (security)
- Read Kaggle credentials only from the environment (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or `~/.kaggle/kaggle.json` (chmod 600).
- Never print, log, commit or write the key into notebooks, datasets, outputs or reports. Add `kaggle.json`, `.env` and `outputs/` secrets patterns to `.gitignore`.
- Never put credentials inside a Kaggle kernel. Kernels receive code and data only.

### 1.2 Kaggle constraints to respect
- GPU quota is limited (about 30 GPU-hours/week on a typical account; check the account page). Total planned GPU use is budgeted in section 9. Stop and report if projected use exceeds 90% of the remaining quota.
- A session has a hard time limit (about 9 hours). Make every kernel run **at most ~3 hours** and **resumable**: results are appended to a CSV after each condition and a run skips conditions already present.
- Working directory limit is about 20 GB. Never cache the full corrupted grid; generate one corruption at a time, evaluate, delete.
- The kernel needs internet enabled for `pip install` and Hugging Face downloads. This requires a phone-verified account (human step).
- Kaggle may assign a T4 x2 or a P100. **Every run must call `nvidia-smi` first and record the GPU name.** Accuracy results are valid on any GPU. **Latency and throughput numbers are valid only on a T4**; use device 0 only for timing and tag each row with the GPU name. Check `kaggle kernels push --help` for an accelerator option on the installed CLI version and use it to request a T4 if available.

### 1.3 Human steps (ask once, up front, then proceed)
1. Confirm Kaggle phone verification (needed for internet in kernels).
2. Accept the rules of the ImageNet competition on the Kaggle website if the competition route (section 3.1b) is needed.
3. Provide `KAGGLE_USERNAME` and `KAGGLE_KEY` through the environment.
4. Optionally provide a Hugging Face token (only if a gated checkpoint is required).

### 1.4 Reproducibility rules
- Pin versions: record `pip freeze`, `torch`, `timm`, CUDA, driver, GPU name and git commit hash in `run_meta.json` for every run.
- Fix all seeds. Corruption randomness is seeded per `(image_id, corruption, severity)` via a hash, so any condition is regenerated identically in any run.
- Inference in fp16 autocast. Run a 500-image fp32 vs fp16 check per model; accuracy must agree within 0.3 points or use fp32 for that model.
- `torch.backends.cudnn.benchmark=False`; `torch.use_deterministic_algorithms(True, warn_only=True)` for accuracy runs. Latency runs may use benchmark mode but must say so in the metadata.
- Report every number with a 95% bootstrap CI over images (1,000 resamples). Paired comparisons use the same images and a paired bootstrap or McNemar test.

## 2. Repository layout (create exactly this)

```
knobs-under-corruption/
  README.md                  # how to reproduce, one command per phase
  pyproject.toml / requirements.txt   # pinned
  src/knobs/
    data.py                  # ImageNet val subset loader, label mapping
    corrupt.py               # deterministic ImageNet-C corruptions + sensor-noise model
    models.py                # model zoo loader (timm, open_clip), eval transforms
    resize.py                # resize operators (antialias on/off, interpolation modes)
    tokens.py                # ToMe wrapper, training-free pruning, token counters
    flops.py                 # FLOP counting with torch.utils.flop_counter.FlopCounterMode
    latency.py               # CUDA-event timing harness
    spectral.py              # spectral / noise / blur statistics
    selector.py              # zero-parameter gate + learned selector
    exits.py                 # (stretch) exit heads on frozen backbones
    stats.py                 # bootstrap CI, McNemar, Pareto, AUC
    run_grid.py              # resumable experiment runner, appends to results CSV
  kaggle/
    kernels/<name>/kernel-metadata.json + run.py   # one per kernel job
    push_and_wait.sh         # push, poll status, download output
  tests/                     # section 4 sanity tests (must pass before any grid run)
  results/                   # raw CSVs only (one row per observation, no aggregation)
  analysis/                  # notebooks/scripts producing every table and figure
  figures/  tables/  report/
```

## 3. Phase 0: data and tooling (no GPU needed for most)

### 3.1 ImageNet validation subset
Priority order. Use the first that works, and record which in `report/data_provenance.md`.
- **a)** Search Kaggle datasets for a validation-only ImageNet-1k upload (50,000 images with labels). Check file count, label file, and class-index convention before using it.
- **b)** Attach the Kaggle ImageNet competition (`imagenet-object-localization-challenge`) as a competition source. Validation labels are in `LOC_val_solution.csv`; class indices follow the sorted order of synsets in `LOC_synset_mapping.txt`. Requires the rules to be accepted (human step).
- **c)** Hugging Face `imagenet-1k` validation split if a token is available.

Build three **disjoint** subsets from validation, stratified by class, saved as a file list with a fixed seed:
- `EVAL`: 2 images/class = 2,000 images. Main grid.
- `EVAL_BIG`: 5 images/class = 5,000 images, superset of `EVAL`. Focus conditions and clean numbers.
- `SEL_TRAIN`: 2 images/class, disjoint from `EVAL_BIG`. Used only to fit the learned selector and any calibration.
- `CALIB`: 2,000 **training**-set images (unlabeled use only), for BatchNorm re-estimation. If the train set is unavailable, use `SEL_TRAIN` without labels.

Standard preprocessing: resize 256, center crop 224, as in ImageNet-C. Corruptions are applied to this 224x224 uint8 image. Lower and higher resolutions are produced **after** corruption by the resize operators (section 5.1), except in the sensor-noise protocol (5.4).

### 3.2 Corruptions
- Use the `imagecorruptions` package (15 ImageNet-C corruptions x 5 severities; it ships a numba-accelerated glass blur). If installation fails, implement from the original ImageNet-C code.
- Wrap in `corrupt.py` with a seeded RNG per `(image_id, corruption, severity)`. Test determinism.
- Group the corruptions: **noise** (gaussian, shot, impulse), **blur** (defocus, glass, motion, zoom), **weather** (snow, frost, fog, brightness), **digital** (contrast, elastic, pixelate, jpeg).
- Generate **one corruption x all 5 severities for `EVAL`** at a time in RAM (about 2,000 x 5 x 150 KB = 1.5 GB), run all models and knobs on it, write results, free it. Do not cache the whole grid.

### 3.3 Models (load through `models.py`, record exact weight tags)
| Key | Source | Reference clean top-1 (use to validate) |
|---|---|---|
| `resnet50` | timm `resnet50.tv_in1k` | about 76.1 |
| `convnext_tiny` | timm `convnext_tiny.fb_in1k` | about 82.1 |
| `deit_small` | timm `deit_small_patch16_224.fb_in1k` | about 79.9 |
| `deit_base` | timm `deit_base_patch16_224.fb_in1k` | about 81.8 |
| `flexivit_base` (optional) | timm `flexivit_base.1200ep_in1k` | verify; trained for variable patch size |
| `clip_vitb16` (optional) | `open_clip` ViT-B/16 zero-shot with the ImageNet prompt templates | verify against the published zero-shot number |
| `dinov2_base` (optional) | HF `facebook/dinov2-base-imagenet1k-1-layer` if it loads | verify |

Use each checkpoint's own mean/std and interpolation. If a tag is missing in the installed timm, find the nearest equivalent and record the substitution. Clean top-1 on `EVAL_BIG` must be within about 1.5 points of the reference or the loader is wrong (stop and debug).

ViTs at non-224 sizes: use timm's dynamic image size support (`dynamic_img_size=True`) or `set_input_size`, with position-embedding interpolation. Record which was used.

### 3.4 Token reduction implementations (`tokens.py`)
- **ToMe (merging):** use the official `facebookresearch/ToMe` package patching timm ViTs, off the shelf (no retraining), size-weighted average with proportional attention as in the paper. The package targets older timm internals. If it fails on the installed timm, create a pinned environment with the timm version the ToMe repo requires, or port the merge step. Verify with test 4.6 (r=0 identical to baseline).
- **Training-free pruning (the fair counterpart):** in `tokens.py`, at layers {3, 6, 9} keep the top-k patch tokens ranked by the CLS-to-patch attention of the previous block (EViT-style inference rule, **no fine-tuning**, optional fused "inattentive" token as in EViT). Set k per layer to hit target FLOP fractions.
- **Official fine-tuned EViT / DynamicViT checkpoints (secondary):** use authors' released DeiT checkpoints only for a clearly labeled secondary comparison, since their training recipe differs from the training-free setting.
- Report token counts per layer and measured FLOPs for every configuration.
- Everything must run on the **same** DeiT checkpoint so the only difference is the reduction rule.

### 3.5 FLOPs and latency
- FLOPs: `torch.utils.flop_counter.FlopCounterMode` on the actual forward pass (counts ops really executed, including after merging). Report GFLOPs per image (multiply-adds x2 convention stated explicitly) and cross-check ResNet-50 at 224 is about 4.1 GMACs and DeiT-B about 17.6 GMACs.
- Latency: CUDA events, 50 warmup runs, 200 timed runs, batch sizes 1 and 64, fp16, report median and IQR, `torch.cuda.synchronize()` around every timing, record GPU name. Include the cost of resizing, of the gate (section 7) and of token merging bookkeeping. Throughput = images/s at batch 64.

## 4. Sanity tests (all must pass before any GPU grid; put in `tests/`)
1. Corruption determinism: same `(image_id, corruption, severity)` gives bit-identical output twice; different seeds differ.
2. Severity monotonicity: for a pooled sample of 500 images, mean absolute pixel change from clean increases with severity for each corruption.
3. Clean accuracy at native resolution matches the reference table within about 1.5 points on `EVAL_BIG`.
4. Label mapping check: top-1 of a random-label shuffle is about 0.1%.
5. fp16 vs fp32 agreement (section 1.4).
6. ToMe with r=0 produces logits identical to the unpatched model (max abs diff below 1e-3 in fp32). Pruning with keep ratio 1.0 likewise.
7. FLOP counter: monotone decrease with r and with resolution; resolution scaling for the ResNet is about quadratic.
8. Resume test: kill a run halfway, restart, final CSV has no duplicates and no gaps.
9. Resize operators: a constant image stays constant; resizing a Gaussian-noise image with antialias on reduces measured noise std by about the scale factor (check the spectral effect is real).

## 5. Experiments

### 5.1 Resize operators and the resolution knob (Experiment A)
Resolutions: {96, 112, 128, 160, 192, 224, 256, 288}. Values above 224 upsample a corrupted 224 image, so they are a control showing that more pixels without more information do not help. ViT patch constraints: use multiples of 16 for DeiT (112, 128, 160, 192, 224, 256, 288).

Resize operators (ablate on `resnet50` and `deit_base`):
- bilinear with antialias, bicubic with antialias, Lanczos (PIL), area/box, bilinear **without** antialias, nearest.
- Baselines at native resolution: Gaussian blur (sigma matched to the low-pass of the downsample), median filter, bilateral filter, non-local means.

Conditions: 15 corruptions x 5 severities + clean, x 8 resolutions, on `EVAL` for all models in 3.3 (CNNs and ViTs). Record top-1, GFLOPs, and per-image correctness (bit-packed) so any paired test can be done offline.

**Mismatch controls (essential, otherwise a reviewer will say you only measured mismatch):**
- *Clean sweep:* the same resolution sweep on clean images quantifies the pure mismatch cost.
- *BatchNorm re-estimation (CNNs):* for each resolution, recompute BN statistics on `CALIB` clean images resized to that resolution (label-free), then evaluate. Report plain vs recalibrated.
- *Inversion criterion:* resolution r* shows inversion for a (model, corruption, severity) if accuracy at r* exceeds accuracy at 224 with the lower bound of the paired 95% CI above 0. Report both plain and recalibrated; the recalibrated result is the headline.
- *Mismatch-free probe:* repeat the key conditions on `flexivit_base` (trained across sizes) and, if available, a resolution-robust CNN. If inversion survives there, it is not only mismatch.

### 5.2 Token reduction under corruption (Experiment B)
Models: `deit_base`, `deit_small` (and `flexivit_base`/CLIP if loaded).
- ToMe r sweep: r in {0, 2, 4, 6, 8, 10, 12, 14, 16} tokens merged per layer (cap so at least a few tokens remain).
- Pruning sweep: keep ratios at layers {3,6,9} chosen so the FLOPs match each ToMe setting within 2% (solve for the match, do not eyeball).
- All 15 corruptions x severities {1,3,5} + clean on `EVAL` (full 5 severities for the noise group and defocus/motion blur).
- Output: accuracy vs GFLOPs and vs measured throughput, per corruption.
- **Mechanism analysis (answers "why", needed for acceptance):** on `EVAL_BIG` for gaussian noise severity 3 and 5 and clean:
  - Fraction of **object-region tokens** dropped by pruning vs merged by ToMe, using the ImageNet bounding boxes if available, otherwise the CLS-attention map of the unreduced model as a proxy mask.
  - Per-layer attention entropy and Jensen-Shannon divergence between clean and corrupted attention maps.
  - Token-embedding noise statistics: variance of the difference between corrupted and clean token embeddings, before and after the reduction step, per layer. **Do not assume a 1/sqrt(k) averaging law**; report the measured ratio and the average similarity of merged tokens (merged tokens are similar, not independent noise samples, so the clean theory may not hold).
  - Merge-cluster size distribution under noise vs clean.

### 5.3 Joint knob grid (Experiment C)
For `deit_small` and `deit_base`: resolution in {112, 160, 224} x ToMe r grid scaled to each resolution's token count (r_max(res) x {0, 0.25, 0.5, 0.75}). Conditions: noise group, blur group, and 4 other corruptions at severities {1,3,5} + clean on `EVAL`. Output the **knob dominance map**: for each (corruption, severity) and each FLOP budget in {75%, 50%, 35%, 25%} of baseline FLOPs, which configuration gives the highest accuracy, with CIs, and whether the winner is resolution-only, tokens-only or a mix.

### 5.4 Sensor-noise protocol (realism, Experiment D)
ImageNet-C noise is applied at 224 pixels, which differs from camera noise followed by downscaling. Add a second protocol: take images at higher base resolution (resize 512, center crop 448), add a signal-dependent Poisson-Gaussian noise model (shot-noise gain and read-noise sigma at three ISO-like levels, plus a simple demosaic-free approximation; document parameters), then downsample to the model resolution. Compare against ImageNet-C noise at equal PSNR. If any real camera-noise classification set is found on Kaggle with labels, add it as a labeled extra; do not invent one if none exists.

### 5.5 Stretch: exit-depth knob (only if Phases 1-3 pass)
Attach small exit heads (linear or 2-layer MLP) to frozen `deit_small` at layers {4, 6, 8, 10}, trained on features of a 50k-image ImageNet **train** subset (extract features once, train heads on the cached features; about one hour on a T4). Evaluate exits under corruption. Compare: max-softmax thresholding, a lower global threshold (matched FLOPs), and a "marginal gain" probe predicting that deeper layers fix a currently wrong prediction (train on held-out corruption types). **Pilot first:** if probe AUROC for "recoverable" is not clearly above MSP on held-out corruptions, drop this stretch and say so in the report.

## 6. Pilot and decision gates

### Phase 1 pilot (about 1 GPU-hour, run before everything else)
`resnet50` and `deit_base`, `EVAL`, gaussian noise and defocus blur at severities {3,5}, resolutions {112, 160, 224}, plain and BN-recalibrated, plus ToMe r in {0, 8, 16} on DeiT-B. Produce one plot and `report/pilot.md`.

**Gate A (resolution claim):** if for at least 2 of the 4 corruption-severity pairs lower resolution beats native by at least 1 point with paired CI above 0 after recalibration, proceed with the full A.
**If not:** keep A as a short "mismatch explains the effect" negative-result section, and put the weight on Experiments B and C.

**Gate B (token claim):** if ToMe and pruning differ by at least 2 points anywhere in the pilot under corruption at matched FLOPs, proceed with the full B mechanism study.
**If not:** the finding is "no meaningful difference", which is still reportable; run B at reduced size.

Write the decision and the numbers to `report/gates.md` before spending more quota. Do not change the criteria after seeing results.

## 7. The selector (Experiment E)

### 7.1 Zero-parameter spectral gate (`spectral.py`)
Computed on the 224x224 corrupted input, target below 1 ms per image on GPU or under 3 ms on CPU:
- radially averaged power spectral density slope (log-log fit over mid frequencies) and high-to-low frequency energy ratio,
- noise sigma estimate (Immerkaer method and the median absolute deviation of the wavelet HH band),
- blur measure (variance of Laplacian), mean luminance, RMS contrast.
Gate rule: thresholds on noise sigma and blur measure choose among resolution settings; thresholds fit on `SEL_TRAIN` only.

### 7.2 Learned selector
Features above (about 10 numbers) to a small model (multinomial logistic regression and a depth-limited gradient-boosted tree). Predict per-configuration correctness for configurations in the joint grid and choose the argmax predicted correctness under each FLOP budget. Train on `SEL_TRAIN` with corruption types from a **training group** and evaluate on `EVAL` with **held-out corruption groups** (leave-one-group-out over noise/blur/weather/digital). Never train and test on the same corruption type or the same image.

### 7.3 Comparisons (matched FLOPs, per budget)
- best fixed configuration chosen on `SEL_TRAIN`,
- per-condition oracle (best configuration for each corruption-severity; upper bound with condition knowledge),
- per-image oracle (upper bound),
- noise-estimate-only gate, zero-parameter gate, learned selector,
- (stretch) confidence-threshold early exit.
Count the selector's own FLOPs and measured latency inside the total. Report accuracy, GFLOPs, latency on T4, and the area under the accuracy-vs-FLOPs curve across severities.

## 8. Analysis outputs (produce all; each from raw CSVs by a script)

Tables:
- T1 clean accuracy vs reference (validation of the setup).
- T2 resolution inversion table: model x corruption group x severity, best resolution, gain over 224 with CI, plain vs recalibrated.
- T3 token reduction: matched-FLOPs accuracy, ToMe minus pruning, with CI, by corruption group.
- T4 knob dominance counts (how often each knob wins at each budget).
- T5 selector results vs baselines at 4 budgets.
- T6 latency and throughput on T4 for the main configurations.

Figures:
- F1 accuracy vs resolution curves per corruption (small multiples), clean and corrupted, plain vs recalibrated.
- F2 accuracy vs GFLOPs Pareto per corruption for the three knobs.
- F3 the mechanism plot for tokens (object-token drop fraction, attention JS divergence, embedding noise ratio by layer).
- F4 knob dominance heatmap (corruption x budget).
- F5 selector accuracy vs FLOPs vs the fixed and oracle curves.
- F6 resize-operator ablation (antialias, interpolation, blur-at-native control).
- F7 sensor-noise protocol vs ImageNet-C noise.

Report (`report/findings.md`): per question, the result, the effect size with CI, whether the pre-registered gate criterion was met, what contradicted expectations, and an explicit **limitations** list (two or three architectures; ImageNet-C is synthetic; training-free reduction differs from fine-tuned reduction; upsampled resolutions add no information; subset size and its CI width). Negative results stay in the body.

## 9. GPU budget and scheduling

Measure real throughput in the pilot (images/s per model at fp16) and recompute this table; adjust `EVAL` size if the projection exceeds the quota.

| Job | Rough T4 time |
|---|---|
| Tests and pilot | 1-1.5 h |
| Exp A, CNNs (`resnet50`, `convnext_tiny`), 8 res x 76 conditions x 2,000 images | 2-3 h |
| Exp A, ViTs (`deit_small`, `deit_base`) | 3-4 h |
| Exp B token sweeps + mechanism analysis | 4-6 h |
| Exp C joint grid | 3-4 h |
| Exp D sensor noise | 1-2 h |
| Selector features + training + eval | 1-2 h |
| Latency tables | 0.5 h |
| Stretch exits | 2-3 h |
| Optional CLIP / DINOv2 / FlexiViT passes | 2-3 h |
| **Total (core, without stretch/optional)** | **about 15-22 h** |

Schedule core jobs across at most two weekly quotas. Split each job into kernels of at most ~3 h, one corruption group per kernel run where possible. Run order: tests, pilot, gates, A, B, C, D, E, latency, then optional/stretch.

## 10. Kaggle execution protocol

For every kernel:
1. Create `kaggle/kernels/<name>/kernel-metadata.json` with: `id` (`<username>/<slug>`), `title`, `code_file`, `language: python`, `kernel_type: script`, `is_private: true`, `enable_gpu: true`, `enable_internet: true`, `dataset_sources` (the code dataset and any data dataset), `competition_sources` if used.
2. Ship the repo to the kernel as a private Kaggle dataset (`kaggle datasets create` / `kaggle datasets version`) so the kernel script does `pip install -e` from it. Re-version after every code change and record the version number in the run metadata.
3. Push: `kaggle kernels push -p kaggle/kernels/<name>`. Poll: `kaggle kernels status <user>/<slug>` every 5 minutes. On completion: `kaggle kernels output <user>/<slug> -p results/raw/<name>/`.
4. Verify downloaded output: row counts equal expected conditions, no duplicate keys, `run_meta.json` shows the GPU name. If a kernel failed or timed out, read the log, fix, and re-run; the resume logic skips finished conditions.
5. Merge raw CSVs into `results/all.csv` with a script. Never hand-edit results.
6. Log every kernel run (name, version, GPU, start, end, rows) in `results/run_log.csv`.

## 11. Results CSV schema (one row per observation, no aggregation)

`run_id, git_hash, gpu, model, weights_tag, knob_type, resolution, resize_op, bn_recal, tome_r, prune_keep, exit_layer, gflops, corruption, group, severity, protocol, image_id, label, pred, correct, top5_correct, conf, latency_ms (nullable), notes`

Per-image rows for the main grids can be bit-packed into per-condition files if size becomes a problem, but the schema must still allow paired tests.

## 12. Definition of done

- All sanity tests pass and are in the repo.
- Gates A and B are documented before the full runs, with the criteria unchanged.
- Tables T1-T6 and figures F1-F7 regenerate from `results/all.csv` with one command.
- `report/findings.md` answers all three questions with CIs, states which gate outcomes occurred, lists limitations, and states clearly what is **not** claimed.
- The README reproduces any phase with a single command and states the Kaggle quota used.
- No credential appears anywhere in the repo, logs or outputs (run a grep for the key string and the username pattern before finishing).

## 13. Guardrails for the agent

- Do not invent numbers, dataset names or checkpoint names. If a named resource does not exist, search for the nearest real one, record the substitution, and flag it in the report.
- Any claim in the report must trace to a row in `results/all.csv`. Do not copy expected effect sizes from this brief (none are given on purpose).
- Do not tune thresholds, resolutions or selectors on `EVAL` or `EVAL_BIG`. Fit only on `SEL_TRAIN` and held-out corruption groups.
- If a result is surprising (for example a large gain from lower resolution), first look for a bug or a mismatch artifact: label order, double resizing, antialias flag, BN mode, wrong checkpoint preprocessing. Then report it with the checks you ran.
- Stop and report instead of improvising if: clean accuracy fails test 3, the GPU quota projection exceeds 90%, or a gate fails and the pivot changes the study scope.
