# Experimental Provenance and Corruption Backend Audit

This document records the exact provenance, environment tracking, and backend consistency status across Kaggle execution kernels in this repository.

## 1. Kernel Provenance Status (k0–k7 vs. k8+)

- **Kernels k0–k7 (Headline Runs)**:
  - `run_meta.json` records `"git_commit": "unknown_git_commit"`.
  - Neither `corruption_backend` nor `imagecorruptions_version` was recorded in the environment metadata.
  - As documented in `docs/CHANGELOG_FIXES.md` (A3), earlier code in `src/knobs/corrupt.py` silently fell back to internal fallback corruption functions if `imagecorruptions` was uninstalled or failed to import.
- **Kernels k8–k13 (Mechanistic & Control Runs)**:
  - `run_meta.json` explicitly records full git commit hashes (e.g. `c1a8cb839339ef57081aa2826a6fc8b27e5569e1-dirty`).
  - Recorded `"corruption_backend": "imagecorruptions"` and `imagecorruptions_version`.
  - Fallback is strictly guarded via `KNOBS_ALLOW_FALLBACK=1`; code raises `RuntimeError` by default if `imagecorruptions` is unavailable.

## 2. Cross-Kernel Corruption Consistency

- **Clean and Gaussian Noise**:
  - Clean and Gaussian-noise evaluations are bit-identical across kernels (same sample images yield 100% per-image prediction agreement).
  - Both fallback and `imagecorruptions` use identical zero-mean Gaussian noise with sigmas $\sigma \in [0.08, 0.12, 0.18, 0.26, 0.38]$.
- **Contrast and JPEG Compression**:
  - The fallback formulas for contrast and JPEG compression match ImageNet-C exactly (`contrast`: linear scaling towards channel mean; `jpeg_compression`: PIL JPEG save/load with quality scale $[25, 18, 15, 10, 7]$).
  - Maximum pixel difference between fallback and `imagecorruptions` outputs for contrast and JPEG is 0.
- **Defocus Blur Discrepancy**:
  - Defocus blur is **not** consistent between headline kernels (k2–k4) and later kernels (k8, k12):
    - On the same 5,000 PILOT images at 224x224 (DeiT-B): headline kernels (k2–k4) reported 76.30% accuracy, whereas k12 (with `imagecorruptions` installed) reported 74.08%. Per-image prediction agreement is only 95.7%.
    - At 448x448 on 2,000 MECH images (K8 vs. headline): DeiT-B is 64.35% (K8) vs. 66.70% (headline); EfficientNet-B3 is 67.55% (K8) vs. 70.40% (headline).
  - Cause: The fallback implementation used `scipy.ndimage.gaussian_filter` (a continuous Gaussian blur), whereas ImageNet-C's real `defocus_blur` uses optical disk point spread function kernels (`cv2.filter2D` with disk kernels).
  - **Status**: Headline defocus rows (k0–k7) probably used the Gaussian-filter fallback rather than ImageNet-C disk defocus, and are therefore treated as **unverified**. Verified ImageNet-C defocus measurements are taken from K8 ($N=2000$), with scaled confirmation planned in k14 ($N=10,000$).
