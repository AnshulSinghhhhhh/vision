# Model Pretrained Configurations & Architecture Details

This document records the exact pretrained configurations, input resolutions, architecture dimensional adaptations across resolution scales, and acquisition preprocessing pipeline for the models studied.

## 1. Official Pretrained Configurations (`timm`)

| Model Key | Pretrained Tag | Input Size (Train) | Test Input Size | Crop Pct (Train / Test) | Interpolation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `deit_base` | `deit_base_patch16_224.fb_in1k` | `(3, 224, 224)` | `None` | `0.9` | `bicubic` |
| `deit_base_384` | `deit_base_patch16_384.fb_in1k` | `(3, 384, 384)` | `None` | `1.0` | `bicubic` |
| `efficientnet_b3` | `efficientnet_b3.ra2_in1k` | `(3, 288, 288)` | `(3, 320, 320)` | `0.875 / 1.0` | `bicubic` |
| `flexivit_base` | `flexivit_base.1200ep_in1k` | `(3, 240, 240)` | `None` | `0.95` | `bicubic` |

**Key Pretrained Observations:**
- **EfficientNet-B3 (`efficientnet_b3.ra2_in1k`)**: Native training resolution is 288x288 (`crop_pct=0.875`), but official `test_input_size` is **(3, 320, 320)** with `test_crop_pct=1.0`. The paper text 'approx 300-320' corresponds precisely to 288 train / 320 test.
- **FlexiViT-B (`flexivit_base.1200ep_in1k`)**: Native training resolution is **240x240** (`crop_pct=0.95`). With native patch size 16x16, the native training grid is **15x15 = 225 tokens**.
- **DeiT-B/16 (`deit_base_patch16_224.fb_in1k`)**: Native training resolution is 224x224 with 14x14 = 196 tokens (`crop_pct=0.9`).
- **DeiT-B/16-384 (`deit_base_patch16_384.fb_in1k`)**: Fine-tuned natively at 384x384 with 24x24 = 576 tokens (`crop_pct=1.0`).

## 2. FlexiViT-B Resolution Arms (F-p vs F-t)

FlexiViT adapts to varying resolutions through two distinct operational regimes:

- **Arm F-p (Constant Patch Size = 16)**: Patch size remains 16x16. As resolution increases, the token grid grows. Positional embeddings are bicubically interpolated from native 15x15 (225 tokens).
- **Arm F-t (Constant Token Count = 256)**: Grid size is fixed at 16x16. Patch size scales with resolution ($p = R / 16$). Patch embedding weights are resampled (PI-resize), and positional embeddings are interpolated from 15x15 to 16x16.

| Arm | Resolution | Patch Size | Grid Size | Patch Count | Total Tokens (+CLS) | Pos Embed Shape | Operations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `F-p` | 224 | `(16, 16)` | `(14, 14)` | 196 | 197 | `[1, 197, 768]` | Fixed patch weights; pos_embed bicubic interpolated |
| `F-p` | 320 | `(16, 16)` | `(20, 20)` | 400 | 401 | `[1, 401, 768]` | Fixed patch weights; pos_embed bicubic interpolated |
| `F-p` | 384 | `(16, 16)` | `(24, 24)` | 576 | 577 | `[1, 577, 768]` | Fixed patch weights; pos_embed bicubic interpolated |
| `F-p` | 448 | `(16, 16)` | `(28, 28)` | 784 | 785 | `[1, 785, 768]` | Fixed patch weights; pos_embed bicubic interpolated |
| `F-t` | 224 | `(14, 14)` | `(16, 16)` | 256 | 257 | `[1, 257, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |
| `F-t` | 320 | `(20, 20)` | `(16, 16)` | 256 | 257 | `[1, 257, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |
| `F-t` | 384 | `(24, 24)` | `(16, 16)` | 256 | 257 | `[1, 257, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |
| `F-t` | 448 | `(28, 28)` | `(16, 16)` | 256 | 257 | `[1, 257, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |

## 3. DeiT-B Dynamic Resolution Adaptation

DeiT-B/16 keeps patch size fixed at 16x16 and bicubically interpolates its positional embeddings (from native 14x14 = 196 tokens):

| Resolution | Patch Size | Grid Size | Patch Count | Total Tokens (+CLS) | Pos Embed Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 224 | `(16, 16)` | `(14, 14)` | 196 | **197** | `[1, 197, 768]` |
| 320 | `(16, 16)` | `(20, 20)` | 400 | **401** | `[1, 401, 768]` |
| 384 | `(16, 16)` | `(24, 24)` | 576 | **577** | `[1, 577, 768]` |
| 448 | `(16, 16)` | `(28, 28)` | 784 | **785** | `[1, 785, 768]` |

At 448 px, DeiT-B/16 processes $28 \times 28 = 784$ image patches plus 1 class token, totaling **785 tokens** with self-attention complexity $\mathcal{O}(785^2)$.

## 4. Acquisition and Resizing Pipeline

The empirical evaluation uses a unified acquisition pipeline:
1. **Acquisition Frame**: Input image short side is resized to 512 px using PIL bilinear interpolation, followed by a center crop of 448x448 px (crop fraction $448/512 = 0.875$).
2. **Deterministic Degradation**: Corruptions (Gaussian noise, defocus blur, JPEG compression, contrast adjustments) are applied directly in this 448x448 acquisition frame using deterministic per-image SHA-256 seeds.
3. **Target Resolution Resizing**: The degraded 448x448 frame is resized to the target model resolution $R \in \{224, 320, 384, 448\}$ using PyTorch `F.interpolate(mode="bilinear", antialias=True)`.
   - At $R = 448$: Resize is the mathematical **identity** (no filtering). Full injected noise passes through unmodified ($\sigma_{\text{eff}} = 1.000 \times \sigma_{\text{inj}}$).
   - At $R < 448$: Downsampling applies an antialiasing triangle lowpass filter, attenuating high-frequency noise ($\sigma_{\text{eff}} = 0.565 \times$ at 384 px, $0.481 \times$ at 320 px, and $0.313 \times$ at 224 px).
