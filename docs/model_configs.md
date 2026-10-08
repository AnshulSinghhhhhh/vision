# Model Pretrained Configurations & Architecture Details

This document records the exact pretrained configurations, input resolutions, and architecture dimensional adaptations across resolution scales for the models studied.

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

| Arm | Resolution | Patch Size | Grid Size | Token Count | Pos Embed Shape | Confounds & Operations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `F-p` | 224 | `(16, 16)` | `(14, 14)` | 196 | `[1, 196, 768]` | Fixed weights (16x16); pos_embed interpolated |
| `F-p` | 320 | `(16, 16)` | `(20, 20)` | 400 | `[1, 400, 768]` | Fixed weights (16x16); pos_embed interpolated |
| `F-p` | 384 | `(16, 16)` | `(24, 24)` | 576 | `[1, 576, 768]` | Fixed weights (16x16); pos_embed interpolated |
| `F-p` | 448 | `(16, 16)` | `(28, 28)` | 784 | `[1, 784, 768]` | Fixed weights (16x16); pos_embed interpolated |
| `F-t` | 224 | `(14, 14)` | `(16, 16)` | 256 | `[1, 256, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |
| `F-t` | 320 | `(20, 20)` | `(16, 16)` | 256 | `[1, 256, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |
| `F-t` | 384 | `(24, 24)` | `(16, 16)` | 256 | `[1, 256, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |
| `F-t` | 448 | `(28, 28)` | `(16, 16)` | 256 | `[1, 256, 768]` | Patch weights resampled (PI-resize); fixed 16x16 grid |

## 3. DeiT-B Dynamic Resolution Adaptation

DeiT-B keeps patch size fixed at 16x16 and bicubically interpolates its positional embeddings (native 14x14 = 196 tokens):

| Resolution | Patch Size | Grid Size | Token Count | Pos Embed Shape |
| :--- | :--- | :--- | :--- | :--- |
| 224 | `(16, 16)` | `(14, 14)` | 196 | `[1, 197, 768]` |
| 320 | `(16, 16)` | `(14, 14)` | 196 | `[1, 197, 768]` |
| 384 | `(16, 16)` | `(14, 14)` | 196 | `[1, 197, 768]` |
| 448 | `(16, 16)` | `(14, 14)` | 196 | `[1, 197, 768]` |
