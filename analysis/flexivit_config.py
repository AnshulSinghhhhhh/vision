"""Model configuration and architecture inspection script.

Inspects pretrained configs and architecture dimensions (patch size, grid size,
positional embedding shapes, interpolation/resampling properties) for:
- DeiT-B/16 (224)
- DeiT-B/16 (384 native)
- EfficientNet-B3
- FlexiViT-B (Arms F-p and F-t across resolutions 224, 320, 384, 448)

Outputs findings and saves a Markdown report to `docs/model_configs.md`.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import timm

from knobs.models import MODEL_TAGS, create_model_instance, get_model_cfg


def get_pretrained_metadata() -> Dict[str, Dict[str, Any]]:
    """Retrieves pretrained config dictionary for each model tag."""
    metadata = {}
    for name, tag in MODEL_TAGS.items():
        cfg = get_model_cfg(tag)
        metadata[name] = {
            "tag": tag,
            "input_size": cfg.get("input_size"),
            "test_input_size": cfg.get("test_input_size"),
            "crop_pct": cfg.get("crop_pct"),
            "test_crop_pct": cfg.get("test_crop_pct"),
            "interpolation": cfg.get("interpolation"),
            "mean": cfg.get("mean"),
            "std": cfg.get("std"),
            "num_classes": cfg.get("num_classes"),
        }
    return metadata


def inspect_flexivit_architectures() -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Inspects FlexiViT patch_size, grid_size, and pos_embed across resolutions."""
    resolutions = [224, 320, 384, 448]
    arms = ["F-p", "F-t"]
    results = {arm: {} for arm in arms}
    for arm in arms:
        for r in resolutions:
            m = create_model_instance("flexivit_base", resolution=r, arm=arm, pretrained=False)
            patch_size = getattr(m.patch_embed, "patch_size", None)
            grid_size = getattr(m.patch_embed, "grid_size", None)
            pos_shape = list(m.pos_embed.shape) if hasattr(m, "pos_embed") and m.pos_embed is not None else None
            results[arm][r] = {
                "patch_size": patch_size,
                "grid_size": grid_size,
                "pos_embed_shape": pos_shape,
                "num_tokens": grid_size[0] * grid_size[1] if grid_size else None,
            }
    return results


def inspect_deit_architectures() -> Dict[int, Dict[str, Any]]:
    """Inspects DeiT-B grid_size and pos_embed across resolutions."""
    resolutions = [224, 320, 384, 448]
    results = {}
    for r in resolutions:
        m = create_model_instance("deit_base", resolution=r, pretrained=False)
        patch_size = getattr(m.patch_embed, "patch_size", None)
        grid_size = getattr(m.patch_embed, "grid_size", None)
        pos_shape = list(m.pos_embed.shape) if hasattr(m, "pos_embed") and m.pos_embed is not None else None
        results[r] = {
            "patch_size": patch_size,
            "grid_size": grid_size,
            "pos_embed_shape": pos_shape,
            "num_tokens": grid_size[0] * grid_size[1] if grid_size else None,
        }
    return results


def generate_markdown_report(
    metadata: Dict[str, Dict[str, Any]],
    flexivit_info: Dict[str, Dict[int, Dict[str, Any]]],
    deit_info: Dict[int, Dict[str, Any]],
) -> str:
    """Formats inspection details into a markdown report."""
    md = []
    md.append("# Model Pretrained Configurations & Architecture Details\n")
    md.append("This document records the exact pretrained configurations, input resolutions, and architecture dimensional adaptations across resolution scales for the models studied.\n")
    
    md.append("## 1. Official Pretrained Configurations (`timm`)\n")
    md.append("| Model Key | Pretrained Tag | Input Size (Train) | Test Input Size | Crop Pct (Train / Test) | Interpolation |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, d in metadata.items():
        crop_str = f"{d['crop_pct']} / {d['test_crop_pct']}" if d.get('test_crop_pct') else f"{d['crop_pct']}"
        md.append(f"| `{name}` | `{d['tag']}` | `{d['input_size']}` | `{d['test_input_size']}` | `{crop_str}` | `{d['interpolation']}` |")
    md.append("\n**Key Pretrained Observations:**")
    md.append("- **EfficientNet-B3 (`efficientnet_b3.ra2_in1k`)**: Native training resolution is 288x288 (`crop_pct=0.875`), but official `test_input_size` is **(3, 320, 320)** with `test_crop_pct=1.0`. The paper text 'approx 300-320' corresponds precisely to 288 train / 320 test.")
    md.append("- **FlexiViT-B (`flexivit_base.1200ep_in1k`)**: Native training resolution is **240x240** (`crop_pct=0.95`). With native patch size 16x16, the native training grid is **15x15 = 225 tokens**.")
    md.append("- **DeiT-B/16 (`deit_base_patch16_224.fb_in1k`)**: Native training resolution is 224x224 with 14x14 = 196 tokens (`crop_pct=0.9`).")
    md.append("- **DeiT-B/16-384 (`deit_base_patch16_384.fb_in1k`)**: Fine-tuned natively at 384x384 with 24x24 = 576 tokens (`crop_pct=1.0`).\n")

    md.append("## 2. FlexiViT-B Resolution Arms (F-p vs F-t)\n")
    md.append("FlexiViT adapts to varying resolutions through two distinct operational regimes:\n")
    md.append("- **Arm F-p (Constant Patch Size = 16)**: Patch size remains 16x16. As resolution increases, the token grid grows. Positional embeddings are bicubically interpolated from native 15x15 (225 tokens).")
    md.append("- **Arm F-t (Constant Token Count = 256)**: Grid size is fixed at 16x16. Patch size scales with resolution ($p = R / 16$). Patch embedding weights are resampled (PI-resize), and positional embeddings are interpolated from 15x15 to 16x16.\n")
    
    md.append("| Arm | Resolution | Patch Size | Grid Size | Token Count | Pos Embed Shape | Confounds & Operations |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for arm in ["F-p", "F-t"]:
        for r, info in flexivit_info[arm].items():
            confound = "Fixed weights (16x16); pos_embed interpolated" if arm == "F-p" else "Patch weights resampled (PI-resize); fixed 16x16 grid"
            md.append(f"| `{arm}` | {r} | `{info['patch_size']}` | `{info['grid_size']}` | {info['num_tokens']} | `{info['pos_embed_shape']}` | {confound} |")
    
    md.append("\n## 3. DeiT-B Dynamic Resolution Adaptation\n")
    md.append("DeiT-B keeps patch size fixed at 16x16 and bicubically interpolates its positional embeddings (native 14x14 = 196 tokens):\n")
    md.append("| Resolution | Patch Size | Grid Size | Token Count | Pos Embed Shape |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for r, info in deit_info.items():
        md.append(f"| {r} | `{info['patch_size']}` | `{info['grid_size']}` | {info['num_tokens']} | `{info['pos_embed_shape']}` |")

    return "\n".join(md) + "\n"


def main():
    print("Inspecting pretrained model metadata...")
    meta = get_pretrained_metadata()
    print("Inspecting FlexiViT architecture...")
    flexivit_info = inspect_flexivit_architectures()
    print("Inspecting DeiT architecture...")
    deit_info = inspect_deit_architectures()

    md_content = generate_markdown_report(meta, flexivit_info, deit_info)
    out_path = Path("docs/model_configs.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"Wrote model configs report to {out_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
