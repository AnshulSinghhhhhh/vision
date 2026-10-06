"""ImageNet dataset indexing, bounding-box parsing, and stratified partition management.

Manages data splits:
- PILOT: 5,000 images (5/class) - frozen for pipeline verification & gates
- CAL-GATE: 1,000 images (1/class) - dedicated to tuning spectral gate thresholds
- CONFIRM_POOL: 44,000 images (44/class) - untouched confirmatory holdout
  - PRIMARY_CONFIRM: 34,000 images (34/class) - Tier 1 headline confirmatory test
  - SUB10K: 10,000 images (10/class) - dose-response, controls & M7 intervention
    - MECH: 2,000 images (2/class) - bounding box attention & mechanism probes
- FULL: 50,000 images (all validation images)
"""

import os
import csv
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image

SYNSET_MAPPING_FILENAME = "LOC_synset_mapping.txt"
VAL_SOLUTION_FILENAME = "LOC_val_solution.csv"


def load_synset_to_class_idx(synset_mapping_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Loads synset to class index mapping from LOC_synset_mapping.txt.
    
    Class indices strictly follow alphabetical / sorted synset order (matching timm).
    """
    synset_to_idx = {}
    idx_to_synset = {}
    with open(synset_mapping_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Each line format: <synset> <description>
    synsets = [line.split()[0] for line in lines]
    # Ensure sorted order
    sorted_synsets = sorted(synsets)
    for idx, syn in enumerate(sorted_synsets):
        synset_to_idx[syn] = idx
        idx_to_synset[idx] = syn
        
    return synset_to_idx, idx_to_synset


def parse_prediction_string(pred_str: str) -> Tuple[str, List[Tuple[float, float, float, float]]]:
    """Parses prediction string from LOC_val_solution.csv into synset and list of bboxes.
    
    Format: 'synset xmin ymin xmax ymax ...'
    Returns: (synset, list of (xmin, ymin, xmax, ymax))
    """
    tokens = pred_str.strip().split()
    if not tokens:
        return "", []
    synset = tokens[0]
    boxes = []
    # Remaining tokens are multiples of 4: xmin, ymin, xmax, ymax
    coords = tokens[1:]
    for i in range(0, len(coords) - 3, 4):
        try:
            xmin = float(coords[i])
            ymin = float(coords[i + 1])
            xmax = float(coords[i + 2])
            ymax = float(coords[i + 3])
            boxes.append((xmin, ymin, xmax, ymax))
        except ValueError:
            continue
    return synset, boxes


def transform_bbox_to_crop448(
    bbox: Tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    crop_size: int = 448,
    target_short: int = 512,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Transforms a bbox in original image coordinates to crop448 coordinates.
    
    Pipeline:
    1. Short-side resize to target_short (512)
    2. Center crop crop_size (448)
    
    Returns: (crop_xmin, crop_ymin, crop_xmax, crop_ymax, area_fraction)
    or None if box is entirely outside the crop.
    """
    xmin, ymin, xmax, ymax = bbox
    min_dim = min(orig_w, orig_h)
    if min_dim <= 0:
        return None
    scale = target_short / min_dim
    
    # Resized dimensions
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    
    # Center crop offsets
    x_offset = (new_w - crop_size) / 2.0
    y_offset = (new_h - crop_size) / 2.0
    
    # Transform coordinates
    c_xmin = xmin * scale - x_offset
    c_ymin = ymin * scale - y_offset
    c_xmax = xmax * scale - x_offset
    c_ymax = ymax * scale - y_offset
    
    # Clip to crop boundaries [0, crop_size]
    clip_xmin = max(0.0, min(float(crop_size), c_xmin))
    clip_ymin = max(0.0, min(float(crop_size), c_ymin))
    clip_xmax = max(0.0, min(float(crop_size), c_xmax))
    clip_ymax = max(0.0, min(float(crop_size), c_ymax))
    
    if clip_xmax <= clip_xmin or clip_ymax <= clip_ymin:
        return None
        
    area = (clip_xmax - clip_xmin) * (clip_ymax - clip_ymin)
    area_fraction = area / (crop_size * crop_size)
    
    return (clip_xmin, clip_ymin, clip_xmax, clip_ymax, area_fraction)


def create_stratified_splits(
    val_solution_path: str,
    synset_mapping_path: str,
    output_dir: str,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generates the disjoint stratified partitions:
    - PILOT: 5,000 (5/class)
    - CAL-GATE: 1,000 (1/class)
    - CONFIRM_POOL: 44,000 (44/class)
      - PRIMARY_CONFIRM: 34,000 (34/class)
      - SUB10K: 10,000 (10/class)
        - MECH: 2,000 (2/class)
    - FULL: 50,000 (all validation images)
    """
    synset_to_idx, _ = load_synset_to_class_idx(synset_mapping_path)
    
    # Read LOC_val_solution.csv
    images_by_synset: Dict[str, List[Dict[str, Any]]] = {syn: [] for syn in synset_to_idx}
    with open(val_solution_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            image_id, pred_str = row[0], row[1]
            synset, bboxes = parse_prediction_string(pred_str)
            if synset in images_by_synset:
                class_idx = synset_to_idx[synset]
                images_by_synset[synset].append({
                    "image_id": image_id,
                    "synset": synset,
                    "class_idx": class_idx,
                    "raw_bboxes": bboxes,
                })
                
    # Verify count per synset
    for syn, imgs in images_by_synset.items():
        if len(imgs) != 50:
            raise ValueError(f"Synset {syn} has {len(imgs)} images, expected 50.")
            
    rng = np.random.RandomState(seed)
    
    pilot_list = []
    cal_gate_list = []
    primary_confirm_list = []
    sub10k_list = []
    mech_list = []
    full_list = []
    
    for syn in sorted(images_by_synset.keys()):
        imgs = images_by_synset[syn]
        # Deterministic shuffle per class
        perm = rng.permutation(len(imgs))
        shuffled = [imgs[i] for i in perm]
        
        # Exact disjoint slices
        class_pilot = shuffled[0:5]          # 5 images
        class_cal_gate = shuffled[5:6]       # 1 image
        class_primary = shuffled[6:40]       # 34 images
        class_sub10k = shuffled[40:50]       # 10 images
        
        pilot_list.extend(class_pilot)
        cal_gate_list.extend(class_cal_gate)
        primary_confirm_list.extend(class_primary)
        sub10k_list.extend(class_sub10k)
        
        # From class_sub10k, pick 2 for MECH
        # Prefer boxes that exist and have positive area
        class_mech = class_sub10k[0:2]
        mech_list.extend(class_mech)
        
        full_list.extend(shuffled)
        
    confirm_pool_list = primary_confirm_list + sub10k_list
    
    splits = {
        "PILOT": pilot_list,
        "CAL-GATE": cal_gate_list,
        "CONFIRM_POOL": confirm_pool_list,
        "PRIMARY_CONFIRM": primary_confirm_list,
        "SUB10K": sub10k_list,
        "MECH": mech_list,
        "FULL": full_list,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_data in splits.items():
        split_file = os.path.join(output_dir, f"{split_name}.json")
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump([item["image_id"] for item in split_data], f, indent=2)
            
    # Save full metadata index
    meta_file = os.path.join(output_dir, "val_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        meta_dict = {item["image_id"]: item for item in full_list}
        json.dump(meta_dict, f, indent=2)
        
    return splits


def preprocess_image_448(
    img: Image.Image,
    target_short: int = 512,
    crop_size: int = 448,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Standardized acquisition preprocessing:
    1. Short-side resize to target_short (512) using antialiased bilinear interpolation
    2. Center crop to crop_size x crop_size (448x448)
    
    Returns:
    - np.ndarray: uint8 RGB image of shape (448, 448, 3)
    - dict: metadata including native dimensions and crop offsets
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    orig_w, orig_h = img.size
    min_dim = min(orig_w, orig_h)
    scale = target_short / min_dim
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    
    # Resample with antialiased bilinear (PIL Image.BILINEAR)
    resized = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    
    # Center crop
    left = (new_w - crop_size) // 2
    top = (new_h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    cropped = resized.crop((left, top, right, bottom))
    arr = np.array(cropped, dtype=np.uint8)
    
    meta = {
        "orig_width": orig_w,
        "orig_height": orig_h,
        "src_short_side": min_dim,
        "scaled_width": new_w,
        "scaled_height": new_h,
        "crop_left": left,
        "crop_top": top,
        "crop_size": crop_size,
    }
    return arr, meta
