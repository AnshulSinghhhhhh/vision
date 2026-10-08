"""Tests for bounding box transforms and split partitioning logic."""

import pytest
from knobs.data import transform_bbox_to_crop448, balanced_subset


def test_bbox_transform_to_crop448():
    """Verifies bounding box transformation from original image to 448 center crop."""
    orig_w, orig_h = 600, 400  # min_dim = 400
    # Box in original: xmin=100, ymin=100, xmax=300, ymax=300
    raw_box = (100.0, 100.0, 300.0, 300.0)
    
    transformed = transform_bbox_to_crop448(raw_box, orig_w=orig_w, orig_h=orig_h)
    assert transformed is not None
    
    c_xmin, c_ymin, c_xmax, c_ymax, area_frac = transformed
    assert 0.0 <= c_xmin < c_xmax <= 448.0
    assert 0.0 <= c_ymin < c_ymax <= 448.0
    assert 0.0 < area_frac <= 1.0


def test_bbox_outside_crop():
    """A bounding box completely outside the 448 crop returns None."""
    orig_w, orig_h = 1000, 400  # scaled to 1280 x 512, crop is center [416, 864] horizontally
    raw_box = (10.0, 10.0, 50.0, 50.0)  # On far left edge, outside crop
    
    transformed = transform_bbox_to_crop448(raw_box, orig_w=orig_w, orig_h=orig_h)
    assert transformed is None


def test_balanced_subset():
    """Verifies that balanced_subset returns equal images per class and is deterministic."""
    # Synthetic dataset mimicking ImageNet: 1000 classes, 5 images per class
    ids = []
    meta = {}
    for c in range(1000):
        for i in range(5):
            img_id = f"img_{c}_{i}"
            ids.append(img_id)
            meta[img_id] = {"class_idx": c, "image_id": img_id}
            
    # Draw 2000 images (2 per class)
    sub1 = balanced_subset(ids, meta, n=2000, seed=42)
    assert len(sub1) == 2000
    
    # Check class distribution: exactly 2 per class
    counts = {}
    for img_id in sub1:
        cls_idx = meta[img_id]["class_idx"]
        counts[cls_idx] = counts.get(cls_idx, 0) + 1
    assert len(counts) == 1000
    assert all(cnt == 2 for cnt in counts.values())
    
    # Determinism: same seed yields identical result
    sub2 = balanced_subset(ids, meta, n=2000, seed=42)
    assert sub1 == sub2
    
    # Different seed yields different sample
    sub3 = balanced_subset(ids, meta, n=2000, seed=123)
    assert sub1 != sub3
    assert len(sub3) == 2000
    
    # Test remainder handling: 10 classes, request 25 images
    small_ids = []
    small_meta = {}
    for c in range(10):
        for i in range(5):
            iid = f"s_img_{c}_{i}"
            small_ids.append(iid)
            small_meta[iid] = {"class_idx": c}
    sub_rem = balanced_subset(small_ids, small_meta, n=25, seed=0)
    assert len(sub_rem) == 25
    small_counts = {}
    for iid in sub_rem:
        cls_idx = small_meta[iid]["class_idx"]
        small_counts[cls_idx] = small_counts.get(cls_idx, 0) + 1
    # 5 classes get 3, 5 classes get 2
    assert sorted(list(small_counts.values())) == [2]*5 + [3]*5
