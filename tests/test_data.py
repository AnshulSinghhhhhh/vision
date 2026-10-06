"""Tests for bounding box transforms and split partitioning logic."""

import pytest
from knobs.data import transform_bbox_to_crop448


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
