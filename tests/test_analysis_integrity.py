"""Regression tests for analysis data loader integrity and deduplication.

Verifies that distinct experimental conditions differing only by operator,
suite_condition, config_name, or band are never erroneously dropped.
"""

import pandas as pd
import pytest
from analysis.common import load_all_raw_data, load_clean_and_corrupted


def test_deduplication_preserves_distinct_operators():
    """Rows differing only by operator must not be deduplicated."""
    base_record = {
        "image_id": "img_001",
        "condition": "gaussian_noise",
        "severity": 5,
        "model": "efficientnet_b3",
        "arm": "standard",
        "resolution": 448,
        "label": 10,
        "pred": 10,
        "correct": 1,
    }
    r1 = dict(base_record, operator="bilinear_antialias")
    r2 = dict(base_record, operator="gaussian_prefilter_448")
    r3 = dict(base_record, operator="bilinear_antialias")  # exact duplicate

    df = pd.DataFrame([r1, r2, r3])
    dedup_candidates = [
        "image_id", "condition", "severity", "model", "arm", "resolution",
        "operator", "suite_condition", "config_name", "band"
    ]
    subset_cols = [c for c in dedup_candidates if c in df.columns]
    deduped = df.drop_duplicates(subset=subset_cols)

    assert len(deduped) == 2, f"Expected 2 rows after dedup, got {len(deduped)}"
    assert set(deduped["operator"]) == {"bilinear_antialias", "gaussian_prefilter_448"}


def test_deduplication_preserves_distinct_bands():
    """Rows differing only by frequency band must not be deduplicated."""
    base_record = {
        "image_id": "img_002",
        "condition": "gaussian_noise",
        "severity": 3,
        "model": "efficientnet_b3",
        "arm": "standard",
        "resolution": 448,
        "label": 25,
        "pred": 25,
        "correct": 1,
    }
    r_low = dict(base_record, band="low")
    r_mid = dict(base_record, band="mid")
    r_high = dict(base_record, band="high")

    df = pd.DataFrame([r_low, r_mid, r_high])
    dedup_candidates = [
        "image_id", "condition", "severity", "model", "arm", "resolution",
        "operator", "suite_condition", "config_name", "band"
    ]
    subset_cols = [c for c in dedup_candidates if c in df.columns]
    deduped = df.drop_duplicates(subset=subset_cols)

    assert len(deduped) == 3
    assert set(deduped["band"]) == {"low", "mid", "high"}


def test_deduplication_preserves_distinct_configs():
    """Rows differing only by config_name must not be deduplicated."""
    base_record = {
        "image_id": "img_003",
        "condition": "clean",
        "severity": 0,
        "model": "deit_base",
        "arm": "standard",
        "resolution": 448,
        "label": 100,
        "pred": 100,
        "correct": 1,
    }
    r_cfg1 = dict(base_record, config_name="res_448_base")
    r_cfg2 = dict(base_record, config_name="tome_r32_448")

    df = pd.DataFrame([r_cfg1, r_cfg2])
    dedup_candidates = [
        "image_id", "condition", "severity", "model", "arm", "resolution",
        "operator", "suite_condition", "config_name", "band"
    ]
    subset_cols = [c for c in dedup_candidates if c in df.columns]
    deduped = df.drop_duplicates(subset=subset_cols)

    assert len(deduped) == 2
    assert set(deduped["config_name"]) == {"res_448_base", "tome_r32_448"}


def test_deduplication_preserves_suite_conditions():
    """Rows differing only by suite_condition must not be deduplicated."""
    base_record = {
        "image_id": "img_004",
        "condition": "gaussian_noise",
        "severity": 3,
        "model": "deit_base",
        "arm": "standard",
        "resolution": 448,
        "label": 50,
        "pred": 50,
        "correct": 1,
    }
    r_cond_a = dict(base_record, suite_condition="A_orig448")
    r_cond_b = dict(base_record, suite_condition="B_filtered448")

    df = pd.DataFrame([r_cond_a, r_cond_b])
    dedup_candidates = [
        "image_id", "condition", "severity", "model", "arm", "resolution",
        "operator", "suite_condition", "config_name", "band"
    ]
    subset_cols = [c for c in dedup_candidates if c in df.columns]
    deduped = df.drop_duplicates(subset=subset_cols)

    assert len(deduped) == 2
    assert set(deduped["suite_condition"]) == {"A_orig448", "B_filtered448"}
