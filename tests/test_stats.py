"""Tests for statistical validation, DiD, Holm correction, and TOST."""

import numpy as np
import pytest
from knobs.stats import (
    paired_bootstrap_ci,
    compute_did_bootstrap,
    holm_bonferroni_correction,
    exact_mcnemar_test,
    tost_equivalence_test,
)


def test_paired_bootstrap_ci():
    """Verifies bootstrap CI coverage on synthetic differences."""
    diffs = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.05, 0.01, 0.03] * 50)
    point, low, high = paired_bootstrap_ci(diffs, n_resamples=1000)
    
    assert low < point < high
    assert abs(point - 2.625) < 0.1


def test_holm_bonferroni_correction():
    """Verifies Holm-Bonferroni step-down multiplier."""
    raw_p = [0.001, 0.02, 0.04, 0.20]
    adj_p = holm_bonferroni_correction(raw_p)
    
    assert len(adj_p) == 4
    # Smallest p is multiplied by 4
    assert abs(adj_p[0] - 0.004) < 1e-4
    # All adjusted p-values are >= raw p-values
    for r, a in zip(raw_p, adj_p):
        assert a >= r


def test_mcnemar_test():
    """Verifies exact McNemar on paired binary outcomes."""
    c1 = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    c2 = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    res = exact_mcnemar_test(c1, c2)
    
    assert "p_value" in res
    assert res["b_discordant"] == 2
    assert res["c_discordant"] == 1


def test_tost_equivalence():
    """Verifies TOST equivalence when difference is well within margin."""
    diffs = np.zeros(200)  # Identical outcomes
    res = tost_equivalence_test(diffs, margin_pp=0.5)
    
    assert res["is_equivalent"] is True
    assert res["tost_p_value"] < 0.05
