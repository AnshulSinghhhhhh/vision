"""Tests for statistical validation, DiD, Holm correction, and TOST."""

import numpy as np
import pytest
from knobs.stats import (
    paired_bootstrap_ci,
    compute_did_bootstrap,
    paired_did_ztest,
    holm_bonferroni_correction,
    holm_adjust,
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


def test_did_sign_and_ci():
    """Verifies that synthetic arrays with known negative DiD yield a negative point estimate inside CI."""
    n = 200
    c448 = np.ones(n, dtype=int)
    c224 = np.ones(n, dtype=int)
    d448 = np.ones(n, dtype=int)
    d448[:50] = 0  # 50 degraded errors at 448
    d224 = np.ones(n, dtype=int)
    
    # 1. Test bootstrap
    boot_res = compute_did_bootstrap(c448, c224, d448, d224, n_resamples=1000, seed=42)
    assert boot_res["did_point_pp"] < 0.0
    assert abs(boot_res["did_point_pp"] - (-25.0)) < 1e-5
    assert boot_res["ci_lower_pp"] <= boot_res["did_point_pp"] <= boot_res["ci_upper_pp"]
    
    # 2. Test analytic z-test
    z_res = paired_did_ztest(c448, c224, d448, d224)
    assert z_res.did_pp < 0.0
    assert abs(z_res.did_pp - (-25.0)) < 1e-5
    assert z_res.ci_lo <= z_res.did_pp <= z_res.ci_hi
    assert z_res.z < 0.0
    assert z_res.p_two_sided < 0.001
    
    # Verify dict-like and tuple unpacking behavior
    did_pp, se_pp, ci_lo, ci_hi, z_val, p_val = z_res
    assert did_pp == z_res["did_pp"]
    assert se_pp == z_res["se_pp"]


def test_holm_bonferroni_hand_computed():
    """Verifies Holm-Bonferroni on a hand-computed 4-test list."""
    raw_p = [0.01, 0.04, 0.03, 0.005]
    adj_p = holm_bonferroni_correction(raw_p)
    expected = [0.03, 0.06, 0.06, 0.02]
    for a, e in zip(adj_p, expected):
        assert abs(a - e) < 1e-6
        
    # Verify holm_adjust alias
    assert holm_adjust(raw_p) == adj_p
