"""Statistical validation and hypothesis testing framework.

Features:
- Paired bootstrap confidence intervals (percentage points scale).
- Difference-in-Differences (DiD) paired bootstrap testing.
- Holm-Bonferroni correction for primary family (m=12 tests).
- Exact McNemar test for pairwise resolution comparisons.
- Two One-Sided Tests (TOST) equivalence testing (margin ±0.5 pp).
- Class-clustered bootstrap for hierarchical sensitivity.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes paired bootstrap confidence interval for a 1D array of differences.
    
    Returns: (point_estimate, ci_lower, ci_upper) in percentage points.
    """
    rng = np.random.RandomState(seed)
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    point_est = float(np.mean(diffs) * 100.0)
    boot_means = np.zeros(n_resamples, dtype=np.float64)
    
    # Vectorized bootstrap batches
    batch_size = 1000
    for i in range(0, n_resamples, batch_size):
        curr_batch = min(batch_size, n_resamples - i)
        idx = rng.randint(0, n, size=(curr_batch, n))
        boot_means[i:i + curr_batch] = np.mean(diffs[idx], axis=1) * 100.0
        
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    ci_lower = float(np.percentile(boot_means, low_pct))
    ci_upper = float(np.percentile(boot_means, high_pct))
    
    return point_est, ci_lower, ci_upper


def compute_did_bootstrap(
    correct_clean_448: np.ndarray,
    correct_clean_224: np.ndarray,
    correct_corr_448: np.ndarray,
    correct_corr_224: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes paired Difference-in-Differences:
    DiD = (Acc_448 - Acc_224)_clean - (Acc_448 - Acc_224)_corr
    across paired observations.
    """
    delta_clean = correct_clean_448.astype(np.float64) - correct_clean_224.astype(np.float64)
    delta_corr = correct_corr_448.astype(np.float64) - correct_corr_224.astype(np.float64)
    did_diff = delta_clean - delta_corr  # Individual observation level difference
    
    point_est, ci_low, ci_high = paired_bootstrap_ci(did_diff, n_resamples=n_resamples, seed=seed)
    
    # Calculate empirical two-tailed p-value
    rng = np.random.RandomState(seed)
    n = len(did_diff)
    boot_means = np.mean(did_diff[rng.randint(0, n, size=(n_resamples, n))], axis=1) * 100.0
    p_val = float(2.0 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    p_val = min(1.0, max(1.0 / n_resamples, p_val))
    
    return {
        "did_point_pp": point_est,
        "ci_lower_pp": ci_low,
        "ci_upper_pp": ci_high,
        "p_value": p_val,
    }


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []
        
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    adjusted = np.zeros(m)
    current_max = 0.0
    for k, p in enumerate(sorted_p):
        mult = m - k
        adj_p = min(1.0, p * mult)
        current_max = max(current_max, adj_p)
        adjusted[k] = current_max
        
    # Invert sort back to original order
    orig_order_adjusted = np.zeros(m)
    orig_order_adjusted[sorted_indices] = adjusted
    return [float(x) for x in orig_order_adjusted]


def exact_mcnemar_test(
    correct_1: np.ndarray,
    correct_2: np.ndarray,
) -> Dict[str, Any]:
    """Performs exact McNemar test for paired binary classification outcomes.
    
    Contingency table:
                Model 2 Correct  Model 2 Incorrect
    Model 1 Correct     a              b
    Model 1 Incorrect   c              d
    Discordant pairs: b (1 correct, 2 incorrect), c (1 incorrect, 2 correct)
    """
    b = int(np.sum((correct_1 == 1) & (correct_2 == 0)))
    c = int(np.sum((correct_1 == 0) & (correct_2 == 1)))
    
    n_discordant = b + c
    if n_discordant == 0:
        p_val = 1.0
    else:
        # Binomial test with p=0.5
        p_val = float(stats.binomtest(min(b, c), n=n_discordant, p=0.5).pvalue)
        
    return {
        "b_discordant": b,
        "c_discordant": c,
        "total_discordant": n_discordant,
        "p_value": p_val,
    }


def tost_equivalence_test(
    diffs: np.ndarray,
    margin_pp: float = 0.5,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two One-Sided Tests (TOST) against equivalence margin ±margin_pp in percentage points.
    
    H0_lower: diff <= -margin
    H0_upper: diff >= +margin
    Equivalent if both nulls rejected at level alpha.
    """
    diffs_pp = diffs * 100.0
    mean_diff = float(np.mean(diffs_pp))
    se = float(stats.sem(diffs_pp))
    n = len(diffs)
    
    # t-statistics for the two one-sided tests
    t_lower = (mean_diff - (-margin_pp)) / (se + 1e-12)
    t_upper = (margin_pp - mean_diff) / (se + 1e-12)
    
    df = n - 1
    p_lower = 1.0 - stats.t.cdf(t_lower, df=df)
    p_upper = 1.0 - stats.t.cdf(t_upper, df=df)
    
    tost_p = float(max(p_lower, p_upper))
    is_equivalent = bool(tost_p < alpha)
    
    return {
        "mean_diff_pp": mean_diff,
        "margin_pp": margin_pp,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "tost_p_value": tost_p,
        "is_equivalent": is_equivalent,
    }
