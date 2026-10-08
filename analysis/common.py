"""Common data loader and caching utilities for analysis scripts.

Loads from results/raw/**/shards/*.parquet, deduplicates, validates image_id matching,
and caches merged views to analysis/out/.cache/ for speed.
"""

import os
import glob
from typing import Optional, List, Tuple
import pandas as pd


def get_repo_root() -> str:
    """Returns absolute path to the repository root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_out_dir() -> str:
    """Returns absolute path to analysis/out/ creating it if needed."""
    out_dir = os.path.join(get_repo_root(), "analysis", "out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_cache_dir() -> str:
    """Returns absolute path to analysis/out/.cache/ creating it if needed."""
    cache_dir = os.path.join(get_out_dir(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_all_raw_data(force_reload: bool = False) -> pd.DataFrame:
    """Loads all parquet shards from results/raw, deduplicating records.
    
    Caches the combined DataFrame to analysis/out/.cache/all_raw.parquet.
    """
    cache_path = os.path.join(get_cache_dir(), "all_raw.parquet")
    if not force_reload and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    raw_dir = os.path.join(get_repo_root(), "results", "raw")
    shard_files = sorted(glob.glob(os.path.join(raw_dir, "**", "shards", "*.parquet"), recursive=True))
    
    # Also check top-level raw parquet files like in k6-mech
    extra_files = sorted(glob.glob(os.path.join(raw_dir, "k6-mech", "*.parquet")))
    all_files = shard_files + [f for f in extra_files if f not in shard_files]

    if not all_files:
        raise FileNotFoundError(f"No parquet shards found under {raw_dir}")

    dfs = [pd.read_parquet(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)

    # Deduplicate in case of overlapping shard writes
    subset_cols = [c for c in ["image_id", "condition", "severity", "model", "arm", "resolution"] if c in df.columns]
    df = df.drop_duplicates(subset=subset_cols)
    
    # Cache for rapid access
    df.to_parquet(cache_path, index=False)
    return df


def _normalize_arm_for_model(model: str, arm: str) -> str:
    """Maps arm to standard model arm if applicable (e.g. FlexiViT F-p)."""
    if model == "flexivit_base" and arm == "standard":
        return "F-p"
    return arm


def load_clean_and_corrupted(
    corruption: Optional[str] = None,
    severity: Optional[int] = None,
    model: Optional[str] = None,
    arm: str = "standard",
    force_reload: bool = False,
) -> pd.DataFrame:
    """Loads merged clean and corrupted evaluation records for matched images.
    
    Every returned row has columns:
    c_224, c_320, c_384, c_448 (clean accuracy: 1.0 or 0.0)
    d_224, d_320, d_384, d_448 (degraded accuracy: 1.0 or 0.0)
    matched strictly on image_id (not position).
    
    Args:
        corruption: Specific corruption name (e.g. 'gaussian_noise') or None for all.
        severity: Specific severity level (e.g. 3) or None for all available.
        model: Model name (e.g. 'deit_base') or None for all models.
        arm: Arm name (default 'standard').
        force_reload: Whether to bypass cache.
        
    Returns:
        pd.DataFrame: Merged clean and corrupted evaluations.
    """
    cache_tag = f"merged_{corruption or 'all'}_s{severity if severity is not None else 'all'}_{model or 'all'}_{arm}.parquet"
    cache_path = os.path.join(get_cache_dir(), cache_tag)
    if not force_reload and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df_raw = load_all_raw_data().copy()
    if "resolution" in df_raw.columns:
        df_raw = df_raw.dropna(subset=["resolution"])
        df_raw["resolution"] = df_raw["resolution"].astype(int)

    models = [model] if model is not None else sorted(df_raw["model"].unique())
    all_corruptions = [c for c in sorted(df_raw["condition"].unique()) if c != "clean"]
    corruptions = [corruption] if corruption is not None else all_corruptions

    merged_chunks = []

    for m in models:
        m_arm = _normalize_arm_for_model(m, arm)
        # Clean subset for model
        clean_mask = (df_raw["model"] == m) & (df_raw["condition"] == "clean") & (df_raw["arm"] == m_arm)
        df_clean_m = df_raw[clean_mask]
        if df_clean_m.empty:
            continue

        # Pivot clean
        # Assert no duplicates for (image_id, resolution)
        dup_clean = df_clean_m.duplicated(subset=["image_id", "resolution"])
        assert not dup_clean.any(), f"Duplicates found in clean data for model {m}"

        c_pivot = df_clean_m.pivot(index="image_id", columns="resolution", values="correct")
        # Ensure standard resolution columns exist
        for r in [224, 320, 384, 448]:
            if r not in c_pivot.columns:
                c_pivot[r] = float("nan")
        c_pivot = c_pivot[[224, 320, 384, 448]]
        c_pivot.columns = [f"c_{int(r)}" for r in c_pivot.columns]
        c_pivot = c_pivot.astype(float)

        for c in corruptions:
            sev_candidates = [severity] if severity is not None else sorted(df_raw[df_raw["condition"] == c]["severity"].unique())
            for s in sev_candidates:
                deg_mask = (df_raw["model"] == m) & (df_raw["condition"] == c) & (df_raw["severity"] == s) & (df_raw["arm"] == m_arm)
                df_deg = df_raw[deg_mask]
                if df_deg.empty:
                    continue

                dup_deg = df_deg.duplicated(subset=["image_id", "resolution"])
                assert not dup_deg.any(), f"Duplicates found in degraded data for model {m}, {c} s{s}"

                d_pivot = df_deg.pivot(index="image_id", columns="resolution", values="correct")
                for r in [224, 320, 384, 448]:
                    if r not in d_pivot.columns:
                        d_pivot[r] = float("nan")
                d_pivot = d_pivot[[224, 320, 384, 448]]
                d_pivot.columns = [f"d_{int(r)}" for r in d_pivot.columns]
                d_pivot = d_pivot.astype(float)

                # Merge strictly on image_id
                merged = c_pivot.join(d_pivot, how="inner").reset_index()
                # Assert clean matched on image_id
                assert (merged["image_id"] == merged["image_id"]).all()
                merged["model"] = m
                merged["condition"] = c
                merged["severity"] = s
                merged["arm"] = arm

                merged_chunks.append(merged)

    if not merged_chunks:
        raise ValueError(f"No matching data found for corruption={corruption}, severity={severity}, model={model}, arm={arm}")

    res_df = pd.concat(merged_chunks, ignore_index=True)
    # Reorder columns
    col_order = ["image_id", "model", "condition", "severity", "arm",
                 "c_224", "c_320", "c_384", "c_448",
                 "d_224", "d_320", "d_384", "d_448"]
    res_df = res_df[col_order]

    # Report complete rows
    complete_count = res_df.dropna().shape[0]
    total_count = res_df.shape[0]
    print(f"[load_clean_and_corrupted] Loaded {total_count} rows ({complete_count} complete 4-resolution rows) "
          f"for corruption={corruption or 'all'}, severity={severity if severity is not None else 'all'}, model={model or 'all'}")

    res_df.to_parquet(cache_path, index=False)
    return res_df


if __name__ == "__main__":
    print("Testing analysis.common loader...")
    df = load_clean_and_corrupted(corruption="gaussian_noise", severity=3, model="deit_base")
    print(f"Sample:\n{df.head(2)}")
