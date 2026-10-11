"""Token-budget analysis from EXISTING raw shards (no GPU).

Run from the repo root:   python analysis/token_budget.py [out_csv]
Reads results/raw/{k2-pilot,k3-headline-a,k4-headline-b,k5-controls,k12-tome-matched,k15-flexivit-grid}.
Writes analysis/out/table_token_budget.csv (or out_csv).

Every row is a paired, per-image contrast with a normal-approximation 95% CI.
Sign convention: DiD = (acc448 - acc224)_corrupted - (acc448 - acc224)_clean.
Negative = the higher resolution hurts more under corruption than on clean images.
Images missing any needed cell are dropped before computing N and the CI.
"""
import glob
import sys

import numpy as np
import pandas as pd

R = "results/raw/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "analysis/out/table_token_budget.csv"
rows = []


def add(exp, contrast, cond, x):
    x = x.dropna()
    rows.append(dict(experiment=exp, contrast=contrast, condition=cond, N=len(x),
                     estimate_pp=round(float(x.mean() * 100), 2),
                     ci95_halfwidth_pp=round(float(1.96 * x.std(ddof=1) / np.sqrt(len(x)) * 100), 2)))


# ---------------- headline shards (k2-k5) ----------------
cols = ["image_id", "condition", "severity", "model", "arm", "resolution", "correct"]
H = pd.concat([pd.read_parquet(f, columns=cols) for k in ["k2-pilot", "k3-headline-a", "k4-headline-b", "k5-controls"]
               for f in glob.glob(R + k + "/shards/*.parquet")], ignore_index=True)
H = H.drop_duplicates(["image_id", "condition", "severity", "model", "arm", "resolution"])
H["c"] = H.correct.astype(float)


def S(m, arm, cond, sev, res):
    x = H[(H.model == m) & (H.arm == arm) & (H.condition == cond) & (H.severity == sev) & (H.resolution == res)]
    return x.set_index("image_id").c


def did(m, arm, cond, sev, hi=448, lo=224):
    df = pd.concat([S(m, arm, cond, sev, hi), S(m, arm, cond, sev, lo),
                    S(m, arm, "clean", 0, hi), S(m, arm, "clean", 0, lo)], axis=1, keys=list("abce"))
    return (df.a - df.b) - (df.c - df.e)


CONDS = [("gaussian_noise", 1), ("gaussian_noise", 3), ("gaussian_noise", 5), ("jpeg_compression", 3), ("contrast", 3)]
for cond, sev in CONDS:
    p, t = did("flexivit_base", "F-p", cond, sev), did("flexivit_base", "F-t", cond, sev)
    c = f"{cond}_s{sev}"
    add("FlexiViT fixed patch vs fixed tokens", "DiD F-p", c, p.where(t.notna()))
    add("FlexiViT fixed patch vs fixed tokens", "DiD F-t (256 tokens at every res)", c, t.where(p.notna()))
    add("FlexiViT fixed patch vs fixed tokens", "F-t minus F-p", c, t - p)
    s = did("deit_base", "standard", cond, sev)
    for arm, m in [("tome_r4", "deit_base_tome_r4"), ("tome_r8", "deit_base_tome_r8")]:
        tm = did(m, arm, cond, sev)
        add("ToMe inside the network (DeiT-B)", f"DiD ToMe-{arm[-2:]} minus DiD standard", c, tm - s)
    # DeiT trained at 384 vs interpolated, test at 384 (reference 224)
    def did384(m, arm):
        df = pd.concat([S(m, arm, cond, sev, 384 if m == "deit_base_384" else 384), S("deit_base", "standard", cond, sev, 224),
                        S(m, arm, "clean", 0, 384), S("deit_base", "standard", "clean", 0, 224)], axis=1, keys=list("abce"))
        return (df.a - df.b) - (df.c - df.e)
    nat, itp = did384("deit_base_384", "native_384"), did384("deit_base", "standard")
    add("Scale mismatch (DeiT-B, test 384 vs 224)", "DiD native-384 checkpoint", c, nat.where(itp.notna()))
    add("Scale mismatch (DeiT-B, test 384 vs 224)", "DiD 224 checkpoint interpolated to 384", c, itp.where(nat.notna()))
    add("Scale mismatch (DeiT-B, test 384 vs 224)", "native minus interpolated", c, nat - itp)

# ---------------- K12: ToMe vs resolution at matched FLOPs ----------------
K = pd.read_parquet(R + "k12-tome-matched/shard_tome_matched.parquet")
K["c"] = K.correct.astype(float)


def Sk(cfg, cond, sev):
    return K[(K.config_name == cfg) & (K.condition == cond) & (K.severity == sev)].set_index("image_id").c


for cfg in ["tome_r32_448", "tome_r64_448"]:
    for cond, sev in [("clean", 0), ("gaussian_noise", 3), ("gaussian_noise", 5), ("defocus_blur", 3)]:
        add("ToMe at 448 vs plain 448 (K12)", f"{cfg} minus res_448_base", f"{cond}_s{sev}", Sk(cfg, cond, sev) - Sk("res_448_base", cond, sev))
for lo, tm in [("res_384_base", "tome_r32_448"), ("res_320_base", "tome_r64_448")]:
    for cond, sev in [("gaussian_noise", 3), ("gaussian_noise", 5)]:
        gap_deg = Sk(lo, cond, sev) - Sk(tm, cond, sev)
        gap_clean = Sk(lo, "clean", 0) - Sk(tm, "clean", 0)
        add("Matched FLOPs (K12)", f"{lo} minus {tm}, minus clean gap", f"{cond}_s{sev}", gap_deg - gap_clean)

# ---------------- K15: FlexiViT resolution x patch size ----------------
G = pd.read_parquet(R + "k15-flexivit-grid/shard_flexivit_grid.parquet")
G["c"] = G.correct.astype(float)


def Sg(r, p, cond, sev):
    return G[(G.resolution == r) & (G.patch_size == p) & (G.condition == cond) & (G.severity == sev)].set_index("image_id").c


for cond, sev in [("clean", 0), ("gaussian_noise", 3), ("gaussian_noise", 5), ("defocus_blur", 3)]:
    c = f"{cond}_s{sev}"
    add("FlexiViT patch grid (K15)", "patch32@448 minus patch16@448 (same pixels)", c, Sg(448, 32, cond, sev) - Sg(448, 16, cond, sev))
    add("FlexiViT patch grid (K15)", "patch32@448 minus patch16@224 (same 196 tokens)", c, Sg(448, 32, cond, sev) - Sg(224, 16, cond, sev))
for p in (16, 32):
    for cond, sev in [("gaussian_noise", 3), ("gaussian_noise", 5)]:
        df = pd.concat([Sg(448, p, cond, sev), Sg(224, p, cond, sev), Sg(448, p, "clean", 0), Sg(224, p, "clean", 0)], axis=1, keys=list("abce"))
        add("FlexiViT patch grid (K15)", f"DiD patch {p}", f"{cond}_s{sev}", (df.a - df.b) - (df.c - df.e))

T = pd.DataFrame(rows)
T.to_csv(OUT, index=False)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 70)
print(T.to_string(index=False))
