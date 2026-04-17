import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


def _default_results_csv() -> Path:
    data_dir = Path(__file__).resolve().parents[1] / "Data" / "aim_data"
    candidates = sorted(data_dir.glob("*_run_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_run_results_*.csv found in {data_dir}")
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=_default_results_csv())
    return parser.parse_args()


args = _parse_args()
df = pd.read_csv(args.csv)

df["ci"] = pd.to_numeric(df["confidence_index"], errors="coerce")

uc_order = [
    "Low uncertainty (high confidence)",
    "Moderate uncertainty",
    "High uncertainty",
]


def match_rate(series):
    valid = series.dropna()
    if len(valid) == 0:
        return "n/a", 0, 0
    n_match = int(valid.sum())
    n_total = len(valid)
    pct = valid.mean() * 100
    return f"{n_match}/{n_total} = {pct:.1f}%", n_match, n_total


# ---- 1. By uncertainty_class ----
print("=" * 60)
print("1. By uncertainty_class")
print("=" * 60)
for uc in uc_order:
    sub = df[df["uncertainty_class"] == uc]
    n = len(sub)
    ci_med = sub["ci"].median()
    b_str, _, _ = match_rate(sub["baseline_qc_ecological_site_match"])
    t_str, _, _ = match_rate(sub["terrain_qc_ecological_site_match"])
    print(f"\n  {uc} (n={n}, median CI={ci_med:.1f})")
    print(f"    baseline_qc ecosite match : {b_str}")
    print(f"    terrain_qc  ecosite match : {t_str}")

# ---- 2. By confidence_index quartile ----
print()
print("=" * 60)
print("2. By confidence_index quartile")
print("=" * 60)
df["ci_q"] = pd.qcut(df["ci"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
for q in ["Q1", "Q2", "Q3", "Q4"]:
    sub = df[df["ci_q"] == q]
    n = len(sub)
    ci_min = sub["ci"].min()
    ci_max = sub["ci"].max()
    b_str, _, _ = match_rate(sub["baseline_qc_ecological_site_match"])
    t_str, _, _ = match_rate(sub["terrain_qc_ecological_site_match"])
    print(f"\n  {q} CI={ci_min:.1f}-{ci_max:.1f} (n={n})")
    print(f"    baseline_qc ecosite match : {b_str}")
    print(f"    terrain_qc  ecosite match : {t_str}")

# ---- 3. Spearman correlation ----
print()
print("=" * 60)
print("3. Spearman correlation: confidence_index vs match")
print("=" * 60)
for col in ["baseline_qc_ecological_site_match", "terrain_qc_ecological_site_match"]:
    sub = df[["ci", col]].dropna()
    r, p = stats.spearmanr(sub["ci"], sub[col].astype(float))
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {col}")
    print(f"    Spearman r={r:.3f}, p={p:.4f} {sig} (n={len(sub)})")

# ---- 4. terrain_qc changed rows only ----
print()
print("=" * 60)
print("4. terrain_qc changed rows (landscape_class_qc_changed=True)")
print("=" * 60)
changed = df[df["landscape_class_qc_changed"] == True]
print(f"  n_changed: {len(changed)}")
for uc in uc_order:
    sub = changed[changed["uncertainty_class"] == uc]
    if len(sub) == 0:
        continue
    t_str, _, _ = match_rate(sub["terrain_qc_ecological_site_match"])
    ci_med = sub["ci"].median()
    print(f"  {uc} (n={len(sub)}, median CI={ci_med:.1f}): terrain_qc ecosite {t_str}")

# ---- 5. Delta (terrain - baseline) by uncertainty_class ----
print()
print("=" * 60)
print("5. Ecosite match delta (terrain_qc - baseline_qc) by uncertainty_class")
print("=" * 60)
for uc in uc_order:
    sub = df[df["uncertainty_class"] == uc].copy()
    b_valid = sub["baseline_qc_ecological_site_match"].dropna()
    t_valid = sub["terrain_qc_ecological_site_match"].dropna()
    b_rate = b_valid.mean() * 100 if len(b_valid) else None
    t_rate = t_valid.mean() * 100 if len(t_valid) else None
    if b_rate is not None and t_rate is not None:
        delta = t_rate - b_rate
        sign = "+" if delta >= 0 else ""
        print(f"  {uc}: {sign}{delta:.1f} pp (baseline={b_rate:.1f}%, terrain={t_rate:.1f}%)")

# ---- 6. Detailed CI bin table ----
print()
print("=" * 60)
print("6. Detailed CI bins (10-unit intervals)")
print("=" * 60)
bins = list(range(55, 101, 5))
df["ci_bin"] = pd.cut(df["ci"], bins=bins, right=True)
print(f"  {'CI bin':<14} {'n':>5}  {'baseline':>10}  {'terrain':>10}")
for interval in df["ci_bin"].cat.categories:
    sub = df[df["ci_bin"] == interval]
    if len(sub) == 0:
        continue
    b_str, bm, bt = match_rate(sub["baseline_qc_ecological_site_match"])
    t_str, tm, tt = match_rate(sub["terrain_qc_ecological_site_match"])
    print(f"  {str(interval):<14} {len(sub):>5}  {b_str:>20}  {t_str:>20}")
