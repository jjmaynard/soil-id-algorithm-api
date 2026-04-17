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
df["dom"] = pd.to_numeric(df["dominant_comppct_r"], errors="coerce")
df["gap"] = pd.to_numeric(df["component_gap"], errors="coerce")
df["match"] = df["baseline_qc_ecological_site_match"].fillna(0).astype(int)

df["dom_b"] = pd.cut(df["dom"], bins=[0, 50, 70, 85, 100], labels=["<50", "50-70", "70-85", "85+"])
df["gap_b"] = pd.cut(df["gap"], bins=[-1, 15, 35, 100], labels=["<15", "15-35", "35+"])

print("=== dom x gap cell match rates ===")
tbl = df.groupby(["dom_b", "gap_b"], observed=True).agg(
    n=("match", "count"), match_pct=("match", "mean")
).reset_index()
tbl["match_pct"] = (tbl["match_pct"] * 100).round(1)
print(tbl.to_string(index=False))

def dom_score(d):
    if pd.isna(d): return 50
    if d < 50:  return 30
    if d < 70:  return 55
    if d < 85:  return 80
    return 100

def gap_score(g):
    if pd.isna(g): return 50
    if g < 15:  return 30
    if g < 35:  return 60
    return 90

df["ds"] = df["dom"].apply(dom_score)
df["gs"] = df["gap"].apply(gap_score)

# Reconstruct approximate CI (Moderate Order3 assumption for most plots)
df["ci_new"] = (0.20 * 80 + 0.10 * 85 + 0.45 * df["ds"] + 0.25 * df["gs"]).round(1)

print()
print("=== Decile 4 (CI 67-72) composition ===")
d4 = df[(df["ci_new"] >= 67) & (df["ci_new"] <= 72)]
n4 = len(d4)
m4 = d4["match"].mean() * 100
print(f"n={n4}, match={m4:.1f}%")
print(d4.groupby(["dom_b", "gap_b"], observed=True)["match"].agg(["count", "mean"]).round(3))

print()
print("=== Decile 3 (CI 60-64) composition ===")
d3 = df[(df["ci_new"] >= 60) & (df["ci_new"] <= 64.2)]
n3 = len(d3)
m3 = d3["match"].mean() * 100
print(f"n={n3}, match={m3:.1f}%")
print(d3.groupby(["dom_b", "gap_b"], observed=True)["match"].agg(["count", "mean"]).round(3))

# What are the discrete CI values that exist?
print()
print("=== Discrete CI values in data (approx, top 15) ===")
print(df["ci_new"].value_counts().head(20).sort_index().to_string())

# Test: what weight combo makes calibration monotone?
# Try reducing gap weight, keeping dom dominant
print()
print("=== Test: alternative weights for monotonicity ===")
for w_dom, w_gap in [(0.45, 0.25), (0.50, 0.20), (0.40, 0.30), (0.35, 0.25)]:
    w_ord = 0.20
    w_muk = 1.0 - w_dom - w_gap - w_ord
    if w_muk < 0: continue
    df["ci_test"] = (w_ord*80 + w_muk*85 + w_dom*df["ds"] + w_gap*df["gs"]).round(1)
    r, p = stats.spearmanr(df["ci_test"], df["match"])
    # Check monotonicity: correlation between CI rank and match rate by decile
    df["ci_test_d"] = pd.qcut(df["ci_test"], q=7, duplicates="drop", labels=False)
    cal = df.groupby("ci_test_d", observed=True)["match"].mean().values
    is_mono = all(cal[i] <= cal[i+1] for i in range(len(cal)-1))
    print(f"  w_ord={w_ord:.2f} w_muk={w_muk:.2f} w_dom={w_dom:.2f} w_gap={w_gap:.2f}  "
          f"r={r:.4f}  monotone={is_mono}  decile_rates={[round(x*100,1) for x in cal]}")
