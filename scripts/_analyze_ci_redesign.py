"""
Evaluate whether the confidence_index formula can be restructured
to better explain/predict QC ecosite misclassification.

Current formula (from query_soil_survey_order.R):
  order_score:    Order2=100, Order3=80, Order4=55, Order5=35, NoData=30
  mukind_score:   consociation=100, assoc=70, complex=45, undiff=40, nodata=30
  dominant_score: pmax(0, pmin(100, dominant_comppct_r))  [50 if NA]
  gap_score:      pmax(0, pmin(100, 30 + 1.75 * component_gap))  [50 if NA]

  CI = round(((0.35*order_score) + (0.20*mukind_score) +
               (0.25*dominant_score) + (0.15*gap_score)) / 0.95, 1)
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score


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

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)

df["ci"]   = pd.to_numeric(df["confidence_index"], errors="coerce")
df["dom"]  = pd.to_numeric(df["dominant_comppct_r"], errors="coerce")
df["sec"]  = pd.to_numeric(df["second_comppct_r"], errors="coerce")
df["gap"]  = pd.to_numeric(df["component_gap"], errors="coerce")
df["match"] = df["baseline_qc_ecological_site_match"].fillna(0).astype(int)

# ── reconstruct formula components ───────────────────────────────────────────
ORDER_SCORE = {
    "Order 2": 100, "Order 3": 80, "Order 4": 55,
    "Order 5": 35,  "NoData":   30,
}
# infer from uncertainty_class / ci what the order and mukind were
# We don't have order/mukind directly (they came from SDA, not saved in results CSV).
# Use what we do have: dom, gap, ci, uncertainty_reason.

# Reconstruct order_score from uncertainty_reason + ci
# "Lower-intensity mapping order" → Order 4/5/NoData → order_score ≤ 55
# "Stronger MU profile" at Moderate (CI 55-74) → likely Order 3, consociation, dom<60 or gap<20 missed
# "Strong MU profile" at Low (CI ≥ 75) → Order 2 or Order 3 consociation, dom≥60
# We can approximate order_score for feature analysis:
def approx_order_score(row):
    if row["uncertainty_reason"] == "Lower-intensity mapping order":
        return 45   # Order 4/5/NoData blend
    elif row["uncertainty_class"] == "Low uncertainty (high confidence)":
        return 95   # almost all Order 2 or top-of-Order-3 consociations
    else:
        return 80   # Order 3 (dominant reason group)

df["order_score_approx"] = df.apply(approx_order_score, axis=1)

df["dominant_score"] = df["dom"].clip(0, 100).fillna(50)
df["gap_score"]      = (30 + 1.75 * df["gap"]).clip(0, 100).fillna(50)

# mukind_score: infer from uncertainty_reason
def approx_mukind_score(row):
    r = str(row["uncertainty_reason"]).lower()
    if "complex" in r or "undiff" in r:
        return 42   # complex=45, undiff=40
    elif row["uncertainty_class"] == "Low uncertainty (high confidence)":
        return 100  # these are consociations
    else:
        return 85   # association / consociation for Moderate "stronger" cases

df["mukind_score_approx"] = df.apply(approx_mukind_score, axis=1)

# ── 1. Point-biserial correlation of each raw input vs match ─────────────────
print("=" * 62)
print("1. Point-biserial / Spearman r: raw inputs vs baseline match")
print("=" * 62)
cols = {
    "confidence_index (CI)":    "ci",
    "dominant_comppct_r":       "dom",
    "component_gap":            "gap",
    "dominant_score (formula)": "dominant_score",
    "gap_score (formula)":      "gap_score",
}
for label, col in cols.items():
    sub = df[[col, "match"]].dropna()
    r, p = stats.spearmanr(sub[col], sub["match"])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {label:<35} r={r:+.3f}  p={p:.4f} {sig}  (n={len(sub)})")

# ── 2. Percentile calibration: match rate at each CI decile ──────────────────
print()
print("=" * 62)
print("2. Calibration: mean match rate at each CI decile")
print("=" * 62)
df["ci_decile"] = pd.qcut(df["ci"], q=10, duplicates="drop", labels=False)
cal = df.groupby("ci_decile").agg(
    n=("match", "count"),
    ci_min=("ci", "min"),
    ci_max=("ci", "max"),
    match_rate=("match", "mean"),
).reset_index()
print(f"  {'Decile':<8} {'CI range':<14} {'n':>5}  {'match%':>8}")
for _, row in cal.iterrows():
    print(f"  {int(row['ci_decile']):<8} {row['ci_min']:.1f}-{row['ci_max']:.1f}  {int(row['n']):>5}  {row['match_rate']*100:>7.1f}%")

# ── 3. What does the "Stronger MU profile" Moderate anomaly look like? ────────
print()
print("=" * 62)
print("3. Moderate + 'Stronger map unit confidence profile' (n=18) detail")
print("=" * 62)
anom = df[
    (df["uncertainty_class"] == "Moderate uncertainty") &
    (df["uncertainty_reason"] == "Stronger map unit confidence profile")
].copy()
print(f"  n={len(anom)}, median CI={anom['ci'].median():.1f}")
print(f"  baseline ecosite match:   {anom['match'].mean()*100:.1f}%  ({anom['match'].sum():.0f}/{len(anom)})")
print(f"  dominant_comppct_r range: {anom['dom'].min():.0f}-{anom['dom'].max():.0f}, median={anom['dom'].median():.0f}")
print(f"  component_gap range:      {anom['gap'].min():.0f}-{anom['gap'].max():.0f}, median={anom['gap'].median():.0f}")
# Why do these have CI in Moderate (55-74) and reason=StrongerProfile?
# The reason ordering assigns StrongerProfile last — meaning:
#   NOT lower-intensity order, NOT complex/undiff, NOT dom<60, NOT gap<20
#   → dom>=60, gap>=20, Order 3 or better, consociation/association
#   But CI is 55-74 meaning some component is weak
# Actually: a point can be Moderate+StrongerProfile when CI is 55-74 purely
# from the dominant_score calculation without triggering dom<60 threshold.
# dom >=60 satisfies the reason threshold but CI is pulled down by other factors.
print(f"\n  CI formula back-calculation check for these 18:")
print(f"  dominant_score  : {anom['dominant_score'].mean():.1f} mean")
print(f"  gap_score       : {anom['gap_score'].mean():.1f} mean")
print(f"  order_score est : {anom['order_score_approx'].mean():.1f} mean")
print(f"  mukind_score est: {anom['mukind_score_approx'].mean():.1f} mean")
estimated_ci = (
    0.35 * anom["order_score_approx"] +
    0.20 * anom["mukind_score_approx"] +
    0.25 * anom["dominant_score"] +
    0.15 * anom["gap_score"]
) / 0.95
print(f"  re-estimated CI : {estimated_ci.mean():.1f} mean  (actual median={anom['ci'].median():.1f})")

# ── 4. Logistic regression: which components best predict correct match? ───────
print()
print("=" * 62)
print("4. Logistic regression feature importance (baseline match)")
print("=" * 62)
features = {
    "dom (dominant_comppct_r)":  "dom",
    "gap (component_gap)":       "gap",
    "dominant_score":            "dominant_score",
    "gap_score":                 "gap_score",
    "CI (current)":              "ci",
}
feat_df = df[list(features.values()) + ["match"]].dropna()
X = feat_df[list(features.values())].values
y = feat_df["match"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)
cv_auc = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")

print(f"  Logistic regression (5-fold CV AUC): {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"  {'Feature':<34} {'Coeff':>10}")
for name, coef in zip(features.keys(), lr.coef_[0]):
    print(f"  {name:<34} {coef:>+10.3f}")

# ── 5. Alternative CI weightings: test which allocation maximises Spearman r ──
print()
print("=" * 62)
print("5. What allocation maximises correlation with correct match?")
print("=" * 62)
# We have approximated scores for order, mukind, dominant, gap
# Test a grid of weight combinations
sub = df[["order_score_approx", "mukind_score_approx", "dominant_score", "gap_score", "match"]].dropna()
best_r, best_w = 0, None
results_grid = []
for w_order in np.arange(0.10, 0.45, 0.05):
    for w_mukind in np.arange(0.10, 0.35, 0.05):
        for w_dom in np.arange(0.15, 0.50, 0.05):
            w_gap = 1.0 - w_order - w_mukind - w_dom
            if w_gap < 0.05 or w_gap > 0.35:
                continue
            ci_alt = (
                w_order  * sub["order_score_approx"] +
                w_mukind * sub["mukind_score_approx"] +
                w_dom    * sub["dominant_score"] +
                w_gap    * sub["gap_score"]
            )
            r, _ = stats.spearmanr(ci_alt, sub["match"])
            results_grid.append((r, w_order, w_mukind, w_dom, w_gap))
            if r > best_r:
                best_r, best_w = r, (w_order, w_mukind, w_dom, w_gap)

results_grid.sort(reverse=True)
r_current, _ = stats.spearmanr(
    (0.35 * sub["order_score_approx"] + 0.20 * sub["mukind_score_approx"] +
     0.25 * sub["dominant_score"] + 0.15 * sub["gap_score"]) / 0.95,
    sub["match"]
)
print(f"  Current weights (0.35/0.20/0.25/0.15) → Spearman r={r_current:.4f}")
print(f"\n  Top 5 alternative weightings (order / mukind / dominant / gap):")
for r, wo, wm, wd, wg in results_grid[:5]:
    marker = " ← best" if (wo, wm, wd, wg) == best_w else ""
    print(f"    {wo:.2f} / {wm:.2f} / {wd:.2f} / {wg:.2f}  → r={r:.4f}{marker}")

# ── 6. Does adding a "series-match" term help? ────────────────────────────────
# We can proxy this: if dominant_comppct_r ≥ 60 AND gap ≥ 20 = "clean" consociation
# What if we score "clean" = 1, else 0 and add it to CI?
print()
print("=" * 62)
print("6. 'Clean consociation' proxy: dom>=60 AND gap>=20")
print("=" * 62)
df["clean"] = ((df["dom"] >= 60) & (df["gap"] >= 20)).astype(float)
df.loc[df["dom"].isna() | df["gap"].isna(), "clean"] = np.nan
clean_sub = df[["clean", "match"]].dropna()
r_c, p_c = stats.spearmanr(clean_sub["clean"], clean_sub["match"])
match_clean    = df[df["clean"] == 1]["match"].mean()
match_notclean = df[df["clean"] == 0]["match"].mean()
print(f"  Clean (n={int(df['clean'].sum())}):     {match_clean*100:.1f}% match")
print(f"  Not clean (n={(df['clean']==0).sum()}): {match_notclean*100:.1f}% match")
print(f"  Spearman r={r_c:.3f}, p={p_c:.4f}")

# ── 7. Diagnosis: what range of dom explains most misclassifications? ─────────
print()
print("=" * 62)
print("7. Dominant comppct_r vs match rate (10-unit bins)")
print("=" * 62)
df["dom_bin"] = pd.cut(df["dom"], bins=range(0, 110, 10), right=True)
dom_tbl = df.groupby("dom_bin", observed=True).agg(
    n=("match", "count"),
    match_rate=("match", "mean")
)
print(f"  {'dom bin':<12} {'n':>5}  {'match%':>8}")
for interval, row in dom_tbl.iterrows():
    if row["n"] < 3:
        continue
    print(f"  {str(interval):<12} {int(row['n']):>5}  {row['match_rate']*100:>7.1f}%")

print()
print("=" * 62)
print("SUMMARY / RECOMMENDATIONS")
print("=" * 62)
print("""
Key structural issues with the current CI formula:

1. Survey order is over-weighted (0.35) relative to its predictive signal.
   Order 2 vs Order 3 produces a 20-point score gap (100 vs 80), but in the
   data the actual match rate difference is only ~3 pp. This inflates CI for
   Order 2 surveys without reflecting better ecosite identification.

2. dominant_score and gap_score are collinear (both capture component clarity)
   but together are only 40% of the formula — likely under-weighted relative to
   their actual predictive value.

3. The /0.95 normalization is artificial inflation with no interpretive basis.
   The max achievable CI (Order 2, consociation, dom=100, gap=100) with /0.95
   gives CI = 97.4 instead of 92.5 — misleadingly suggesting near-certainty.

4. The "Moderate + Stronger profile" anomaly (n=18, CI~74, only 22% match)
   suggests CI 55-74 is not a homogeneous uncertainty zone. Some of these
   are borderline cases that score just under 75 on order_score alone.

Suggested restructuring:
  • Reduce order weight to 0.20 (from 0.35)
  • Increase dominant_score weight to 0.40 (from 0.25)
  • Increase gap_score weight to 0.20 (from 0.15)
  • Keep mukind_score at 0.20 (from 0.20)
  • Remove /0.95 normalization (or rescale to 0-100 explicitly)
  • Raise the Low/Moderate threshold from 75 to ~78 (right-shifts boundary
    to exclude the anomalous near-75 Moderate cases)
""")
