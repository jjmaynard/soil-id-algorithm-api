"""
Reanalyze QC/terrain ecosite match rates using the REVISED CI formula.

New formula (from updated query_soil_survey_order.R):
  order_score:    Order2=100, Order3=80, Order4=55, Order5=35, NoData=30  (unchanged)
  mukind_score:   unchanged

  dominant_score (NEW - step function):
    NA        → 50
        dom < 50  → 20
        dom < 80  → 55
        dom >= 80 → 100

  gap_score (NEW - step function):
    NA        → 50
    gap < 20  → 30
    gap < 40  → 60
    gap >= 35 → 90

  CI = (0.20*order_score) + (0.10*mukind_score) + (0.35*dominant_score) +
      (0.20*gap_score) + (0.15*multiplicity_score)

  uncertainty_class threshold: Low >= 78 (was 75)
  uncertainty_reason Weak dominant: < 80

We reconstruct order_score and mukind_score from the original uncertainty_class/reason
since those raw fields aren't in the results CSV.
"""
import re
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CSV = (
    _REPO / "Data" / "aim_data"
    / "study_plot_characteristics_run_results_20260412T004929Z.csv"
)
_PLOTS_CSV = _REPO / "Data" / "aim_data" / "study_plot_characteristics.csv"
_MULT_CSV  = _REPO / "Data" / "aim_data" / "compname_mlra_ecosite_multiplicity.csv"


def _norm_mlra(s: str) -> str:
    """Map survey-area MLRA code (e.g. '28BY', '026X') to SDA mlrasymbol (e.g. '28b', '26')."""
    m = re.match(r'^0*(\d+)([ab])?', str(s).lower().strip())
    if not m:
        return str(s).lower().strip()
    return m.group(1) + (m.group(2) or '')


def _norm_compname(s) -> str | None:
    if pd.isna(s):
        return None
    return re.sub(r'\s+', ' ', str(s).lower().strip())


def _enrich_mlra_multiplicity(df: pd.DataFrame) -> pd.DataFrame:
    """Merge mlrasymbol and multiplicity_score from lookup tables when absent."""
    # -- mlrasymbol --
    if "mlrasymbol" not in df.columns or df["mlrasymbol"].isna().all():
        plots = pd.read_csv(_PLOTS_CSV, usecols=["PrimaryKey", "MLRA", "aim_series_component_name"])
        plots["mlrasymbol"] = plots["MLRA"].apply(_norm_mlra)
        plots["compname_norm"] = plots["aim_series_component_name"].apply(_norm_compname)
        df = df.merge(plots[["PrimaryKey", "mlrasymbol", "compname_norm"]], on="PrimaryKey", how="left")
    else:
        if "compname_norm" not in df.columns:
            plots = pd.read_csv(_PLOTS_CSV, usecols=["PrimaryKey", "aim_series_component_name"])
            plots["compname_norm"] = plots["aim_series_component_name"].apply(_norm_compname)
            df = df.merge(plots[["PrimaryKey", "compname_norm"]], on="PrimaryKey", how="left")

    # -- multiplicity_score --
    if "multiplicity_score" not in df.columns or df["multiplicity_score"].isna().all():
        mult = pd.read_csv(_MULT_CSV)
        mult["mlrasymbol"] = mult["mlrasymbol"].astype(str).str.strip().str.lower().str.replace(r'\.0$', '', regex=True)
        df = df.merge(mult.rename(columns={"n_ecosites": "n_ecosites_dominant"}),
                      on=["compname_norm", "mlrasymbol"], how="left")
        # multiplicity_score: 1 ecosite→100, ≥2→30, missing→50
        def _ms(n):
            if pd.isna(n): return 50
            return 100 if int(n) == 1 else 30
        df["multiplicity_score"] = df["n_ecosites_dominant"].apply(_ms)

    return df


df = pd.read_csv(_DEFAULT_CSV)
df = _enrich_mlra_multiplicity(df)

df["dom"] = pd.to_numeric(df["dominant_comppct_r"], errors="coerce")
df["gap"] = pd.to_numeric(df["component_gap"], errors="coerce")
df["ci_old"] = pd.to_numeric(df["confidence_index"], errors="coerce")

# --- Reconstruct order_score from original uncertainty_reason/class ---
# "Lower-intensity mapping order" → Order 4/5/NoData  → order_score ~45 blend
# Low uncertainty (high confidence) → predominantly Order 2 or Order 3 consociation
# Moderate "Stronger profile" at CI~74 → Order 3 dominant → 80
# Everything else Moderate → Order 3 → 80
def approx_order_score(row):
    if row["uncertainty_reason"] == "Lower-intensity mapping order":
        return 45
    elif row["uncertainty_class"] == "Low uncertainty (high confidence)":
        return 92   # mix of Order 2 and top Order 3
    else:
        return 80   # Order 3 dominant for all Moderate groups

def approx_mukind_score(row):
    r = str(row["uncertainty_reason"]).lower()
    if "complex" in r or "undiff" in r:
        return 42
    elif row["uncertainty_class"] == "Low uncertainty (high confidence)":
        return 100
    else:
        return 85

df["order_score"] = df.apply(approx_order_score, axis=1)
df["mukind_score"] = df.apply(approx_mukind_score, axis=1)

# --- New step-function scores ---
def dominant_score_new(dom):
    if pd.isna(dom):  return 50
    if dom < 50:      return 20
    if dom < 80:      return 55
    return 100

def gap_score_new(gap):
    if pd.isna(gap):  return 50
    if gap < 20:      return 30
    if gap < 40:      return 60
    return 90

df["dom_score_new"] = df["dom"].apply(dominant_score_new)
df["gap_score_new"] = df["gap"].apply(gap_score_new)

# --- Reconstruct old scores for comparison ---
df["dom_score_old"] = df["dom"].clip(0, 100).fillna(50)
df["gap_score_old"] = (30 + 1.75 * df["gap"]).clip(0, 100).fillna(50)

# --- New CI ---
if "multiplicity_score" in df.columns:
    df["multiplicity_score_new"] = pd.to_numeric(df["multiplicity_score"], errors="coerce").fillna(50)
else:
    # Fallback for legacy result files created before multiplicity integration.
    df["multiplicity_score_new"] = 50

df["ci_new"] = (
    0.20 * df["order_score"] +
    0.10 * df["mukind_score"] +
    0.35 * df["dom_score_new"] +
    0.20 * df["gap_score_new"] +
    0.15 * df["multiplicity_score_new"]
).round(1)

# --- New uncertainty_class (threshold 78) ---
def new_class(ci):
    if ci >= 78: return "Low uncertainty (high confidence)"
    if ci >= 55: return "Moderate uncertainty"
    return "High uncertainty"

df["uc_new"] = df["ci_new"].apply(new_class)

# --- New uncertainty_reason (weak dominant threshold <80) ---
def new_reason(row):
    if row["uncertainty_reason"] == "Lower-intensity mapping order":
        return "Lower-intensity mapping order"
    r = str(row["uncertainty_reason"]).lower()
    if "complex" in r or "undiff" in r:
        return "Complex/undifferentiated map unit"
    dom = row["dom"]
    gap = row["gap"]
    if not pd.isna(dom) and dom < 80:
        return "Weak dominant component"
    if not pd.isna(gap) and gap < 20:
        return "Top components have similar proportion"
    return "Stronger map unit confidence profile"

df["reason_new"] = df.apply(new_reason, axis=1)

df["match"] = df["baseline_qc_ecological_site_match"].fillna(0).astype(int)
df["match_t"] = df["terrain_qc_ecological_site_match"].fillna(0).astype(int)

uc_order = [
    "Low uncertainty (high confidence)",
    "Moderate uncertainty",
    "High uncertainty",
]

def match_rate(series):
    valid = series.dropna()
    if len(valid) == 0: return "n/a"
    return f"{int(valid.sum())}/{len(valid)} = {valid.mean()*100:.1f}%"

def wilson_rate(series):
    valid = pd.to_numeric(series, errors="coerce").dropna()
    n = len(valid)
    if n == 0:
        return (0, 0, np.nan, np.nan, np.nan)
    x = int(valid.sum())
    rate = x / n
    ci = stats.binomtest(x, n).proportion_ci(confidence_level=0.95, method="wilson")
    return (x, n, rate, ci.low, ci.high)

def delta(b, t):
    bv, tv = b.dropna(), t.dropna()
    if len(bv) == 0 or len(tv) == 0: return "n/a"
    d = (tv.mean() - bv.mean()) * 100
    return f"{'+' if d >= 0 else ''}{d:.1f} pp"

# ── 1. Class distribution shift ───────────────────────────────────────────────
print("=" * 65)
print("1. Uncertainty class distribution: OLD vs NEW formula")
print("=" * 65)
old_counts = df["uncertainty_class"].value_counts()
new_counts = df["uc_new"].value_counts()
print(f"  {'Class':<40} {'OLD':>6}  {'NEW':>6}")
for uc in uc_order:
    o = old_counts.get(uc, 0)
    n = new_counts.get(uc, 0)
    print(f"  {uc:<40} {o:>6}  {n:>6}")

# ── 2. Reason distribution shift ─────────────────────────────────────────────
print()
print("=" * 65)
print("2. Uncertainty reason distribution: OLD vs NEW")
print("=" * 65)
old_r = df["uncertainty_reason"].value_counts()
new_r = df["reason_new"].value_counts()
all_reasons = sorted(set(old_r.index) | set(new_r.index))
print(f"  {'Reason':<45} {'OLD':>6}  {'NEW':>6}")
for r in all_reasons:
    o = old_r.get(r, 0)
    n = new_r.get(r, 0)
    chg = n - o
    marker = f" ({'+' if chg>=0 else ''}{chg})" if chg != 0 else ""
    print(f"  {r:<45} {o:>6}  {n:>6}{marker}")

# ── 3. CI distribution shift ─────────────────────────────────────────────────
print()
print("=" * 65)
print("3. CI summary statistics: OLD vs NEW")
print("=" * 65)
for label, col in [("OLD", "ci_old"), ("NEW", "ci_new")]:
    s = df[col].describe()
    print(f"  {label}: min={s['min']:.1f}  Q1={s['25%']:.1f}  med={s['50%']:.1f}  Q3={s['75%']:.1f}  max={s['max']:.1f}")

# ── 4. Match rate by NEW uncertainty_class ───────────────────────────────────
print()
print("=" * 65)
print("4. Match rates by NEW uncertainty_class")
print("=" * 65)
for uc in uc_order:
    sub = df[df["uc_new"] == uc]
    n = len(sub)
    med = sub["ci_new"].median()
    b = match_rate(sub["baseline_qc_ecological_site_match"])
    t = match_rate(sub["terrain_qc_ecological_site_match"])
    d = delta(sub["baseline_qc_ecological_site_match"], sub["terrain_qc_ecological_site_match"])
    print(f"\n  {uc} (n={n}, median CI={med:.1f})")
    print(f"    baseline_qc ecosite : {b}")
    print(f"    terrain_qc  ecosite : {t}")
    print(f"    delta               : {d}")

# ── 5. Match rate by NEW class x reason ──────────────────────────────────────
print()
print("=" * 65)
print("5. Match rates by NEW class x reason")
print("=" * 65)
for uc in uc_order:
    sub_uc = df[df["uc_new"] == uc]
    uc_short = uc.replace(" (high confidence)", "")
    b_all = match_rate(sub_uc["baseline_qc_ecological_site_match"])
    t_all = match_rate(sub_uc["terrain_qc_ecological_site_match"])
    print(f"\n  {uc_short} (n={len(sub_uc)})  baseline={b_all}  terrain={t_all}")
    for reason in sorted(sub_uc["reason_new"].unique()):
        sub = sub_uc[sub_uc["reason_new"] == reason]
        b = match_rate(sub["baseline_qc_ecological_site_match"])
        t = match_rate(sub["terrain_qc_ecological_site_match"])
        d = delta(sub["baseline_qc_ecological_site_match"], sub["terrain_qc_ecological_site_match"])
        print(f"    └ {reason[:46]:<46} n={len(sub):>4}  CI med={sub['ci_new'].median():.1f}  base={b}  terrain={t}  Δ={d}")

# ── 6. Primary 3-class rates with Wilson 95% CI ──────────────────────────────
print()
print("=" * 65)
print("6. Primary 3-class match rates with Wilson 95% CI")
print("=" * 65)
print(f"  {'Class':<34} {'Baseline':<27} {'Terrain':<27}")
for uc in uc_order:
    sub = df[df["uc_new"] == uc]
    bx, bn, br, blo, bhi = wilson_rate(sub["baseline_qc_ecological_site_match"])
    tx, tn, tr, tlo, thi = wilson_rate(sub["terrain_qc_ecological_site_match"])

    btxt = "n/a" if bn == 0 else f"{bx}/{bn} ({br*100:.1f}%) [{blo*100:.1f}, {bhi*100:.1f}]"
    ttxt = "n/a" if tn == 0 else f"{tx}/{tn} ({tr*100:.1f}%) [{tlo*100:.1f}, {thi*100:.1f}]"
    print(f"  {uc:<34} {btxt:<27} {ttxt:<27}")

# ── 7. Chi-square test: match x uncertainty class ───────────────────────────
print()
print("=" * 65)
print("7. Chi-square test: baseline match x NEW uncertainty class")
print("=" * 65)
chi_df = df[["uc_new", "match"]].dropna().copy()
cont = pd.crosstab(chi_df["match"], chi_df["uc_new"]).reindex(columns=uc_order, fill_value=0)
chi2, p, dof, _ = stats.chi2_contingency(cont)
print(cont.to_string())
print(f"\n  chi2={chi2:.4f}, dof={dof}, p={p:.6g}")

# ── 8. Per-MLRA class breakdown (top-10 by n) ───────────────────────────────
print()
print("=" * 65)
print("8. Per-MLRA baseline match rates by class (top 10 MLRAs by n)")
print("=" * 65)
if "mlrasymbol" in df.columns:
    mlra_df = df[["mlrasymbol", "uc_new", "baseline_qc_ecological_site_match"]].copy()
    mlra_df["mlrasymbol"] = mlra_df["mlrasymbol"].astype(str).str.strip()
    mlra_df = mlra_df[mlra_df["mlrasymbol"] != ""]

    top_mlras = (
        mlra_df.groupby("mlrasymbol", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    for mlra in top_mlras:
        sub_mlra = mlra_df[mlra_df["mlrasymbol"] == mlra]
        print(f"\n  MLRA {mlra} (n={len(sub_mlra)})")
        for uc in uc_order:
            sub = sub_mlra[sub_mlra["uc_new"] == uc]
            x, n, rate, lo, hi = wilson_rate(sub["baseline_qc_ecological_site_match"])
            if n == 0:
                print(f"    {uc:<34} n=0")
            else:
                print(
                    f"    {uc:<34} {x}/{n} ({rate*100:.1f}%) "
                    f"[{lo*100:.1f}, {hi*100:.1f}]"
                )
else:
    print("  mlrasymbol column not present in current CSV; rerun query script first.")

# ── 9. Calibration: new CI decile vs match ───────────────────────────────────
print()
print("=" * 65)
print("9. Calibration: NEW CI decile vs baseline match rate")
print("=" * 65)
df["ci_new_decile"] = pd.qcut(df["ci_new"], q=10, duplicates="drop", labels=False)
cal = df.groupby("ci_new_decile", observed=True).agg(
    n=("match", "count"),
    ci_min=("ci_new", "min"),
    ci_max=("ci_new", "max"),
    match_rate=("match", "mean"),
).reset_index()
print(f"  {'Decile':<8} {'CI range':<14} {'n':>5}  {'match%':>8}")
for _, row in cal.iterrows():
    print(f"  {int(row['ci_new_decile']):<8} {row['ci_min']:.1f}-{row['ci_max']:.1f}  {int(row['n']):>5}  {row['match_rate']*100:>7.1f}%")

# ── 10. Spearman r: old vs new CI ────────────────────────────────────────────
print()
print("=" * 65)
print("10. Spearman r vs baseline match: OLD vs NEW CI")
print("=" * 65)
for label, col in [("OLD CI", "ci_old"), ("NEW CI", "ci_new")]:
    sub = df[[col, "match"]].dropna()
    r, p = stats.spearmanr(sub[col], sub["match"])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {label}: r={r:+.4f}  p={p:.4f} {sig}  (n={len(sub)})")

# ── 11. Former anomaly group: now reclassified? ──────────────────────────────
print()
print("=" * 65)
print("11. Former anomaly: Moderate + 'Stronger profile' (old) -> now?")
print("=" * 65)
anomaly = df[
    (df["uncertainty_class"] == "Moderate uncertainty") &
    (df["uncertainty_reason"] == "Stronger map unit confidence profile")
].copy()
print(f"  n={len(anomaly)}, dom range={anomaly['dom'].min():.0f}-{anomaly['dom'].max():.0f}")
print(f"  Old reason: Stronger map unit confidence profile × {len(anomaly)}")
print(f"  New reason distribution:")
print(anomaly["reason_new"].value_counts().to_string())
print(f"  New class distribution:")
print(anomaly["uc_new"].value_counts().to_string())
print(f"  Baseline match rate (this group): {match_rate(anomaly['baseline_qc_ecological_site_match'])}")
