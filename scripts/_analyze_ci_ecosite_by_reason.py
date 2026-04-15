import argparse
from pathlib import Path

import pandas as pd


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


def match_rate(series):
    valid = series.dropna()
    if len(valid) == 0:
        return "n/a"
    pct = valid.mean() * 100
    return f"{int(valid.sum())}/{len(valid)} = {pct:.1f}%"


def delta(b_series, t_series):
    bv = b_series.dropna()
    tv = t_series.dropna()
    if len(bv) == 0 or len(tv) == 0:
        return "n/a"
    d = (tv.mean() - bv.mean()) * 100
    return f"{'+' if d >= 0 else ''}{d:.1f} pp"


# Cross-tab: uncertainty_class x uncertainty_reason
print("=" * 72)
print("Ecosite match rate by uncertainty_class x uncertainty_reason")
print("=" * 72)

uc_order = [
    "Low uncertainty (high confidence)",
    "Moderate uncertainty",
    "High uncertainty",
]

groups = df.groupby(["uncertainty_class", "uncertainty_reason"], sort=False)

for uc in uc_order:
    sub_uc = df[df["uncertainty_class"] == uc]
    reasons = sub_uc["uncertainty_reason"].unique()

    print(f"\n{'─'*72}")
    print(f"  {uc}  (n={len(sub_uc)}, overall median CI={sub_uc['ci'].median():.1f})")
    print(f"{'─'*72}")

    for reason in sorted(reasons):
        sub = sub_uc[sub_uc["uncertainty_reason"] == reason]
        n = len(sub)
        ci_med = sub["ci"].median()
        ci_rng = f"{sub['ci'].min():.1f}-{sub['ci'].max():.1f}"
        b = match_rate(sub["baseline_qc_ecological_site_match"])
        t = match_rate(sub["terrain_qc_ecological_site_match"])
        d = delta(
            sub["baseline_qc_ecological_site_match"],
            sub["terrain_qc_ecological_site_match"],
        )
        print(f"\n    Reason: {reason}")
        print(f"      n={n}  median CI={ci_med:.1f}  range={ci_rng}")
        print(f"      baseline_qc ecosite : {b}")
        print(f"      terrain_qc  ecosite : {t}")
        print(f"      delta               : {d}")

# Summary table
print()
print("=" * 72)
print("Summary table")
print("=" * 72)
hdr = f"{'Class / Reason':<50} {'n':>5}  {'med CI':>7}  {'baseline':>12}  {'terrain':>12}  {'delta':>8}"
print(hdr)
print("-" * 100)

for uc in uc_order:
    sub_uc = df[df["uncertainty_class"] == uc]
    b_all = match_rate(sub_uc["baseline_qc_ecological_site_match"])
    t_all = match_rate(sub_uc["terrain_qc_ecological_site_match"])
    d_all = delta(
        sub_uc["baseline_qc_ecological_site_match"],
        sub_uc["terrain_qc_ecological_site_match"],
    )
    uc_short = uc.replace(" (high confidence)", "")
    print(f"  {uc_short:<48} {len(sub_uc):>5}  {sub_uc['ci'].median():>7.1f}  {b_all:>20}  {t_all:>20}  {d_all:>8}")

    for reason in sorted(sub_uc["uncertainty_reason"].unique()):
        sub = sub_uc[sub_uc["uncertainty_reason"] == reason]
        b = match_rate(sub["baseline_qc_ecological_site_match"])
        t = match_rate(sub["terrain_qc_ecological_site_match"])
        d = delta(
            sub["baseline_qc_ecological_site_match"],
            sub["terrain_qc_ecological_site_match"],
        )
        reason_short = reason[:44]
        print(f"    └ {reason_short:<46} {len(sub):>5}  {sub['ci'].median():>7.1f}  {b:>20}  {t:>20}  {d:>8}")
    print()
