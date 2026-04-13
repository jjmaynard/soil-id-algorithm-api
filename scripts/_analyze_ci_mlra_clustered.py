"""Cluster-robust CI analysis with MLRA grouping.

Usage:
  python scripts/_analyze_ci_mlra_clustered.py
  python scripts/_analyze_ci_mlra_clustered.py --csv Data/aim_data/study_plot_characteristics_run_results_20260412T004929Z.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError as exc:
    raise SystemExit(
        "statsmodels is required. Install with: pip install statsmodels"
    ) from exc


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "Data"
    / "aim_data"
    / "study_plot_characteristics_run_results_20260412T004929Z.csv"
)

DEFAULT_MLRA_CSV = (
    Path(__file__).resolve().parents[1]
    / "Data"
    / "aim_data"
    / "study_plot_characteristics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--mlra-csv", type=Path, default=DEFAULT_MLRA_CSV)
    return parser.parse_args()


def ci_or_table(beta: float, se: float) -> tuple[float, float, float]:
    lo = beta - 1.96 * se
    hi = beta + 1.96 * se
    return math.exp(beta), math.exp(lo), math.exp(hi)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    if "mlrasymbol" not in df.columns and args.mlra_csv.exists():
        if "PrimaryKey" in df.columns:
            mlra_df = pd.read_csv(args.mlra_csv)
            mlra_col = None
            for candidate in ["mlrasymbol", "MLRA", "mlra"]:
                if candidate in mlra_df.columns:
                    mlra_col = candidate
                    break

            if mlra_col is not None and "PrimaryKey" in mlra_df.columns:
                df = df.merge(
                    mlra_df[["PrimaryKey", mlra_col]].rename(columns={mlra_col: "mlrasymbol"}),
                    on="PrimaryKey",
                    how="left",
                )

    # Normalize survey-area MLRA codes (e.g. '28BY' → '28b', '026X' → '26')
    if "mlrasymbol" in df.columns:
        import re
        def _norm_mlra(s: str) -> str:
            m = re.match(r'^0*(\d+)([ab])?', str(s).lower().strip())
            if not m:
                return str(s).lower().strip()
            return m.group(1) + (m.group(2) or '')
        df["mlrasymbol"] = df["mlrasymbol"].apply(_norm_mlra)

    required = {"confidence_index", "baseline_qc_ecological_site_match", "mlrasymbol"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Compute ecosite base-rate: fraction of plots in each MLRA that share
    # the same expected_ecological_site as this plot.  This captures the
    # "mega-ecosite" effect where SoilID can match by prior alone.
    # We use expected_ecological_site from the run-results CSV if present;
    # otherwise fall back to the mlra_csv.
    # ------------------------------------------------------------------
    ecosite_col = None
    for candidate in ["expected_ecological_site", "EcolSite"]:
        if candidate in df.columns:
            ecosite_col = candidate
            break
    if ecosite_col is None and args.mlra_csv.exists():
        mlra_df2 = pd.read_csv(args.mlra_csv, usecols=["PrimaryKey", "EcolSite"])
        df = df.merge(mlra_df2, on="PrimaryKey", how="left")
        ecosite_col = "EcolSite"

    if ecosite_col is not None:
        ecosite_counts = (
            df.groupby(["mlrasymbol", ecosite_col])
            .size()
            .rename("ecosite_n")
            .reset_index()
        )
        mlra_totals = df.groupby("mlrasymbol").size().rename("mlra_n").reset_index()
        ecosite_counts = ecosite_counts.merge(mlra_totals, on="mlrasymbol")
        ecosite_counts["ecosite_base_rate"] = ecosite_counts["ecosite_n"] / ecosite_counts["mlra_n"]
        df = df.merge(
            ecosite_counts[["mlrasymbol", ecosite_col, "ecosite_base_rate"]],
            on=["mlrasymbol", ecosite_col],
            how="left",
        )
    else:
        df["ecosite_base_rate"] = np.nan

    base_rate_available = df["ecosite_base_rate"].notna().any()

    model_cols = ["confidence_index", "baseline_qc_ecological_site_match", "mlrasymbol"]
    if base_rate_available:
        model_cols.append("ecosite_base_rate")
    model_df = df[model_cols].copy()
    model_df["confidence_index"] = pd.to_numeric(model_df["confidence_index"], errors="coerce")
    model_df["match"] = pd.to_numeric(
        model_df["baseline_qc_ecological_site_match"], errors="coerce"
    )
    model_df["mlrasymbol"] = model_df["mlrasymbol"].astype(str).str.strip()

    model_df = model_df.dropna(subset=["confidence_index", "match", "mlrasymbol"])
    model_df = model_df[model_df["mlrasymbol"] != ""]
    model_df["match"] = (model_df["match"] > 0).astype(int)

    print("=" * 72)
    print("MLRA Cluster-Robust Logistic Regression")
    print("=" * 72)
    print(f"Rows used: {len(model_df):,}")
    print(f"Unique MLRAs: {model_df['mlrasymbol'].nunique():,}")

    cluster_model = smf.logit("match ~ confidence_index", data=model_df).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": model_df["mlrasymbol"]},
    )

    beta = float(cluster_model.params["confidence_index"])
    se = float(cluster_model.bse["confidence_index"])
    pval = float(cluster_model.pvalues["confidence_index"])
    odds_ratio, or_lo, or_hi = ci_or_table(beta, se)

    print("\n1) Cluster-robust model: match ~ confidence_index")
    print(f"   beta(confidence_index): {beta:+.5f}")
    print(f"   robust SE:              {se:.5f}")
    print(f"   p-value:                {pval:.6f}")
    print(f"   OR per +1 CI point:     {odds_ratio:.4f} [{or_lo:.4f}, {or_hi:.4f}]")

    print("\n2) Fixed-effects model: match ~ confidence_index + C(mlrasymbol)")
    try:
        fe_model = smf.logit("match ~ confidence_index + C(mlrasymbol)", data=model_df).fit(
            disp=False, maxiter=200
        )

        fe_beta = float(fe_model.params["confidence_index"])
        fe_se = float(fe_model.bse["confidence_index"])
        fe_pval = float(fe_model.pvalues["confidence_index"])
        fe_or, fe_or_lo, fe_or_hi = ci_or_table(fe_beta, fe_se)

        converged = bool(getattr(fe_model, "mle_retvals", {}).get("converged", True))

        intercept = float(fe_model.params.get("Intercept", 0.0))
        mlra_intercepts = [intercept]
        for name, value in fe_model.params.items():
            if name.startswith("C(mlrasymbol)"):
                mlra_intercepts.append(intercept + float(value))

        var_u = float(np.var(mlra_intercepts, ddof=1)) if len(mlra_intercepts) > 1 else 0.0
        icc = var_u / (var_u + (math.pi**2) / 3.0) if var_u > 0 else 0.0

        print(f"   beta(confidence_index): {fe_beta:+.5f}")
        print(f"   model SE:               {fe_se:.5f}")
        print(f"   p-value:                {fe_pval:.6f}")
        print(f"   OR per +1 CI point:     {fe_or:.4f} [{fe_or_lo:.4f}, {fe_or_hi:.4f}]")
        if converged:
            print(f"   between-MLRA intercept variance (proxy): {var_u:.5f}")
            print(f"   ICC proxy (logit scale):                {icc:.4f}")
        else:
            print("   Fixed-effects model did not converge; MLRA variance/ICC proxy not reported.")
    except Exception as exc:  # noqa: BLE001
        print(f"   Fixed-effects fit failed: {exc}")

    # ------------------------------------------------------------------
    # 3) Ecosite base-rate covariate model
    #    Keeps all data; ecosite_base_rate soaks up "mega-ecosite" luck,
    #    leaving beta(confidence_index) as the CI's incremental contribution.
    # ------------------------------------------------------------------
    if base_rate_available:
        br_df = model_df.dropna(subset=["ecosite_base_rate"]).copy()
        print(f"\n3) Base-rate-adjusted model: match ~ confidence_index + ecosite_base_rate")
        print(f"   (ecosite_base_rate = fraction of that MLRA's plots sharing the same")
        print(f"    expected ecosite — captures mega-ecosite base-rate; all {len(br_df)} plots retained)")
        try:
            br_model = smf.logit(
                "match ~ confidence_index + ecosite_base_rate", data=br_df
            ).fit(
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": br_df["mlrasymbol"]},
            )
            br_beta = float(br_model.params["confidence_index"])
            br_se   = float(br_model.bse["confidence_index"])
            br_pval = float(br_model.pvalues["confidence_index"])
            br_or, br_or_lo, br_or_hi = ci_or_table(br_beta, br_se)

            er_beta = float(br_model.params["ecosite_base_rate"])
            er_se   = float(br_model.bse["ecosite_base_rate"])
            er_pval = float(br_model.pvalues["ecosite_base_rate"])
            er_or, er_or_lo, er_or_hi = ci_or_table(er_beta, er_se)

            print(f"\n   confidence_index (incremental CI signal after base-rate):")
            print(f"     beta: {br_beta:+.5f}  robust SE: {br_se:.5f}  p: {br_pval:.6f}")
            print(f"     OR per +1 CI point: {br_or:.4f} [{br_or_lo:.4f}, {br_or_hi:.4f}]")
            print(f"\n   ecosite_base_rate (mega-ecosite prior):")
            print(f"     beta: {er_beta:+.5f}  robust SE: {er_se:.5f}  p: {er_pval:.6f}")
            print(f"     OR per +0.1 base-rate: {math.exp(er_beta*0.1):.4f}")

            # Attenuation ratio: how much does base-rate adjustment reduce CI beta?
            attenuation = (beta - br_beta) / beta if beta != 0 else float("nan")
            print(f"\n   CI beta attenuation from adding base-rate: {attenuation*100:+.1f}%")
            print(f"   (positive = CI beta shrinks after accounting for base-rate;")
            print(f"    near 0 = CI is doing real work independent of ecosite prior)")
        except Exception as exc:  # noqa: BLE001
            print(f"   Base-rate-adjusted fit failed: {exc}")

        # Per-MLRA: ecosite_base_rate distribution to flag high-concentration MLRAs
        print(f"\n4) Per-MLRA ecosite concentration (mean base-rate of plots' expected ecosite)")
        print(f"   High values indicate MLRAs where one ecosite dominates the sample —")
        print(f"   raw match rates in those MLRAs conflate CI signal with base-rate luck.")
        print(f"   {'MLRA':<8} {'n':>4}  {'mean_base_rate':>14}  {'max_base_rate':>13}  {'match%':>7}")
        mlra_stats = (
            br_df.groupby("mlrasymbol")
            .agg(
                n=("match", "count"),
                mean_br=("ecosite_base_rate", "mean"),
                max_br=("ecosite_base_rate", "max"),
                match_rate=("match", "mean"),
            )
            .sort_values("mean_br", ascending=False)
        )
        for mlra, row in mlra_stats.iterrows():
            flag = " <-- high concentration" if row["mean_br"] > 0.25 else ""
            print(
                f"   {mlra:<8} {int(row['n']):>4}  {row['mean_br']:>14.3f}  "
                f"{row['max_br']:>13.3f}  {row['match_rate']*100:>6.1f}%{flag}"
            )
    else:
        print("\n3) Base-rate-adjusted model: skipped (ecosite column not available)")


if __name__ == "__main__":
    main()
