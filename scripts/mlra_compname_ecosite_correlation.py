"""
mlra_compname_ecosite_correlation.py

Within all MLRAs represented in the study_plot_characteristics.csv, query SDA
to find every soil component name (compname, all compkind values) and its
associated ecological site IDs (coecoclass.ecoclassid).

Reports:
  - Total unique series in those MLRAs (with ≥1 ecosite)
  - How many are correlated to more than one distinct ecosite ID
  - The percentage of series with >1 ecosite
  - Per-MLRA breakdown
  - Full detail CSV: one row per series showing all ecosite IDs

Usage
-----
  python scripts/mlra_compname_ecosite_correlation.py
  python scripts/mlra_compname_ecosite_correlation.py --plot-csv Data/aim_data/study_plot_characteristics.csv
"""

import argparse
import logging
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data" / "aim_data"
SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"
SDA_TIMEOUT = 60

# ---------------------------------------------------------------------------
# MLRA code mapping:  study CSV codes  →  SDA mapunit.mlrasymbol
# ---------------------------------------------------------------------------
# Study CSV uses codes like '023X', '23XY', '28AY', '30XA' etc.
# SDA stores them as short codes: '23', '28A', '28B', '30', etc.
# Strip leading zeros, strip trailing 'X'/'Y' except where the letter preceding
# it identifies a sub-MLRA (28A, 28B).

def _csv_mlra_to_sda(code: str) -> str:
    """Convert a study-CSV MLRA code to its SDA mlrasymbol equivalent."""
    code = str(code).strip().upper()
    # Extract leading digits and optional sub-letter
    m = re.match(r"^0*(\d+)([AB])?", code)
    if not m:
        return code
    num = m.group(1)
    sub = m.group(2) or ""
    return num + sub


# ---------------------------------------------------------------------------
# SDA helper
# ---------------------------------------------------------------------------

def _sda_query(sql: str, timeout: int = SDA_TIMEOUT) -> pd.DataFrame | None:
    """POST a T-SQL query to SDA, return DataFrame or None."""
    payload = {"format": "JSON+COLUMNNAME", "query": sql}
    try:
        resp = requests.post(SDA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "Table" not in data:
            return None
        raw = data["Table"]
        if not raw:
            return None
        headers = raw[0]
        rows = raw[1:]
        if not headers or not rows:
            return None
        return pd.DataFrame(rows, columns=headers)
    except requests.ConnectionError as e:
        logger.warning(f"SDA connection error: {e}")
    except requests.Timeout:
        logger.warning(f"SDA timed out (>{timeout}s)")
    except requests.RequestException as e:
        logger.warning(f"SDA request error: {e}")
    return None


# ---------------------------------------------------------------------------
# Main query: all (mlrasymbol, compname, ecoclassid) tuples in target MLRAs
# ---------------------------------------------------------------------------

def query_mlra_compname_ecosites(
    sda_mlras: list[str],
    state: str = "NV",
) -> "pd.DataFrame | None":
    """Query SDA for all compname + state ecoclassid pairs within state survey areas.

    Returns DataFrame with columns: mlrasymbol, compname, ecoclassid, ecoclassname,
    survey_order (from mapunit.invesintens, MIN per group).
    Restricted to legend.areasymbol LIKE '<STATE>%' so only components mapped in
    that state's soil surveys are included, regardless of which other states share
    the MLRA.  Ecosite IDs are filtered to those containing the state code.
    All compkind values (Series, Miscellaneous area, etc.) are included.

    Pass sda_mlras=[] to query all MLRAs present in the state (no MLRA filter).
    """
    state = state.strip().upper()
    mlra_clause = ""
    if sda_mlras:
        quoted = ", ".join(f"'{m}'" for m in sda_mlras)
        mlra_clause = f"  AND mu.mlrasymbol IN ({quoted}) "
    sql = (
        "SELECT "
        "  mu.mlrasymbol, "
        "  c.compname, "
        "  ce.ecoclassid, "
        "  ce.ecoclassname, "
        "  MIN(mu.invesintens) AS survey_order "
        "FROM component c "
        "JOIN mapunit mu ON c.mukey = mu.mukey "
        "JOIN legend l ON mu.lkey = l.lkey "
        "JOIN coecoclass ce ON c.cokey = ce.cokey "
        f"WHERE l.areasymbol LIKE '{state}%' "
        + mlra_clause +
        "  AND ce.ecoclassid IS NOT NULL "
        f"  AND ce.ecoclassid LIKE '%{state}%' "
        "GROUP BY mu.mlrasymbol, c.compname, ce.ecoclassid, ce.ecoclassname "
        "ORDER BY c.compname, ce.ecoclassid"
    )
    scope = f"areasymbol LIKE '{state}%'"
    logger.info(f"Querying SDA for compname-ecosite pairs ({scope}, all compkind) ...")
    df = _sda_query(sql, timeout=SDA_TIMEOUT)
    if df is None:
        logger.error("SDA returned no data.")
    else:
        logger.info(f"  -> {len(df)} rows returned ({df['compname'].nunique()} unique compnames)")
        orders = df['survey_order'].value_counts(dropna=False).to_dict()
        logger.info(f"  invesintens distribution: {orders}")
    return df


def query_total_compnames_by_state(
    sda_mlras: list[str],
    state: str = "NV",
) -> "pd.DataFrame | None":
    """Query SDA for total compname count per MLRA within a state's survey areas.

    Includes all components regardless of compkind or whether they have an ecosite.
    Returns DataFrame with columns: mlrasymbol, total_compnames

    Pass sda_mlras=[] to query all MLRAs present in the state.
    """
    state = state.strip().upper()
    mlra_clause = ""
    if sda_mlras:
        quoted = ", ".join(f"'{m}'" for m in sda_mlras)
        mlra_clause = f"  AND mu.mlrasymbol IN ({quoted}) "
    sql = (
        "SELECT mu.mlrasymbol, COUNT(DISTINCT c.compname) AS total_compnames "
        "FROM component c "
        "JOIN mapunit mu ON c.mukey = mu.mukey "
        "JOIN legend l ON mu.lkey = l.lkey "
        f"WHERE l.areasymbol LIKE '{state}%' "
        + mlra_clause +
        "GROUP BY mu.mlrasymbol"
    )
    logger.info(f"Querying SDA for total {state} compnames per MLRA ...")
    df = _sda_query(sql, timeout=SDA_TIMEOUT)
    if df is None:
        logger.error("SDA returned no data for total compnames.")
    else:
        df["total_compnames"] = df["total_compnames"].astype(int)
        logger.info(f"  -> {len(df)} MLRAs; total {state} compnames: {df['total_compnames'].sum()}")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _norm_ecosite(v):
    """Normalise ecosite ID: strip R/F prefix and _NNN numeric suffix."""
    s = str(v).strip().upper()
    if s.startswith(("R", "F")):
        s = s[1:]
    s = re.sub(r"_\d+$", "", s)          # remove underscore + numeric suffix (e.g. _001)
    return re.sub(r"[^A-Z0-9]", "", s)


def _norm_series_name(v: str) -> str:
    """Normalise a series/compname for matching: uppercase, strip non-alphanumeric."""
    s = str(v).strip().upper()
    return re.sub(r"[^A-Z0-9 ]", "", s).strip()


# ---------------------------------------------------------------------------
# AIM/QC study-plot subset analysis
# ---------------------------------------------------------------------------

def analyse_aim_qc_subset(
    plot_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    aim_col: str = "aim_series_component_name",
    qc_col: str = "qc_series_component_name",
) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """Filter raw_df to compnames present in the AIM or QC series columns of plot_df.

    Returns:
        matched_df   - raw_df rows for the matched compnames (with _enorm column)
        per_mlra_df  - per-MLRA breakdown for the subset
        missing      - set of normalised series names not found in SDA NV data
    """
    aim_norm = set(_norm_series_name(s) for s in plot_df[aim_col].dropna())
    qc_norm = set(_norm_series_name(s) for s in plot_df[qc_col].dropna())
    all_norm = aim_norm | qc_norm

    rdf = raw_df.copy()
    rdf["_compname_norm"] = rdf["compname"].apply(_norm_series_name)
    rdf["_enorm"] = rdf["ecoclassid"].apply(_norm_ecosite)

    matched = rdf[rdf["_compname_norm"].isin(all_norm)].copy()

    # Per-MLRA breakdown
    mlra_compnames = (
        matched.groupby(["mlrasymbol", "compname"])["_enorm"]
        .nunique()
        .reset_index()
    )
    mlra_compnames.columns = ["mlrasymbol", "compname", "n_eco"]
    per_mlra = mlra_compnames.groupby("mlrasymbol").agg(
        compnames_with_ecosite=("compname", "nunique"),
        multi_eco=("n_eco", lambda x: (x > 1).sum()),
    ).reset_index()
    per_mlra["pct_multiple"] = (
        per_mlra["multi_eco"] / per_mlra["compnames_with_ecosite"] * 100
    ).round(1)

    sda_norms = set(rdf["_compname_norm"].unique())
    missing = all_norm - sda_norms

    # Store source breakdown on the function object for the summary print
    analyse_aim_qc_subset._aim_norm = aim_norm
    analyse_aim_qc_subset._qc_norm = qc_norm
    analyse_aim_qc_subset._all_norm = all_norm

    return matched, per_mlra, missing


def analyse_by_order(raw_df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Return a per-survey-order breakdown of compname multiplicity.

    Groups compnames by their survey_order value (mapunit.invesintens) and reports
    single vs. multiple ecosite rates for each order tier.
    """
    df = raw_df.copy()
    df["_eco_norm"] = df["ecoclassid"].apply(_norm_ecosite)
    # Assign each (compname, survey_order) pair a normalised-ecosite count.
    # A compname may appear across multiple orders; count it separately per order.
    order_label = df["survey_order"].fillna("NoData")
    df["_order"] = order_label

    rows = []
    for order, grp in df.groupby("_order", sort=True):
        per_series = grp.groupby("compname")["_eco_norm"].nunique()
        total = len(per_series)
        single = int((per_series == 1).sum())
        multi = int((per_series > 1).sum())
        rows.append({
            "survey_order": order,
            "compnames_with_ecosite": total,
            "single_ecosite": single,
            "multiple_ecosites": multi,
            "pct_single": round(100.0 * single / total, 1) if total else 0.0,
            "pct_multiple": round(100.0 * multi / total, 1) if total else 0.0,
        })
    return pd.DataFrame(rows)


def analyse(raw_df: pd.DataFrame, sda_mlra_set: set, totals_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (series_summary_df, per_mlra_df)."""
    df = raw_df.copy()

    # Normalised ecosite for deduplication
    df["_eco_norm"] = df["ecoclassid"].apply(_norm_ecosite)

    # -----------------------------------------------------------------------
    # Series-level summary (across all MLRAs combined)
    # -----------------------------------------------------------------------
    def agg_series(grp):
        # Unique normalised ecosites
        unique_eco = sorted(grp["_eco_norm"].unique().tolist())
        # Keep original IDs for display (first occurrence per normalised)
        eco_orig = {}
        for _, row in grp.iterrows():
            en = row["_eco_norm"]
            if en not in eco_orig:
                eco_orig[en] = row["ecoclassid"]
        # MLRAs where this series occurs
        mlras_present = sorted(
            [m for m in grp["mlrasymbol"].unique().tolist() if m is not None]
        )
        return pd.Series({
            "n_ecosites": len(unique_eco),
            "ecosite_ids": "; ".join(eco_orig[e] for e in unique_eco),
            "ecosite_ids_normalised": "; ".join(unique_eco),
            "n_mlras": len(mlras_present),
            "mlras": "; ".join(mlras_present),
        })

    series_df = (
        df.groupby("compname", sort=True)
        .apply(agg_series, include_groups=False)
        .reset_index()
    )
    series_df = series_df.sort_values(["n_ecosites", "compname"], ascending=[False, True])

    # -----------------------------------------------------------------------
    # Per-MLRA breakdown: for each MLRA, how many series have >1 ecosite
    # -----------------------------------------------------------------------
    per_mlra_rows = []
    for mlra in sorted(sda_mlra_set):
        sub = df[df["mlrasymbol"] == mlra]
        if sub.empty:
            continue
        mlra_series = sub.groupby("compname")["_eco_norm"].nunique()
        total = len(mlra_series)
        multi = (mlra_series > 1).sum()
        per_mlra_rows.append({
            "mlrasymbol": mlra,
            "total_series_with_ecosite": total,
            "series_with_multiple_ecosites": int(multi),
            "pct_multiple": round(100.0 * multi / total, 1) if total else 0.0,
        })
    per_mlra_df = pd.DataFrame(per_mlra_rows)

    # Merge in state total compname counts if provided
    if totals_df is not None:
        per_mlra_df = per_mlra_df.merge(totals_df, on="mlrasymbol", how="left")
        per_mlra_df["total_compnames"] = per_mlra_df["total_compnames"].fillna(0).astype(int)
    else:
        per_mlra_df["total_compnames"] = None

    return series_df, per_mlra_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse compname–ecosite correlation multiplicity within a state's surveys"
    )
    parser.add_argument(
        "--plot-csv",
        default=str(DATA_DIR / "study_plot_characteristics.csv"),
        help="Plot CSV with MLRA column (used to derive MLRA scope when --mlras not given)",
    )
    parser.add_argument(
        "--state",
        default="NV",
        metavar="XX",
        help="Two-letter state code to restrict areasymbol and ecoclassid filters (default: NV)",
    )
    parser.add_argument(
        "--mlras",
        nargs="+",
        default=None,
        metavar="MLRA",
        help=(
            "One or more SDA mlrasymbol values to restrict the query "
            "(e.g. --mlras 28A 28B 34).  When omitted the script reads the "
            "MLRA column from --plot-csv to derive the list.  "
            "Pass --mlras ALL to query all MLRAs present in the state with no filter."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--by-order",
        action="store_true",
        default=False,
        help="Print and save a breakdown of compname-ecosite multiplicity by survey order (mapunit.invesintens)",
    )
    args = parser.parse_args()
    state = args.state.strip().upper()

    plot_path = Path(args.plot_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Determine MLRA scope
    # -----------------------------------------------------------------------
    plot_df = None
    if args.mlras and args.mlras[0].upper() == "ALL":
        sda_mlras = []   # empty list → no MLRA filter in queries
        logger.info(f"MLRA scope: ALL (no filter) for state {state}")
    elif args.mlras:
        sda_mlras = sorted(set(m.strip().upper() for m in args.mlras))
        logger.info(f"MLRA scope from --mlras: {sda_mlras}")
    else:
        logger.info(f"Loading plot CSV: {plot_path}")
        plot_df = pd.read_csv(plot_path, dtype=str)
        csv_mlras = sorted(plot_df["MLRA"].dropna().unique().tolist())
        logger.info(f"CSV MLRA codes ({len(csv_mlras)}): {csv_mlras}")
        sda_mlras_raw = [_csv_mlra_to_sda(c) for c in csv_mlras]
        sda_mlras = sorted(set(sda_mlras_raw))
        logger.info(f"SDA mlrasymbol targets ({len(sda_mlras)}): {sda_mlras}")
        mapping = {c: _csv_mlra_to_sda(c) for c in csv_mlras}
        for csv_code, sda_code in sorted(mapping.items()):
            logger.info(f"  {csv_code} → {sda_code}")

    # -----------------------------------------------------------------------
    # 2. Query SDA
    # -----------------------------------------------------------------------
    raw_df = query_mlra_compname_ecosites(sda_mlras, state=state)
    if raw_df is None or raw_df.empty:
        logger.error("No data returned from SDA. Exiting.")
        return

    # Collect the MLRAs actually returned (may differ from requested when ALL)
    returned_mlras = sorted(
        [m for m in raw_df["mlrasymbol"].unique().tolist() if m is not None]
    )
    if not sda_mlras:
        sda_mlras = returned_mlras

    # -----------------------------------------------------------------------
    # 2b. Query total compnames per MLRA (all, with or without ecosite)
    # -----------------------------------------------------------------------
    totals_df = query_total_compnames_by_state(sda_mlras, state=state)

    # -----------------------------------------------------------------------
    # 3. Analyse
    # -----------------------------------------------------------------------
    series_df, per_mlra_df = analyse(raw_df, set(sda_mlras), totals_df)

    # -----------------------------------------------------------------------
    # 4. Save CSVs
    # -----------------------------------------------------------------------
    # Raw pairs
    state_lc = state.lower()
    raw_path = output_dir / f"{state_lc}_compname_ecosite_raw_pairs.csv"
    raw_df.drop(columns=["_eco_norm"] if "_eco_norm" in raw_df.columns else []).to_csv(raw_path, index=False)
    logger.info(f"Raw pairs saved: {raw_path}")

    # Series summary
    series_path = output_dir / f"{state_lc}_compname_ecosite_series_summary.csv"
    series_df.to_csv(series_path, index=False)
    logger.info(f"Series summary saved: {series_path}")

    # Per-MLRA
    mlra_path = output_dir / f"{state_lc}_compname_ecosite_per_mlra.csv"
    per_mlra_df.to_csv(mlra_path, index=False)
    logger.info(f"Per-MLRA summary saved: {mlra_path}")

    # -----------------------------------------------------------------------
    # 5. Print summary
    # -----------------------------------------------------------------------
    total_series = len(series_df)
    multi_eco = (series_df["n_ecosites"] > 1).sum()
    single_eco = (series_df["n_ecosites"] == 1).sum()
    pct_multi = round(100.0 * multi_eco / total_series, 1) if total_series else 0.0

    print(f"\n=== Compname–Ecosite Correlation Summary ({state}) ===\n")
    mlra_scope = ', '.join(sda_mlras) if sda_mlras else "ALL"
    print(f"State                  : {state}")
    print(f"MLRAs queried          : {mlra_scope}")
    print(f"Total compname rows    : {len(raw_df)}")
    print(f"Unique compnames with >=1 ecosite: {total_series}")
    print(f"  Single ecosite only  : {single_eco}  ({round(100.0*single_eco/total_series,1)}%)")
    print(f"  Multiple ecosites    : {multi_eco}  ({pct_multi}%)")

    # Distribution of ecosite counts
    print(f"\nEcosite count distribution:")
    for n, cnt in series_df["n_ecosites"].value_counts().sort_index().items():
        bar = "#" * min(cnt, 60)
        print(f"  {n:3d} ecosite(s): {cnt:5d} series  {bar}")

    print(f"\nPer-MLRA breakdown:")
    cols = [c for c in ["mlrasymbol", "total_compnames", "total_series_with_ecosite", "series_with_multiple_ecosites", "pct_multiple"] if c in per_mlra_df.columns]
    print(per_mlra_df[cols].to_string(index=False))

    print(f"\nTop 20 series with most ecosites:")
    cols = ["compname", "n_ecosites", "n_mlras", "mlras", "ecosite_ids"]
    print(series_df.head(20)[cols].to_string(index=False))

    # -----------------------------------------------------------------------
    # 5b. Optional: by-order breakdown
    # -----------------------------------------------------------------------
    if args.by_order:
        order_df = analyse_by_order(raw_df, state)
        order_path = output_dir / f"{state_lc}_compname_ecosite_by_order.csv"
        order_df.to_csv(order_path, index=False)
        logger.info(f"By-order summary saved: {order_path}")
        print(f"\nBreakdown by survey order (mapunit.invesintens):")
        print(order_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # 6. AIM/QC study-plot subset analysis (only when plot_df was loaded)
    # -----------------------------------------------------------------------
    aim_col = "aim_series_component_name"
    qc_col = "qc_series_component_name"
    if plot_df is not None and aim_col in plot_df.columns and qc_col in plot_df.columns:
        logger.info(f"Running AIM/QC subset analysis ({aim_col} / {qc_col}) ...")
        sub_matched, sub_per_mlra, sub_missing = analyse_aim_qc_subset(
            plot_df, raw_df, aim_col=aim_col, qc_col=qc_col
        )
        aim_n = len(analyse_aim_qc_subset._aim_norm)
        qc_n = len(analyse_aim_qc_subset._qc_norm)
        both_n = len(analyse_aim_qc_subset._aim_norm & analyse_aim_qc_subset._qc_norm)
        all_n = len(analyse_aim_qc_subset._all_norm)

        sub_per_series = (
            sub_matched.groupby("compname")["_enorm"].nunique().reset_index()
        )
        sub_per_series.columns = ["compname", "n_eco"]
        sub_total = len(sub_per_series)
        sub_single = int((sub_per_series["n_eco"] == 1).sum())
        sub_multi = int((sub_per_series["n_eco"] > 1).sum())

        print(f"\n=== AIM/QC Study Plot Series Subset ===\n")
        print(f"AIM series ({aim_col})     : {aim_n}")
        print(f"QC series ({qc_col})      : {qc_n}")
        print(f"In both                   : {both_n}")
        print(f"Combined unique           : {all_n}")
        print(f"Matched to SDA {state} data   : {sub_total} ({round(sub_total/all_n*100,1)}%)")
        print(f"Not found in SDA {state} data : {len(sub_missing)}")
        if sub_missing:
            print(f"  Missing: {sorted(sub_missing)}")
        print(f"SDA rows in subset        : {len(sub_matched)}")
        print(f"Unique raw ecosite IDs    : {sub_matched['ecoclassid'].nunique()}")
        print(f"Unique norm ecosite IDs   : {sub_matched['_enorm'].nunique()}")
        print(f"Single {state} ecosite      : {sub_single} ({round(sub_single/sub_total*100,1)}%)")
        print(f"Multiple {state} ecosites   : {sub_multi} ({round(sub_multi/sub_total*100,1)}%)")
        print(f"\nPer-MLRA (AIM/QC subset):")
        print(sub_per_mlra.to_string(index=False))
        sub_top = (
            sub_matched.groupby("compname")
            .agg(n_eco=("_enorm", "nunique"), n_mlras=("mlrasymbol", "nunique"),
                 mlras=("mlrasymbol", lambda x: ", ".join(sorted(str(v) for v in x.unique()))))
            .reset_index()
            .sort_values("n_eco", ascending=False)
            .head(10)
        )
        print(f"\nTop 10 subset compnames:")
        print(sub_top.to_string(index=False))


if __name__ == "__main__":
    main()
