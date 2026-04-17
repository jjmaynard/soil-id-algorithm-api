"""
sda_series_ecosite_distance.py

For each terrain_qc "series-match / ecosite-mismatch" plot, this script:

  1. Takes the predicted soil series (rank_soils top result) and the expected
     ecological site (QC reference).
  2. Queries SDA to find every map unit in SSURGO where that series IS
     associated with that expected ecological site.
  3. Computes the Haversine distance from the plot to the centroid of each
     matching map unit polygon.
  4. Reports the nearest association, how many map units share it, and which
     survey areas they fall in.

Output columns
--------------
  PrimaryKey                  – plot identifier
  plot_lat / plot_lon         – NADS83 coordinates
  terrain_qc_soil_series      – series predicted by rank_soils
  soilweb_ecological_site     – ecosite SoilWeb returned for that series
  qc_expected_ecological_site – ecosite the QC reference records for this plot
  n_matching_mukeys           – number of SSURGO map units where series ∩ expected_ecosite
  n_survey_areas              – number of distinct survey areas that contain the combo
  association_found           – True if ≥1 matching map unit was found
  closest_dist_km             – Haversine distance to nearest matching polygon centroid
  closest_mukey               – mukey of that nearest polygon
  closest_areasymbol          – survey area (e.g. NV025)
  closest_areaname            – survey area full name
  closest_centroid_lat/lon    – centroid of nearest polygon

Usage
-----
  python scripts/sda_series_ecosite_distance.py
  python scripts/sda_series_ecosite_distance.py --results-csv Data/aim_data/my_run_results.csv
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
SDA_TIMEOUT = 30   # seconds per query
MUKEY_CHUNK = 250  # max mukeys per mupolygon query (IN clause limit)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _norm_text(v):
    if v is None:
        return ""
    if isinstance(v, float):
        try:
            if math.isnan(v):
                return ""
        except Exception:
            pass
    text = str(v).strip()
    return "" if text.lower() in {"", "nan", "none", "na"} else text


def _norm_ecosite_bare(value):
    """Strip the leading R/F prefix and all non-alphanumeric chars, lowercase.
    Used to build a safe LIKE pattern for SDA comparisons."""
    site = _norm_text(value).lower()
    if not site:
        return ""
    if site.startswith(("r", "f")):
        site = site[1:]
    site = re.sub(r"_\d+$", "", site)   # strip underscore + numeric suffix (e.g. _001)
    return re.sub(r"[^a-z0-9]", "", site)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two WGS84 points in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# SDA helpers
# ---------------------------------------------------------------------------

def _sda_query(sql: str, timeout: int = SDA_TIMEOUT):
    """Send a T-SQL query to SDA, return a DataFrame or None on failure."""
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
        # SDA JSON+COLUMNNAME: raw[0] is column-name list; raw[1:] are data rows
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


def query_series_ecosite_polygons(compname: str, ecosite_bare: str) -> pd.DataFrame | None:
    """Two-phase SDA lookup for all mupolygon centroids where compname is
    associated with the given ecosite.

    Phase 1 (tabular): Get all mukeys + areasymbol where the series has that ecosite.
    Phase 2 (spatial): Get mupolygon centroid coordinates for those mukeys.

    Returns a DataFrame with columns:
        mukey, areasymbol, areaname, centroid_lat, centroid_lon
    or None if nothing found.
    """
    compname_esc = compname.replace("'", "''")
    ecosite_upper = ecosite_bare.upper()

    # ------------------------------------------------------------------
    # Phase 1: tabular join — get mukeys, survey areas
    # ------------------------------------------------------------------
    sql1 = (
        f"SELECT DISTINCT mu.mukey, l.areasymbol, l.areaname "
        f"FROM component c "
        f"JOIN mapunit mu ON c.mukey = mu.mukey "
        f"JOIN legend l ON mu.lkey = l.lkey "
        f"JOIN coecoclass ce ON c.cokey = ce.cokey "
        f"WHERE UPPER(c.compname) = UPPER('{compname_esc}') "
        f"AND UPPER(ce.ecoclassid) LIKE '%{ecosite_upper}%'"
    )
    tab_df = _sda_query(sql1, timeout=SDA_TIMEOUT)
    if tab_df is None or tab_df.empty:
        return None

    mukeys = tab_df["mukey"].dropna().astype(str).tolist()
    # Build a lookup mukey → (areasymbol, areaname)
    meta_map = {
        str(r["mukey"]): (str(r["areasymbol"]), str(r["areaname"]))
        for _, r in tab_df.iterrows()
    }

    # ------------------------------------------------------------------
    # Phase 2: mupolygon spatial — get centroid coords per polygon part
    # Chunk to avoid IN-clause overflow
    # ------------------------------------------------------------------
    centroid_rows = []
    for i in range(0, len(mukeys), MUKEY_CHUNK):
        chunk = mukeys[i : i + MUKEY_CHUNK]
        qs = ", ".join(f"'{k}'" for k in chunk)
        sql2 = (
            f"SELECT mukey, "
            f"mupolygongeo.STCentroid().STY AS centroid_lat, "
            f"mupolygongeo.STCentroid().STX AS centroid_lon "
            f"FROM mupolygon "
            f"WHERE mukey IN ({qs})"
        )
        geo_df = _sda_query(sql2, timeout=SDA_TIMEOUT)
        if geo_df is not None and not geo_df.empty:
            centroid_rows.append(geo_df)
        time.sleep(0.2)

    if not centroid_rows:
        # Fall back to sapolygon survey area centroids
        areasymbols = tab_df["areasymbol"].dropna().unique().tolist()
        qs = ", ".join(f"'{a}'" for a in areasymbols)
        sql_sa = (
            f"SELECT areasymbol, "
            f"sapolygongeo.STCentroid().STY AS centroid_lat, "
            f"sapolygongeo.STCentroid().STX AS centroid_lon "
            f"FROM sapolygon "
            f"WHERE areasymbol IN ({qs})"
        )
        sa_df = _sda_query(sql_sa, timeout=SDA_TIMEOUT)
        if sa_df is None or sa_df.empty:
            return None
        # Attach areaname from tab_df
        sa_meta = tab_df[["areasymbol", "areaname"]].drop_duplicates("areasymbol")
        sa_df = sa_df.merge(sa_meta, on="areasymbol", how="left")
        sa_df["mukey"] = ""
        return sa_df[["mukey", "areasymbol", "areaname", "centroid_lat", "centroid_lon"]]

    combined = pd.concat(centroid_rows, ignore_index=True)
    # Attach areasymbol + areaname from the meta map
    combined["areasymbol"] = combined["mukey"].map(lambda k: meta_map.get(str(k), ("", ""))[0])
    combined["areaname"] = combined["mukey"].map(lambda k: meta_map.get(str(k), ("", ""))[1])
    return combined[["mukey", "areasymbol", "areaname", "centroid_lat", "centroid_lon"]]


# ---------------------------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------------------------

def nearest_polygon(plot_lat: float, plot_lon: float, poly_df: pd.DataFrame):
    """Given a DataFrame with centroid_lat / centroid_lon columns, return the
    row whose centroid is closest to (plot_lat, plot_lon), plus its distance."""
    if poly_df is None or poly_df.empty:
        return None, None

    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    best_row = None
    best_dist = float("inf")
    for _, row in poly_df.iterrows():
        clat = safe_float(row.get("centroid_lat"))
        clon = safe_float(row.get("centroid_lon"))
        if clat is None or clon is None:
            continue
        d = haversine_km(plot_lat, plot_lon, clat, clon)
        if d < best_dist:
            best_dist = d
            best_row = row

    return best_row, best_dist if best_dist < float("inf") else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_latest_results(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*_run_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No run_results CSV found in {data_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Find nearest SSURGO association of predicted series + expected ecosite"
    )
    parser.add_argument("--results-csv", default=None,
                        help="run_results CSV (default: latest in Data/aim_data/)")
    parser.add_argument("--plot-csv",
                        default=str(DATA_DIR / "study_plot_characteristics.csv"),
                        help="Plot CSV with Latitude_NAD83 / Longitude_NAD83")
    parser.add_argument("--output-dir", default=str(DATA_DIR),
                        help="Directory for output CSV")
    args = parser.parse_args()

    results_path = (Path(args.results_csv) if args.results_csv
                    else _find_latest_results(DATA_DIR))
    plot_path = Path(args.plot_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results: {results_path}")
    results_df = pd.read_csv(results_path, dtype=str)

    logger.info(f"Loading plot data: {plot_path}")
    plot_df = pd.read_csv(plot_path, dtype=str)

    # -----------------------------------------------------------------------
    # 1. Filter to terrain_qc series-match / ecosite-mismatch rows
    # -----------------------------------------------------------------------
    passed = results_df[results_df["status"] == "passed"].copy()
    mismatch = passed[
        (passed["terrain_qc_soil_series_match"].str.strip().str.upper() == "TRUE")
        & (passed["terrain_qc_ecological_site_match"].str.strip().str.upper() == "FALSE")
    ].copy()
    logger.info(f"terrain_qc series-match / ecosite-mismatch rows: {len(mismatch)}")

    # Merge lat/lon
    coord_df = plot_df[["PrimaryKey", "Latitude_NAD83", "Longitude_NAD83"]].drop_duplicates("PrimaryKey")
    mismatch = mismatch.merge(coord_df, on="PrimaryKey", how="left")

    missing_coords = mismatch[mismatch["Latitude_NAD83"].isna() | mismatch["Longitude_NAD83"].isna()]
    if not missing_coords.empty:
        logger.warning(f"{len(missing_coords)} rows missing coordinates; they will be skipped.")

    # -----------------------------------------------------------------------
    # 2. Cache SDA polygon lookups by unique (series, expected_ecosite)
    # -----------------------------------------------------------------------
    pairs = (
        mismatch[["terrain_qc_soil_series", "qc_expected_ecological_site"]]
        .drop_duplicates()
        .values.tolist()
    )
    logger.info(f"Unique (series, expected_ecosite) pairs to query: {len(pairs)}")

    polygon_cache: dict[tuple, pd.DataFrame | None] = {}

    for idx, (series, expected_ecosite) in enumerate(pairs, 1):
        ecosite_bare = _norm_ecosite_bare(expected_ecosite)
        series_clean = _norm_text(series)
        key = (series_clean.lower(), ecosite_bare)

        if not series_clean or not ecosite_bare:
            logger.info(f"  [{idx}/{len(pairs)}] Skipping empty series or ecosite")
            polygon_cache[key] = None
            continue

        logger.info(
            f"  [{idx}/{len(pairs)}] Querying: '{series_clean}' + '%{ecosite_bare.upper()}%' ..."
        )
        poly_df = query_series_ecosite_polygons(series_clean, ecosite_bare)

        if poly_df is None or poly_df.empty:
            logger.info(f"    → No SSURGO associations found")
        else:
            n_mu = poly_df["mukey"].nunique()
            n_sa = poly_df["areasymbol"].nunique()
            logger.info(f"    → {n_mu} map units in {n_sa} survey areas")

        polygon_cache[key] = poly_df
        time.sleep(0.3)   # polite rate-limiting

    # -----------------------------------------------------------------------
    # 3. For each mismatch row compute distances
    # -----------------------------------------------------------------------
    out_rows = []
    for _, row in mismatch.iterrows():
        series_clean = _norm_text(row["terrain_qc_soil_series"])
        expected_ecosite = _norm_text(row["qc_expected_ecological_site"])
        ecosite_bare = _norm_ecosite_bare(expected_ecosite)
        key = (series_clean.lower(), ecosite_bare)

        poly_df = polygon_cache.get(key)

        try:
            plot_lat = float(row["Latitude_NAD83"])
            plot_lon = float(row["Longitude_NAD83"])
            coords_ok = True
        except (TypeError, ValueError):
            plot_lat = plot_lon = None
            coords_ok = False

        if poly_df is not None and not poly_df.empty:
            n_mukeys = poly_df["mukey"].nunique()
            n_survey_areas = poly_df["areasymbol"].nunique()
            survey_area_list = "; ".join(sorted(poly_df["areasymbol"].dropna().unique()))
            association_found = True

            if coords_ok:
                best_row, best_dist = nearest_polygon(plot_lat, plot_lon, poly_df)
            else:
                best_row, best_dist = None, None
        else:
            n_mukeys = 0
            n_survey_areas = 0
            survey_area_list = ""
            association_found = False
            best_row, best_dist = None, None

        out_rows.append(
            {
                "PrimaryKey": _norm_text(row.get("PrimaryKey")),
                "source": _norm_text(row.get("source")),
                "plot_lat": plot_lat,
                "plot_lon": plot_lon,
                "terrain_qc_soil_series": series_clean,
                "soilweb_ecological_site": _norm_text(row.get("terrain_qc_ecological_site")),
                "qc_expected_ecological_site": expected_ecosite,
                "n_matching_mukeys": n_mukeys,
                "n_survey_areas": n_survey_areas,
                "all_survey_areas": survey_area_list,
                "association_found": association_found,
                "closest_dist_km": round(best_dist, 2) if best_dist is not None else "",
                "closest_mukey": _norm_text(best_row["mukey"]) if best_row is not None else "",
                "closest_areasymbol": _norm_text(best_row["areasymbol"]) if best_row is not None else "",
                "closest_areaname": _norm_text(best_row["areaname"]) if best_row is not None else "",
                "closest_centroid_lat": (
                    round(float(best_row["centroid_lat"]), 5) if best_row is not None else ""
                ),
                "closest_centroid_lon": (
                    round(float(best_row["centroid_lon"]), 5) if best_row is not None else ""
                ),
            }
        )

    # -----------------------------------------------------------------------
    # 4. Write output and print summary
    # -----------------------------------------------------------------------
    out_df = pd.DataFrame(out_rows)
    stem = results_path.stem
    out_path = output_dir / f"series_ecosite_distance_{stem}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"\nOutput written: {out_path}  ({len(out_df)} rows)")

    # Summary
    print("\n=== terrain_qc Series–Ecosite Distance Summary ===\n")
    total = len(out_df)
    found = out_df["association_found"].sum()
    not_found = total - found
    print(f"Total mismatch plots : {total}")
    print(f"Association found    : {found}")
    print(f"Not found in SSURGO  : {not_found}")

    if found > 0:
        dist_rows = out_df[out_df["closest_dist_km"] != ""].copy()
        dist_rows["closest_dist_km"] = dist_rows["closest_dist_km"].astype(float)
        print(f"\nDistance to nearest SSURGO occurrence (km):")
        print(f"  Min    : {dist_rows['closest_dist_km'].min():.1f}")
        print(f"  Median : {dist_rows['closest_dist_km'].median():.1f}")
        print(f"  Max    : {dist_rows['closest_dist_km'].max():.1f}")

        bins = [0, 10, 50, 100, 250, 500, float("inf")]
        labels = ["<10", "10–50", "50–100", "100–250", "250–500", ">500"]
        dist_rows["dist_bin"] = pd.cut(dist_rows["closest_dist_km"], bins=bins, labels=labels, right=False)
        print(f"\nDistance distribution:")
        for label, cnt in dist_rows["dist_bin"].value_counts().sort_index().items():
            print(f"  {label:>8} km : {cnt}")

        print(f"\nPlots with association found (sorted by distance):")
        cols = ["PrimaryKey", "terrain_qc_soil_series", "qc_expected_ecological_site",
                "closest_dist_km", "closest_areasymbol", "n_matching_mukeys"]
        print(
            dist_rows.sort_values("closest_dist_km")[cols]
            .to_string(index=False)
        )

    if not_found > 0:
        print(f"\nPlots with NO SSURGO association found:")
        nf = out_df[~out_df["association_found"]][
            ["PrimaryKey", "terrain_qc_soil_series", "qc_expected_ecological_site"]
        ]
        print(nf.to_string(index=False))


if __name__ == "__main__":
    main()
