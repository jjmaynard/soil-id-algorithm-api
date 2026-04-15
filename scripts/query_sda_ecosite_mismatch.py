"""
query_sda_ecosite_mismatch.py

For each "series-match / ecosite-mismatch" point in a run-results CSV, query
SDA's coecoclass table directly using the matched cokey reported by rank_soils
and compare the SDA ecoclassid back against:
  - the ecological site that SoilWeb returned via rank_soils
  - the expected ecological site recorded in the AIM/QC reference data

This helps diagnose *why* the ecosite is wrong:

  soilweb_matches_sda == True  & sda_matches_expected == False
      → SoilWeb and SDA agree, but the reference data differs (reference issue)

  soilweb_matches_sda == False & sda_matches_expected == True
      → SDA has the correct site but SoilWeb returned a different one (SoilWeb cache / lag)

  soilweb_matches_sda == False & sda_matches_expected == False
      → All three differ; both SoilWeb and SDA disagree with the reference

Usage
-----
  python scripts/query_sda_ecosite_mismatch.py
  python scripts/query_sda_ecosite_mismatch.py --results-csv Data/aim_data/my_results.csv
  python scripts/query_sda_ecosite_mismatch.py --output-dir Data/aim_data/sda_check
"""

import argparse
import logging
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
SDA_CHUNK = 200   # max cokeys per SDA POST request
SDA_TIMEOUT = 30  # seconds

# Comparison modes: (mode_name, component_id_col, soilweb_ecosite_col, expected_ecosite_col)
MODES = [
    ("baseline_aim",  "baseline_component_id",    "baseline_ecological_site",    "aim_expected_ecological_site"),
    ("terrain_aim",   "terrain_aim_component_id",  "terrain_aim_ecological_site", "aim_expected_ecological_site"),
    ("baseline_qc",   "baseline_component_id",     "baseline_ecological_site",    "qc_expected_ecological_site"),
    ("terrain_qc",    "terrain_qc_component_id",   "terrain_qc_ecological_site",  "qc_expected_ecological_site"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_text(v):
    if v is None or (isinstance(v, float)):
        try:
            import math
            if math.isnan(v):
                return ""
        except Exception:
            pass
    text = str(v).strip()
    return "" if text.lower() in {"", "nan", "none", "na"} else text


def _norm_ecosite(value):
    """Normalise an ecological site ID for comparison.

    Strips a leading 'R' or 'F' (range/forest prefix), removes non-alphanumeric
    characters and lowercases — matching the logic in run_all_aim_examples.py.
    """
    site = _norm_text(value).lower()
    if not site:
        return ""
    if site.startswith(("r", "f")):
        site = site[1:]
    site = re.sub(r"_\d+$", "", site)   # strip underscore + numeric suffix (e.g. _001)
    # strip non-alphanumeric
    site = re.sub(r"[^a-z0-9]", "", site)
    return site


def _ecosites_match(a, b):
    na, nb = _norm_ecosite(a), _norm_ecosite(b)
    if not na or not nb:
        return None  # can't judge
    return na == nb


def _cokey_str(raw):
    """Return canonical string cokey (strip trailing .0 etc.)."""
    s = _norm_text(raw)
    if not s:
        return ""
    return re.sub(r"\.0+$", "", s)


# ---------------------------------------------------------------------------
# SDA query
# ---------------------------------------------------------------------------

def _sda_return(query: str):
    """POST a T-SQL query to SDA and return a DataFrame, or None on failure."""
    payload = {"format": "JSON+COLUMNNAME", "query": query}
    try:
        resp = requests.post(SDA_URL, json=payload, timeout=SDA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "Table" not in data:
            return None
        from pandas import json_normalize
        result = json_normalize(data)
        return result
    except requests.ConnectionError as e:
        logger.warning(f"SDA connection error: {e}")
    except requests.Timeout:
        logger.warning("SDA request timed out")
    except requests.RequestException as e:
        logger.warning(f"SDA request error: {e}")
    return None


def query_sda_ecoclassid(cokeys: list[str]) -> dict[str, list[dict]]:
    """Query SDA coecoclass for a list of cokeys.

    Returns a dict keyed by cokey → list of {"ecoclassid": ..., "ecoclassname": ...}
    (a cokey can have >1 ecosite assignment in SDA).
    """
    cokeys = [k for k in cokeys if k]
    result: dict[str, list[dict]] = {k: [] for k in cokeys}

    for i in range(0, len(cokeys), SDA_CHUNK):
        chunk = cokeys[i : i + SDA_CHUNK]
        quoted = ", ".join(f"'{k}'" for k in chunk)
        query = (
            "SELECT cokey, ecoclassid, ecoclassname "
            "FROM coecoclass "
            f"WHERE cokey IN ({quoted}) "
            "ORDER BY cokey"
        )
        logger.info(
            f"  Querying SDA for cokeys {i+1}–{min(i+len(chunk), len(cokeys))} / {len(cokeys)} ..."
        )
        out = _sda_return(query)
        if out is None:
            logger.warning(f"  SDA returned no data for chunk starting at index {i}")
            time.sleep(1)
            continue

        # Parse nested table: out["Table"].iloc[0] is a list; [0] = col headers, [1:] = rows
        try:
            raw_table = out["Table"].iloc[0]
            headers = raw_table[0]
            rows = raw_table[1:]
            sda_df = pd.DataFrame(rows, columns=headers)
        except Exception as e:
            logger.warning(f"  Failed to parse SDA response: {e}")
            continue

        for _, row in sda_df.iterrows():
            ck = _cokey_str(row.get("cokey", ""))
            if ck not in result:
                result[ck] = []
            ecoclassid = _norm_text(row.get("ecoclassid", ""))
            ecoclassname = _norm_text(row.get("ecoclassname", ""))
            if ecoclassid and ecoclassid.lower() not in {"none", "nan"}:
                result[ck].append({"ecoclassid": ecoclassid, "ecoclassname": ecoclassname})

        time.sleep(0.25)  # be polite to SDA

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_latest_results(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*_run_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No run_results CSV found in {data_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Query SDA for ecosite data on mismatched points")
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Path to run-results CSV (default: latest in Data/aim_data/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="Directory for output CSV (default: Data/aim_data/)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline_aim", "terrain_aim", "baseline_qc", "terrain_qc"],
        default=["baseline_aim", "terrain_aim", "baseline_qc", "terrain_qc"],
        help="Comparison modes to analyse (default: all four)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_csv) if args.results_csv else _find_latest_results(DATA_DIR)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results: {results_path}")
    df = pd.read_csv(results_path, dtype=str)

    passed = df[df["status"] == "passed"].copy()
    logger.info(f"Passed rows: {len(passed)}")

    active_modes = [m for m in MODES if m[0] in args.modes]

    # -----------------------------------------------------------------------
    # 1. Build mismatch table
    # -----------------------------------------------------------------------
    mismatch_rows = []

    for mode_name, comp_id_col, sw_eco_col, exp_eco_col in active_modes:
        series_match_col = f"{mode_name}_soil_series_match"
        ecosite_match_col = f"{mode_name}_ecological_site_match"

        if series_match_col not in passed.columns or ecosite_match_col not in passed.columns:
            logger.warning(f"Mode {mode_name}: required columns missing, skipping")
            continue

        subset = passed[
            (passed[series_match_col].str.strip().str.upper() == "TRUE")
            & (passed[ecosite_match_col].str.strip().str.upper() == "FALSE")
        ].copy()

        logger.info(f"Mode {mode_name}: {len(subset)} series-match / ecosite-mismatch rows")

        for _, row in subset.iterrows():
            cokey = _cokey_str(row.get(comp_id_col, ""))
            mismatch_rows.append(
                {
                    "PrimaryKey": _norm_text(row.get("PrimaryKey")),
                    "source": _norm_text(row.get("source")),
                    "comparison_mode": mode_name,
                    "cokey": cokey,
                    "soil_series_predicted": _norm_text(row.get(sw_eco_col.replace("ecological_site", "soil_series"), "")),
                    "expected_ecological_site": _norm_text(row.get(exp_eco_col, "")),
                    "soilweb_ecological_site": _norm_text(row.get(sw_eco_col, "")),
                }
            )

    if not mismatch_rows:
        logger.info("No mismatch rows found — nothing to query.")
        return

    mismatch_df = pd.DataFrame(mismatch_rows)
    logger.info(f"Total mismatch rows across all modes: {len(mismatch_df)}")

    # Collect unique cokeys
    unique_cokeys = [k for k in mismatch_df["cokey"].unique().tolist() if k]
    logger.info(f"Unique cokeys to query in SDA: {len(unique_cokeys)}")

    # -----------------------------------------------------------------------
    # 2. Query SDA
    # -----------------------------------------------------------------------
    logger.info("Querying SDA coecoclass ...")
    sda_map = query_sda_ecoclassid(unique_cokeys)
    logger.info("SDA query complete.")

    # Stats
    n_found = sum(1 for v in sda_map.values() if v)
    logger.info(f"Cokeys with ≥1 SDA ecoclassid: {n_found} / {len(unique_cokeys)}")

    # -----------------------------------------------------------------------
    # 3. Augment mismatch table with SDA data and comparison columns
    # -----------------------------------------------------------------------
    out_rows = []
    for _, row in mismatch_df.iterrows():
        ck = row["cokey"]
        sda_entries = sda_map.get(ck, [])

        # Primary SDA ecosite (first entry) — SDA can return multiple per cokey
        sda_ecoclassid = sda_entries[0]["ecoclassid"] if sda_entries else ""
        sda_ecoclassname = sda_entries[0]["ecoclassname"] if sda_entries else ""
        sda_all_ecoclassids = "; ".join(e["ecoclassid"] for e in sda_entries)

        expected = row["expected_ecological_site"]
        soilweb = row["soilweb_ecological_site"]

        soilweb_matches_sda = _ecosites_match(soilweb, sda_ecoclassid)
        sda_matches_expected = _ecosites_match(sda_ecoclassid, expected)
        soilweb_matches_expected = _ecosites_match(soilweb, expected)

        # Check if ANY of the SDA entries matches expected (multi-ecosite cokeys)
        sda_any_matches_expected = (
            any(_ecosites_match(e["ecoclassid"], expected) for e in sda_entries)
            if sda_entries
            else None
        )

        # Diagnosis
        no_sw = not _norm_ecosite(soilweb)
        no_sda = not sda_ecoclassid

        if no_sda:
            if no_sw:
                diagnosis = "No SoilWeb or SDA ecosite data; reference has value"
            else:
                diagnosis = "No SDA ecosite data; SoilWeb returned value but differs from reference"
        elif no_sw:
            # SDA has data, SoilWeb returned nothing
            if sda_matches_expected is True:
                diagnosis = "SoilWeb returned no ecosite; SDA matches reference"
            elif sda_matches_expected is False:
                diagnosis = "SoilWeb returned no ecosite; SDA also differs from reference"
            else:
                diagnosis = "SoilWeb returned no ecosite; SDA has value but reference is unknown"
        elif soilweb_matches_sda is True and sda_matches_expected is False:
            diagnosis = "SoilWeb==SDA but both differ from reference"
        elif soilweb_matches_sda is False and sda_matches_expected is True:
            diagnosis = "SDA==reference but SoilWeb differs (SoilWeb lag/different data)"
        elif soilweb_matches_sda is True and sda_matches_expected is True:
            diagnosis = "All match (unexpected — should not appear in mismatch set)"
        elif soilweb_matches_sda is False and sda_matches_expected is False:
            diagnosis = "All three differ (SoilWeb, SDA, reference all different)"
        else:
            diagnosis = "Unknown"

        out_rows.append(
            {
                **row.to_dict(),
                "sda_ecoclassid": sda_ecoclassid,
                "sda_ecoclassname": sda_ecoclassname,
                "sda_all_ecoclassids": sda_all_ecoclassids,
                "soilweb_matches_sda": soilweb_matches_sda,
                "sda_matches_expected": sda_matches_expected,
                "sda_any_matches_expected": sda_any_matches_expected,
                "soilweb_matches_expected": soilweb_matches_expected,
                "diagnosis": diagnosis,
            }
        )

    out_df = pd.DataFrame(out_rows)

    # -----------------------------------------------------------------------
    # 4. Write output
    # -----------------------------------------------------------------------
    stem = results_path.stem
    out_path = output_dir / f"sda_ecosite_check_{stem}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Output written: {out_path}  ({len(out_df)} rows)")

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    print("\n=== SDA Ecosite Mismatch Summary ===\n")
    for mode_name, *_ in active_modes:
        sub = out_df[out_df["comparison_mode"] == mode_name]
        if sub.empty:
            continue
        print(f"[{mode_name}]  {len(sub)} mismatch rows")
        for diag, cnt in sub["diagnosis"].value_counts().items():
            print(f"  {cnt:3d}  {diag}")
        no_sda = (sub["sda_ecoclassid"] == "").sum()
        any_match = sub["sda_any_matches_expected"].dropna()
        n_any_match = (any_match == True).sum()  # noqa: E712
        print(f"  {no_sda:3d}  cokeys with no SDA ecosite record")
        print(f"  {n_any_match:3d}  cokeys where ≥1 SDA ecosite matches expected")
        print()


if __name__ == "__main__":
    main()
