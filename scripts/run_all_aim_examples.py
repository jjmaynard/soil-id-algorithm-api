import argparse
import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import soil_id.config
from soil_id.color import find_closest_rgb_in_reference, munsell2rgb
from soil_id.tests.us.test_study_dataset_inputs import (
    HORIZONS_CSV,
    TERRAIN_DATA_DIR,
    _to_float,
)
from soil_id.landscape_crosswalk import (
    aim_to_standard_class,
    build_sda_landscape_label,
    crosswalk_landscape_class,
    ssurgo_to_standard_class,
)
from soil_id.us_soil import list_soils, rank_soils


_COLOR_REF = None
_MUNSELL_REF = None

TEXTURE_ABBREV_MAP = {
    "S": "SAND",
    "LS": "LOAMY SAND",
    "SL": "SANDY LOAM",
    "L": "LOAM",
    "SIL": "SILT LOAM",
    "SI": "SILT",
    "SCL": "SANDY CLAY LOAM",
    "CL": "CLAY LOAM",
    "SICL": "SILTY CLAY LOAM",
    "SC": "SANDY CLAY",
    "SIC": "SILTY CLAY",
    "C": "CLAY",
}


def _norm_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "na"} else text


def _norm_compare(value):
    return _norm_text(value).lower()


def _norm_name(value):
    # normalise to lowercase, replace underscores with spaces, collapse whitespace
    return " ".join(_norm_compare(value).replace("_", " ").split())


def _norm_source(value):
    src = _norm_compare(value)
    if src.startswith("aim"):
        return "AIM"
    if src.startswith("qc"):
        return "QC"
    return "UNKNOWN"


def _pick_expected_soil_series(row):
    source = _norm_source(row.get("source") or row.get("Source"))
    if source == "AIM":
        return _first_nonempty(row, ["aim_series_component_name"])
    if source == "QC":
        return _first_nonempty(row, ["qc_series_component_name"])
    return _first_nonempty(row, ["aim_series_component_name", "qc_series_component_name"])


def _norm_ecological_site(value):
    site = _norm_compare(value)
    if not site:
        return ""
    if site.startswith(("r", "f")):
        return site[1:]
    return site


def _extract_top_metadata(rank_result, comp_meta_by_id, rank_method="rank_data_loc"):
    soil_rank = rank_result.get("soilRank") or []
    if not soil_rank:
        return {
            "soil_series": "",
            "ecological_site": "",
            "landscape_class": "",
            "component_id": "",
        }

    top = next(
        (item for item in soil_rank if _norm_text(item.get(rank_method)) == "1"),
        soil_rank[0],
    )
    component_id = _norm_text(top.get("componentID"))
    # Use canonical component group name (e.g. "Acuff") for matching/reporting.
    soil_series = _norm_text(top.get("component") or top.get("name"))
    meta = comp_meta_by_id.get(component_id, {})
    if not meta:
        meta = comp_meta_by_id.get(f"name::{_norm_name(soil_series)}", {})

    # Prefer landscape_class from the rank output (site_match.landscape.mapped),
    # which is populated in all execution modes including api mode.
    rank_landscape_class = _norm_text(
        top.get("site_match", {}).get("landscape", {}).get("mapped")
    )
    landscape_class = rank_landscape_class or _norm_text(meta.get("landscape_class"))

    horizon_match = top.get("horizon_match") or {}
    site_match = top.get("site_match") or {}

    def _sm(field, key):
        return site_match.get(field, {}).get(key)

    return {
        "soil_series": soil_series,
        "ecological_site": _norm_text(meta.get("ecological_site")),
        "landscape_class": landscape_class,
        "component_id": component_id,
        "score_data_horz": top.get("score_data_horz"),
        "score_data_site": top.get("score_data_site"),
        "horz_texture": horizon_match.get("texture"),
        "horz_rock_fragments": horizon_match.get("rock_fragments"),
        "horz_color": horizon_match.get("color"),
        "horz_observed_depth": horizon_match.get("observed_depth"),
        "horz_mapped_depth": horizon_match.get("mapped_depth"),
        "horz_depth_coverage": horizon_match.get("depth_coverage"),
        "site_slope_observed": _sm("slope", "observed"),
        "site_slope_mapped": _sm("slope", "mapped"),
        "site_elev_observed": _sm("elevation", "observed"),
        "site_elev_mapped": _sm("elevation", "mapped"),
        "site_aspect_north_observed": _sm("aspect_northerness", "observed"),
        "site_aspect_north_mapped": _sm("aspect_northerness", "mapped"),
        "site_aspect_east_observed": _sm("aspect_easterness", "observed"),
        "site_aspect_east_mapped": _sm("aspect_easterness", "mapped"),
        "site_shape_vert_observed": _sm("slope_shape_vertical", "observed"),
        "site_shape_vert_mapped": _sm("slope_shape_vertical", "mapped"),
        "site_shape_horiz_observed": _sm("slope_shape_horizontal", "observed"),
        "site_shape_horiz_mapped": _sm("slope_shape_horizontal", "mapped"),
        "site_landscape_observed": _sm("landscape", "observed"),
        "site_landscape_mapped": _sm("landscape", "mapped"),
    }


def _extract_expected_rank(rank_result, expected_soil_series, rank_method="rank_data_loc"):
    expected = _norm_name(expected_soil_series)
    if not expected:
        return {"component_id": "", "rank": "", "score_data_horz": None, "score_data_site": None}

    soil_rank = rank_result.get("soilRank") or []
    for item in soil_rank:
        item_component = _norm_name(item.get("component"))
        if item_component == expected:
            rank = _norm_text(item.get(rank_method))
            return {
                "component_id": _norm_text(item.get("componentID")),
                "rank": rank,
                "score_data_horz": item.get("score_data_horz"),
                "score_data_site": item.get("score_data_site"),
            }

    return {"component_id": "", "rank": "", "score_data_horz": None, "score_data_site": None}


def _extract_expected_ecosite_rank(rank_result, expected_ecosite, rank_method="rank_data_loc"):
    """Return the rank of the first soilRank item whose ecoclassid matches expected_ecosite."""
    expected_norm = _norm_ecological_site(expected_ecosite)
    if not expected_norm:
        return {"rank": ""}

    soil_rank = rank_result.get("soilRank") or []
    for item in soil_rank:
        ecoclassid = _norm_ecological_site(_norm_text(item.get("ecoclassid") or ""))
        if ecoclassid and ecoclassid == expected_norm:
            return {"rank": _norm_text(item.get(rank_method))}

    return {"rank": ""}


def _metadata_for_component(comp_meta_by_id, component_id="", component_name=""):
    component_id = _norm_text(component_id)
    if component_id and component_id in comp_meta_by_id:
        return comp_meta_by_id[component_id]

    name_key = f"name::{_norm_name(component_name)}"
    if _norm_name(component_name) and name_key in comp_meta_by_id:
        return comp_meta_by_id[name_key]

    return {"ecological_site": "", "landscape_type": "", "landscape_class": ""}


def _build_component_metadata(list_output_data):
    by_id = {}
    map_unit_component_data_csv = _norm_text(
        getattr(list_output_data, "map_unit_component_data_csv", "")
    )
    if not map_unit_component_data_csv and isinstance(list_output_data, dict):
        map_unit_component_data_csv = _norm_text(list_output_data.get("map_unit_component_data_csv"))

    if map_unit_component_data_csv:
        comp_df = pd.read_csv(io.StringIO(map_unit_component_data_csv))
        comp_df.columns = [str(c).strip() for c in comp_df.columns]

        for _, row in comp_df.iterrows():
            cokey = _norm_text(row.get("cokey"))
            compname = _norm_text(row.get("compname"))
            landscape_type = _norm_text(
                build_sda_landscape_label(
                    row.get("geomftname"),
                    row.get("geomfname"),
                    row.get("geomfmod"),
                    row.get("geomposmntn"),
                    row.get("geomposhill"),
                    row.get("geompostrce"),
                    row.get("geomposflats"),
                    row.get("shapeacross"),
                    row.get("shapedown"),
                )
            )
            metadata = {
                "landscape_type": landscape_type,
                "landscape_class": _norm_text(
                    row.get("landscape_class")
                    or ssurgo_to_standard_class(
                        geomftname=row.get("geomftname"),
                        geomfname=row.get("geomfname"),
                        geomfmod=row.get("geomfmod"),
                        geomposmntn=row.get("geomposmntn"),
                        geomposhill=row.get("geomposhill"),
                        geompostrce=row.get("geompostrce"),
                        geomposflats=row.get("geomposflats"),
                        shapeacross=row.get("shapeacross"),
                        shapedown=row.get("shapedown"),
                        mode="base",
                    )
                ),
                "ecological_site": _norm_ecological_site(
                    row.get("ecoclassid_update") or row.get("ecoclassid")
                ).upper(),
            }
            if not cokey:
                if compname:
                    by_id[f"name::{_norm_name(compname)}"] = metadata
                continue
            by_id[cokey] = metadata
            if compname and f"name::{_norm_name(compname)}" not in by_id:
                by_id[f"name::{_norm_name(compname)}"] = metadata

    soil_list_json = getattr(list_output_data, "soil_list_json", None)
    if soil_list_json is None and isinstance(list_output_data, dict):
        soil_list_json = list_output_data.get("soil_list_json")
    if not isinstance(soil_list_json, dict):
        soil_list_json = {}

    for entry in soil_list_json.get("soilList", []):
        id_obj = entry.get("id", {}) if isinstance(entry, dict) else {}
        site_data = (
            entry.get("site", {}).get("siteData", {}) if isinstance(entry, dict) else {}
        )
        comp_id = _norm_text(id_obj.get("componentID") or site_data.get("componentID"))
        comp_name = _norm_text(id_obj.get("component") or id_obj.get("name"))
        if not comp_id:
            comp_id = f"name::{_norm_name(comp_name)}" if comp_name else ""
        if not comp_id:
            continue
        esd = entry.get("esd", {}).get("ESD", {})
        ecoclassid = esd.get("ecoclassid")
        if isinstance(ecoclassid, list):
            ecoclassid = ecoclassid[0] if ecoclassid else ""
        ecoclassid = _norm_ecological_site(ecoclassid).upper()
        if comp_id not in by_id:
            by_id[comp_id] = {"landscape_type": "", "landscape_class": "", "ecological_site": ""}
        if not by_id[comp_id].get("ecological_site") and ecoclassid:
            by_id[comp_id]["ecological_site"] = ecoclassid

    return by_id


def _call_analyze_soil_api(api_url, payload, timeout_seconds):
    try:
        response = requests.post(api_url, json=payload, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc

    if response.status_code >= 400:
        try:
            err_json = response.json()
            detail = err_json.get("detail") or err_json
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"API analyze-soil failed ({response.status_code}): {detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("API analyze-soil returned non-JSON response") from exc


def _build_analyze_payload(
    lon,
    lat,
    rank_inputs,
    p_slope,
    p_elev,
    bedrock_depth,
    p_aspect=None,
    p_shape_vert=None,
    p_shape_horiz=None,
    p_landscape=None,
    max_distance_m=1000,
):
    payload = {
        "lon": lon,
        "lat": lat,
        # Required by current API request schema; ignored by combined endpoint.
        "soil_list_json": {"metadata": {}, "soilList": []},
        "rank_data_csv": "compname,sandpct_intpl",
        "map_unit_component_data_csv": "mukey,cokey",
        "sim": False,
        "max_distance_m": max_distance_m,
        "soilHorizon": rank_inputs.get("soilHorizon"),
        "topDepth": rank_inputs.get("topDepth"),
        "bottomDepth": rank_inputs.get("bottomDepth"),
        "rfvDepth": rank_inputs.get("rfvDepth"),
        "claypct_est": rank_inputs.get("claypct_est"),
        "lab_Color": rank_inputs.get("lab_Color"),
        "pSlope": p_slope,
        "pElev": p_elev,
        "bedrock": bedrock_depth,
        "cracks": False,
        "pLandscapeMode": "base",
    }
    if p_aspect is not None:
        payload["pAspect"] = p_aspect
    if _norm_text(p_shape_vert):
        payload["pSlopeShapeVert"] = p_shape_vert
    if _norm_text(p_shape_horiz):
        payload["pSlopeShapeHoriz"] = p_shape_horiz
    if _norm_text(p_landscape):
        payload["pLandscape"] = p_landscape
    return payload


def _match(expected, predicted):
    exp = _norm_compare(expected)
    pred = _norm_compare(predicted)
    if not exp:
        return None
    return exp == pred


def _match_ecological_site(expected, predicted):
    exp = _norm_ecological_site(expected)
    pred = _norm_ecological_site(predicted)
    if not exp:
        return None
    return exp == pred


def _first_nonempty(row, columns):
    for col in columns:
        value = _norm_text(row.get(col))
        if value:
            return value
    return ""


def _score_summary(rows, mode_prefix, expected_col, predicted_col, source=None, passed_only=False):
    subset = [r for r in rows if _norm_text(r.get(expected_col))]
    if source:
        subset = [r for r in subset if _norm_source(r.get("source")) == source]
    if passed_only:
        subset = [r for r in subset if r.get("status") == "passed"]
    compared = len(subset)
    matched = sum(1 for r in subset if r.get(f"{mode_prefix}_{predicted_col}_match") is True)
    return {
        "compared": compared,
        "matched": matched,
        "accuracy": None if compared == 0 else round(matched / compared, 4),
    }


def _score_summary_with_match_col(rows, expected_col, match_col, passed_only=False):
    subset = [r for r in rows if _norm_text(r.get(expected_col))]
    if passed_only:
        subset = [r for r in subset if r.get("status") == "passed"]
    compared = len(subset)
    matched = sum(1 for r in subset if r.get(match_col) is True)
    return {
        "compared": compared,
        "matched": matched,
        "accuracy": None if compared == 0 else round(matched / compared, 4),
    }


def _score_match_col(rows, match_col, require_col=None, changed_only=False, passed_only=False):
    subset = rows
    if changed_only:
        subset = [r for r in subset if r.get("landscape_class_qc_changed") is True]
    if require_col:
        subset = [r for r in subset if _norm_text(r.get(require_col))]
    if passed_only:
        subset = [r for r in subset if r.get("status") == "passed"]
    compared = len(subset)
    matched = sum(1 for r in subset if r.get(match_col) is True)
    return {
        "compared": compared,
        "matched": matched,
        "accuracy": None if compared == 0 else round(matched / compared, 4),
    }


def _extract_plot_metadata(row):
    return {
        "mlrasymbol": _norm_text(row.get("mlrasymbol")),
        "confidence_index": _norm_text(row.get("confidence_index")),
        "uncertainty_class": _norm_text(
            row.get("uncertainty_class") or row.get("uncertainty_class")
        ),
        "uncertainty_reason": _norm_text(row.get("uncertainty_reason")),
        "dominant_comppct_r": _norm_text(row.get("dominant_comppct_r")),
        "second_comppct_r": _norm_text(row.get("second_comppct_r")),
        "component_gap": _norm_text(row.get("component_gap")),
        "n_ecosites_dominant": _norm_text(row.get("n_ecosites_dominant")),
        "multiplicity_score": _norm_text(row.get("multiplicity_score")),
    }


def _aim_landscape_class(value):
    mapped = aim_to_standard_class(value)
    if mapped in (None, "other"):
        return _norm_text(crosswalk_landscape_class(value, mode="base"))
    return _norm_text(mapped)


def _rfv_bucket(value):
    v = _to_float(value)
    if v is None:
        return "0-1%"
    lower = max(0, int(round(v)))
    upper = lower + 1
    return f"{lower}-{upper}%"


def _normalize_texture_value(texture_value):
    raw = _norm_text(texture_value).upper()
    if not raw:
        return "LOAM"
    return TEXTURE_ABBREV_MAP.get(raw, raw)


def _to_optional_float(value):
    txt = _norm_text(value)
    if not txt:
        return None
    try:
        return float(txt)
    except (TypeError, ValueError):
        return None


def _build_rank_inputs_all_horizons(plot_key, horizons_df):
    hz = horizons_df[horizons_df["PrimaryKey"] == plot_key].copy()
    hz = hz.sort_values("HorizonDepthUpper", kind="stable")

    if hz.empty:
        raise RuntimeError(f"No horizons found for plot {plot_key}")

    hz["Texture"] = hz["Texture"].apply(_normalize_texture_value)
    hz["HorizonDepthUpper"] = pd.to_numeric(hz["HorizonDepthUpper"], errors="coerce").fillna(0)
    hz["HorizonDepthLower"] = pd.to_numeric(hz["HorizonDepthLower"], errors="coerce").fillna(0)

    # Keep only horizons with a valid interval after coercion.
    hz = hz[hz["HorizonDepthLower"] >= hz["HorizonDepthUpper"]].copy()
    if hz.empty:
        raise RuntimeError(f"No valid horizon intervals for plot {plot_key}")

    return {
        "soilHorizon": hz["Texture"].tolist(),
        "topDepth": hz["HorizonDepthUpper"].astype(int).tolist(),
        "bottomDepth": hz["HorizonDepthLower"].astype(int).tolist(),
        "rfvDepth": hz["RockFragments"].apply(_rfv_bucket).tolist(),
        "claypct_est": hz["ClayPct"].apply(_to_optional_float).tolist(),
        "lab_Color": [],
    }


def _get_color_refs():
    global _COLOR_REF, _MUNSELL_REF
    if _COLOR_REF is None or _MUNSELL_REF is None:
        _COLOR_REF = pd.read_csv(soil_id.config.MUNSELL_RGB_LAB_PATH)
        _MUNSELL_REF = _COLOR_REF[["hue", "value", "chroma"]]
    return _COLOR_REF, _MUNSELL_REF


def _munsell_to_lab(hue, value, chroma):
    hue_txt = _norm_text(hue)
    value_txt = _norm_text(value)
    chroma_txt = _norm_text(chroma)
    if not hue_txt or not value_txt or not chroma_txt:
        return None

    try:
        value_num = int(float(value_txt))
        chroma_num = int(float(chroma_txt))
    except (TypeError, ValueError):
        return None

    color_ref, munsell_ref = _get_color_refs()
    try:
        rgb = munsell2rgb(color_ref, munsell_ref, [hue_txt, value_num, chroma_num])
        l_val, a_val, b_val = find_closest_rgb_in_reference(rgb[0], rgb[1], rgb[2], color_ref)
        return [round(float(l_val), 2), round(float(a_val), 2), round(float(b_val), 2)]
    except Exception:
        return None


def _build_lab_color_from_horizons(plot_key, horizons_df, n_layers):
    hz = horizons_df[horizons_df["PrimaryKey"] == plot_key].copy()
    hz = hz.sort_values("HorizonDepthUpper", kind="stable")
    if n_layers is not None and n_layers > 0:
        hz = hz.head(n_layers)

    if hz.empty:
        return [None for _ in range(max(1, n_layers or 1))]

    colors = []
    for _, hz_row in hz.iterrows():
        colors.append(_munsell_to_lab(hz_row.get("Hue"), hz_row.get("Value"), hz_row.get("Chroma")))

    if not colors:
        return [None for _ in range(max(1, n_layers or 1))]

    if n_layers and len(colors) < n_layers:
        colors.extend([None] * (n_layers - len(colors)))

    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-csv",
        default="study_plot_characteristics_enriched.csv",
        help="Plot characteristics CSV path (absolute, workspace-relative, or under Data/aim_data)",
    )
    parser.add_argument(
        "--horizons-csv",
        default=str(HORIZONS_CSV),
        help="Soil horizons CSV path containing Texture/RockFragments/ClayPct",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TERRAIN_DATA_DIR),
        help="Directory to write result artifacts",
    )
    parser.add_argument(
        "--list-source",
        choices=["live", "synthetic"],
        default="live",
        help="Use live list_soils per row or synthetic candidate data",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit processing to the first N rows (useful for quick live-mode tests)",
    )
    parser.add_argument(
        "--rank-method",
        choices=["rank_data_loc", "rank_data", "rank_loc"],
        default="rank_data_loc",
        help="Rank method used for top-result selection and expected-rank lookup (default: rank_data_loc)",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["local", "api"],
        default="local",
        help="Execution backend: local Python functions or API endpoint (default: local)",
    )
    parser.add_argument(
        "--soilid-api-url",
        default=os.getenv("SOILID_API_URL", "https://soil-id-algorithm-api.vercel.app/api/analyze-soil"),
        help="Full analyze-soil API URL used when --execution-mode=api",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for API mode (default: 120)",
    )
    parser.add_argument(
        "--buffer-meters",
        type=int,
        default=1000,
        help="Search radius in metres used for list_soils in live/local mode (default: 1000)",
    )
    parser.add_argument(
        "--retry-csv",
        default=None,
        help="Path to a previous run-results CSV. Only rows with status != 'passed' will be re-run; "
             "results are merged back and a new combined CSV is written.",
    )
    parser.add_argument(
        "--legacy-gower",
        action="store_true",
        default=False,
        help="Use pre-44cad9e Gower normalization: separate sand/clay/l/a/b features with equal "
             "weights instead of texture_dist + color_delta_e with HORIZON_FEATURE_WEIGHTS. "
             "Only applies to --execution-mode local.",
    )
    args = parser.parse_args()

    plot_csv_arg = Path(args.plot_csv)
    if plot_csv_arg.is_absolute():
        plot_csv = plot_csv_arg
    elif plot_csv_arg.exists():
        plot_csv = plot_csv_arg
    elif plot_csv_arg.parent != Path("."):
        plot_csv = plot_csv_arg
    else:
        plot_csv = TERRAIN_DATA_DIR / args.plot_csv
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = pd.read_csv(plot_csv)
    horizons_df = pd.read_csv(Path(args.horizons_csv))

    matched = plot_df.merge(
        horizons_df[["PrimaryKey"]].drop_duplicates(), on="PrimaryKey", how="inner"
    )
    required_cols = [
        "Slope",
        "Elevation",
        "Aspect",
        "SlopeShapeVertical",
        "SlopeShapeHorizontal",
        "LandscapeType",
    ]
    valid = matched.dropna(subset=required_cols).copy()
    if args.max_rows is not None:
        valid = valid.iloc[: args.max_rows]

    # Retry mode: load previous results and restrict processing to non-passed rows.
    prior_rows_df = None
    if args.retry_csv:
        prior_rows_df = pd.read_csv(args.retry_csv, dtype=str)
        retry_keys = set(
            prior_rows_df.loc[prior_rows_df["status"] != "passed", "PrimaryKey"]
        )
        n_prior_passed = int((prior_rows_df["status"] == "passed").sum())
        valid = valid[valid["PrimaryKey"].isin(retry_keys)].copy()
        print(f"retry_mode=True  prior_passed={n_prior_passed}  retrying={len(valid)}", flush=True)

    synthetic_list_output_data = None

    total = len(valid)
    passed = 0
    failed = 0
    skipped = 0
    failures = []
    row_results = []

    for plot_idx, (_, row) in enumerate(valid.iterrows(), start=1):
        key = row["PrimaryKey"]
        try:
            rank_inputs = _build_rank_inputs_all_horizons(key, horizons_df)
            rank_inputs["lab_Color"] = _build_lab_color_from_horizons(
                row["PrimaryKey"],
                horizons_df,
                len(rank_inputs.get("soilHorizon") or []),
            )
            p_slope = _to_float(row.get("Slope"))
            p_elev = _to_float(row.get("Elevation"))
            p_aspect = _to_float(row.get("Aspect"))
            bedrock_depth = _to_float(row.get("bedrock"))
            if bedrock_depth is not None and bedrock_depth <= 0:
                bedrock_depth = None
            if bedrock_depth is not None:
                bedrock_depth = int(round(bedrock_depth))
            p_shape_vert = row.get("SlopeShapeVertical")
            p_shape_horiz = row.get("SlopeShapeHorizontal")
            p_landscape = row.get("LandscapeType")

            row_source = _norm_source(row.get("source") or row.get("Source"))
            expected_soil_series = _pick_expected_soil_series(row)
            expected_ecological_site = _norm_text(row.get("EcolSite"))
            expected_landscape_raw = _norm_text(row.get("LandscapeType"))
            expected_landscape_class = _aim_landscape_class(expected_landscape_raw)
            plot_metadata = _extract_plot_metadata(row)

            # AIM reference values — prefer SDA-matched component name over raw recorded name
            aim_expected_soil_series = _norm_text(row.get("aim_series_component_name")) or expected_soil_series
            aim_expected_eco_site = _norm_text(row.get("EcolSite_AIM")) or expected_ecological_site
            _aim_lc_raw = _norm_text(row.get("LandscapeType_AIM")) or expected_landscape_raw
            aim_expected_landscape_class = _aim_landscape_class(_aim_lc_raw) or expected_landscape_class

            # QC reference values — prefer SDA-matched component name over raw recorded name
            qc_expected_soil_series = _norm_text(row.get("qc_series_component_name"))
            qc_expected_eco_site = _norm_text(row.get("EcolSite_QC")) or expected_ecological_site
            _qc_lc_raw = _norm_text(row.get("LandscapeType_QC")) or expected_landscape_raw
            qc_expected_landscape_class = _aim_landscape_class(_qc_lc_raw)

            # Change flags from QC review
            landscape_class_qc_changed = (
                str(row.get("LandscapeType_qc_changed", "")).strip().upper() == "TRUE"
            )
            any_qc_changed = str(row.get("Any_qc_changed", "")).strip().upper() == "TRUE"

            lon = _to_float(row.get("Longitude_NAD83"))
            lat = _to_float(row.get("Latitude_NAD83"))

            if args.execution_mode == "api":
                if lon is None or lat is None:
                    raise RuntimeError("Missing Longitude_NAD83/Latitude_NAD83 for API mode")

                baseline_payload = _build_analyze_payload(
                    lon=lon,
                    lat=lat,
                    rank_inputs=rank_inputs,
                    p_slope=p_slope,
                    p_elev=p_elev,
                    bedrock_depth=bedrock_depth,
                    max_distance_m=args.buffer_meters,
                )
                baseline_api_response = _call_analyze_soil_api(
                    args.soilid_api_url,
                    baseline_payload,
                    args.request_timeout,
                )

                terrain_payload = _build_analyze_payload(
                    lon=lon,
                    lat=lat,
                    rank_inputs=rank_inputs,
                    p_slope=p_slope,
                    p_elev=p_elev,
                    bedrock_depth=bedrock_depth,
                    p_aspect=p_aspect,
                    p_shape_vert=p_shape_vert,
                    p_shape_horiz=p_shape_horiz,
                    p_landscape=p_landscape,
                    max_distance_m=args.buffer_meters,
                )
                terrain_api_response = _call_analyze_soil_api(
                    args.soilid_api_url,
                    terrain_payload,
                    args.request_timeout,
                )

                list_output_data = {
                    "soil_list_json": baseline_api_response.get("soil_list_json") or {},
                    "map_unit_component_data_csv": baseline_api_response.get("map_unit_component_data_csv") or "",
                }
                baseline = baseline_api_response.get("ranking_result") or {}
                with_terrain = terrain_api_response.get("ranking_result") or {}
            elif args.list_source == "live":
                if lon is None or lat is None:
                    raise RuntimeError("Missing Longitude_NAD83/Latitude_NAD83 for live list_soils")
                list_output_data = list_soils(lon=lon, lat=lat, sim=False, max_distance_m=args.buffer_meters)
                if not hasattr(list_output_data, "map_unit_component_data_csv"):
                    skipped += 1
                    row_results.append(
                        {
                            "plot_id": plot_idx,
                            "PrimaryKey": key,
                            "source": row_source,
                            "status": "skipped",
                            "error": _norm_text(list_output_data) or "No SoilListOutputData returned",
                            **plot_metadata,
                            "aim_expected_rank_baseline": "",
                            "aim_expected_rank_terrain": "",
                            "qc_expected_rank_baseline": "",
                            "qc_expected_rank_terrain": "",
                            "expected_component_id_baseline": "",
                            "expected_component_id_terrain": "",
                            "expected_sda_ecological_site": "",
                            "expected_sda_landscape_type": "",
                            "expected_sda_landscape_class": "",
                            "baseline_soil_series": "",
                            "baseline_ecological_site": "",
                            "baseline_landscape_class": "",
                            "terrain_aim_soil_series": "",
                            "terrain_aim_ecological_site": "",
                            "terrain_aim_landscape_class": "",
                            "baseline_component_id": "",
                            "terrain_aim_component_id": "",
                            "baseline_aim_soil_series_match": None,
                            "baseline_aim_ecological_site_match": None,
                            "baseline_aim_landscape_class_match": None,
                            "terrain_aim_soil_series_match": None,
                            "terrain_aim_ecological_site_match": None,
                            "terrain_aim_landscape_class_match": None,
                            "top_changed": False,
                            "aim_expected_soil_series": aim_expected_soil_series,
                            "aim_expected_ecological_site": aim_expected_eco_site,
                            "aim_expected_landscape_class": aim_expected_landscape_class,
                            "qc_expected_soil_series": qc_expected_soil_series,
                            "qc_expected_ecological_site": qc_expected_eco_site,
                            "qc_expected_landscape_class": qc_expected_landscape_class,
                            "landscape_class_qc_changed": landscape_class_qc_changed,
                            "any_qc_changed": any_qc_changed,
                            "baseline_qc_soil_series_match": None,
                            "baseline_qc_ecological_site_match": None,
                            "baseline_qc_landscape_class_match": None,
                            "terrain_qc_soil_series": "",
                            "terrain_qc_ecological_site": "",
                            "terrain_qc_landscape_class": "",
                            "terrain_qc_component_id": "",
                            "terrain_qc_soil_series_match": None,
                            "terrain_qc_ecological_site_match": None,
                            "terrain_qc_landscape_class_match": None,
                            "aim_qc_soil_series_match": None,
                            "aim_qc_ecological_site_match": None,
                            "aim_qc_landscape_class_match": None,
                        }
                    )
                    continue

                baseline = rank_soils(
                    lon=0.0,
                    lat=0.0,
                    list_output_data=list_output_data,
                    soilHorizon=rank_inputs["soilHorizon"],
                    topDepth=rank_inputs["topDepth"],
                    bottomDepth=rank_inputs["bottomDepth"],
                    rfvDepth=rank_inputs["rfvDepth"],
                    claypct_est=rank_inputs["claypct_est"],
                    lab_Color=rank_inputs["lab_Color"],
                    pSlope=p_slope,
                    pElev=p_elev,
                    bedrock=bedrock_depth,
                    cracks=False,
                    legacy_gower=args.legacy_gower,
                )

                with_terrain = rank_soils(
                    lon=0.0,
                    lat=0.0,
                    list_output_data=list_output_data,
                    soilHorizon=rank_inputs["soilHorizon"],
                    topDepth=rank_inputs["topDepth"],
                    bottomDepth=rank_inputs["bottomDepth"],
                    rfvDepth=rank_inputs["rfvDepth"],
                    claypct_est=rank_inputs["claypct_est"],
                    lab_Color=rank_inputs["lab_Color"],
                    pSlope=p_slope,
                    pElev=p_elev,
                    bedrock=bedrock_depth,
                    cracks=False,
                    pAspect=p_aspect,
                    pSlopeShapeVert=p_shape_vert,
                    pSlopeShapeHoriz=p_shape_horiz,
                    pLandscape=p_landscape,
                    legacy_gower=args.legacy_gower,
                )
            else:
                # Synthetic candidates are only allowed when explicitly requested.
                if synthetic_list_output_data is None:
                    from soil_id.tests.us.test_study_dataset_inputs import _build_list_output_data

                    synthetic_list_output_data = _build_list_output_data()
                list_output_data = synthetic_list_output_data

                baseline = rank_soils(
                    lon=0.0,
                    lat=0.0,
                    list_output_data=list_output_data,
                    soilHorizon=rank_inputs["soilHorizon"],
                    topDepth=rank_inputs["topDepth"],
                    bottomDepth=rank_inputs["bottomDepth"],
                    rfvDepth=rank_inputs["rfvDepth"],
                    claypct_est=rank_inputs["claypct_est"],
                    lab_Color=rank_inputs["lab_Color"],
                    pSlope=p_slope,
                    pElev=p_elev,
                    bedrock=bedrock_depth,
                    cracks=False,
                    legacy_gower=args.legacy_gower,
                )

                with_terrain = rank_soils(
                    lon=0.0,
                    lat=0.0,
                    list_output_data=list_output_data,
                    soilHorizon=rank_inputs["soilHorizon"],
                    topDepth=rank_inputs["topDepth"],
                    bottomDepth=rank_inputs["bottomDepth"],
                    rfvDepth=rank_inputs["rfvDepth"],
                    claypct_est=rank_inputs["claypct_est"],
                    lab_Color=rank_inputs["lab_Color"],
                    pSlope=p_slope,
                    pElev=p_elev,
                    bedrock=bedrock_depth,
                    cracks=False,
                    pAspect=p_aspect,
                    pSlopeShapeVert=p_shape_vert,
                    pSlopeShapeHoriz=p_shape_horiz,
                    pLandscape=p_landscape,
                    legacy_gower=args.legacy_gower,
                )

            comp_meta_by_id = _build_component_metadata(list_output_data)

            if not baseline.get("soilRank") or not with_terrain.get("soilRank"):
                raise RuntimeError("Empty soilRank in output")

            baseline_expected = _extract_expected_rank(baseline, expected_soil_series, args.rank_method)
            terrain_expected = _extract_expected_rank(with_terrain, expected_soil_series, args.rank_method)
            expected_meta = _metadata_for_component(
                comp_meta_by_id,
                component_id=baseline_expected["component_id"] or terrain_expected["component_id"],
                component_name=expected_soil_series,
            )

            # AIM/QC specific baseline and AIM-terrain rank lookups
            aim_baseline_rank = _extract_expected_rank(baseline, aim_expected_soil_series, args.rank_method)
            aim_terrain_rank = _extract_expected_rank(with_terrain, aim_expected_soil_series, args.rank_method)
            qc_baseline_rank = _extract_expected_rank(baseline, qc_expected_soil_series, args.rank_method)

            baseline_top = _extract_top_metadata(baseline, comp_meta_by_id, args.rank_method)
            terrain_top = _extract_top_metadata(with_terrain, comp_meta_by_id, args.rank_method)

            # QC terrain run: re-rank with QC landscape type (only when it changed from AIM)
            p_landscape_qc = _norm_text(row.get("LandscapeType_QC")) or p_landscape
            if landscape_class_qc_changed:
                if args.execution_mode == "api":
                    if lon is None or lat is None:
                        raise RuntimeError("Missing Longitude_NAD83/Latitude_NAD83 for API mode")
                    qc_payload = _build_analyze_payload(
                        lon=lon,
                        lat=lat,
                        rank_inputs=rank_inputs,
                        p_slope=p_slope,
                        p_elev=p_elev,
                        bedrock_depth=bedrock_depth,
                        p_aspect=p_aspect,
                        p_shape_vert=p_shape_vert,
                        p_shape_horiz=p_shape_horiz,
                        p_landscape=p_landscape_qc,
                        max_distance_m=args.buffer_meters,
                    )
                    with_terrain_qc = (
                        _call_analyze_soil_api(
                            args.soilid_api_url,
                            qc_payload,
                            args.request_timeout,
                        ).get("ranking_result")
                        or {}
                    )
                else:
                    with_terrain_qc = rank_soils(
                        lon=0.0,
                        lat=0.0,
                        list_output_data=list_output_data,
                        soilHorizon=rank_inputs["soilHorizon"],
                        topDepth=rank_inputs["topDepth"],
                        bottomDepth=rank_inputs["bottomDepth"],
                        rfvDepth=rank_inputs["rfvDepth"],
                        claypct_est=rank_inputs["claypct_est"],
                        lab_Color=rank_inputs["lab_Color"],
                        pSlope=p_slope,
                        pElev=p_elev,
                        bedrock=bedrock_depth,
                        cracks=False,
                        pAspect=p_aspect,
                        pSlopeShapeVert=p_shape_vert,
                        pSlopeShapeHoriz=p_shape_horiz,
                        pLandscape=p_landscape_qc,
                        legacy_gower=args.legacy_gower,
                    )
                if not with_terrain_qc.get("soilRank"):
                    raise RuntimeError("Empty soilRank in QC terrain output")
                terrain_qc_top = _extract_top_metadata(with_terrain_qc, comp_meta_by_id, args.rank_method)
            else:
                # landscape class unchanged — QC terrain result equals AIM terrain result
                terrain_qc_top = terrain_top.copy()

            # Rank lookups against the terrain_qc result for expected series/ecosites
            qc_rank_result = with_terrain_qc if landscape_class_qc_changed else with_terrain
            aim_expected_rank_qc = _extract_expected_rank(
                qc_rank_result, aim_expected_soil_series, args.rank_method
            )
            qc_expected_rank_qc = _extract_expected_rank(
                qc_rank_result, qc_expected_soil_series, args.rank_method
            )
            aim_expected_ecosite_rank_qc = _extract_expected_ecosite_rank(
                qc_rank_result, aim_expected_eco_site, args.rank_method
            )
            qc_expected_ecosite_rank_qc = _extract_expected_ecosite_rank(
                qc_rank_result, qc_expected_eco_site, args.rank_method
            )

            passed += 1
            row_results.append(
                {
                    "plot_id": plot_idx,
                    "PrimaryKey": key,
                    "source": row_source,
                    "status": "passed",
                    "error": "",
                    **plot_metadata,
                    "aim_expected_rank_baseline": aim_baseline_rank["rank"],
                    "aim_expected_rank_terrain": aim_terrain_rank["rank"],
                    "qc_expected_rank_baseline": qc_baseline_rank["rank"],
                    "expected_component_id_baseline": baseline_expected["component_id"],
                    "expected_component_id_terrain": terrain_expected["component_id"],
                    "expected_sda_ecological_site": _norm_text(expected_meta.get("ecological_site")),
                    "expected_sda_landscape_type": _norm_text(expected_meta.get("landscape_type")),
                    "expected_sda_landscape_class": _norm_text(expected_meta.get("landscape_class")),
                    "baseline_soil_series": baseline_top["soil_series"],
                    "baseline_ecological_site": baseline_top["ecological_site"],
                    "baseline_landscape_class": baseline_top["landscape_class"],
                    "terrain_aim_soil_series": terrain_top["soil_series"],
                    "terrain_aim_ecological_site": terrain_top["ecological_site"],
                    "terrain_aim_landscape_class": terrain_top["landscape_class"],
                    "baseline_component_id": baseline_top["component_id"],
                    "terrain_aim_component_id": terrain_top["component_id"],
                    "baseline_aim_soil_series_match": _match(
                        expected_soil_series, baseline_top["soil_series"]
                    ),
                    "baseline_aim_ecological_site_match": _match_ecological_site(
                        expected_ecological_site, baseline_top["ecological_site"]
                    ),
                    "baseline_aim_landscape_class_match": _match(
                        expected_landscape_class, baseline_top["landscape_class"]
                    ),
                    "terrain_aim_soil_series_match": _match(
                        expected_soil_series, terrain_top["soil_series"]
                    ),
                    "terrain_aim_ecological_site_match": _match_ecological_site(
                        expected_ecological_site, terrain_top["ecological_site"]
                    ),
                    "terrain_aim_landscape_class_match": _match(
                        expected_landscape_class, terrain_top["landscape_class"]
                    ),
                    "top_changed": baseline_top["soil_series"] != terrain_top["soil_series"],
                    # AIM vs QC reference fields
                    "aim_expected_soil_series": aim_expected_soil_series,
                    "aim_expected_ecological_site": aim_expected_eco_site,
                    "aim_expected_landscape_class": aim_expected_landscape_class,
                    "qc_expected_soil_series": qc_expected_soil_series,
                    "qc_expected_ecological_site": qc_expected_eco_site,
                    "qc_expected_landscape_class": qc_expected_landscape_class,
                    "landscape_class_qc_changed": landscape_class_qc_changed,
                    "any_qc_changed": any_qc_changed,
                    # rank_soils (baseline) vs QC reference matches
                    "baseline_qc_soil_series_match": _match(
                        qc_expected_soil_series, baseline_top["soil_series"]
                    ),
                    "baseline_qc_ecological_site_match": _match_ecological_site(
                        qc_expected_eco_site, baseline_top["ecological_site"]
                    ),
                    "baseline_qc_landscape_class_match": _match(
                        qc_expected_landscape_class, baseline_top["landscape_class"]
                    ),
                    # rank_soils (QC landscape inputs) top result
                    "terrain_qc_soil_series": terrain_qc_top["soil_series"],
                    "terrain_qc_ecological_site": terrain_qc_top["ecological_site"],
                    "terrain_qc_landscape_class": terrain_qc_top["landscape_class"],
                    "terrain_qc_component_id": terrain_qc_top["component_id"],
                    # rank_soils (QC landscape) vs QC reference matches
                    "terrain_qc_soil_series_match": _match(
                        qc_expected_soil_series, terrain_qc_top["soil_series"]
                    ),
                    "terrain_qc_ecological_site_match": _match_ecological_site(
                        qc_expected_eco_site, terrain_qc_top["ecological_site"]
                    ),
                    "terrain_qc_landscape_class_match": _match(
                        qc_expected_landscape_class, terrain_qc_top["landscape_class"]
                    ),
                    # AIM vs QC reference agreement
                    "aim_qc_soil_series_match": _match(qc_expected_soil_series, aim_expected_soil_series),
                    "aim_qc_ecological_site_match": _match_ecological_site(
                        qc_expected_eco_site, aim_expected_eco_site
                    ),
                    "aim_qc_landscape_class_match": _match(
                        qc_expected_landscape_class, aim_expected_landscape_class
                    ),
                    # Expected series/ecosite ranks in the terrain_qc result
                    "qc_expected_rank_terrain": qc_expected_rank_qc["rank"],
                    "aim_expected_ecosite_rank": aim_expected_ecosite_rank_qc["rank"],
                    "qc_expected_ecosite_rank": qc_expected_ecosite_rank_qc["rank"],
                    # QC expected series score detail from terrain_qc result
                    "qc_expected_score_data_horz": qc_expected_rank_qc["score_data_horz"],
                    "qc_expected_score_data_site": qc_expected_rank_qc["score_data_site"],
                    # Terrain (QC) rank-1 component match detail scores
                    "terrain_qc_score_data_horz": terrain_qc_top.get("score_data_horz"),
                    "terrain_qc_score_data_site": terrain_qc_top.get("score_data_site"),
                    "terrain_qc_horz_texture": terrain_qc_top.get("horz_texture"),
                    "terrain_qc_horz_rock_fragments": terrain_qc_top.get("horz_rock_fragments"),
                    "terrain_qc_horz_color": terrain_qc_top.get("horz_color"),
                    "terrain_qc_horz_observed_depth": terrain_qc_top.get("horz_observed_depth"),
                    "terrain_qc_horz_mapped_depth": terrain_qc_top.get("horz_mapped_depth"),
                    "terrain_qc_horz_depth_coverage": terrain_qc_top.get("horz_depth_coverage"),
                    "terrain_qc_site_slope_observed": terrain_qc_top.get("site_slope_observed"),
                    "terrain_qc_site_slope_mapped": terrain_qc_top.get("site_slope_mapped"),
                    "terrain_qc_site_elev_observed": terrain_qc_top.get("site_elev_observed"),
                    "terrain_qc_site_elev_mapped": terrain_qc_top.get("site_elev_mapped"),
                    "terrain_qc_site_aspect_north_observed": terrain_qc_top.get("site_aspect_north_observed"),
                    "terrain_qc_site_aspect_north_mapped": terrain_qc_top.get("site_aspect_north_mapped"),
                    "terrain_qc_site_aspect_east_observed": terrain_qc_top.get("site_aspect_east_observed"),
                    "terrain_qc_site_aspect_east_mapped": terrain_qc_top.get("site_aspect_east_mapped"),
                    "terrain_qc_site_shape_vert_observed": terrain_qc_top.get("site_shape_vert_observed"),
                    "terrain_qc_site_shape_vert_mapped": terrain_qc_top.get("site_shape_vert_mapped"),
                    "terrain_qc_site_shape_horiz_observed": terrain_qc_top.get("site_shape_horiz_observed"),
                    "terrain_qc_site_shape_horiz_mapped": terrain_qc_top.get("site_shape_horiz_mapped"),
                    "terrain_qc_site_landscape_observed": terrain_qc_top.get("site_landscape_observed"),
                    "terrain_qc_site_landscape_mapped": terrain_qc_top.get("site_landscape_mapped"),
                }
            )
        except Exception as exc:
            failed += 1
            failures.append((key, str(exc)))
            row_results.append(
                {
                    "plot_id": plot_idx,
                    "PrimaryKey": key,
                    "source": _norm_source(row.get("source") or row.get("Source")),
                    "status": "failed",
                    "error": str(exc),
                    **_extract_plot_metadata(row),
                    "aim_expected_rank_baseline": "",
                    "aim_expected_rank_terrain": "",
                    "qc_expected_rank_baseline": "",
                    "qc_expected_rank_terrain": "",
                    "expected_component_id_baseline": "",
                    "expected_component_id_terrain": "",
                    "expected_sda_ecological_site": "",
                    "expected_sda_landscape_type": "",
                    "expected_sda_landscape_class": "",
                    "baseline_soil_series": "",
                    "baseline_ecological_site": "",
                    "baseline_landscape_class": "",
                    "terrain_aim_soil_series": "",
                    "terrain_aim_ecological_site": "",
                    "terrain_aim_landscape_class": "",
                    "baseline_component_id": "",
                    "terrain_aim_component_id": "",
                    "baseline_aim_soil_series_match": None,
                    "baseline_aim_ecological_site_match": None,
                    "baseline_aim_landscape_class_match": None,
                    "terrain_aim_soil_series_match": None,
                    "terrain_aim_ecological_site_match": None,
                    "terrain_aim_landscape_class_match": None,
                    "top_changed": False,
                    "aim_expected_soil_series": _norm_text(row.get("aim_series_component_name")) or _pick_expected_soil_series(row),
                    "aim_expected_ecological_site": _norm_text(row.get("EcolSite_AIM")) or _norm_text(row.get("EcolSite")),
                    "aim_expected_landscape_class": _aim_landscape_class(
                        _norm_text(row.get("LandscapeType_AIM")) or _norm_text(row.get("LandscapeType"))
                    ),
                    "qc_expected_soil_series": _norm_text(row.get("qc_series_component_name")),
                    "qc_expected_ecological_site": _norm_text(row.get("EcolSite_QC")) or _norm_text(row.get("EcolSite")),
                    "qc_expected_landscape_class": _aim_landscape_class(
                        _norm_text(row.get("LandscapeType_QC")) or _norm_text(row.get("LandscapeType"))
                    ),
                    "landscape_class_qc_changed": str(row.get("LandscapeType_qc_changed", "")).strip().upper() == "TRUE",
                    "any_qc_changed": str(row.get("Any_qc_changed", "")).strip().upper() == "TRUE",
                    "baseline_qc_soil_series_match": None,
                    "baseline_qc_ecological_site_match": None,
                    "baseline_qc_landscape_class_match": None,
                    "terrain_qc_soil_series": "",
                    "terrain_qc_ecological_site": "",
                    "terrain_qc_landscape_class": "",
                    "terrain_qc_component_id": "",
                    "terrain_qc_soil_series_match": None,
                    "terrain_qc_ecological_site_match": None,
                    "terrain_qc_landscape_class_match": None,
                    "aim_qc_soil_series_match": None,
                    "aim_qc_ecological_site_match": None,
                    "aim_qc_landscape_class_match": None,
                    "aim_expected_rank": "",
                    "qc_expected_rank": "",
                    "aim_expected_ecosite_rank": "",
                    "qc_expected_ecosite_rank": "",
                    "qc_expected_score_data_horz": None,
                    "qc_expected_score_data_site": None,
                    "terrain_qc_score_data_horz": None,
                    "terrain_qc_score_data_site": None,
                    "terrain_qc_horz_texture": None,
                    "terrain_qc_horz_rock_fragments": None,
                    "terrain_qc_horz_color": None,
                    "terrain_qc_horz_observed_depth": None,
                    "terrain_qc_horz_mapped_depth": None,
                    "terrain_qc_horz_depth_coverage": None,
                    "terrain_qc_site_slope_observed": None,
                    "terrain_qc_site_slope_mapped": None,
                    "terrain_qc_site_elev_observed": None,
                    "terrain_qc_site_elev_mapped": None,
                    "terrain_qc_site_aspect_north_observed": None,
                    "terrain_qc_site_aspect_north_mapped": None,
                    "terrain_qc_site_aspect_east_observed": None,
                    "terrain_qc_site_aspect_east_mapped": None,
                    "terrain_qc_site_shape_vert_observed": None,
                    "terrain_qc_site_shape_vert_mapped": None,
                    "terrain_qc_site_shape_horiz_observed": None,
                    "terrain_qc_site_shape_horiz_mapped": None,
                    "terrain_qc_site_landscape_observed": None,
                    "terrain_qc_site_landscape_mapped": None,
                }
            )

    stem = Path(args.plot_csv).stem
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    rows_path = output_dir / f"{stem}_run_results_{timestamp}.csv"
    summary_json_path = output_dir / f"{stem}_run_summary_{timestamp}.json"
    summary_txt_path = output_dir / f"{stem}_run_summary_{timestamp}.txt"

    retry_df = pd.DataFrame(row_results)

    # In retry mode: merge fresh results over the prior passed rows.
    if prior_rows_df is not None and not retry_df.empty:
        passed_df = prior_rows_df[prior_rows_df["status"] == "passed"].copy()
        # Align dtypes so concat doesn't break numeric columns
        combined_df = pd.concat([passed_df, retry_df], ignore_index=True)
        # Restore original plot_id order
        combined_df["plot_id"] = pd.to_numeric(combined_df["plot_id"], errors="coerce")
        combined_df = combined_df.sort_values("plot_id").reset_index(drop=True)
        # Recount totals from merged results
        total = len(combined_df)
        passed = int((combined_df["status"] == "passed").sum())
        failed = int((combined_df["status"] == "failed").sum())
        skipped = int((combined_df["status"] == "skipped").sum())
    else:
        combined_df = retry_df

    combined_df.to_csv(rows_path, index=False)

    # Use combined (merged) rows for all summary scoring
    score_rows = combined_df.to_dict("records")
    # Fix boolean columns that may have been stringified by CSV round-trip
    for _r in score_rows:
        for _col in list(_r.keys()):
            if str(_r[_col]).lower() == "true":
                _r[_col] = True
            elif str(_r[_col]).lower() == "false":
                _r[_col] = False

    n_landscape_changed = sum(
        1 for _r in score_rows if _r.get("landscape_class_qc_changed") is True
    )

    baseline_scores = {
        "soil_series": _score_summary(score_rows, "baseline_aim", "aim_expected_soil_series", "soil_series"),
        "ecological_site": _score_summary(
            score_rows, "baseline_aim", "aim_expected_ecological_site", "ecological_site"
        ),
        "landscape_class": _score_summary(
            score_rows, "baseline_aim", "aim_expected_landscape_class", "landscape_class"
        ),
    }
    terrain_scores = {
        "soil_series": _score_summary(score_rows, "terrain_aim", "aim_expected_soil_series", "soil_series"),
        "ecological_site": _score_summary(
            score_rows, "terrain_aim", "aim_expected_ecological_site", "ecological_site"
        ),
        "landscape_class": _score_summary(
            score_rows, "terrain_aim", "aim_expected_landscape_class", "landscape_class"
        ),
    }
    # Passed-only variants (exclude failed rows from denominator)
    baseline_scores_passed = {
        "soil_series": _score_summary(score_rows, "baseline_aim", "aim_expected_soil_series", "soil_series", passed_only=True),
        "ecological_site": _score_summary(
            score_rows, "baseline_aim", "aim_expected_ecological_site", "ecological_site", passed_only=True
        ),
        "landscape_class": _score_summary(
            score_rows, "baseline_aim", "aim_expected_landscape_class", "landscape_class", passed_only=True
        ),
    }
    terrain_scores_passed = {
        "soil_series": _score_summary(score_rows, "terrain_aim", "aim_expected_soil_series", "soil_series", passed_only=True),
        "ecological_site": _score_summary(
            score_rows, "terrain_aim", "aim_expected_ecological_site", "ecological_site", passed_only=True
        ),
        "landscape_class": _score_summary(
            score_rows, "terrain_aim", "aim_expected_landscape_class", "landscape_class", passed_only=True
        ),
    }

    baseline_scores_by_source = {
        "AIM": {
            "soil_series": _score_summary(
                score_rows, "baseline_aim", "aim_expected_soil_series", "soil_series", source="AIM"
            ),
            "ecological_site": _score_summary(
                score_rows,
                "baseline_aim",
                "aim_expected_ecological_site",
                "ecological_site",
                source="AIM",
            ),
            "landscape_class": _score_summary(
                score_rows,
                "baseline_aim",
                "aim_expected_landscape_class",
                "landscape_class",
                source="AIM",
            ),
        },
        "QC": {
            "soil_series": _score_summary_with_match_col(
                score_rows,
                "qc_expected_soil_series",
                "baseline_qc_soil_series_match",
            ),
            "ecological_site": _score_summary_with_match_col(
                score_rows,
                "qc_expected_ecological_site",
                "baseline_qc_ecological_site_match",
            ),
            "landscape_class": _score_summary_with_match_col(
                score_rows,
                "qc_expected_landscape_class",
                "baseline_qc_landscape_class_match",
            ),
        },
    }
    terrain_scores_by_source = {
        "AIM": {
            "soil_series": _score_summary(
                score_rows, "terrain_aim", "aim_expected_soil_series", "soil_series", source="AIM"
            ),
            "ecological_site": _score_summary(
                score_rows,
                "terrain_aim",
                "aim_expected_ecological_site",
                "ecological_site",
                source="AIM",
            ),
            "landscape_class": _score_summary(
                score_rows,
                "terrain_aim",
                "aim_expected_landscape_class",
                "landscape_class",
                source="AIM",
            ),
        },
        "QC": {
            "soil_series": _score_summary_with_match_col(
                score_rows,
                "qc_expected_soil_series",
                "terrain_qc_soil_series_match",
            ),
            "ecological_site": _score_summary_with_match_col(
                score_rows,
                "qc_expected_ecological_site",
                "terrain_qc_ecological_site_match",
            ),
            "landscape_class": _score_summary_with_match_col(
                score_rows,
                "qc_expected_landscape_class",
                "terrain_qc_landscape_class_match",
            ),
        },
    }

    # rank_soils (QC landscape inputs) vs QC reference — evaluate across all comparable rows
    baseline_qc_scores = {
        "soil_series": _score_match_col(
            score_rows,
            "baseline_qc_soil_series_match",
            require_col="qc_expected_soil_series",
        ),
        "ecological_site": _score_match_col(
            score_rows,
            "baseline_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
        ),
        "landscape_class": _score_match_col(
            score_rows,
            "baseline_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
        ),
    }
    terrain_qc_scores = {
        "soil_series": _score_match_col(
            score_rows,
            "terrain_qc_soil_series_match",
            require_col="qc_expected_soil_series",
        ),
        "ecological_site": _score_match_col(
            score_rows,
            "terrain_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
        ),
        "landscape_class": _score_match_col(
            score_rows,
            "terrain_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
        ),
    }
    # Passed-only QC variants
    baseline_qc_scores_passed = {
        "soil_series": _score_match_col(
            score_rows,
            "baseline_qc_soil_series_match",
            require_col="qc_expected_soil_series",
            passed_only=True,
        ),
        "ecological_site": _score_match_col(
            score_rows,
            "baseline_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
            passed_only=True,
        ),
        "landscape_class": _score_match_col(
            score_rows,
            "baseline_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
            passed_only=True,
        ),
    }
    terrain_qc_scores_passed = {
        "soil_series": _score_match_col(
            score_rows,
            "terrain_qc_soil_series_match",
            require_col="qc_expected_soil_series",
            passed_only=True,
        ),
        "ecological_site": _score_match_col(
            score_rows,
            "terrain_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
            passed_only=True,
        ),
        "landscape_class": _score_match_col(
            score_rows,
            "terrain_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
            passed_only=True,
        ),
    }

    # AIM vs QC reference agreement (how often AIM and QC agree on soil series / eco site / landscape)
    aim_qc_agreement = {
        "soil_series": _score_match_col(
            score_rows,
            "aim_qc_soil_series_match",
            require_col="qc_expected_soil_series",
        ),
        "ecological_site": _score_match_col(
            score_rows,
            "aim_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
        ),
        "landscape_class": _score_match_col(
            score_rows,
            "aim_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
        ),
    }

    summary = {
        "run_utc": timestamp,
        "execution_mode": args.execution_mode,
        "soilid_api_url": args.soilid_api_url if args.execution_mode == "api" else "",
        "list_source": args.list_source,
        "rank_method": args.rank_method,
        "plot_csv": str(plot_csv),
        "horizons_csv": str(HORIZONS_CSV),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "n_landscape_changed": n_landscape_changed,
        "baseline_match_aim": baseline_scores,
        "terrain_aim_match": terrain_scores,
        "baseline_match_aim_passed_only": baseline_scores_passed,
        "terrain_aim_match_passed_only": terrain_scores_passed,
        "baseline_qc_match": baseline_qc_scores,
        "terrain_qc_match": terrain_qc_scores,
        "baseline_qc_match_passed_only": baseline_qc_scores_passed,
        "terrain_qc_match_passed_only": terrain_qc_scores_passed,
        "aim_qc_reference_agreement": aim_qc_agreement,
        "baseline_match_by_source": baseline_scores_by_source,
        "terrain_match_by_source": terrain_scores_by_source,
        "output_rows_csv": str(rows_path),
        "output_summary_json": str(summary_json_path),
        "output_summary_txt": str(summary_txt_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_txt_path.write_text(
        "\n".join(
            [
                f"Run UTC: {timestamp}",
                f"Execution mode: {args.execution_mode}",
                f"Analyze API URL: {args.soilid_api_url if args.execution_mode == 'api' else 'n/a'}",
                f"List source: {args.list_source}",
                f"Rank method: {args.rank_method}",
                f"Plot CSV: {plot_csv}",
                f"Horizons CSV: {HORIZONS_CSV}",
                f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}",
                f"Rows with QC landscape change: {n_landscape_changed}",
                "",
                "=" * 60,
                "rank_soils MATCH RATES vs AIM REFERENCE (all rows incl. failed)",
                "=" * 60,
                "",
                "Baseline (no terrain inputs):",
                f"  Soil series:     {baseline_scores['soil_series']['matched']}/{baseline_scores['soil_series']['compared']} (accuracy={baseline_scores['soil_series']['accuracy']})",
                f"  Ecological site: {baseline_scores['ecological_site']['matched']}/{baseline_scores['ecological_site']['compared']} (accuracy={baseline_scores['ecological_site']['accuracy']})",
                f"  Landscape class: {baseline_scores['landscape_class']['matched']}/{baseline_scores['landscape_class']['compared']} (accuracy={baseline_scores['landscape_class']['accuracy']})",
                "",
                "With terrain (AIM landscape inputs):",
                f"  Soil series:     {terrain_scores['soil_series']['matched']}/{terrain_scores['soil_series']['compared']} (accuracy={terrain_scores['soil_series']['accuracy']})",
                f"  Ecological site: {terrain_scores['ecological_site']['matched']}/{terrain_scores['ecological_site']['compared']} (accuracy={terrain_scores['ecological_site']['accuracy']})",
                f"  Landscape class: {terrain_scores['landscape_class']['matched']}/{terrain_scores['landscape_class']['compared']} (accuracy={terrain_scores['landscape_class']['accuracy']})",
                "",
                "=" * 60,
                "rank_soils MATCH RATES vs AIM REFERENCE (passed rows only)",
                "=" * 60,
                "",
                "Baseline (no terrain inputs):",
                f"  Soil series:     {baseline_scores_passed['soil_series']['matched']}/{baseline_scores_passed['soil_series']['compared']} (accuracy={baseline_scores_passed['soil_series']['accuracy']})",
                f"  Ecological site: {baseline_scores_passed['ecological_site']['matched']}/{baseline_scores_passed['ecological_site']['compared']} (accuracy={baseline_scores_passed['ecological_site']['accuracy']})",
                f"  Landscape class: {baseline_scores_passed['landscape_class']['matched']}/{baseline_scores_passed['landscape_class']['compared']} (accuracy={baseline_scores_passed['landscape_class']['accuracy']})",
                "",
                "With terrain (AIM landscape inputs):",
                f"  Soil series:     {terrain_scores_passed['soil_series']['matched']}/{terrain_scores_passed['soil_series']['compared']} (accuracy={terrain_scores_passed['soil_series']['accuracy']})",
                f"  Ecological site: {terrain_scores_passed['ecological_site']['matched']}/{terrain_scores_passed['ecological_site']['compared']} (accuracy={terrain_scores_passed['ecological_site']['accuracy']})",
                f"  Landscape class: {terrain_scores_passed['landscape_class']['matched']}/{terrain_scores_passed['landscape_class']['compared']} (accuracy={terrain_scores_passed['landscape_class']['accuracy']})",
                "",
                "=" * 60,
                f"rank_soils MATCH RATES vs QC REFERENCE (all rows incl. failed; landscape-changed n={n_landscape_changed})",
                "=" * 60,
                "",
                "Baseline (no terrain inputs):",
                f"  Soil series:     {baseline_qc_scores['soil_series']['matched']}/{baseline_qc_scores['soil_series']['compared']} (accuracy={baseline_qc_scores['soil_series']['accuracy']})",
                f"  Ecological site: {baseline_qc_scores['ecological_site']['matched']}/{baseline_qc_scores['ecological_site']['compared']} (accuracy={baseline_qc_scores['ecological_site']['accuracy']})",
                f"  Landscape class: {baseline_qc_scores['landscape_class']['matched']}/{baseline_qc_scores['landscape_class']['compared']} (accuracy={baseline_qc_scores['landscape_class']['accuracy']})",
                "",
                "With terrain (QC landscape inputs):",
                f"  Soil series:     {terrain_qc_scores['soil_series']['matched']}/{terrain_qc_scores['soil_series']['compared']} (accuracy={terrain_qc_scores['soil_series']['accuracy']})",
                f"  Ecological site: {terrain_qc_scores['ecological_site']['matched']}/{terrain_qc_scores['ecological_site']['compared']} (accuracy={terrain_qc_scores['ecological_site']['accuracy']})",
                f"  Landscape class: {terrain_qc_scores['landscape_class']['matched']}/{terrain_qc_scores['landscape_class']['compared']} (accuracy={terrain_qc_scores['landscape_class']['accuracy']})",
                "",
                "=" * 60,
                f"rank_soils MATCH RATES vs QC REFERENCE (passed rows only; landscape-changed n={n_landscape_changed})",
                "=" * 60,
                "",
                "Baseline (no terrain inputs):",
                f"  Soil series:     {baseline_qc_scores_passed['soil_series']['matched']}/{baseline_qc_scores_passed['soil_series']['compared']} (accuracy={baseline_qc_scores_passed['soil_series']['accuracy']})",
                f"  Ecological site: {baseline_qc_scores_passed['ecological_site']['matched']}/{baseline_qc_scores_passed['ecological_site']['compared']} (accuracy={baseline_qc_scores_passed['ecological_site']['accuracy']})",
                f"  Landscape class: {baseline_qc_scores_passed['landscape_class']['matched']}/{baseline_qc_scores_passed['landscape_class']['compared']} (accuracy={baseline_qc_scores_passed['landscape_class']['accuracy']})",
                "",
                "With terrain (QC landscape inputs):",
                f"  Soil series:     {terrain_qc_scores_passed['soil_series']['matched']}/{terrain_qc_scores_passed['soil_series']['compared']} (accuracy={terrain_qc_scores_passed['soil_series']['accuracy']})",
                f"  Ecological site: {terrain_qc_scores_passed['ecological_site']['matched']}/{terrain_qc_scores_passed['ecological_site']['compared']} (accuracy={terrain_qc_scores_passed['ecological_site']['accuracy']})",
                f"  Landscape class: {terrain_qc_scores_passed['landscape_class']['matched']}/{terrain_qc_scores_passed['landscape_class']['compared']} (accuracy={terrain_qc_scores_passed['landscape_class']['accuracy']})",
                "",
                "=" * 60,
                "AIM vs QC REFERENCE AGREEMENT",
                "=" * 60,
                "",
                f"  Soil series:     {aim_qc_agreement['soil_series']['matched']}/{aim_qc_agreement['soil_series']['compared']} agree (accuracy={aim_qc_agreement['soil_series']['accuracy']})",
                f"  Ecological site: {aim_qc_agreement['ecological_site']['matched']}/{aim_qc_agreement['ecological_site']['compared']} agree (accuracy={aim_qc_agreement['ecological_site']['accuracy']})",
                f"  Landscape class: {aim_qc_agreement['landscape_class']['matched']}/{aim_qc_agreement['landscape_class']['compared']} agree (accuracy={aim_qc_agreement['landscape_class']['accuracy']})",
                "",
                f"Rows file:    {rows_path}",
                f"Summary JSON: {summary_json_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"total={total}")
    print(f"passed={passed}")
    print(f"failed={failed}")
    print(f"rows_csv={rows_path}")
    print(f"summary_json={summary_json_path}")
    print(f"summary_txt={summary_txt_path}")
    if failures:
        print("first_failures:")
        for key, msg in failures[:10]:
            print(f"  {key} -> {msg}")


if __name__ == "__main__":
    main()
