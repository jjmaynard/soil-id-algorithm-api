import argparse
import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

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

    return {
        "soil_series": soil_series,
        "ecological_site": _norm_text(meta.get("ecological_site")),
        "landscape_class": _norm_text(meta.get("landscape_class")),
        "component_id": component_id,
    }


def _extract_expected_rank(rank_result, expected_soil_series, rank_method="rank_data_loc"):
    expected = _norm_name(expected_soil_series)
    if not expected:
        return {"component_id": "", "rank": ""}

    soil_rank = rank_result.get("soilRank") or []
    for item in soil_rank:
        item_component = _norm_name(item.get("component"))
        if item_component == expected:
            rank = _norm_text(item.get(rank_method))
            return {
                "component_id": _norm_text(item.get("componentID")),
                "rank": rank,
            }

    return {"component_id": "", "rank": ""}


def _metadata_for_component(comp_meta_by_id, component_id="", component_name=""):
    component_id = _norm_text(component_id)
    if component_id and component_id in comp_meta_by_id:
        return comp_meta_by_id[component_id]

    name_key = f"name::{_norm_name(component_name)}"
    if _norm_name(component_name) and name_key in comp_meta_by_id:
        return comp_meta_by_id[name_key]

    return {"ecological_site": "", "landscape_type": "", "landscape_class": ""}


def _build_component_metadata(list_output_data):
    comp_df = pd.read_csv(io.StringIO(list_output_data.map_unit_component_data_csv))
    comp_df.columns = [str(c).strip() for c in comp_df.columns]

    by_id = {}
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

    for entry in list_output_data.soil_list_json.get("soilList", []):
        comp_id = _norm_text(entry.get("id", {}).get("componentID"))
        comp_name = _norm_text(entry.get("id", {}).get("name"))
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


def _score_summary(rows, mode_prefix, expected_col, predicted_col, source=None):
    subset = [r for r in rows if _norm_text(r.get(expected_col))]
    if source:
        subset = [r for r in subset if _norm_source(r.get("source")) == source]
    compared = len(subset)
    matched = sum(1 for r in subset if r.get(f"{mode_prefix}_{predicted_col}_match") is True)
    return {
        "compared": compared,
        "matched": matched,
        "accuracy": None if compared == 0 else round(matched / compared, 4),
    }


def _score_summary_with_match_col(rows, expected_col, match_col):
    subset = [r for r in rows if _norm_text(r.get(expected_col))]
    compared = len(subset)
    matched = sum(1 for r in subset if r.get(match_col) is True)
    return {
        "compared": compared,
        "matched": matched,
        "accuracy": None if compared == 0 else round(matched / compared, 4),
    }


def _score_match_col(rows, match_col, require_col=None, changed_only=False):
    subset = rows
    if changed_only:
        subset = [r for r in subset if r.get("landscape_class_qc_changed") is True]
    if require_col:
        subset = [r for r in subset if _norm_text(r.get(require_col))]
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
        default="study_plot_characteristics_AIM.csv",
        help="Plot characteristics CSV path (absolute, workspace-relative, or under Data/aim_data)",
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
        help="Execution backend flag accepted for compatibility with master runner",
    )
    parser.add_argument(
        "--soilid-api-url",
        default=os.getenv("SOILID_API_URL", "https://soil-id-algorithm-api.vercel.app/api/analyze-soil"),
        help="Analyze-soil API URL (currently informational in this branch runner)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout for API mode compatibility",
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
    horizons_df = pd.read_csv(HORIZONS_CSV)

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

    synthetic_list_output_data = None

    total = len(valid)
    passed = 0
    failed = 0
    skipped = 0
    failures = []
    row_results = []

    for _, row in valid.iterrows():
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

            if args.list_source == "live":
                lon = _to_float(row.get("Longitude_NAD83"))
                lat = _to_float(row.get("Latitude_NAD83"))
                if lon is None or lat is None:
                    raise RuntimeError("Missing Longitude_NAD83/Latitude_NAD83 for live list_soils")
                list_output_data = list_soils(lon=lon, lat=lat, sim=False)
                if not hasattr(list_output_data, "map_unit_component_data_csv"):
                    skipped += 1
                    row_results.append(
                        {
                            "PrimaryKey": key,
                            "source": row_source,
                            "status": "skipped",
                            "error": _norm_text(list_output_data) or "No SoilListOutputData returned",
                            **plot_metadata,
                            "expected_soil_series": expected_soil_series,
                            "expected_ecological_site": expected_ecological_site,
                            "expected_landscape_type": expected_landscape_raw,
                            "expected_landscape_class": expected_landscape_class,
                            "expected_rank_baseline": "",
                            "expected_rank_terrain": "",
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
            else:
                # Synthetic candidates are only allowed when explicitly requested.
                if synthetic_list_output_data is None:
                    from soil_id.tests.us.test_study_dataset_inputs import _build_list_output_data

                    synthetic_list_output_data = _build_list_output_data()
                list_output_data = synthetic_list_output_data

            comp_meta_by_id = _build_component_metadata(list_output_data)

            baseline = rank_soils(
                lon=0.0,
                lat=0.0,
                list_output_data=list_output_data,
                soilHorizon=rank_inputs["soilHorizon"],
                topDepth=rank_inputs["topDepth"],
                bottomDepth=rank_inputs["bottomDepth"],
                rfvDepth=rank_inputs["rfvDepth"],
                lab_Color=rank_inputs["lab_Color"],
                pSlope=p_slope,
                pElev=p_elev,
                bedrock=bedrock_depth,
                cracks=False,
            )

            with_terrain = rank_soils(
                lon=0.0,
                lat=0.0,
                list_output_data=list_output_data,
                soilHorizon=rank_inputs["soilHorizon"],
                topDepth=rank_inputs["topDepth"],
                bottomDepth=rank_inputs["bottomDepth"],
                rfvDepth=rank_inputs["rfvDepth"],
                lab_Color=rank_inputs["lab_Color"],
                pSlope=p_slope,
                pElev=p_elev,
                bedrock=bedrock_depth,
                cracks=False,
                pAspect=p_aspect,
                pSlopeShapeVert=p_shape_vert,
                pSlopeShapeHoriz=p_shape_horiz,
                pLandscape=p_landscape,
            )

            if not baseline.get("soilRank") or not with_terrain.get("soilRank"):
                raise RuntimeError("Empty soilRank in output")

            baseline_expected = _extract_expected_rank(baseline, expected_soil_series, args.rank_method)
            terrain_expected = _extract_expected_rank(with_terrain, expected_soil_series, args.rank_method)
            expected_meta = _metadata_for_component(
                comp_meta_by_id,
                component_id=baseline_expected["component_id"] or terrain_expected["component_id"],
                component_name=expected_soil_series,
            )

            baseline_top = _extract_top_metadata(baseline, comp_meta_by_id, args.rank_method)
            terrain_top = _extract_top_metadata(with_terrain, comp_meta_by_id, args.rank_method)

            # QC terrain run: re-rank with QC landscape type (only when it changed from AIM)
            p_landscape_qc = _norm_text(row.get("LandscapeType_QC")) or p_landscape
            if landscape_class_qc_changed:
                with_terrain_qc = rank_soils(
                    lon=0.0,
                    lat=0.0,
                    list_output_data=list_output_data,
                    soilHorizon=rank_inputs["soilHorizon"],
                    topDepth=rank_inputs["topDepth"],
                    bottomDepth=rank_inputs["bottomDepth"],
                    rfvDepth=rank_inputs["rfvDepth"],
                    lab_Color=rank_inputs["lab_Color"],
                    pSlope=p_slope,
                    pElev=p_elev,
                    bedrock=bedrock_depth,
                    cracks=False,
                    pAspect=p_aspect,
                    pSlopeShapeVert=p_shape_vert,
                    pSlopeShapeHoriz=p_shape_horiz,
                    pLandscape=p_landscape_qc,
                )
                if not with_terrain_qc.get("soilRank"):
                    raise RuntimeError("Empty soilRank in QC terrain output")
                terrain_qc_top = _extract_top_metadata(with_terrain_qc, comp_meta_by_id, args.rank_method)
            else:
                # landscape class unchanged — QC terrain result equals AIM terrain result
                terrain_qc_top = terrain_top.copy()

            passed += 1
            row_results.append(
                {
                    "PrimaryKey": key,
                    "source": row_source,
                    "status": "passed",
                    "error": "",
                    **plot_metadata,
                    "expected_soil_series": expected_soil_series,
                    "expected_ecological_site": expected_ecological_site,
                    "expected_landscape_type": expected_landscape_raw,
                    "expected_landscape_class": expected_landscape_class,
                    "expected_rank_baseline": baseline_expected["rank"],
                    "expected_rank_terrain": terrain_expected["rank"],
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
                }
            )
        except Exception as exc:
            failed += 1
            failures.append((key, str(exc)))
            row_results.append(
                {
                    "PrimaryKey": key,
                    "source": _norm_source(row.get("source") or row.get("Source")),
                    "status": "failed",
                    "error": str(exc),
                    **_extract_plot_metadata(row),
                    "expected_soil_series": _pick_expected_soil_series(row),
                    "expected_ecological_site": _norm_text(row.get("EcolSite")),
                    "expected_landscape_type": _norm_text(row.get("LandscapeType")),
                    "expected_landscape_class": _aim_landscape_class(
                        _norm_text(row.get("LandscapeType"))
                    ),
                    "expected_rank_baseline": "",
                    "expected_rank_terrain": "",
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
                }
            )

    stem = Path(args.plot_csv).stem
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    rows_path = output_dir / f"{stem}_run_results_{timestamp}.csv"
    summary_json_path = output_dir / f"{stem}_run_summary_{timestamp}.json"
    summary_txt_path = output_dir / f"{stem}_run_summary_{timestamp}.txt"

    pd.DataFrame(row_results).to_csv(rows_path, index=False)

    baseline_scores = {
        "soil_series": _score_summary(row_results, "baseline_aim", "expected_soil_series", "soil_series"),
        "ecological_site": _score_summary(
            row_results, "baseline_aim", "expected_ecological_site", "ecological_site"
        ),
        "landscape_class": _score_summary(
            row_results, "baseline_aim", "expected_landscape_class", "landscape_class"
        ),
    }
    terrain_scores = {
        "soil_series": _score_summary(row_results, "terrain_aim", "expected_soil_series", "soil_series"),
        "ecological_site": _score_summary(
            row_results, "terrain_aim", "expected_ecological_site", "ecological_site"
        ),
        "landscape_class": _score_summary(
            row_results, "terrain_aim", "expected_landscape_class", "landscape_class"
        ),
    }

    baseline_scores_by_source = {
        "AIM": {
            "soil_series": _score_summary(
                row_results, "baseline_aim", "expected_soil_series", "soil_series", source="AIM"
            ),
            "ecological_site": _score_summary(
                row_results,
                "baseline_aim",
                "expected_ecological_site",
                "ecological_site",
                source="AIM",
            ),
            "landscape_class": _score_summary(
                row_results,
                "baseline_aim",
                "expected_landscape_class",
                "landscape_class",
                source="AIM",
            ),
        },
        "QC": {
            "soil_series": _score_summary_with_match_col(
                row_results,
                "qc_expected_soil_series",
                "baseline_qc_soil_series_match",
            ),
            "ecological_site": _score_summary_with_match_col(
                row_results,
                "qc_expected_ecological_site",
                "baseline_qc_ecological_site_match",
            ),
            "landscape_class": _score_summary_with_match_col(
                row_results,
                "qc_expected_landscape_class",
                "baseline_qc_landscape_class_match",
            ),
        },
    }
    terrain_scores_by_source = {
        "AIM": {
            "soil_series": _score_summary(
                row_results, "terrain_aim", "expected_soil_series", "soil_series", source="AIM"
            ),
            "ecological_site": _score_summary(
                row_results,
                "terrain_aim",
                "expected_ecological_site",
                "ecological_site",
                source="AIM",
            ),
            "landscape_class": _score_summary(
                row_results,
                "terrain_aim",
                "expected_landscape_class",
                "landscape_class",
                source="AIM",
            ),
        },
        "QC": {
            "soil_series": _score_summary_with_match_col(
                row_results,
                "qc_expected_soil_series",
                "terrain_qc_soil_series_match",
            ),
            "ecological_site": _score_summary_with_match_col(
                row_results,
                "qc_expected_ecological_site",
                "terrain_qc_ecological_site_match",
            ),
            "landscape_class": _score_summary_with_match_col(
                row_results,
                "qc_expected_landscape_class",
                "terrain_qc_landscape_class_match",
            ),
        },
    }

    # rank_soils (QC landscape inputs) vs QC reference — all comparable rows
    n_landscape_changed = sum(
        1 for r in row_results if r.get("landscape_class_qc_changed") is True
    )
    baseline_qc_scores = {
        "soil_series": _score_match_col(
            row_results,
            "baseline_qc_soil_series_match",
            require_col="qc_expected_soil_series",
        ),
        "ecological_site": _score_match_col(
            row_results,
            "baseline_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
        ),
        "landscape_class": _score_match_col(
            row_results,
            "baseline_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
        ),
    }
    terrain_qc_scores = {
        "soil_series": _score_match_col(
            row_results,
            "terrain_qc_soil_series_match",
            require_col="qc_expected_soil_series",
        ),
        "ecological_site": _score_match_col(
            row_results,
            "terrain_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
        ),
        "landscape_class": _score_match_col(
            row_results,
            "terrain_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
        ),
    }

    # AIM vs QC reference agreement (how often AIM and QC agree on soil series / eco site / landscape)
    aim_qc_agreement = {
        "soil_series": _score_match_col(
            row_results,
            "aim_qc_soil_series_match",
            require_col="qc_expected_soil_series",
        ),
        "ecological_site": _score_match_col(
            row_results,
            "aim_qc_ecological_site_match",
            require_col="qc_expected_ecological_site",
        ),
        "landscape_class": _score_match_col(
            row_results,
            "aim_qc_landscape_class_match",
            require_col="qc_expected_landscape_class",
        ),
    }

    summary = {
        "run_utc": timestamp,
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
        "baseline_qc_match": baseline_qc_scores,
        "terrain_qc_match": terrain_qc_scores,
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
                f"List source: {args.list_source}",
                f"Rank method: {args.rank_method}",
                f"Plot CSV: {plot_csv}",
                f"Horizons CSV: {HORIZONS_CSV}",
                f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}",
                f"Rows with QC landscape change: {n_landscape_changed}",
                "",
                "=" * 60,
                "rank_soils MATCH RATES vs AIM REFERENCE (all rows)",
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
                f"rank_soils MATCH RATES vs QC REFERENCE (all comparable rows; landscape-changed n={n_landscape_changed})",
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
