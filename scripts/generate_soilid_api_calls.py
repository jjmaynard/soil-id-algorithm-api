#!/usr/bin/env python3
import argparse
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


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


def _to_float(value):
    txt = _norm_text(value)
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


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


def _build_rank_inputs_with_munsell(plot_key, horizons_df):
    hz = horizons_df[horizons_df["PrimaryKey"] == plot_key].copy()
    hz = hz.sort_values("HorizonDepthUpper", kind="stable")
    if hz.empty:
        raise RuntimeError(f"No horizons found for plot {plot_key}")

    hz["Texture"] = hz["Texture"].apply(_normalize_texture_value)
    hz["HorizonDepthUpper"] = pd.to_numeric(hz["HorizonDepthUpper"], errors="coerce").fillna(0)
    hz["HorizonDepthLower"] = pd.to_numeric(hz["HorizonDepthLower"], errors="coerce").fillna(0)

    hz = hz[hz["HorizonDepthLower"] >= hz["HorizonDepthUpper"]].copy()
    if hz.empty:
        raise RuntimeError(f"No valid horizon intervals for plot {plot_key}")

    munsell_colors = []
    for _, row in hz.iterrows():
        hue = _norm_text(row.get("Hue"))
        value = _norm_text(row.get("Value"))
        chroma = _norm_text(row.get("Chroma"))
        if hue and value and chroma:
            munsell_colors.append(f"{hue} {value}/{chroma}")
        else:
            munsell_colors.append(None)

    return {
        "soilHorizon": hz["Texture"].tolist(),
        "topDepth": hz["HorizonDepthUpper"].astype(int).tolist(),
        "bottomDepth": hz["HorizonDepthLower"].astype(int).tolist(),
        "rfvDepth": hz["RockFragments"].apply(_rfv_bucket).tolist(),
        "munsell_Color": munsell_colors,
    }


def _build_analyze_payload(row, rank_inputs):
    lon = _to_float(row.get("Longitude_NAD83"))
    lat = _to_float(row.get("Latitude_NAD83"))
    if lon is None or lat is None:
        raise RuntimeError("Missing Longitude_NAD83/Latitude_NAD83")

    p_slope = _to_float(row.get("Slope"))
    p_elev = _to_float(row.get("Elevation"))
    p_aspect = _to_float(row.get("Aspect"))

    payload = {
        # Keep field order aligned with requested API body format.
        "bottomDepth": rank_inputs.get("bottomDepth"),
        "cracks": False,
        "munsell_Color": rank_inputs.get("munsell_Color"),
        "lat": lat,
        "lon": lon,
        "map_unit_component_data_csv": "mukey,cokey...",
        "pAspect": p_aspect,
        "pElev": p_elev,
        "pLandscape": _norm_text(row.get("LandscapeType")),
        "pLandscapeMode": "base",
        "pSlope": p_slope,
        "pSlopeShapeHoriz": _norm_text(row.get("SlopeShapeHorizontal")),
        "pSlopeShapeVert": _norm_text(row.get("SlopeShapeVertical")),
        "rank_data_csv": "compname,sandpct_intpl...",
        "rfvDepth": rank_inputs.get("rfvDepth"),
        "soilHorizon": rank_inputs.get("soilHorizon"),
        "soil_list_json": {"metadata": {}, "soilList": []},
        "topDepth": rank_inputs.get("topDepth"),
    }

    return payload


def _build_curl_command(api_url, payload):
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return (
        "curl -sS -X POST "
        + shlex.quote(api_url)
        + " -H 'accept: application/json' -H 'Content-Type: application/json' --data-raw "
        + shlex.quote(payload_json)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate one SoilID analyze-soil API call per plot and export to CSV."
    )
    parser.add_argument(
        "--plot-csv",
        default="Data/aim_data/study_plot_characteristics_enriched.csv",
        help="Plot characteristics CSV path.",
    )
    parser.add_argument(
        "--horizons-csv",
        default="Data/aim_data/study_soil_horizons.csv",
        help="Soil horizons CSV path.",
    )
    parser.add_argument(
        "--api-url",
        default="https://soil-id-algorithm-api.vercel.app/api/analyze-soil",
        help="Analyze-soil endpoint URL.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit output to first N valid rows.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path (default: outputs/aim_qc/<plot_stem>_api_calls_<UTC>.csv).",
    )
    args = parser.parse_args()

    plot_csv = Path(args.plot_csv)
    horizons_csv = Path(args.horizons_csv)

    if not plot_csv.exists():
        raise FileNotFoundError(f"Plot CSV not found: {plot_csv}")
    if not horizons_csv.exists():
        raise FileNotFoundError(f"Horizons CSV not found: {horizons_csv}")

    plot_df = pd.read_csv(plot_csv)
    horizons_df = pd.read_csv(horizons_csv)

    required_cols = [
        "Slope",
        "Elevation",
        "Aspect",
        "SlopeShapeVertical",
        "SlopeShapeHorizontal",
        "LandscapeType",
        "Longitude_NAD83",
        "Latitude_NAD83",
    ]

    matched = plot_df.merge(
        horizons_df[["PrimaryKey"]].drop_duplicates(), on="PrimaryKey", how="inner"
    )
    valid = matched.dropna(subset=required_cols).copy()
    if args.max_rows is not None:
        valid = valid.iloc[: args.max_rows]

    rows = []
    failures = []
    for _, row in valid.iterrows():
        key = row.get("PrimaryKey")
        try:
            rank_inputs = _build_rank_inputs_with_munsell(key, horizons_df)
            payload = _build_analyze_payload(row, rank_inputs)
            payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            payload_pretty_json = json.dumps(payload, indent=2, ensure_ascii=True)
            api_call = _build_curl_command(args.api_url, payload)
            rows.append(
                {
                    "PrimaryKey": key,
                    "api_payload_json": payload_json,
                    "api_payload_pretty_json": payload_pretty_json,
                    "api_call": api_call,
                }
            )
        except Exception as exc:
            failures.append({"PrimaryKey": key, "error": str(exc)})

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_csv = (
            Path("Data")
            / "aim_data"
            / "R_evaluation"
            / "outputs"
            / "aim_qc"
            / f"{plot_csv.stem}_api_calls_{timestamp}.csv"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"valid_rows={len(valid)}")
    print(f"generated_calls={len(rows)}")
    print(f"failed_rows={len(failures)}")
    print(f"output_csv={output_csv}")
    if failures:
        print("first_failures:")
        for item in failures[:10]:
            print(f"  {item['PrimaryKey']} -> {item['error']}")


if __name__ == "__main__":
    main()
