from pathlib import Path
import re

import pandas as pd
import pytest

from soil_id.us_soil import SoilListOutputData, rank_soils


REPO_ROOT = Path(__file__).resolve().parents[3]
TERRAIN_DATA_DIR = REPO_ROOT / "Data" / "aim_data"
HORIZONS_CSV = TERRAIN_DATA_DIR / "study_soil_horizons.csv"
DIAGNOSTICS_CSV = TERRAIN_DATA_DIR / "aim_filter_diagnostics.csv"
SUMMARY_TXT = TERRAIN_DATA_DIR / "aim_filter_summary.txt"


def _summary_first_int(summary_text, label):
    match = re.search(rf"{re.escape(label)}\s*:\s*([\d,]+)", summary_text)
    assert match, f"Could not find '{label}' in summary file"
    return int(match.group(1).replace(",", ""))


def _summary_all_unmatched(summary_text):
    return [
        int(value.replace(",", ""))
        for value in re.findall(r"Unmatched PrimaryKeys\s*:\s*([\d,]+)", summary_text)
    ]


def _to_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rfv_bucket(value):
    v = _to_float(value)
    if v is None:
        return "0-1%"
    lower = max(0, int(round(v)))
    upper = lower + 1
    return f"{lower}-{upper}%"


def _build_rank_inputs(plot_row, horizons_df):
    plot_key = plot_row["PrimaryKey"]
    hz = horizons_df[horizons_df["PrimaryKey"] == plot_key].copy()
    hz = hz.sort_values("HorizonDepthUpper", kind="stable")
    hz = hz.head(1)

    assert not hz.empty, f"No horizons found for plot {plot_key}"

    soil_horizon = hz["Texture"].fillna("LOAM").astype(str).str.upper().tolist()
    top_depth = pd.to_numeric(hz["HorizonDepthUpper"], errors="coerce").fillna(0).astype(int).tolist()
    bottom_depth = (
        pd.to_numeric(hz["HorizonDepthLower"], errors="coerce").fillna(0).astype(int).tolist()
    )
    rfv_depth = hz["RockFragments"].apply(_rfv_bucket).tolist()

    # Keep a constant LAB color vector so the test is deterministic.
    lab_color = [[41.24, 2.54, 21.17] for _ in soil_horizon]

    return {
        "soilHorizon": soil_horizon,
        "topDepth": top_depth,
        "bottomDepth": bottom_depth,
        "rfvDepth": rfv_depth,
        "lab_Color": lab_color,
    }


def _select_plot_row(plot_csv, plot_df, horizons_df):
    matched = plot_df.merge(
        horizons_df[["PrimaryKey"]].drop_duplicates(),
        on="PrimaryKey",
        how="inner",
    )
    assert not matched.empty, f"No overlapping plot/horizon keys found for {plot_csv}"

    if "AIM" in plot_csv and DIAGNOSTICS_CSV.exists():
        diag_df = pd.read_csv(DIAGNOSTICS_CSV)
        flagged = diag_df[
            (diag_df["has_texture"] == True)
            & (diag_df["has_clay_pct"] == True)
            & (diag_df["has_rock_frag"] == True)
            & (diag_df["has_color"] == True)
            & (diag_df["n_horizons"] > 0)
        ][["PrimaryKey"]]

        matched = matched.merge(flagged, on="PrimaryKey", how="inner")
        assert not matched.empty, "No AIM rows matched required diagnostics flags"

    required_cols = [
        "Slope",
        "Elevation",
        "Aspect",
        "SlopeShapeVertical",
        "SlopeShapeHorizontal",
        "LandscapeType",
    ]
    valid = matched.dropna(subset=required_cols)
    assert not valid.empty, f"No rows with required terrain fields for {plot_csv}"

    return valid.iloc[0]


def _build_list_output_data():
    depth_count = 200
    rank_rows = []
    for compname in ("alpha", "beta"):
        for _ in range(depth_count):
            rank_rows.append(
                {
                    "compname": compname,
                    "sandpct_intpl": 40.0,
                    "claypct_intpl": 20.0,
                    "rfv_intpl": 5.0,
                    "l": 50.0,
                    "a": 5.0,
                    "b": 20.0,
                }
            )
    rank_df = pd.DataFrame(rank_rows)

    comp_df = pd.DataFrame(
        {
            "compname": ["alpha", "beta"],
            "compname_grp": ["alpha", "beta"],
            "comp_max_bottom": [150, 150],
            "slope_r": [5.0, 5.0],
            "slope_l": [0.0, 0.0],
            "slope_h": [10.0, 10.0],
            "elev_r": [1500.0, 1500.0],
            "elev_l": [1200.0, 1200.0],
            "elev_h": [1800.0, 1800.0],
            "aspect_northerness": [1.0, -1.0],
            "aspect_easterness": [0.0, 0.0],
            "shape_vert_class": ["convex", "concave"],
            "shape_horiz_class": ["convex", "concave"],
            "landscape_class": ["fans", "hills_mountains"],
            "cokey": ["100", "200"],
            "cond_prob": [0.5, 0.5],
            "clay": ["No", "No"],
            "taxorder": ["Entisols", "Entisols"],
            "taxsubgrp": ["Typic Torrifluvents", "Typic Torrifluvents"],
            "OSD_text_int": ["No", "No"],
            "OSD_rfv_int": ["No", "No"],
            "data_source": ["SSURGO", "SSURGO"],
            "Rank_Loc": ["1", "2"],
        }
    )

    return SoilListOutputData(
        soil_list_json={"metadata": {"location": "us"}, "soilList": []},
        rank_data_csv=rank_df.to_csv(index=False),
        map_unit_component_data_csv=comp_df.to_csv(index=False),
    )


def _extract_scores(rank_result):
    return {
        entry["name"]: entry.get("score_data_loc")
        for entry in rank_result["soilRank"]
        if not entry.get("not_displayed", False)
    }


@pytest.mark.parametrize(
    "plot_csv",
    [
        "study_plot_characteristics_AIM.csv",
        "study_plot_characteristics_QC.csv",
    ],
)
def test_rank_soils_with_study_plot_datasets(plot_csv):
    plot_df = pd.read_csv(TERRAIN_DATA_DIR / plot_csv)
    horizons_df = pd.read_csv(HORIZONS_CSV)

    # Choose a row that is explicitly flagged as data-complete for AIM runs.
    plot_row = _select_plot_row(plot_csv, plot_df, horizons_df)
    rank_inputs = _build_rank_inputs(plot_row, horizons_df)
    list_output_data = _build_list_output_data()

    p_slope = _to_float(plot_row.get("Slope"))
    p_elev = _to_float(plot_row.get("Elevation"))
    p_aspect = _to_float(plot_row.get("Aspect"))
    p_shape_vert = plot_row.get("SlopeShapeVertical")
    p_shape_horiz = plot_row.get("SlopeShapeHorizontal")
    p_landscape = plot_row.get("LandscapeType")

    assert p_slope is not None
    assert p_elev is not None
    assert p_aspect is not None
    assert isinstance(p_shape_vert, str) and p_shape_vert.strip()
    assert isinstance(p_shape_horiz, str) and p_shape_horiz.strip()
    assert isinstance(p_landscape, str) and p_landscape.strip()

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
        bedrock=None,
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
        bedrock=None,
        cracks=False,
        pAspect=p_aspect,
        pSlopeShapeVert=p_shape_vert,
        pSlopeShapeHoriz=p_shape_horiz,
        pLandscape=p_landscape,
    )

    baseline_scores = _extract_scores(baseline)
    terrain_scores = _extract_scores(with_terrain)

    assert baseline_scores
    assert terrain_scores
    assert set(baseline_scores.keys()) == set(terrain_scores.keys())
    assert baseline_scores != terrain_scores


def test_aim_filter_coverage_summary_matches_diagnostics():
    assert DIAGNOSTICS_CSV.exists(), "aim_filter_diagnostics.csv not found"
    assert SUMMARY_TXT.exists(), "aim_filter_summary.txt not found"

    diag_df = pd.read_csv(DIAGNOSTICS_CSV)
    summary_text = SUMMARY_TXT.read_text(encoding="utf-8")

    study_primarykeys = _summary_first_int(summary_text, "Study PrimaryKeys")
    output_rows = _summary_first_int(summary_text, "Output rows")
    total_horizon_rows = _summary_first_int(summary_text, "Total horizon rows")

    unmatched_counts = _summary_all_unmatched(summary_text)
    assert unmatched_counts, "No unmatched-key sections found in summary"
    assert all(value == 0 for value in unmatched_counts)

    assert diag_df["PrimaryKey"].nunique() == study_primarykeys
    assert len(diag_df) == output_rows
    assert int(diag_df["n_horizons"].sum()) == total_horizon_rows

    assert diag_df["in_plot_characteristics"].all()
    assert diag_df["in_soil_horizon"].all()

    assert diag_df["n_horizons"].min() >= 1
    assert diag_df["n_horizons"].max() <= 6
