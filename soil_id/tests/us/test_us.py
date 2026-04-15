# Copyright © 2024 Technology Matters
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

import logging
import time

import pandas as pd
import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from soil_id.tests.utils import clean_soil_list_json
from soil_id.us_soil import SoilListOutputData, list_soils, rank_soils

test_locations = [
    {"lon": -121.5111084, "lat": 45.6508331},
    {"lon": -101.9733687, "lat": 33.81246789},
    {"lon": -121.0347381, "lat": 45.88932423},
    {"lon": -85.50621214, "lat": 39.26009312},
    {"lon": -94.31005777, "lat": 42.63413723},
    {"lon": -99.55016693, "lat": 37.48216451},
    {"lon": -157.2767099, "lat": 62.32776717},
    {"lon": -156.4422738, "lat": 63.52666854},
    {"lon": -119.4596489, "lat": 43.06450312},
    {"lon": -69.28246582, "lat": 47.21392200},
    {"lon": -158.4018264, "lat": 60.42282639},
    {"lon": -121.8166, "lat": 48.6956},
    {"lat": 34.92816, "lon": -114.80764},  # NOTCOM
    {"lat": 35.599180, "lon": -120.491439},  # previous crash: no objects to concatenate
    {"lon": -122.084000, "lat": 37.422000},  # missing LCC
    {"lat": 42.494912, "lon": -123.064531},  # crash: could not broadcast input array
    # {"lat": 40.79861, "lon": -112.35477},  # crash: str object has no attribute rank_data_csv
]

test_params = []
for idx, coords in enumerate(test_locations):
    test_params.append(pytest.param(coords, id=f"{coords['lat']},{coords['lon']}"))


@pytest.mark.parametrize("location", test_params)
def test_soil_location(location, snapshot):
    # Dummy Soil Profile Data (replicating the structure provided)
    soilHorizon = ["LOAM"] * 7
    topDepth = [0, 1, 10, 20, 50, 70, 100]
    bottomDepth = [1, 10, 20, 50, 70, 100, 120]
    rfvDepth = ["0-1%"] * 7
    lab_Color = [[41.24, 2.54, 21.17]] * 7
    bedrock = None
    pSlope = "15"
    pElev = None
    cracks = False

    start_time = time.perf_counter()
    list_soils_result = list_soils(location["lon"], location["lat"])
    logging.info(f"...time: {(time.perf_counter() - start_time):.2f}s")
    rank_result = rank_soils(
        location["lon"],
        location["lat"],
        list_soils_result,
        soilHorizon,
        topDepth,
        bottomDepth,
        rfvDepth,
        lab_Color,
        pSlope,
        pElev,
        bedrock,
        cracks,
    )

    assert snapshot.with_defaults(extension_class=JSONSnapshotExtension) == {
        "list": clean_soil_list_json(list_soils_result.soil_list_json),
        "rank": rank_result,
    }


def test_empty_rank():
    SoilListOutputData = list_soils(test_locations[0]["lon"], test_locations[0]["lat"])
    rank_soils(
        test_locations[0]["lon"],
        test_locations[0]["lat"],
        SoilListOutputData,
        soilHorizon=[],
        topDepth=[],
        bottomDepth=[],
        rfvDepth=[],
        lab_Color=[],
        pSlope=None,
        pElev=None,
        bedrock=None,
        cracks=None,
    )


def test_rank_soils_terrain_inputs_change_scores():
    rank_df = pd.DataFrame(
        {
            "compname": ["alpha", "beta"],
            "sandpct_intpl": [40.0, 40.0],
            "claypct_intpl": [20.0, 20.0],
            "rfv_intpl": [5.0, 5.0],
            "l": [50.0, 50.0],
            "a": [5.0, 5.0],
            "b": [20.0, 20.0],
        }
    )

    comp_df = pd.DataFrame(
        {
            "compname": ["alpha", "beta"],
            "compname_grp": ["alpha", "beta"],
            "comp_max_bottom": [150, 150],
            "slope_r": [35.0, 20.0],
            "slope_l": [30.0, 15.0],
            "slope_h": [40.0, 25.0],
            "elev_r": [1200.0, 1000.0],
            "elev_l": [1100.0, 900.0],
            "elev_h": [1300.0, 1100.0],
            "aspect_northerness": [1.0, 0.0],
            "aspect_easterness": [0.0, -1.0],
            "shape_vert_class": ["concave", "convex"],
            "shape_horiz_class": ["concave", "convex"],
            "landscape_class": ["alluvial_fan", "hill_mountain"],
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

    list_output_data = SoilListOutputData(
        soil_list_json={"metadata": {"location": "us"}, "soilList": []},
        rank_data_csv=rank_df.to_csv(index=False),
        map_unit_component_data_csv=comp_df.to_csv(index=False),
    )

    baseline = rank_soils(
        lon=-106.0,
        lat=32.0,
        list_output_data=list_output_data,
        soilHorizon=[],
        topDepth=[],
        bottomDepth=[],
        rfvDepth=[],
        lab_Color=[],
        pSlope=20.0,
        pElev=1000.0,
        bedrock=None,
        cracks=False,
    )

    with_terrain = rank_soils(
        lon=-106.0,
        lat=32.0,
        list_output_data=list_output_data,
        soilHorizon=[],
        topDepth=[],
        bottomDepth=[],
        rfvDepth=[],
        lab_Color=[],
        pSlope=20.0,
        pElev=1000.0,
        bedrock=None,
        cracks=False,
        pAspect=0.0,
        pSlopeShapeVert="Concave",
        pSlopeShapeHoriz="Concave",
        pLandscape="alluvial fan",
    )

    baseline_top = baseline["soilRank"][0]["name"]
    terrain_top = with_terrain["soilRank"][0]["name"]

    assert baseline_top == "Beta"
    assert terrain_top == "Alpha"
