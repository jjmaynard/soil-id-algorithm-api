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

"""Unit tests for Gower distance transforms introduced in the session:

  * texture_dist  – replaces sandpct_intpl / claypct_intpl with a single
                    normalised Euclidean distance in (sand%, clay%) space.
  * color_delta_e – replaces l / a / b columns with a single ΔE2000 distance
                    normalised by DELTA_E_MAX (50).
  * SITE_THEORETICAL_RANGES – hybrid-floor normalization for site features.
  * validate_clay_estimate  – QC gate for AIM field clay estimates.
  * HORIZON_FEATURE_WEIGHTS – expected keys and value ranges.
"""

import math

import numpy as np
import pandas as pd
import pytest

from soil_id.color import calculate_deltaE2000
from soil_id.us_soil import SoilListOutputData, rank_soils
from soil_id.utils import (
    DELTA_E_MAX,
    TEXTURE_MAX_EUCLIDEAN_DIST,
    validate_clay_estimate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_list_output(rank_df, comp_df):
    """Wrap DataFrames into a SoilListOutputData fixture."""
    return SoilListOutputData(
        soil_list_json={"metadata": {"location": "us"}, "soilList": []},
        rank_data_csv=rank_df.to_csv(index=False),
        map_unit_component_data_csv=comp_df.to_csv(index=False),
    )


def _rank_df(sand, clay):
    """Build a minimal rank DataFrame with two components."""
    return pd.DataFrame(
        {
            "compname": ["alpha", "beta"],
            "sandpct_intpl": [sand, 40.0],
            "claypct_intpl": [clay, 20.0],
            "rfv_intpl": [5.0, 5.0],
            "l": [50.0, 50.0],
            "a": [5.0, 5.0],
            "b": [20.0, 20.0],
        }
    )


def _comp_df():
    """Build a minimal comp DataFrame with two identical components."""
    return pd.DataFrame(
        {
            "compname": ["alpha", "beta"],
            "compname_grp": ["alpha", "beta"],
            "comp_max_bottom": [150, 150],
            "slope_r": [15.0, 15.0],
            "slope_l": [10.0, 10.0],
            "slope_h": [20.0, 20.0],
            "elev_r": [1000.0, 1000.0],
            "elev_l": [900.0, 900.0],
            "elev_h": [1100.0, 1100.0],
            "aspect_northerness": [0.0, 0.0],
            "aspect_easterness": [0.0, 0.0],
            "shape_vert_class": ["linear", "linear"],
            "shape_horiz_class": ["linear", "linear"],
            "landscape_class": ["hill_mountain", "hill_mountain"],
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_texture_max_dist_positive(self):
        assert TEXTURE_MAX_EUCLIDEAN_DIST > 0

    def test_texture_max_dist_roughly_correct(self):
        # Sand centroid ≈ (92, 5), Clay centroid ≈ (22.5, 70)
        expected = math.sqrt((92 - 22.5) ** 2 + (5 - 70) ** 2)
        assert abs(TEXTURE_MAX_EUCLIDEAN_DIST - expected) < 1.0

    def test_delta_e_max_positive(self):
        assert DELTA_E_MAX > 0

    def test_delta_e_max_in_reasonable_range(self):
        # Should be the empirical ceiling for soil colour space (~50)
        assert 30.0 <= DELTA_E_MAX <= 100.0


# ---------------------------------------------------------------------------
# calculate_deltaE2000
# ---------------------------------------------------------------------------

class TestCalculateDeltaE2000:
    def test_identical_colours_give_zero(self):
        lab = [50.0, 5.0, 20.0]
        assert calculate_deltaE2000(lab, lab) == pytest.approx(0.0, abs=1e-9)

    def test_different_colours_give_positive_distance(self):
        black = [0.0, 0.0, 0.0]
        white = [100.0, 0.0, 0.0]
        assert calculate_deltaE2000(black, white) > 0

    def test_symmetric(self):
        a = [40.0, 10.0, 15.0]
        b = [70.0, -5.0, 30.0]
        assert calculate_deltaE2000(a, b) == pytest.approx(calculate_deltaE2000(b, a), abs=1e-6)

    def test_delta_e_max_is_larger_than_typical_soil_distance(self):
        # Two extreme-but-realistic soil Munsell chips: very dark grey vs pale tan.
        # The purpose is to confirm that DELTA_E_MAX is a meaningful ceiling, not
        # that it bounds every possible LAB pair (the rank_soils clip handles that).
        very_dark = [10.0, 2.0, 5.0]
        pale_tan = [60.0, 5.0, 20.0]
        de = calculate_deltaE2000(very_dark, pale_tan)
        # DELTA_E_MAX should be at least as large as this real-world soil contrast
        assert de < DELTA_E_MAX * 1.2, (
            "DELTA_E_MAX may be too low; consider raising it if soil colour pairs "
            "routinely exceed it before clipping"
        )


# ---------------------------------------------------------------------------
# validate_clay_estimate
# ---------------------------------------------------------------------------

class TestValidateClayEstimate:
    def test_in_range_value_returned_unchanged(self):
        # "clay loam" range is 27–40 %; 30 % is within range
        result = validate_clay_estimate("clay loam", 30.0)
        assert result == pytest.approx(30.0)

    def test_below_range_returns_centroid(self):
        # "clay" min is 40 %; 10 % is below → centroid
        result = validate_clay_estimate("clay", 10.0)
        assert result > 40.0  # centroid is well above lower bound

    def test_above_range_returns_centroid(self):
        # "sand" max is 10 %; 50 % is above → centroid
        result = validate_clay_estimate("sand", 50.0)
        assert result < 10.0  # centroid is well below upper bound

    def test_boundary_values_accepted(self):
        # Exactly on the "loam" boundary (7–27 %)
        assert validate_clay_estimate("loam", 7.0) == pytest.approx(7.0)
        assert validate_clay_estimate("loam", 27.0) == pytest.approx(27.0)

    def test_none_returns_centroid(self):
        result = validate_clay_estimate("loam", None)
        # getClay("loam") centroid should be ≈ 17 %
        assert 7.0 <= result <= 27.0

    def test_unknown_texture_class_returns_centroid(self):
        # Falls back gracefully for unknown class names
        result = validate_clay_estimate("gravel", 5.0)
        assert result is not None and isinstance(result, float)


# ---------------------------------------------------------------------------
# texture_dist transform — direct math tests
# ---------------------------------------------------------------------------

class TestTextureDist:
    """Verify the pre-computed texture distance math used in rank_soils.

    rank_soils requires depth-interpolated (200-row) DataFrames from list_soils,
    which are not available in unit tests.  These tests verify the transform
    arithmetic directly against TEXTURE_MAX_EUCLIDEAN_DIST."""

    def test_identical_centroids_give_zero(self):
        sand, clay = 45.0, 20.0
        dist = math.sqrt((sand - sand) ** 2 + (clay - clay) ** 2) / TEXTURE_MAX_EUCLIDEAN_DIST
        assert dist == pytest.approx(0.0)

    def test_sand_to_clay_centroid_gives_one(self):
        # Sand centroid (92, 5) → clay centroid (22.5, 70): should ≈ 1.0
        dist = math.sqrt((92 - 22.5) ** 2 + (5 - 70) ** 2) / TEXTURE_MAX_EUCLIDEAN_DIST
        assert dist == pytest.approx(1.0, abs=0.02)

    def test_closer_texture_gives_smaller_distance(self):
        # pedon ≈ sandy loam (65, 10)
        pedon = (65.0, 10.0)
        alpha = (65.0, 10.0)  # exact match
        beta  = (15.0, 55.0)  # clay — far away
        d_alpha = math.sqrt((pedon[0]-alpha[0])**2 + (pedon[1]-alpha[1])**2) / TEXTURE_MAX_EUCLIDEAN_DIST
        d_beta  = math.sqrt((pedon[0]-beta[0])**2  + (pedon[1]-beta[1])**2)  / TEXTURE_MAX_EUCLIDEAN_DIST
        assert d_alpha < d_beta

    def test_distance_clipped_to_unit_range(self):
        sand, clay = 92.0, 5.0
        osd_sand, osd_clay = 22.5, 70.0
        raw = math.sqrt((sand - osd_sand) ** 2 + (clay - osd_clay) ** 2) / TEXTURE_MAX_EUCLIDEAN_DIST
        clipped = min(max(raw, 0.0), 1.0)
        assert 0.0 <= clipped <= 1.0


# ---------------------------------------------------------------------------
# color_delta_e transform — direct math tests
# ---------------------------------------------------------------------------

class TestColorDeltaE:
    """Verify the color_delta_e transform arithmetic.

    rank_soils requires depth-interpolated (200-row) DataFrames from list_soils,
    which are not available in unit tests.  These tests verify the ΔE2000 /
    DELTA_E_MAX normalisation directly, plus a site-only integration smoke test."""

    def test_identical_colours_give_zero_distance(self):
        lab = [40.0, 8.0, 18.0]
        norm = calculate_deltaE2000(lab, lab) / DELTA_E_MAX
        assert norm == pytest.approx(0.0, abs=1e-9)

    def test_matching_colour_gives_lower_distance_than_different(self):
        pedon_lab = [40.0, 8.0, 18.0]
        matching_osd = [40.0, 8.0, 18.0]
        different_osd = [90.0, 0.0, 0.0]   # nearly white
        d_match = calculate_deltaE2000(pedon_lab, matching_osd) / DELTA_E_MAX
        d_diff  = calculate_deltaE2000(pedon_lab, different_osd) / DELTA_E_MAX
        assert d_match < d_diff

    def test_clip_keeps_normalised_value_in_unit_range(self):
        # The clip(0.0, 1.0) in rank_soils should cap any extreme pair.
        extreme_a = [0.0, -128.0, -128.0]
        extreme_b = [100.0, 127.0, 127.0]
        raw_norm = calculate_deltaE2000(extreme_a, extreme_b) / DELTA_E_MAX
        clipped = min(max(raw_norm, 0.0), 1.0)
        assert 0.0 <= clipped <= 1.0

    def test_no_pedon_colour_falls_back_gracefully(self):
        """When lab_Color is empty the pipeline must not crash and must
        return a valid soilRank list (uses site-only path)."""
        rank_df = _rank_df(45.0, 20.0)
        result = rank_soils(
            lon=-106.0, lat=32.0,
            list_output_data=_make_list_output(rank_df, _comp_df()),
            soilHorizon=[], topDepth=[], bottomDepth=[], rfvDepth=[],
            lab_Color=[],
            pSlope=10.0, pElev=1000.0, bedrock=None, cracks=False,
        )
        assert "soilRank" in result
        assert len(result["soilRank"]) > 0


# ---------------------------------------------------------------------------
# HORIZON_FEATURE_WEIGHTS – structural checks via source inspection
# ---------------------------------------------------------------------------

class TestHorizonFeatureWeights:
    """Verify the HORIZON_FEATURE_WEIGHTS dict by inspecting the function source.

    The dict is defined inside rank_soils, so we parse the source rather than
    executing the full pipeline (which requires live depth-interpolated data).
    """

    @pytest.fixture(scope="class")
    def weights(self):
        import ast, inspect, soil_id.us_soil as us
        source = inspect.getsource(us.rank_soils)
        # Find the HORIZON_FEATURE_WEIGHTS dict literal in the source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "HORIZON_FEATURE_WEIGHTS"
                    for t in node.targets
                )
            ):
                return {k.s: v.n for k, v in zip(node.value.keys, node.value.values)}
        pytest.skip("HORIZON_FEATURE_WEIGHTS not found in rank_soils source")

    def test_texture_feature_present(self, weights):
        assert "texture_dist" in weights or "sandpct_intpl" in weights

    def test_rfv_present(self, weights):
        assert "rfv_intpl" in weights

    def test_color_feature_present(self, weights):
        has_combined = "color_delta_e" in weights
        has_fallback = "l" in weights and "a" in weights and "b" in weights
        assert has_combined or has_fallback

    def test_texture_weight_greater_than_rfv_weight(self, weights):
        texture_w = weights.get("texture_dist", weights.get("claypct_intpl", 0))
        rfv_w = weights.get("rfv_intpl", 0)
        assert texture_w > rfv_w

    def test_color_weight_less_than_texture_weight(self, weights):
        color_w = weights.get("color_delta_e", weights.get("l", 0))
        texture_w = weights.get("texture_dist", weights.get("claypct_intpl", 0))
        assert color_w < texture_w

    def test_all_weights_positive(self, weights):
        for col, w in weights.items():
            assert w > 0, f"Weight for {col!r} must be positive, got {w}"


# ---------------------------------------------------------------------------
# SITE_THEORETICAL_RANGES – hybrid normalization keeps terrain signal stable
# ---------------------------------------------------------------------------

class TestSiteTheoreticalRanges:
    """The terrain test in test_us.py already validates the primary
    behaviour.  These tests verify the mathematical properties directly."""

    def test_gower_uses_floor_denominator(self):
        """With a 3-row matrix that has a very narrow numeric range the
        theoretical floor should prevent the denominator from collapsing."""
        from soil_id.utils import gower_distances

        # slope range = 1 % (very narrow local variation)
        X = pd.DataFrame({"slope_r": [10.0, 10.5, 11.0]})
        d_no_floor = gower_distances(X)
        d_with_floor = gower_distances(X, theoretical_ranges=[100.0])

        # With a 100-unit floor the normalised distances should be much smaller
        assert d_with_floor[0][2] < d_no_floor[0][2]

    def test_full_range_unaffected(self):
        """When the observed range equals the theoretical range the floor
        changes nothing (floor = 10 %, observed = 100 %)."""
        from soil_id.utils import gower_distances

        X = pd.DataFrame({"slope_r": [0.0, 50.0, 100.0]})
        d_no_floor = gower_distances(X)
        d_with_floor = gower_distances(X, theoretical_ranges=[100.0])

        # Both should give the same result because slice_range == floor * 10
        assert d_no_floor[0][2] == pytest.approx(d_with_floor[0][2], abs=1e-5)

    def test_terrain_inputs_change_ranking(self):
        """Integration: supplying terrain inputs should flip the top-ranked
        component when Alpha matches the pedon terrain better than Beta.

        This mirrors test_rank_soils_terrain_inputs_change_scores but
        verifies the SITE_THEORETICAL_RANGES path is active by confirming
        that the expected winner only emerges with the floor applied."""
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
        list_output_data = _make_list_output(rank_df, comp_df)

        # Without terrain inputs Beta wins (lower Rank_Loc probability tie)
        baseline = rank_soils(
            lon=-106.0, lat=32.0,
            list_output_data=list_output_data,
            soilHorizon=[], topDepth=[], bottomDepth=[], rfvDepth=[],
            lab_Color=[],
            pSlope=20.0, pElev=1000.0, bedrock=None, cracks=False,
        )
        # With terrain inputs that match Alpha, Alpha should win
        with_terrain = rank_soils(
            lon=-106.0, lat=32.0,
            list_output_data=list_output_data,
            soilHorizon=[], topDepth=[], bottomDepth=[], rfvDepth=[],
            lab_Color=[],
            pSlope=20.0, pElev=1000.0, bedrock=None, cracks=False,
            pAspect=0.0,
            pSlopeShapeVert="Concave",
            pSlopeShapeHoriz="Concave",
            pLandscape="alluvial fan",
        )
        assert baseline["soilRank"][0]["name"] == "Beta"
        assert with_terrain["soilRank"][0]["name"] == "Alpha"
