import math

import pytest

from soil_id.terrain_crosswalk import (
    crosswalk_slope_shape,
    slope_shape_gowers_distance,
)


def test_crosswalk_slope_shape_planar_to_linear():
    assert crosswalk_slope_shape("Planar") == "linear"


def test_crosswalk_slope_shape_undulate_to_undulating():
    assert crosswalk_slope_shape("Undulate") == "undulating"


def test_crosswalk_slope_shape_none():
    assert crosswalk_slope_shape(None) is None


def test_slope_shape_gowers_distance_linear_planar_match():
    assert slope_shape_gowers_distance("Linear", "Planar") == 0.0


def test_slope_shape_gowers_distance_convex_concave_mismatch():
    assert slope_shape_gowers_distance("Convex", "Concave") == 1.0


def test_aspect_vector_roundtrip_cardinal_directions():
    north = (math.cos(math.radians(0.0)), math.sin(math.radians(0.0)))
    east = (math.cos(math.radians(90.0)), math.sin(math.radians(90.0)))
    south = (math.cos(math.radians(180.0)), math.sin(math.radians(180.0)))
    west = (math.cos(math.radians(270.0)), math.sin(math.radians(270.0)))

    assert north[0] == pytest.approx(1.0)
    assert north[1] == pytest.approx(0.0)
    assert east[0] == pytest.approx(0.0)
    assert east[1] == pytest.approx(1.0)
    assert south[0] == pytest.approx(-1.0)
    assert south[1] == pytest.approx(0.0)
    assert west[0] == pytest.approx(0.0)
    assert west[1] == pytest.approx(-1.0)
