from soil_id.landscape_crosswalk import (
    aim_to_standard_class,
    crosswalk_landscape_class,
    landscape_gowers_distance,
    ssurgo_to_standard_class,
)


def test_aim_to_standard_class_alluvial_fan():
    assert aim_to_standard_class("Alluvial Fan") == "alluvial_fan"


def test_crosswalk_landscape_class_alluvial_fan():
    assert crosswalk_landscape_class("alluvial fan") == "alluvial_fan"


def test_ssurgo_to_standard_class_priority_tread_over_fan():
    assert (
        ssurgo_to_standard_class(geomftname="fan remnant", geompostrce="tread")
        == "terrace_tread"
    )


def test_crosswalk_landscape_class_none():
    assert crosswalk_landscape_class(None) is None


def test_landscape_gowers_distance_exact_match():
    assert landscape_gowers_distance("alluvial_fan", "alluvial_fan") == 0.0


def test_landscape_gowers_distance_partial_related_pair():
    assert landscape_gowers_distance("alluvial_fan", "terrace_tread") == 0.5


def test_landscape_gowers_distance_unrelated_pair():
    assert landscape_gowers_distance("hill_mountain", "playa") == 1.0


def test_landscape_gowers_distance_none():
    assert landscape_gowers_distance(None, "alluvial_fan") is None
