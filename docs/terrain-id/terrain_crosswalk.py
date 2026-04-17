"""
terrain_crosswalk.py
====================
Crosswalk and Gower's distance functions for five BLM AIM terrain/site
variables that are collected once per plot (same value for AIM and QC):

  slope              – % slope at the point
  elevation          – elevation at the point (field units typically feet)
  aspect             – compass direction of slope face (0-360°)
  slopeshapevertical – down-slope profile curvature (concave/convex/linear)
  slopeshapehorizontal – across-slope planform curvature (concave/convex/linear)

Each variable is compared to the equivalent SDA SSURGO component attribute
returned by the 1000 m buffer query:

  slope      <- component.slope_r  (representative, %)
                component.slope_l  (low, %)
                component.slope_h  (high, %)
  elevation  <- component.elev_r   (representative, metres)
                component.elev_l   (low, metres)
                component.elev_h   (high, metres)
  aspect     <- cosurfmorphss.aspectrep  (degrees, 0-360)
  shape vert <- cosurfmorphss.shapedown  (free text)
  shape horiz<- cosurfmorphss.shapeacross (free text)

Slope shape canonical classes (applied to both BLM and SDA text)
-----------------------------------------------------------------
  concave    – Concave, concave
  convex     – Convex, convex
  linear     – Linear, linear, Planar, planar, Flat, Straight
  undulating – Undulating, Undulate, wavy, irregular, rolling
  other      – catch-all for non-null unrecognised text

Gower's distance conventions
-----------------------------
  Numeric variables (slope, elevation, aspect*):
      d = |observed - sda_r| / norm_range                      (simple)
      d = 0  if sda_l <= observed <= sda_h  (range-aware, preferred)
      d = min(|obs-sda_l|, |obs-sda_h|) / norm_range           (out-of-range)
  Categorical variables (slope shape):
      d = 0.0 if same canonical class
      d = 1.0 if different canonical class
  Missing (either side):
      d = None  (variable excluded from Gower's weighted denominator)

  * Aspect uses a circular distance formula; see aspect_gowers_distance().

Usage
-----
  from terrain_crosswalk import (
      crosswalk_slope_shape,
      slope_gowers_distance,
      elevation_gowers_distance,
      aspect_gowers_distance,
      slope_shape_gowers_distance,
      compute_terrain_gowers,
  )

  # Per-candidate convenience (all five variables at once)
  results = compute_terrain_gowers(
      obs_slope_pct=12.0,
      obs_elev=5200.0,       # feet by default
      obs_aspect_deg=225.0,
      obs_shape_vert="Concave",
      obs_shape_horiz="Linear",
      candidates=[
          {
              "cokey": "123",
              "slope_r": 10.0, "slope_l": 5.0, "slope_h": 20.0,
              "elev_r": 1560.0, "elev_l": 1500.0, "elev_h": 1620.0,  # metres
              "aspectrep": 215.0,
              "shapedown": "Concave",
              "shapeacross": "Linear",
          },
          ...
      ],
      elevation_units="feet",   # observed elevation units
  )
"""

from __future__ import annotations

import re
from typing import Optional, Literal

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Canonical slope shape classes; first match in crosswalk wins.
SLOPE_SHAPE_CLASSES: list[str] = ["concave", "convex", "linear", "undulating", "other"]

#: Normalization range for slope (percent).  Covers 0-100 % which encompasses
#: virtually all CONUS agricultural and rangeland slopes.
SLOPE_NORM_RANGE_PCT: float = 100.0

#: Fallback normalization range for elevation (metres) when the candidate set
#: has zero spread.  2 500 m spans most intra-survey-area elevation variation.
ELEVATION_FALLBACK_RANGE_M: float = 2500.0

#: Minimum allowed elevation normalization range (metres) to avoid
#: near-zero-division when all candidates cluster tightly.
ELEVATION_MIN_RANGE_M: float = 100.0

#: Conversion factor: US survey feet → metres.
FEET_TO_METRES: float = 0.3048

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(x) -> Optional[float]:
    """Cast to float; return None on failure or for null-equivalent strings."""
    if x is None:
        return None
    if isinstance(x, float):
        return None if x != x else x   # NaN guard
    try:
        v = float(str(x).strip())
        return None if v != v else v
    except (ValueError, TypeError):
        return None


def _normalize_shape_txt(x: Optional[str]) -> Optional[str]:
    """Lowercase and strip; return None for blank / NA-equivalent strings."""
    if x is None:
        return None
    s = re.sub(r"\s+", " ", str(x).strip().lower())
    if s in ("", "na", "null", "none", "nan", "n/a", "not recorded"):
        return None
    return s


# Compiled shape rules – evaluated in order; first match wins.
_SHAPE_RULES: list[tuple[str, re.Pattern]] = [
    ("concave",    re.compile(r"concav", re.IGNORECASE)),
    ("convex",     re.compile(r"convex", re.IGNORECASE)),
    ("linear",     re.compile(r"linear|planar|flat|straight", re.IGNORECASE)),
    ("undulating", re.compile(r"undulat|wavy|irregular|rolling", re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Slope shape crosswalk
# ---------------------------------------------------------------------------

def crosswalk_slope_shape(x: Optional[str]) -> Optional[str]:
    """Map a free-text slope shape description to a canonical class.

    Handles both BLM AIM recorded values and SDA ``shapedown`` /
    ``shapeacross`` text.

    Parameters
    ----------
    x:
        Raw slope shape string (e.g. ``"Concave"``, ``"Planar"``,
        ``"Undulating"``).  Pass ``None`` or blank to receive ``None``.

    Returns
    -------
    str | None
        One of ``"concave"``, ``"convex"``, ``"linear"``, ``"undulating"``,
        ``"other"``, or ``None`` when the input is missing.

    Examples
    --------
    >>> crosswalk_slope_shape("Concave")
    'concave'
    >>> crosswalk_slope_shape("Planar")
    'linear'
    >>> crosswalk_slope_shape("Undulate")
    'undulating'
    >>> crosswalk_slope_shape(None)
    None
    """
    lx = _normalize_shape_txt(x)
    if lx is None:
        return None
    for class_name, pattern in _SHAPE_RULES:
        if pattern.search(lx):
            return class_name
    return "other"


# ---------------------------------------------------------------------------
# Per-variable Gower's distance functions
# ---------------------------------------------------------------------------

def slope_gowers_distance(
    obs_pct: Optional[float],
    sda_r_pct: Optional[float],
    sda_l_pct: Optional[float] = None,
    sda_h_pct: Optional[float] = None,
    norm_range: float = SLOPE_NORM_RANGE_PCT,
) -> Optional[float]:
    """Gower's numeric distance for slope (percent).

    When both low and high bounds are available the distance is 0 if the
    observed value falls within the component's slope range, otherwise the
    distance to the nearest bound (range-aware).  Falls back to a simple
    representative-value comparison when bounds are absent.

    Parameters
    ----------
    obs_pct:
        Observed slope in percent (BLM AIM ``Slope`` field).
    sda_r_pct:
        SDA ``component.slope_r`` (representative percent slope).
    sda_l_pct:
        SDA ``component.slope_l`` (low bound, optional).
    sda_h_pct:
        SDA ``component.slope_h`` (high bound, optional).
    norm_range:
        Normalization range in percent (default 100).

    Returns
    -------
    float | None
        Gower's distance in [0, 1], or ``None`` when either value is missing.

    Examples
    --------
    >>> slope_gowers_distance(12.0, 15.0)
    0.03
    >>> slope_gowers_distance(12.0, 20.0, sda_l_pct=5.0, sda_h_pct=25.0)
    0.0   # obs within range
    >>> slope_gowers_distance(35.0, 20.0, sda_l_pct=5.0, sda_h_pct=25.0)
    0.1   # 10 pct above high bound
    """
    obs = _safe_float(obs_pct)
    sda_r = _safe_float(sda_r_pct)
    if obs is None or sda_r is None:
        return None

    sda_l = _safe_float(sda_l_pct)
    sda_h = _safe_float(sda_h_pct)

    if sda_l is not None and sda_h is not None and sda_l <= sda_h:
        if sda_l <= obs <= sda_h:
            raw_dist = 0.0
        else:
            raw_dist = min(abs(obs - sda_l), abs(obs - sda_h))
    else:
        raw_dist = abs(obs - sda_r)

    return min(1.0, raw_dist / norm_range)


def elevation_gowers_distance(
    obs_elev: Optional[float],
    sda_r_m: Optional[float],
    sda_l_m: Optional[float] = None,
    sda_h_m: Optional[float] = None,
    elevation_units: Literal["feet", "metres", "meters"] = "feet",
    norm_range_m: Optional[float] = None,
) -> Optional[float]:
    """Gower's numeric distance for elevation.

    Converts the observed elevation from field units to metres before
    comparison (since SDA ``elev_r`` is stored in metres).

    Parameters
    ----------
    obs_elev:
        Observed elevation in *elevation_units*.
    sda_r_m:
        SDA ``component.elev_r`` in **metres**.
    sda_l_m:
        SDA ``component.elev_l`` in **metres** (optional).
    sda_h_m:
        SDA ``component.elev_h`` in **metres** (optional).
    elevation_units:
        Units of *obs_elev* – ``"feet"`` (default, common in BLM AIM
        geodatabases) or ``"metres"`` / ``"meters"``.
    norm_range_m:
        Normalization range in metres.  Provide the spread of ``elev_r``
        values across all candidates in the buffer query (recommended) or
        leave ``None`` to use the :data:`ELEVATION_FALLBACK_RANGE_M` constant.

    Returns
    -------
    float | None
        Gower's distance in [0, 1], or ``None`` when either value is missing.

    Examples
    --------
    >>> elevation_gowers_distance(5200.0, 1580.0)          # 5200 ft ≈ 1585 m
    0.002   # very close
    >>> elevation_gowers_distance(5200.0, 1000.0, norm_range_m=2500.0)
    0.234
    """
    obs = _safe_float(obs_elev)
    sda_r = _safe_float(sda_r_m)
    if obs is None or sda_r is None:
        return None

    # Unit conversion
    if elevation_units in ("metres", "meters"):
        obs_m = obs
    else:
        obs_m = obs * FEET_TO_METRES

    norm = norm_range_m if (norm_range_m is not None and norm_range_m > 0) else ELEVATION_FALLBACK_RANGE_M

    sda_l = _safe_float(sda_l_m)
    sda_h = _safe_float(sda_h_m)

    if sda_l is not None and sda_h is not None and sda_l <= sda_h:
        if sda_l <= obs_m <= sda_h:
            raw_dist = 0.0
        else:
            raw_dist = min(abs(obs_m - sda_l), abs(obs_m - sda_h))
    else:
        raw_dist = abs(obs_m - sda_r)

    return min(1.0, raw_dist / norm)


def aspect_gowers_distance(
    obs_deg: Optional[float],
    sda_deg: Optional[float],
) -> Optional[float]:
    """Gower's circular distance for aspect (compass bearing, 0-360°).

    Uses the minimum arc between two bearings (wraps at 360°) scaled to
    [0, 1] where 180° difference is the maximum (1.0).

    Parameters
    ----------
    obs_deg:
        Observed aspect in degrees (BLM AIM ``Aspect`` field, 0-360).
    sda_deg:
        SDA ``cosurfmorphss.aspectrep`` in degrees.

    Returns
    -------
    float | None
        Circular Gower's distance in [0, 1], or ``None`` when either value
        is missing.

    Notes
    -----
    Flat sites or sites coded as ``-1`` / ``999`` in BLM data (meaning "no
    aspect") should be passed as ``None`` so they are excluded from the
    Gower's sum rather than producing a spurious distance.

    Examples
    --------
    >>> aspect_gowers_distance(10.0, 350.0)   # 20° arc across N
    0.111
    >>> aspect_gowers_distance(0.0, 180.0)    # directly opposite
    1.0
    >>> aspect_gowers_distance(225.0, 225.0)
    0.0
    """
    obs = _safe_float(obs_deg)
    sda = _safe_float(sda_deg)
    if obs is None or sda is None:
        return None

    diff = abs(obs - sda) % 360.0
    arc = min(diff, 360.0 - diff)          # shortest arc, 0-180°
    return round(arc / 180.0, 6)


def slope_shape_gowers_distance(
    obs_label: Optional[str],
    sda_label: Optional[str],
) -> Optional[float]:
    """Gower's categorical distance for slope shape (vertical or horizontal).

    Both labels are mapped through :func:`crosswalk_slope_shape` before
    comparison.  Works identically for ``slopeshapevertical`` vs
    ``shapedown`` and ``slopeshapehorizontal`` vs ``shapeacross``.

    Parameters
    ----------
    obs_label:
        BLM AIM observed value (e.g. ``"Concave"``, ``"Linear"``).
    sda_label:
        SDA value from ``cosurfmorphss.shapedown`` or ``.shapeacross``.

    Returns
    -------
    float | None
        ``0.0`` – same canonical class  |  ``1.0`` – different class
        ``None`` – one or both labels missing

    Examples
    --------
    >>> slope_shape_gowers_distance("Concave", "concave")
    0.0
    >>> slope_shape_gowers_distance("Linear", "Planar")
    0.0   # both → "linear"
    >>> slope_shape_gowers_distance("Convex", "Concave")
    1.0
    >>> slope_shape_gowers_distance(None, "Concave")
    None
    """
    obs_class = crosswalk_slope_shape(obs_label)
    sda_class = crosswalk_slope_shape(sda_label)
    if obs_class is None or sda_class is None:
        return None
    return 0.0 if obs_class == sda_class else 1.0


# ---------------------------------------------------------------------------
# Batch convenience function
# ---------------------------------------------------------------------------

def compute_terrain_gowers(
    obs_slope_pct: Optional[float],
    obs_elev: Optional[float],
    obs_aspect_deg: Optional[float],
    obs_shape_vert: Optional[str],
    obs_shape_horiz: Optional[str],
    candidates: list[dict],
    elevation_units: Literal["feet", "metres", "meters"] = "feet",
    slope_norm_range: float = SLOPE_NORM_RANGE_PCT,
    elev_norm_range_m: Optional[float] = None,
) -> list[dict]:
    """Compute all five terrain Gower's distances for every SDA candidate.

    The elevation normalization range is computed automatically from the
    spread of ``elev_r`` values across *candidates* unless *elev_norm_range_m*
    is provided explicitly.

    Parameters
    ----------
    obs_slope_pct:
        Observed slope in percent (BLM AIM ``Slope``).
    obs_elev:
        Observed elevation in *elevation_units* (BLM AIM ``Elevation``).
    obs_aspect_deg:
        Observed aspect in degrees (BLM AIM ``Aspect``).  Pass ``None`` for
        flat sites or coded no-aspect values (e.g. -1, 999).
    obs_shape_vert:
        BLM AIM ``SlopeShapeVertical`` (down-slope profile).
    obs_shape_horiz:
        BLM AIM ``SlopeShapeHorizontal`` (across-slope planform).
    candidates:
        List of dicts, one per SDA component.  Each must contain at least::

            {
              "cokey":       str,
              "slope_r":     float | None,
              "slope_l":     float | None,   # optional
              "slope_h":     float | None,   # optional
              "elev_r":      float | None,   # metres
              "elev_l":      float | None,   # metres, optional
              "elev_h":      float | None,   # metres, optional
              "aspectrep":   float | None,
              "shapedown":   str   | None,   # vertical
              "shapeacross": str   | None,   # horizontal
            }

    elevation_units:
        Units of *obs_elev* (``"feet"`` by default – BLM AIM field protocol).
    slope_norm_range:
        Normalization range for slope (default 100 %).
    elev_norm_range_m:
        Override the auto-computed elevation normalization range (metres).

    Returns
    -------
    list[dict]
        One dict per candidate (same order as *candidates*), each containing::

            {
              "cokey":                    str,
              "d_slope":                  float | None,
              "d_elevation":              float | None,
              "d_aspect":                 float | None,
              "d_shape_vert":             float | None,
              "d_shape_horiz":            float | None,
              "obs_slope_class":          None,                # not applicable (numeric)
              "obs_shape_vert_class":     str  | None,
              "obs_shape_horiz_class":    str  | None,
              "sda_shape_vert_class":     str  | None,
              "sda_shape_horiz_class":    str  | None,
              "elev_norm_range_m_used":   float,
            }

    Example
    -------
    >>> results = compute_terrain_gowers(
    ...     obs_slope_pct=12.0,
    ...     obs_elev=5200.0,
    ...     obs_aspect_deg=225.0,
    ...     obs_shape_vert="Concave",
    ...     obs_shape_horiz="Linear",
    ...     candidates=[{
    ...         "cokey": "abc",
    ...         "slope_r": 10.0, "slope_l": 5.0, "slope_h": 20.0,
    ...         "elev_r": 1584.96, "elev_l": 1524.0, "elev_h": 1645.0,
    ...         "aspectrep": 215.0,
    ...         "shapedown": "Concave",
    ...         "shapeacross": "Planar",
    ...     }],
    ... )
    >>> results[0]["d_slope"]
    0.0          # 12 % within [5, 20]
    >>> results[0]["d_shape_vert"]
    0.0          # Concave == concave
    >>> results[0]["d_shape_horiz"]
    0.0          # Linear == Planar → both 'linear'
    """
    # --- Auto-compute elevation normalization range from candidates ---
    if elev_norm_range_m is not None:
        norm_elev = max(elev_norm_range_m, ELEVATION_MIN_RANGE_M)
    else:
        sda_elevs = [
            _safe_float(c.get("elev_r"))
            for c in candidates
            if _safe_float(c.get("elev_r")) is not None
        ]
        if len(sda_elevs) >= 2:
            spread = max(sda_elevs) - min(sda_elevs)
            norm_elev = max(spread, ELEVATION_MIN_RANGE_M)
        else:
            norm_elev = ELEVATION_FALLBACK_RANGE_M

    obs_sv_class = crosswalk_slope_shape(obs_shape_vert)
    obs_sh_class = crosswalk_slope_shape(obs_shape_horiz)

    output: list[dict] = []
    for cand in candidates:
        sda_sv_class = crosswalk_slope_shape(cand.get("shapedown"))
        sda_sh_class = crosswalk_slope_shape(cand.get("shapeacross"))

        row = {
            "cokey": cand.get("cokey"),
            "d_slope": slope_gowers_distance(
                obs_pct=obs_slope_pct,
                sda_r_pct=cand.get("slope_r"),
                sda_l_pct=cand.get("slope_l"),
                sda_h_pct=cand.get("slope_h"),
                norm_range=slope_norm_range,
            ),
            "d_elevation": elevation_gowers_distance(
                obs_elev=obs_elev,
                sda_r_m=cand.get("elev_r"),
                sda_l_m=cand.get("elev_l"),
                sda_h_m=cand.get("elev_h"),
                elevation_units=elevation_units,
                norm_range_m=norm_elev,
            ),
            "d_aspect": aspect_gowers_distance(
                obs_deg=obs_aspect_deg,
                sda_deg=cand.get("aspectrep"),
            ),
            "d_shape_vert": None if (obs_sv_class is None or sda_sv_class is None)
                            else (0.0 if obs_sv_class == sda_sv_class else 1.0),
            "d_shape_horiz": None if (obs_sh_class is None or sda_sh_class is None)
                             else (0.0 if obs_sh_class == sda_sh_class else 1.0),
            "obs_shape_vert_class": obs_sv_class,
            "obs_shape_horiz_class": obs_sh_class,
            "sda_shape_vert_class": sda_sv_class,
            "sda_shape_horiz_class": sda_sh_class,
            "elev_norm_range_m_used": norm_elev,
        }
        output.append(row)

    return output
