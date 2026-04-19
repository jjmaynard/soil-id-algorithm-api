"""
landscape_crosswalk.py
======================
Converts BLM AIM landscape position observations and SSURGO geomorphic field
values to a shared set of 12 standard landscape classes, and computes a
partial Gower's distance for use in the SoilID dissimilarity algorithm.

Twelve standard landscape classes (aligned to BLM AIM v2 field form)
----------------------------------------------------------------------
  hill_mountain        – hills, mountains, ridges, uplands (parent landform)
  summit_interfluve    – summit positions; interfluves, mountaintops
  shoulder_backslope   – shoulder and backslope hillslope positions
  alluvial_fan         – alluvial fans, fan remnants, aprons
  terrace              – stream terraces, benches (position unspecified)
  terrace_tread        – tread surface of a terrace or fan
  terrace_riser        – riser face of a terrace or fan
  floodplain_basin     – flood plains, drainageways, basins, valleys, lake plains
  flat_plain           – plains, flats, tablelands, playas (non-lacustrine)
  playa                – playas, salt flats, dry lake beds (geochemically distinct)
  dunes_sands          – dunes, sand sheets, aeolian sands
  rocklands            – rock outcrops, badlands, cliffs, talus slopes
  other                – catch-all for unmatched inputs

BLM AIM structured input
-------------------------
Use ``aim_to_standard_class()`` with the exact checkbox label from the field
form (e.g. "Alluvial Fan", "Tread", "Floodplain/Basin").  This path uses a
static lookup dict — no regex.

SSURGO free-text input
-----------------------
Use ``ssurgo_to_standard_class()`` with individual SDA geomorphic fields from
cogeomordesc / cosurfmorphgc / cosurfmorphss::

    geomftname   – landform type name  (hill, mountain, terrace, fan remnant …)
    geomfname    – specific landform name
    geomfmod     – landform modifier
    geomposmntn  – geomorphic position on mountains  (summit, shoulder, backslope …)
    geomposhill  – geomorphic position on hills
    geompostrce  – geomorphic position on terraces   (tread, riser)
    geomposflats – geomorphic position on flats

Field priority order: geomfname → geomftname → geompos* fields.  This avoids
hillslope-position terms (e.g. "backslope") swamping a landform signal.

Partial Gower's distance (Option A — explicit pairwise matrix)
--------------------------------------------------------------
``landscape_gowers_distance()`` returns values in {0.0, 0.25, 0.5, 0.75, 1.0}
encoding geomorphic similarity rather than a binary match / mismatch.

Distance levels
  0.00  exact match (same standard class)
  0.25  closely related  – same landform, position unspecified vs. specified
  0.50  moderately related – adjacent in landscape sequence, similar genesis
  0.75  loosely related  – same broad domain (upland / lowland) but distinct
  1.00  unrelated        – maximally different geomorphic context

Usage
-----
  from landscape_crosswalk import (
      aim_to_standard_class,
      ssurgo_to_standard_class,
      landscape_gowers_distance,
      landscape_class_info,
  )

  # BLM AIM structured input
  obs = aim_to_standard_class("Alluvial Fan")          # -> "alluvial_fan"

  # SSURGO field-priority input
  sda = ssurgo_to_standard_class(
      geomftname="fan remnant",
      geompostrce="tread",
  )                                                     # -> "terrace_tread"

  # Partial Gower's distance
  dist = landscape_gowers_distance(obs, sda)            # -> 0.25

  # Full diagnostic
  info = landscape_class_info("Alluvial Fan", sda)
"""

from __future__ import annotations

import re
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

LANDSCAPE_CLASSES: list[str] = [
    "hill_mountain",
    "summit_interfluve",
    "shoulder_backslope",
    "alluvial_fan",
    "terrace",
    "terrace_tread",
    "terrace_riser",
    "floodplain_basin",
    "flat_plain",
    "playa",
    "dunes_sands",
    "rocklands",
    "other",
]

LandscapeMode = Literal["base", "strict", "loose"]


# ---------------------------------------------------------------------------
# BLM AIM static lookup  (structured input — no regex)
# ---------------------------------------------------------------------------

# Keys are the exact checkbox labels from the BLM AIM v2 field form.
# Values are standard landscape class names.
AIM_TO_STANDARD: dict[str, str] = {
    "Hill/Mountain":    "hill_mountain",
    "Summit":           "summit_interfluve",
    "Shoulder":         "shoulder_backslope",
    "Backslope":        "shoulder_backslope",
    "Alluvial Fan":     "alluvial_fan",
    "Terrace":          "terrace",
    "Tread":            "terrace_tread",
    "Riser":            "terrace_riser",
    "Floodplain/Basin": "floodplain_basin",
    "Flat/Plain":       "flat_plain",
    "Playa":            "playa",
    "Dunes":            "dunes_sands",
    "Other":            "other",
    # --- AIM field-data CSV aliases (underscore / camelCase / abbreviated variants) ---
    "Floodplain":       "floodplain_basin",   # CSV omits "/Basin"
    "Hillslope":        "hill_mountain",       # generic hillslope context
    "Fan_remnant":      "alluvial_fan",
    "Fan remnant":      "alluvial_fan",
    "BasinFloor":       "floodplain_basin",
    "Basin Floor":      "floodplain_basin",
    "Basin_Floor":      "floodplain_basin",
}


def aim_to_standard_class(aim_label: Optional[str]) -> Optional[str]:
    """Map a BLM AIM checkbox label to a standard landscape class.

    Uses a static lookup — no regex.  Case-insensitive match after stripping
    whitespace.

    Parameters
    ----------
    aim_label:
        Exact checkbox label from the BLM AIM v2 field form, e.g.
        ``"Alluvial Fan"``, ``"Tread"``, ``"Floodplain/Basin"``.

    Returns
    -------
    str | None
        Standard class name, or ``None`` for missing / unrecognised input.

    Examples
    --------
    >>> aim_to_standard_class("Alluvial Fan")
    'alluvial_fan'
    >>> aim_to_standard_class("Tread")
    'terrace_tread'
    >>> aim_to_standard_class("Floodplain/Basin")
    'floodplain_basin'
    >>> aim_to_standard_class(None)
    None
    """
    if aim_label is None:
        return None
    key = aim_label.strip()

    # direct match first
    if key in AIM_TO_STANDARD:
        return AIM_TO_STANDARD[key]

    def _canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    key_canon = _canon(key)
    for k, v in AIM_TO_STANDARD.items():
        if _canon(k) == key_canon:
            return v
    return "other"


# ---------------------------------------------------------------------------
# SSURGO field-priority rules  (free-text input)
# ---------------------------------------------------------------------------

# Rules are (standard_class, compiled_pattern) pairs evaluated top-to-bottom;
# first match wins.  Separate rule sets are compiled for each sensitivity mode.

_SSURGO_RULES: dict[str, list[tuple[str, re.Pattern]]] = {}

# Generic geomftname category labels that carry no landform signal; skip them
# so the more diagnostic geomfname / geompos* fields are not overshadowed.
_GEOMFTNAME_SKIP: frozenset[str] = frozenset(["landform", "landscape"])


def _compile_ssurgo_rules(mode: str) -> list[tuple[str, re.Pattern]]:
    """Return compiled (class, pattern) pairs for SSURGO text matching."""
    if mode == "strict":
        raw = [
            # Tread / riser must come before generic terrace/fan
            ("terrace_tread",     r"\btread\b"),
            ("terrace_riser",     r"\briser\b"),
            ("playa",             r"\bplayas?\b"),
            ("alluvial_fan",      r"\balluvial fans?\b|\bfan remnants?\b|\balluvial apron\b"
                                  r"|\bfan aprons?\b|\binset fans?\b|\bfan piedmonts?\b"
                                  r"|\bfan skirts?\b|\bpiedmonts?\b|\bpediments?\b"),
            ("terrace",           r"\bstream terraces?\b|\bterraces?\b|\bbenches?\b"
                                  r"|\bbeach terraces?\b|\blake terraces?\b"),
            ("floodplain_basin",  r"\bflood plains?\b|\bfloodplains?\b|\bdrainageways?\b"
                                  r"|\bbasins?\b|\blake plains?\b|\bbolsons?\b"
                                  r"|\blagoons?\b|\bintermontane\b"),
            ("flat_plain",        r"\bplains?\b|\bflats?\b|\btablelands?\b|\bvalley floor\b"
                                  r"|\bplateaus?\b"),
            ("summit_interfluve", r"\bsummit\b|\binterfluve\b|\bmountaintop\b"),
            ("shoulder_backslope",r"\bshoulder\b|\bbackslope\b|\bside slope\b|\bhead slope\b|\bnose slope\b"),
            ("hill_mountain",     r"\bhills?\b|\bhillsides?\b|\bmountains?\b|\bmountainflank\b"
                                  r"|\bmountainsides?\b|\bridges?\b|\bescarpments?\b|\bupland\b"
                                  r"|\bballenas?\b"),
            ("dunes_sands",       r"\bdunes?\b|\bsand sheets?\b|\baeolian\b"),
            ("rocklands",         r"\brock outcrops?\b|\bbadlands?\b|\bcliffs?\b|\btalus\b"),
        ]
    elif mode == "loose":
        raw = [
            ("terrace_tread",     r"tread"),
            ("terrace_riser",     r"riser"),
            ("playa",             r"playa|salt flat|dry lake"),
            ("alluvial_fan",      r"alluvial|fan|apron|piedmont|pediment"),
            ("terrace",           r"terrace|bench|tableland"),
            ("floodplain_basin",  r"flood|floodplain|drainageway|basin|valley|lake plain|swale|wash|draw|arroyo|channel|bottom|bolson|lagoon|intermontane|river valley|barrier beach|longshore bar"),
            ("flat_plain",        r"plain|flat|level|plateau|mesa"),
            ("summit_interfluve", r"summit|interfluve|crest|mountaintop"),
            ("shoulder_backslope",r"shoulder|backslope|sideslope|side slope|head slope|nose slope|footslope|toeslope"),
            ("hill_mountain",     r"hills?|hillside|mountainside|mountains?|mountainflank|ridges?|escarpment|upland|slope|ballena"),
            ("dunes_sands",       r"dune|sand|aeolian"),
            ("rocklands",         r"rock|outcrop|badland|cliff|talus|ledge"),
        ]
    else:  # base (default)
        raw = [
            # Position-specific terms evaluated before generic landform terms
            ("terrace_tread",     r"\btread\b"),
            ("terrace_riser",     r"\briser\b"),
            ("playa",             r"\bplayas?\b|\bsalt flat\b|\bdry lake\b"),
            ("alluvial_fan",      r"\balluvial fans?\b|\bfan remnants?\b|\balluvial apron\b"
                                  r"|\bfan aprons?\b|\binset fans?\b|\bfan piedmonts?\b"
                                  r"|\bfan skirts?\b|\bpiedmonts?\b|\bpediments?\b"),
            ("terrace",           r"\bstream terraces?\b|\bterraces?\b|\bbenches?\b"
                                  r"|\bbeach terraces?\b|\blake terraces?\b"),
            ("floodplain_basin",  r"\bflood plains?\b|\bfloodplains?\b|\bdrainageways?\b"
                                  r"|\bbasins?\b|\blake plains?\b|\bswales?\b|\bdraws?\b"
                                  r"|\barroyo\b|\bwash\b|\bbottoms?\b|\bbolsons?\b"
                                  r"|\blagoons?\b|\bintermontane\b|\briver valleys\b"
                                  r"|\bbarrier beaches?\b|\blongshore bars?\b"),
            ("flat_plain",        r"\bplains?\b|\bflats?\b|\btablelands?\b|\bvalley floor\b"
                                  r"|\bplateaus?\b|\bmesas?\b"),
            ("summit_interfluve", r"\bsummit\b|\binterfluve\b|\bmountaintop\b|\bcrest\b"),
            ("shoulder_backslope",r"\bshoulder\b|\bbackslope\b|\bside slope\b|\bhead slope\b|\bnose slope\b"),
            ("hill_mountain",     r"\bhills?\b|\bhillsides?\b|\bmountains?\b|\bmountainflank\b"
                                  r"|\bmountainsides?\b|\bridges?\b|\bescarpments?\b|\bupland\b"
                                  r"|\bballenas?\b"),
            ("dunes_sands",       r"\bdunes?\b|\bsand sheets?\b|\baeolian\b"),
            ("rocklands",         r"\brock outcrops?\b|\bbadlands?\b|\bcliffs?\b|\btalus\b"),
        ]
    return [(cls, re.compile(pat, re.IGNORECASE)) for cls, pat in raw]


for _mode in ("base", "strict", "loose"):
    _SSURGO_RULES[_mode] = _compile_ssurgo_rules(_mode)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_txt(x: Optional[str]) -> Optional[str]:
    """Lowercase, strip, collapse internal whitespace; return None for blanks.

    Mirrors normalize_txt() in the companion R script.
    """
    if x is None:
        return None
    s = re.sub(r"\s+", " ", str(x).strip().lower())
    if s in ("", "na", "null", "none", "nan"):
        return None
    return s


def _match_rules(
    text: str,
    mode: LandscapeMode,
) -> Optional[str]:
    """Apply compiled SSURGO rule set and return first matching class."""
    for class_name, pattern in _SSURGO_RULES[mode]:
        if pattern.search(text):
            return class_name
    return None


# ---------------------------------------------------------------------------
# SSURGO field-priority classifier
# ---------------------------------------------------------------------------

def ssurgo_to_standard_class(
    geomftname: Optional[str] = None,
    geomfname: Optional[str] = None,
    geomfmod: Optional[str] = None,
    geomposmntn: Optional[str] = None,
    geomposhill: Optional[str] = None,
    geompostrce: Optional[str] = None,
    geomposflats: Optional[str] = None,
    shapeacross: Optional[str] = None,
    shapedown: Optional[str] = None,
    mode: LandscapeMode = "base",
) -> Optional[str]:
    """Map SSURGO geomorphic fields to a standard landscape class.

    Uses a field-priority approach rather than concatenating all fields into a
    single string.  Priority order:

    1. ``geomfname``  – specific landform name (most diagnostic)
    2. ``geomftname`` – landform type name
    3. ``geomfmod``   – landform modifier
    4. ``geompostrce`` – terrace/fan position (tread / riser) — checked before
       hill/mountain position fields to avoid suppressing tread/riser signals
    5. ``geomposmntn``, ``geomposhill``, ``geomposflats`` – hillslope/flat position
    6. Full concatenation of all non-null fields as fallback

    Parameters
    ----------
    geomftname:
        Landform type name (e.g. "hill", "mountain", "fan remnant", "terrace").
    geomfname:
        Specific landform name (e.g. "alluvial fan", "flood plain").
    geomfmod:
        Landform modifier (e.g. "dissected", "undulating").
    geomposmntn:
        Geomorphic position on mountains (e.g. "summit", "shoulder", "backslope").
    geomposhill:
        Geomorphic position on hills.
    geompostrce:
        Geomorphic position on terraces (e.g. "tread", "riser").
    geomposflats:
        Geomorphic position on flats.
    shapeacross:
        Shape of surface across the slope (linear, convex, concave).
    shapedown:
        Shape of surface down the slope.
    mode:
        Crosswalk sensitivity – ``"base"`` (default), ``"strict"``, or ``"loose"``.

    Returns
    -------
    str | None
        Standard landscape class name, or ``None`` if all fields are blank.

    Examples
    --------
    >>> ssurgo_to_standard_class(geomftname="fan remnant", geompostrce="tread")
    'terrace_tread'
    >>> ssurgo_to_standard_class(geomfname="alluvial fan")
    'alluvial_fan'
    >>> ssurgo_to_standard_class(geomftname="hill", geomposhill="backslope")
    'shoulder_backslope'
    >>> ssurgo_to_standard_class(geomftname="hill", geomposhill="summit")
    'summit_interfluve'
    """
    # Build field list in priority order.
    #
    # Design rationale:
    #   geomfname    – specific landform name; most diagnostic when present
    #                  (e.g. "alluvial fan", "flood plain")
    #   geompostrce  – tread / riser position on a terrace or fan; must come
    #                  BEFORE geomftname so "fan remnant + tread" resolves to
    #                  terrace_tread rather than alluvial_fan
    #   geomposmntn / geomposhill – hillslope position (summit, backslope …);
    #                  also before geomftname for the same reason
    #   geomftname   – landform type; evaluated AFTER position fields
    #   geomfmod     – modifier (dissected, undulating …); least diagnostic
    #   geomposflats – flat position; typically broad / low-priority
    # geomftname values "Landform" and "Landscape" are generic SDA category
    # labels with no landform signal; skip them so the more diagnostic
    # geomfname / geompos* fields are not obscured.
    geomftname_filtered = (
        None
        if normalize_txt(geomftname) in _GEOMFTNAME_SKIP
        else geomftname
    )

    priority_fields = [
        geomfname,          # 1 – specific landform name (highest priority)
        geompostrce,        # 2 – tread / riser (position on terrace/fan)
        geomposmntn,        # 3 – position on mountain
        geomposhill,        # 3 – position on hill
        geomftname_filtered,# 4 – landform type name (generic labels filtered)
        geomfmod,           # 5 – landform modifier
        geomposflats,       # 6 – position on flats (least specific)
    ]

    # Try each field individually in priority order
    for field in priority_fields:
        norm = normalize_txt(field)
        if norm is not None:
            result = _match_rules(norm, mode)
            if result is not None:
                return result

    # Fallback: concatenate all non-null fields (captures combinations)
    all_tokens = [
        geomftname, geomfname, geomfmod,
        geomposmntn, geomposhill, geompostrce, geomposflats,
        shapeacross, shapedown,
    ]
    combined = " ".join(t for t in (normalize_txt(f) for f in all_tokens) if t)
    if not combined:
        return None
    result = _match_rules(combined, mode)
    return result if result is not None else "other"


# ---------------------------------------------------------------------------
# Free-text crosswalk  (legacy / convenience wrapper)
# ---------------------------------------------------------------------------

def crosswalk_landscape_class(
    x: Optional[str],
    mode: LandscapeMode = "base",
) -> Optional[str]:
    """Map a free-text landscape description to a standard class.

    Convenience wrapper around the SSURGO rule engine for callers that have a
    single pre-assembled string rather than individual SDA fields.  Prefer
    ``ssurgo_to_standard_class()`` when individual fields are available.

    Parameters
    ----------
    x:
        Raw landscape description string.
    mode:
        Crosswalk sensitivity – ``"base"`` (default), ``"strict"``, or ``"loose"``.

    Returns
    -------
    str | None
        Standard class name, or ``None`` for missing / blank input.

    Examples
    --------
    >>> crosswalk_landscape_class("alluvial fan")
    'alluvial_fan'
    >>> crosswalk_landscape_class("fan remnant tread")
    'terrace_tread'
    >>> crosswalk_landscape_class("Mountain Ridge", mode="strict")
    'hill_mountain'
    >>> crosswalk_landscape_class(None)
    None
    """
    lx = normalize_txt(x)
    if lx is None:
        return None
    result = _match_rules(lx, mode)
    return result if result is not None else "other"


# ---------------------------------------------------------------------------
# Partial Gower's distance matrix  (Option A — explicit pairwise)
# ---------------------------------------------------------------------------
#
# Distance levels
#   0.00  exact match
#   0.25  closely related  – same landform, position unspecified vs. specified
#   0.50  moderately related – adjacent landscape position, similar genesis
#   0.75  loosely related  – same broad domain but distinct character
#   1.00  unrelated
#
# The matrix is stored as a dict of frozenset pairs for symmetric lookup.
# Only non-1.0 off-diagonal entries are stored; all others default to 1.0.

_PARTIAL_DIST: dict[frozenset, float] = {

    # ── Hillslope position family ──────────────────────────────────────────
    # summit / shoulder / backslope are positions within hill_mountain
    frozenset({"hill_mountain",       "summit_interfluve"}):   0.25,
    frozenset({"hill_mountain",       "shoulder_backslope"}):  0.25,
    frozenset({"summit_interfluve",   "shoulder_backslope"}):  0.50,

    # ── Terrace / fan family ───────────────────────────────────────────────
    # tread and riser are positions within terrace; terrace is the unspecified parent
    frozenset({"terrace",             "terrace_tread"}):        0.25,
    frozenset({"terrace",             "terrace_riser"}):        0.25,
    frozenset({"terrace_tread",       "terrace_riser"}):        0.50,
    # alluvial fan shares terrace/fan depositional context
    frozenset({"alluvial_fan",        "terrace"}):              0.50,
    frozenset({"alluvial_fan",        "terrace_tread"}):        0.50,
    frozenset({"alluvial_fan",        "terrace_riser"}):        0.75,

    # ── Lowland / depositional family ─────────────────────────────────────
    frozenset({"floodplain_basin",    "flat_plain"}):           0.50,
    frozenset({"floodplain_basin",    "playa"}):                0.50,
    frozenset({"floodplain_basin",    "terrace"}):              0.50,
    frozenset({"floodplain_basin",    "terrace_tread"}):        0.50,
    frozenset({"floodplain_basin",    "alluvial_fan"}):         0.50,
    frozenset({"flat_plain",          "playa"}):                0.50,
    frozenset({"flat_plain",          "terrace"}):              0.75,
    frozenset({"flat_plain",          "terrace_tread"}):        0.75,

    # ── Cross-domain transitions ───────────────────────────────────────────
    # Backslope/shoulder transition to fan/terrace (piedmont zone)
    frozenset({"shoulder_backslope",  "alluvial_fan"}):         0.75,
    frozenset({"shoulder_backslope",  "terrace_riser"}):        0.75,
    # Hill/mountain to shoulder (same landform, different abstraction level)
    # already captured above; riser shares sloping character with backslope
    frozenset({"terrace_riser",       "shoulder_backslope"}):   0.75,
    frozenset({"terrace_riser",       "hill_mountain"}):        0.75,

    # ── Aeolian ────────────────────────────────────────────────────────────
    # Dunes commonly co-occur with flat plains but are genetically distinct
    frozenset({"dunes_sands",         "flat_plain"}):           0.75,
    frozenset({"dunes_sands",         "floodplain_basin"}):     0.75,

    # ── Rocklands ─────────────────────────────────────────────────────────
    # Rocklands share high-relief context with hill/mountain
    frozenset({"rocklands",           "hill_mountain"}):        0.75,
    frozenset({"rocklands",           "shoulder_backslope"}):   0.75,

    # All other pairs default to 1.0 (handled in lookup function below)
}


def partial_landscape_distance(
    class_a: Optional[str],
    class_b: Optional[str],
) -> Optional[float]:
    """Look up the partial Gower's distance between two standard classes.

    Parameters
    ----------
    class_a, class_b:
        Standard landscape class names (from :data:`LANDSCAPE_CLASSES`).
        Pass ``None`` to receive ``None`` (variable excluded from Gower sum).

    Returns
    -------
    float | None
        Value in {0.0, 0.25, 0.50, 0.75, 1.0}, or ``None`` if either input
        is ``None``.

    Notes
    -----
    * ``"other"`` vs ``"other"`` returns 0.0 (treated as a match). If that is
      undesirable, set ``w_landscape = 0`` in the caller when either class is
      ``"other"``.
    * The matrix is symmetric; argument order does not matter.

    Examples
    --------
    >>> partial_landscape_distance("alluvial_fan", "terrace_tread")
    0.5
    >>> partial_landscape_distance("terrace", "terrace_tread")
    0.25
    >>> partial_landscape_distance("hill_mountain", "playa")
    1.0
    >>> partial_landscape_distance(None, "terrace")
    None
    """
    if class_a is None or class_b is None:
        return None
    if class_a == class_b:
        return 0.0
    key = frozenset({class_a, class_b})
    return _PARTIAL_DIST.get(key, 1.0)


# ---------------------------------------------------------------------------
# Public distance API
# ---------------------------------------------------------------------------

def landscape_gowers_distance(
    observed_class: Optional[str],
    sda_class: Optional[str],
) -> Optional[float]:
    """Partial Gower's distance given two pre-resolved standard class names.

    Accepts standard class names directly (i.e. the output of
    ``aim_to_standard_class()`` or ``ssurgo_to_standard_class()``).

    Parameters
    ----------
    observed_class:
        Standard landscape class for the field observation (AIM or QC).
    sda_class:
        Standard landscape class resolved from SSURGO geomorphic fields.

    Returns
    -------
    float | None
        Value in {0.0, 0.25, 0.50, 0.75, 1.0}, or ``None`` if either input
        is missing (variable excluded from the Gower weighted sum).

    Examples
    --------
    >>> landscape_gowers_distance("alluvial_fan", "terrace_tread")
    0.5
    >>> landscape_gowers_distance("hill_mountain", "hill_mountain")
    0.0
    >>> landscape_gowers_distance(None, "terrace")
    None
    """
    return partial_landscape_distance(observed_class, sda_class)


def landscape_gowers_distance_from_labels(
    aim_label: Optional[str],
    geomftname: Optional[str] = None,
    geomfname: Optional[str] = None,
    geomfmod: Optional[str] = None,
    geomposmntn: Optional[str] = None,
    geomposhill: Optional[str] = None,
    geompostrce: Optional[str] = None,
    geomposflats: Optional[str] = None,
    shapeacross: Optional[str] = None,
    shapedown: Optional[str] = None,
    mode: LandscapeMode = "base",
) -> Optional[float]:
    """End-to-end partial Gower's distance from raw BLM AIM + SSURGO inputs.

    Resolves the BLM AIM label via ``aim_to_standard_class()`` and the SSURGO
    fields via ``ssurgo_to_standard_class()``, then looks up the partial
    distance.

    Parameters
    ----------
    aim_label:
        BLM AIM checkbox label (e.g. ``"Alluvial Fan"``).
    geomftname … shapedown:
        Individual SDA geomorphic fields (see ``ssurgo_to_standard_class()``).
    mode:
        SSURGO text-matching sensitivity – ``"base"`` (default), ``"strict"``,
        or ``"loose"``.

    Returns
    -------
    float | None
        Partial Gower's distance, or ``None`` if either label is unresolvable.

    Examples
    --------
    >>> landscape_gowers_distance_from_labels(
    ...     "Alluvial Fan",
    ...     geomftname="fan remnant", geompostrce="tread",
    ... )
    0.25
    >>> landscape_gowers_distance_from_labels(
    ...     "Floodplain/Basin",
    ...     geomfname="flood plain",
    ... )
    0.0
    """
    obs = aim_to_standard_class(aim_label)
    sda = ssurgo_to_standard_class(
        geomftname=geomftname,
        geomfname=geomfname,
        geomfmod=geomfmod,
        geomposmntn=geomposmntn,
        geomposhill=geomposhill,
        geompostrce=geompostrce,
        geomposflats=geomposflats,
        shapeacross=shapeacross,
        shapedown=shapedown,
        mode=mode,
    )
    return partial_landscape_distance(obs, sda)


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def landscape_class_info(
    aim_label: Optional[str],
    geomftname: Optional[str] = None,
    geomfname: Optional[str] = None,
    geomfmod: Optional[str] = None,
    geomposmntn: Optional[str] = None,
    geomposhill: Optional[str] = None,
    geompostrce: Optional[str] = None,
    geomposflats: Optional[str] = None,
    shapeacross: Optional[str] = None,
    shapedown: Optional[str] = None,
    mode: LandscapeMode = "base",
) -> dict:
    """Return a diagnostic dict with resolved classes and partial distance.

    Useful for logging, API response metadata, and unit testing.

    Returns
    -------
    dict with keys:
        observed_class   str | None  – standard class from BLM AIM label
        sda_class        str | None  – standard class from SSURGO fields
        gowers_distance  float | None – partial Gower's distance
        distance_level   str | None  – human-readable distance label
        mode             str

    Examples
    --------
    >>> landscape_class_info("Alluvial Fan", geomftname="fan remnant", geompostrce="tread")
    {
        'observed_class': 'alluvial_fan',
        'sda_class': 'terrace_tread',
        'gowers_distance': 0.5,
        'distance_level': 'moderately related',
        'mode': 'base'
    }
    """
    obs = aim_to_standard_class(aim_label)
    sda = ssurgo_to_standard_class(
        geomftname=geomftname,
        geomfname=geomfname,
        geomfmod=geomfmod,
        geomposmntn=geomposmntn,
        geomposhill=geomposhill,
        geompostrce=geompostrce,
        geomposflats=geomposflats,
        shapeacross=shapeacross,
        shapedown=shapedown,
        mode=mode,
    )
    dist = partial_landscape_distance(obs, sda)

    _LEVELS = {
        0.0:  "exact match",
        0.25: "closely related",
        0.50: "moderately related",
        0.75: "loosely related",
        1.0:  "unrelated",
    }
    level = _LEVELS.get(dist) if dist is not None else None

    return {
        "observed_class":  obs,
        "sda_class":       sda,
        "gowers_distance": dist,
        "distance_level":  level,
        "mode":            mode,
    }


# ---------------------------------------------------------------------------
# Legacy shim — build_sda_landscape_label()
# ---------------------------------------------------------------------------

def build_sda_landscape_label(*fields: Optional[str]) -> Optional[str]:
    """Assemble a single string from multiple SDA geomorphic fields.

    Retained for backward compatibility.  For new code, pass fields directly
    to ``ssurgo_to_standard_class()`` to benefit from field-priority matching.

    Parameters
    ----------
    *fields:
        Positional field values in the order:
        geomftname, geomfname, geomfmod, geomposmntn, geomposhill,
        geompostrce, geomposflats, shapeacross, shapedown.

    Returns
    -------
    str | None
        Space-joined non-null tokens, or ``None`` if all fields are blank.
    """
    tokens = [normalize_txt(f) for f in fields if normalize_txt(f) is not None]
    return " ".join(tokens) if tokens else None
