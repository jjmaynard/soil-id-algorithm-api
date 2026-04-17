# Agent Prompt: Integrate BLM Terrain Variables into SoilID Gower's Dissimilarity

## Task Summary

Add five BLM AIM site-measurement variables as inputs to the SoilID FastAPI
endpoint and include them as Gower's dissimilarity variables compared against
the matched SDA SSURGO component attributes returned from the 1000 m buffer
query. These five fields are recorded once per plot and do not differ between
AIM and QC:

| BLM AIM Field | Type | SDA Component Attribute |
|---|---|---|
| `Slope` | Numeric (%) | `component.slope_r / slope_l / slope_h` |
| `Elevation` | Numeric (ft, default) | `component.elev_r / elev_l / elev_h` (metres) |
| `Aspect` | Numeric (0-360°) | `cosurfmorphss.aspectrep` |
| `SlopeShapeVertical` | Categorical | `cosurfmorphss.shapedown` |
| `SlopeShapeHorizontal` | Categorical | `cosurfmorphss.shapeacross` |

A fully working crosswalk and distance module is provided. **Do not rewrite
that logic.** Your job is to wire it into the FastAPI/SoilID codebase and
extend the SDA SQL query to expose the missing SDA columns.

---

## Attachments

| File | Description |
|---|---|
| `terrain_crosswalk.py` | Drop into the SoilID utils package. Contains all crosswalk and Gower's distance functions. |
| `landscape_crosswalk.py` | Previously provided landscape crosswalk (already integrated or in progress). |

---

## Step 1 — Copy the Module

Place `terrain_crosswalk.py` in the same package directory as your Gower's
distance utilities (e.g. `soilid/utils/`).

---

## Step 2 — Extend the SDA Component SQL Query

The existing 1000 m buffer component query already joins `cosurfmorphss`
(aliased `ss`) and selects `ss.shapeacross`, `ss.shapedown`. You only need
to add three more columns to the **SELECT list** of that query. No new JOINs
are required.

**Add to the SELECT list (component table alias is `c`):**

```sql
c.slope_r,
c.slope_l,
c.slope_h,
c.elev_r,
c.elev_l,
c.elev_h,
ss.aspectrep
```

**Also add these to the fallback/empty-result tibble** (when the SDA call
fails) as `NA_real_` to keep the schema consistent:

```r
slope_r = NA_real_,
slope_l = NA_real_,
slope_h = NA_real_,
elev_r  = NA_real_,
elev_l  = NA_real_,
elev_h  = NA_real_,
aspectrep = NA_real_
```

> `shapeacross` and `shapedown` are already present — do not add them again.

---

## Step 3 — Extend the FastAPI Request Model

Add five optional fields. `aspect` should accept `None` for flat sites (coded
as `-1` or `999` in some BLM geodatabases — caller should convert those to
`None` before sending).

```python
from pydantic import BaseModel, Field
from typing import Optional

class SoilIDRequest(BaseModel):
    # ... existing fields ...

    # BLM terrain site variables (same value for AIM and QC)
    slope_pct: Optional[float] = Field(
        default=None,
        ge=0, le=100,
        description="Observed slope at the point in percent."
    )
    elevation: Optional[float] = Field(
        default=None,
        description="Observed elevation. Units controlled by elevation_units."
    )
    elevation_units: Optional[str] = Field(
        default="feet",
        description="Units of elevation field: 'feet' (default) or 'metres'."
    )
    aspect_deg: Optional[float] = Field(
        default=None,
        ge=0, lt=360,
        description="Slope aspect in degrees (0-360, clockwise from N). "
                    "Omit for flat sites."
    )
    slope_shape_vertical: Optional[str] = Field(
        default=None,
        description="Down-slope profile shape: Concave, Convex, or Linear."
    )
    slope_shape_horizontal: Optional[str] = Field(
        default=None,
        description="Across-slope planform shape: Concave, Convex, or Linear."
    )
```

---

## Step 4 — Call `compute_terrain_gowers()` in the Scoring Function

After the 1000 m buffer SDA query returns your component rows, pass them all
to the batch function before the per-candidate Gower's loop:

```python
from terrain_crosswalk import compute_terrain_gowers

# candidates: list of dicts from SDA (cokey + slope/elev/aspect/shape fields)
terrain_distances = compute_terrain_gowers(
    obs_slope_pct=request.slope_pct,
    obs_elev=request.elevation,
    obs_aspect_deg=request.aspect_deg,
    obs_shape_vert=request.slope_shape_vertical,
    obs_shape_horiz=request.slope_shape_horizontal,
    candidates=candidates,               # list[dict] from SDA buffer query
    elevation_units=request.elevation_units or "feet",
)

# Index by cokey for O(1) lookup in the per-candidate loop
terrain_by_cokey = {row["cokey"]: row for row in terrain_distances}
```

---

## Step 5 — Add Terrain Variables to the Gower's Weighted Sum

Inside the per-candidate scoring loop, retrieve the pre-computed distances and
fold them into the weighted sum. Standard Gower's: variables with `None`
distance are excluded from both numerator **and** denominator.

```python
# Recommended starting weights — validated against ~524 BLM AIM/QC plots
TERRAIN_WEIGHTS = {
    "slope":       0.10,
    "elevation":   0.10,
    "aspect":      0.05,   # noisy in field data; keep low
    "shape_vert":  0.05,
    "shape_horiz": 0.05,
}

def gowers_distance(obs: dict, candidate: dict, request: SoilIDRequest) -> float:
    weighted_sum = 0.0
    weight_total = 0.0

    # --- existing variables (texture, color, depth, etc.) ---
    # ... your current logic, unchanged ...

    # --- NEW: terrain variables ---
    t = terrain_by_cokey.get(candidate["cokey"], {})

    terrain_var_map = {
        "slope":       t.get("d_slope"),
        "elevation":   t.get("d_elevation"),
        "aspect":      t.get("d_aspect"),
        "shape_vert":  t.get("d_shape_vert"),
        "shape_horiz": t.get("d_shape_horiz"),
    }

    for var_name, d in terrain_var_map.items():
        if d is not None:
            w = TERRAIN_WEIGHTS[var_name]
            weighted_sum += w * d
            weight_total += w

    # --- normalise ---
    if weight_total == 0:
        return 1.0
    return weighted_sum / weight_total
```

> **Weight guidance**: The five terrain variables combined carry a total
> possible weight of 0.35. Start conservatively (total = 0.35) and reduce
> `slope` and `elevation` if they over-shadow texture/color signals in
> validation. `aspect` is intentionally low (0.05) because BLM measured
> aspect is single-point and SDA aspect is a component-level generalization.

---

## Step 6 — Optionally Return Diagnostic Metadata

Include terrain matching details per candidate in the response:

```python
candidate_response["terrain_match"] = {
    "d_slope":              t.get("d_slope"),
    "d_elevation":          t.get("d_elevation"),
    "d_aspect":             t.get("d_aspect"),
    "d_shape_vert":         t.get("d_shape_vert"),
    "d_shape_horiz":        t.get("d_shape_horiz"),
    "obs_shape_vert_class": t.get("obs_shape_vert_class"),
    "obs_shape_horiz_class":t.get("obs_shape_horiz_class"),
    "sda_shape_vert_class": t.get("sda_shape_vert_class"),
    "sda_shape_horiz_class":t.get("sda_shape_horiz_class"),
    "elev_norm_range_m":    t.get("elev_norm_range_m_used"),
}
```

---

## Slope Shape Canonical Classes

Both BLM AIM text and SDA `shapedown`/`shapeacross` text are mapped to:

| Canonical class | Matches |
|---|---|
| `concave` | Concave, concave |
| `convex` | Convex, convex |
| `linear` | Linear, Planar, Flat, Straight, planar |
| `undulating` | Undulating, Undulate, wavy, irregular, rolling |
| `other` | Any other non-null text |
| `None` | Blank / NA / null — **excluded from Gower's sum** |

---

## Key Technical Notes

### Elevation unit mismatch
BLM AIM `Elevation` is typically recorded in **feet**. SDA `elev_r` /
`elev_l` / `elev_h` are stored in **metres**. The `elevation_gowers_distance()`
function converts observed feet to metres automatically when
`elevation_units="feet"` (the default). Never pass raw BLM feet directly
against SDA metres.

### Aspect on flat sites
BLM AIM codes aspect as `-1` or `999` on flat terrain. Convert these to
`None` in the API request (or in validation pre-processing) so the variable
is excluded from Gower's rather than producing a meaningless angular distance.

### Elevation normalization is dynamic
`compute_terrain_gowers()` auto-computes the elevation normalization range
from the spread of `elev_r` values across all buffer candidates. This means
points in high-relief terrain naturally get a larger normalization range
(smaller distances for the same absolute difference). The minimum floor is
100 m; the fallback for single-candidate cases is 2 500 m.

### Slope range-awareness
If `slope_l` and `slope_h` are available, a candidate receives distance = 0
when the observed slope falls anywhere within its component slope range. This
is intentional: SSURGO slope ranges are deliberately conservative and
represent valid characterization of the mapping unit.

---

## SDA SQL — Summary of Additions

Find the buffer component SELECT list (the query that returns candidates from
the 1 000 m buffer) and add exactly these columns. No new joins needed.

```sql
-- ADD to existing SELECT (component alias c, cosurfmorphss alias ss):
c.slope_r,
c.slope_l,
c.slope_h,
c.elev_r,
c.elev_l,
c.elev_h,
ss.aspectrep
-- shapeacross and shapedown are already selected — do not duplicate
```

The Python dict key names that `compute_terrain_gowers()` reads from each
candidate dict are:

```
slope_r, slope_l, slope_h, elev_r, elev_l, elev_h,
aspectrep, shapedown, shapeacross
```

Make sure the SDA query result rows use these exact key names before passing
to `compute_terrain_gowers()`.

---

## Testing Checklist

- [ ] `crosswalk_slope_shape("Planar")` returns `"linear"`
- [ ] `crosswalk_slope_shape("Undulate")` returns `"undulating"`
- [ ] `crosswalk_slope_shape(None)` returns `None`
- [ ] `slope_gowers_distance(12.0, 15.0, 5.0, 20.0)` returns `0.0` (within range)
- [ ] `slope_gowers_distance(30.0, 15.0, 5.0, 20.0)` returns `0.10`
- [ ] `elevation_gowers_distance(5249.0, 1600.5)` ≈ `0.0006` (5249 ft ≈ 1600 m)
- [ ] `elevation_gowers_distance(None, 1600.0)` returns `None`
- [ ] `aspect_gowers_distance(10.0, 350.0)` ≈ `0.111` (20° arc)
- [ ] `aspect_gowers_distance(0.0, 180.0)` returns `1.0`
- [ ] `slope_shape_gowers_distance("Linear", "Planar")` returns `0.0`
- [ ] `slope_shape_gowers_distance("Convex", "Concave")` returns `1.0`
- [ ] `compute_terrain_gowers(...)` with all `None` obs returns all `None` distances
- [ ] API endpoint accepts all terrain fields as `null` and scores gracefully
- [ ] No regressions in existing Gower's variables when terrain fields are omitted

---

## Files to Create / Modify

| Action | File |
|---|---|
| **Add** | `soilid/utils/terrain_crosswalk.py` (from attachment) |
| **Modify** | Request model — add 6 terrain fields |
| **Modify** | SDA buffer component SQL — add 7 columns to SELECT (slope_r/l/h, elev_r/l/h, aspectrep) |
| **Modify** | Fallback empty-result schema (add 7 `NA_real_` / `None` entries) |
| **Modify** | Gower's scoring function — add terrain variable block |
| **Modify** | Response schema — optionally add `terrain_match` metadata per candidate |
| **Add** | Unit tests for `terrain_crosswalk.py` |

---

## Do NOT Change

- The rule order or regex patterns in `terrain_crosswalk.py`.
- The `FEET_TO_METRES` constant (0.3048 — US survey foot).
- Any existing Gower's variables, weights, or normalization logic.
- The `"other"` catch-all in slope shape — two `"other"` values are treated as
  a match (distance = 0); document this if it surfaces in QA.
