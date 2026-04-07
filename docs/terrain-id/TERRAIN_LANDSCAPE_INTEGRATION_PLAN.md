# Terrain & Landscape Integration Plan — US SoilID

## Overview

Add seven new site-level variables to the existing `rank_soils` Gower's dissimilarity
scoring pipeline (US path only). Field-collected BLM AIM/QC measurements for
**slope, elevation, aspect, slope shape (vertical & horizontal), and landscape type**
are compared against matched SDA SSURGO component attributes via the existing
`site_mat → gower_distances` framework. No separate scoring path is needed.

**Key design decisions:**
- Aspect is decomposed to **northerness** = cos(θ) and **easterness** = sin(θ),
  both in [−1, 1], so Gower's normalization handles the circular wrap correctly.
- All new variables funnel through the existing `gower_distances` function
  (`categorical_features` param handles shape/landscape classes).
- Terrain data is fetched once in `list_soils` and passed to `rank_soils` via
  the existing `SoilListOutputData` CSV — no structural changes to that dataclass.
- SSURGO path only. STATSGO path gets `None`-filled columns; terrain variables
  are silently excluded from scoring when all values are `None`.

### New Variables Summary

| Observed input | SDA source | Site matrix column(s) | Type |
|---|---|---|---|
| `pSlope` (existing) | `component.slope_r` | `slope_r` (range-adjusted) | Numeric |
| `pElev` (existing) | `component.elev_r` | `elev_r` | Numeric |
| `pAspect` | `cosurfmorphss.aspectrep` | `aspect_northerness`, `aspect_easterness` | Numeric |
| `pSlopeShapeVert` | `cosurfmorphss.shapedown` | `shape_vert_class` | Categorical |
| `pSlopeShapeHoriz` | `cosurfmorphss.shapeacross` | `shape_horiz_class` | Categorical |
| `pLandscape` | assembled from geomorphic fields | `landscape_class` | Categorical |

### Starting Weights

```python
DEFAULT_WEIGHTS = {
    "slope_r":          1.00,   # existing
    "elev_r":           0.50,   # existing
    "bottom_depth":     1.50,   # existing
    "aspect_northerness": 0.25, # new
    "aspect_easterness":  0.25, # new
    "shape_vert_class":   0.25, # new
    "shape_horiz_class":  0.25, # new
    "landscape_class":    0.50, # new
}
# Existing total: 3.0  |  New terrain total: 1.5
```

---

## Task List

### Phase 1 — Add Crosswalk Modules to Package

- [x] **1.1** Copy `docs/terrain-id/landscape_crosswalk.py` → `soil_id/landscape_crosswalk.py`
  *(no logic changes)*
- [x] **1.2** Copy `docs/terrain-id/terrain_crosswalk.py` → `soil_id/terrain_crosswalk.py`
  *(no logic changes)*

---

### Phase 2 — New SDA Query in `list_soils` (SSURGO path only)

- [x] **2.1** After `comp_key` is finalized (post horizon-filtering, ~line 240 in `us_soil.py`),
  add a new `sda_return` call with the following SQL:

  ```sql
  SELECT
      c.cokey,
      c.slope_l, c.slope_h,
      c.elev_l,  c.elev_h,
      ss.aspectrep,
      ss.shapedown,
      ss.shapeacross,
      gm.geomftname, gm.geomfname, gm.geomfmod,
      gc.geomposmntn, gc.geomposhill, gc.geompostrce, gc.geomposflats
  FROM component c
  LEFT JOIN cosurfmorphss ss ON c.cokey = ss.cokey
  LEFT JOIN cogeomordesc gm ON c.cokey = gm.cokey AND gm.rvindicator = 'Yes'
  LEFT JOIN cosurfmorphgc gc ON gm.cogeomdesckey = gc.cogeomdesckey
  WHERE c.cokey IN ({cokey_list})
  ORDER BY c.cokey
  ```

  Return result as `terrain_pd`. If the SDA call fails or returns empty, create a
  `terrain_pd` DataFrame with the same column names, all `None`.

- [x] **2.2** Merge `terrain_pd` into `mucompdata_pd` on `cokey` (left join, keep all
  existing rows).

- [x] **2.3** Crosswalk text columns immediately after merge:
  - `shapedown` → `shape_vert_class` via `crosswalk_slope_shape()`
  - `shapeacross` → `shape_horiz_class` via `crosswalk_slope_shape()`
  - Assemble SDA landscape label via `build_sda_landscape_label(geomftname, geomfname,
    geomfmod, geomposmntn, geomposhill, geompostrce, geomposflats, shapeacross, shapedown)`
    → `landscape_class` via `crosswalk_landscape_class()`
  - `aspectrep` → `aspect_northerness = cos(radians(aspectrep))`,
    `aspect_easterness = sin(radians(aspectrep))` — `None` if `aspectrep` is `None`

- [x] **2.4** Update `process_site_data()` in `soil_id/utils.py` to retain the nine new
  columns in its column keep-list:
  `slope_l`, `slope_h`, `elev_l`, `elev_h`, `aspect_northerness`, `aspect_easterness`,
  `shape_vert_class`, `shape_horiz_class`, `landscape_class`

---

### Phase 3 — Extend `rank_soils` Parameters

- [x] **3.1** Add the following optional parameters to the `rank_soils` function signature
  in `soil_id/us_soil.py`:

  ```python
  pAspect=None,           # degrees 0-360; None or sentinel (-1, 999) for flat sites
  pSlopeShapeVert=None,   # free text e.g. "Concave", "Linear"
  pSlopeShapeHoriz=None,  # free text e.g. "Convex", "Planar"
  pLandscape=None,        # free text e.g. "alluvial fan", "hill slope"
  pLandscapeMode="base",  # crosswalk sensitivity: "base" | "strict" | "loose"
  ```

- [x] **3.2** At the top of `rank_soils` (before the site matrix is built), crosswalk the
  observed inputs:
  - `pAspect`: convert sentinels (`-1`, `999`) to `None`; then
    `obs_northerness = cos(radians(float(pAspect)))`,
    `obs_easterness  = sin(radians(float(pAspect)))` (or `None` if still `None`)
  - `pSlopeShapeVert`  → `obs_shape_vert`  via `crosswalk_slope_shape()`
  - `pSlopeShapeHoriz` → `obs_shape_horiz` via `crosswalk_slope_shape()`
  - `pLandscape`       → `obs_landscape`   via `crosswalk_landscape_class(mode=pLandscapeMode)`

---

### Phase 4 — Extend the Site Matrix in `rank_soils`

- [x] **4.1** Extend the `provided` dict to include all new variables alongside the existing ones:

  ```python
  provided = {
      "slope_r":             pSlope,
      "elev_r":              pElev,
      "bottom_depth":        bedrock,
      "aspect_northerness":  obs_northerness,
      "aspect_easterness":   obs_easterness,
      "shape_vert_class":    obs_shape_vert,
      "shape_horiz_class":   obs_shape_horiz,
      "landscape_class":     obs_landscape,
  }
  ```

- [x] **4.2** Extend `lib_cols` / `lib_df` to pull the new columns from `mucompdata_pd`
  when they are present in `features`.

- [x] **4.3** Identify categorical features for `gower_distances`:

  ```python
  CATEGORICALS = {"shape_vert_class", "shape_horiz_class", "landscape_class"}
  cat_indices = [i for i, f in enumerate(features) if f in CATEGORICALS]
  ```

- [x] **4.4** Extend `DEFAULT_WEIGHTS` dict (see Starting Weights table above).

- [x] **4.5** Pass `categorical_features=cat_indices` to the `gower_distances` call:

  ```python
  D_raw = gower_distances(site_mat, feature_weight=weights, categorical_features=cat_indices)
  ```

---

### Phase 5 — Slope Range-Awareness Pre-Processing

- [x] **5.1** Before building `site_mat`, apply range-aware slope adjustment:
  when the observed slope (`pSlope`) falls within a component's `[slope_l, slope_h]`
  range, temporarily set that component's `slope_r = pSlope` in `lib_df` so that
  Gower's distance for slope collapses to 0 for that candidate.
  *(No changes to `gower_distances` needed — pure pre-processing.)*

---

### Phase 6 — STATSGO Fallback

- [x] **6.1** In the STATSGO code path of `list_soils`, after `mucompdata_pd` is assembled,
  add all nine new columns with `None` values so the schema is consistent with the
  SSURGO path. Terrain variables will be absent from `features` in `rank_soils` because
  all observed and SDA values are `None`.

---

### Phase 7 — Tests

- [x] **7.1** Create `soil_id/tests/test_landscape_crosswalk.py` with unit tests covering:
  - `crosswalk_landscape_class("alluvial fan")` → `"fans"`
  - `crosswalk_landscape_class("hill slope")` → `"hills_mountains"`
  - `crosswalk_landscape_class(None)` → `None`
  - `landscape_gowers_distance("alluvial fan", "fan remnant")` → `0.0`
  - `landscape_gowers_distance("hill slope", "drainageway")` → `1.0`
  - `landscape_gowers_distance(None, "fan remnant")` → `None`

- [x] **7.2** Create `soil_id/tests/test_terrain_crosswalk.py` with unit tests covering:
  - `crosswalk_slope_shape("Planar")` → `"linear"`
  - `crosswalk_slope_shape("Undulate")` → `"undulating"`
  - `crosswalk_slope_shape(None)` → `None`
  - `slope_shape_gowers_distance("Linear", "Planar")` → `0.0`
  - `slope_shape_gowers_distance("Convex", "Concave")` → `1.0`
  - Northerness/easterness decomposition round-trips for N, S, E, W bearings

- [x] **7.3** Add a `rank_soils` test case that passes terrain inputs
  (`pAspect`, `pSlopeShapeVert`, `pSlopeShapeHoriz`, `pLandscape`) and asserts
  that the returned scores differ from the baseline (terrain variables are active).

- [x] **7.4** Regenerate existing US test snapshots after all changes are complete:
  ```
  pytest soil_id/tests/us/test_us.py --snapshot-update
  ```
  *(Scores will shift because new terrain variables are active when SDA returns data.)*

---

## Files to Create / Modify

| Action | File | Notes |
|---|---|---|
| **Create** | `soil_id/landscape_crosswalk.py` | Copy from `docs/terrain-id/landscape_crosswalk.py` |
| **Create** | `soil_id/terrain_crosswalk.py` | Copy from `docs/terrain-id/terrain_crosswalk.py` |
| **Modify** | `soil_id/us_soil.py` | `list_soils` (SDA query + merge + crosswalk); `rank_soils` (new params + site_mat extension + slope range adjustment) |
| **Modify** | `soil_id/utils.py` | `process_site_data()` — add 9 new column names to keep-list |
| **Create** | `soil_id/tests/test_landscape_crosswalk.py` | Unit tests |
| **Create** | `soil_id/tests/test_terrain_crosswalk.py` | Unit tests |
| **Modify** | `soil_id/tests/us/test_us.py` | Add terrain test case; regenerate snapshots |

---

## Open Questions / Future Considerations

1. **Weight validation**: Starting weights are conservative (terrain total 1.5 vs.
   existing 3.0). Validate against BLM AIM/QC test dataset and tune before production.
2. **`pElev` units**: Currently no unit parameter exists for the observed elevation.
   SDA stores `elev_r`/`elev_l`/`elev_h` in **metres**; BLM typically records in
   **feet**. Decide whether unit conversion belongs in `rank_soils` or at the API layer.
3. **SDA query latency**: The new JOIN query adds a round-trip to `list_soils`.
   Consider batching it with the existing horizon query or firing in parallel if
   latency becomes a concern.
4. **Duplicate `cosurfmorphss` rows**: Some components have multiple surface morphology
   rows. The query uses `rvindicator = 'Yes'` for `cogeomordesc` but not for
   `cosurfmorphss`. Confirm whether a `DISTINCT` or `TOP 1` guard is needed.

---

## AIM/QC Batch Evaluation (Current)

The repository includes a batch runner for study datasets:

`scripts/run_all_aim_examples.py`

### Runner Modes

- `--list-source live` (default): uses live `list_soils` calls per row.
- `--list-source synthetic`: uses static synthetic candidate data for controlled tests.

### Typical Commands

Run AIM:

`py -W ignore scripts/run_all_aim_examples.py --plot-csv study_plot_characteristics_AIM.csv --list-source live`

Run QC:

`py -W ignore scripts/run_all_aim_examples.py --plot-csv study_plot_characteristics_QC.csv --list-source live`

### Output Artifacts

Each run writes three timestamped files to `docs/terrain-id/`:

- `*_run_results_<timestamp>.csv`
- `*_run_summary_<timestamp>.json`
- `*_run_summary_<timestamp>.txt`

### Final Results CSV Columns (Current)

In addition to expected labels and baseline/terrain top predictions, the runner
now writes:

- Expected-series position columns:
  - `expected_rank_baseline`
  - `expected_rank_terrain`
  - `expected_component_id_baseline`
  - `expected_component_id_terrain`
- SDA metadata for expected series:
  - `expected_sda_ecological_site`
  - `expected_sda_landscape_type`
  - `expected_sda_landscape_class`

### Matching Semantics (Important)

- Soil series comparison in the runner uses rank output `component` (canonical
  group name, e.g., `Acuff`) rather than `name` (instance label, e.g.,
  `Acuff2`).
- Ecological-site comparisons can differ depending on reference source:
  - `*_ecological_site_match` compares against study label `EcolSite`.
  - `expected_sda_ecological_site` stores SDA-derived ecological site for the
    expected series/component.

This distinction is intentional and helps separate agreement with study labels
from internal SDA consistency.
