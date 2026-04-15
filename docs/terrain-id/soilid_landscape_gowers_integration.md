# Agent Prompt: Integrate Landscape Type into SoilID Gower's Dissimilarity

## Task Summary

Add landscape type as a **categorical variable** in the SoilID FastAPI soilID
endpoint's Gower's dissimilarity algorithm. The user can now supply a free-text
landscape description (e.g., "alluvial fan", "hill slope") as an API input
parameter. That text and the equivalent text from each SDA candidate component
are both mapped to a shared 7-class vocabulary via a fuzzy crosswalk, and the
resulting Gower's categorical distance (0 or 1) is folded into the existing
weighted Gower's sum.

A fully working Python crosswalk module is provided — **do not rewrite that
logic**. Your job is to wire it into the existing FastAPI/SoilID codebase.

---

## Attachments

| File | Description |
|---|---|
| `landscape_crosswalk.py` | Drop this module into the SoilID package. It contains `crosswalk_landscape_class()`, `build_sda_landscape_label()`, `landscape_gowers_distance()`, and `landscape_class_info()`. |

---

## Background: The 7-Class Landscape Vocabulary

The crosswalk maps raw text to these seven classes (order matters — first match wins):

| Class | Canonical examples |
|---|---|
| `flats_plains` | playas, flats, plains, basins, valley floors, lake plains |
| `fans` | alluvial fans, fan remnants, aprons, toeslopes, footslopes |
| `hills_mountains` | hills, mountains, ridges, escarpments, slopes |
| `terraces_plateaus` | terraces, plateaus, mesas, benches |
| `drainages` | drainageways, channels, swales, washes, draws, arroyos |
| `dunes_sands` | dunes, sand sheets, aeolian deposits |
| `rocklands` | rock outcrops, badlands, cliffs, talus |
| `other` | catch-all (including missing/unmappable) |

Three sensitivity modes exist: `"base"` (default, recommended), `"strict"`,
`"loose"`. Use `"base"` unless there is a reason to diverge.

---

## What the SDA Side Provides

For each candidate soil component SDA already returns geomorphic fields from
`cogeomordesc`, `cosurfmorphgc`, and `cosurfmorphss`. Assemble the SDA
landscape label before crosswalking like this:

```python
from landscape_crosswalk import build_sda_landscape_label, crosswalk_landscape_class

sda_label = build_sda_landscape_label(
    component["geomftname"],   # geomorphic feature type
    component["geomfname"],    # geomorphic feature name
    component["geomfmod"],     # geomorphic feature modifier
    component["geomposmntn"],  # geomorphic position – mountains
    component["geomposhill"],  # geomorphic position – hills
    component["geompostrce"],  # geomorphic position – terraces
    component["geomposflats"], # geomorphic position – flats
    component["shapeacross"],  # shape across slope
    component["shapedown"],    # shape down slope
)
sda_class = crosswalk_landscape_class(sda_label, mode="base")
```

If those SDA fields are **not yet included** in the component SQL query, add
them via a JOIN on `cogeomordesc` (linked through `cokey`). See the reference
SQL pattern at the end of this prompt.

---

## Step-by-Step Integration Instructions

### 1  Copy the module

Place `landscape_crosswalk.py` in the same package directory as your Gower's
distance utilities (e.g., `soilid/utils/` or wherever `gower_distance.py`
lives).

### 2  Add the endpoint parameter

In the appropriate FastAPI router (the one that handles `POST /soilID` or
equivalent), add a new optional field to the request model:

```python
from pydantic import BaseModel
from typing import Optional

class SoilIDRequest(BaseModel):
    # ... existing fields ...
    landscape_type: Optional[str] = None   # free-text, e.g. "alluvial fan"
    landscape_crosswalk_mode: Optional[str] = "base"  # "base" | "strict" | "loose"
```

### 3  Compute the Gower landscape distance for each candidate

In the function that assembles the per-candidate Gower variable array, add
landscape type as a new categorical entry. Pseudocode adapter:

```python
from landscape_crosswalk import (
    build_sda_landscape_label,
    landscape_gowers_distance,
    landscape_class_info,
)

def compute_landscape_distance(
    observed_label: str | None,
    component_row: dict,
    mode: str = "base",
) -> float | None:
    """Return Gower's categorical distance (0/1/None) for landscape type."""
    sda_label = build_sda_landscape_label(
        component_row.get("geomftname"),
        component_row.get("geomfname"),
        component_row.get("geomfmod"),
        component_row.get("geomposmntn"),
        component_row.get("geomposhill"),
        component_row.get("geompostrce"),
        component_row.get("geomposflats"),
        component_row.get("shapeacross"),
        component_row.get("shapedown"),
    )
    return landscape_gowers_distance(observed_label, sda_label, mode=mode)
```

### 4  Extend the Gower's weighted sum

Standard Gower's for a pair `(observation, candidate)`:

$$D_{Gower} = \frac{\sum_{k} w_k \cdot \delta_k}{\sum_{k} w_k \cdot \mathbb{1}[\delta_k \neq \text{None}]}$$

Add landscape type as variable `k = "landscape"`:

```python
LANDSCAPE_WEIGHT = 0.10   # tune this — start at 0.10 and validate

def gowers_distance(obs: dict, candidate: dict, request: SoilIDRequest) -> float:
    """
    Extend your existing Gower's function to include landscape type.
    Only the landscape variable addition is shown here; keep all other
    variable contributions unchanged.
    """
    weighted_sum = 0.0
    weight_total = 0.0

    # --- existing variables (texture, color, depth, etc.) ---
    # ... your current logic here (do not change) ...

    # --- NEW: landscape type (categorical, Gower = 0 or 1) ---
    d_landscape = compute_landscape_distance(
        observed_label=request.landscape_type,
        component_row=candidate,
        mode=request.landscape_crosswalk_mode or "base",
    )
    if d_landscape is not None:                        # None = missing → exclude
        weighted_sum  += LANDSCAPE_WEIGHT * d_landscape
        weight_total  += LANDSCAPE_WEIGHT

    # --- normalise ---
    if weight_total == 0:
        return 1.0   # no usable variables → maximum dissimilarity
    return weighted_sum / weight_total
```

> **Weight guidance**: Start with `LANDSCAPE_WEIGHT = 0.10`.  
> Validation against this dataset shows ~58–85 % base-mode agreement
> (depending on field protocol), so landscape is informative but noisy —
> 0.10 is conservative. Do **not** exceed 0.20 without cross-validation.

### 5  Optionally return diagnostic metadata

You can include the crosswalked classes in the API response for transparency:

```python
from landscape_crosswalk import landscape_class_info

info = landscape_class_info(
    observed_label=request.landscape_type,
    sda_label=sda_label,
    mode=request.landscape_crosswalk_mode or "base",
)
# info = {
#   "observed_class": "fans",
#   "sda_class":      "fans",
#   "gowers_distance": 0.0,
#   "mode":           "base"
# }
```

Include `info` in a `landscape_match` field on the per-candidate response
object if your schema supports it.

---

## Reference SQL — Geomorphic Fields

If the SDA component query does not yet return geomorphic fields, add this
pattern (adjust table alias as needed):

```sql
-- Add to existing component-level SELECT
SELECT
    c.cokey,
    c.compname,
    c.comppct_r,
    -- geomorphic fields
    gm.geomftname,
    gm.geomfname,
    gm.geomfmod,
    ssmn.geomposmntn,
    ssh.geomposhill,
    sstr.geompostrce,
    ssfl.geomposflats,
    ssac.shapeacross,
    ssdn.shapedown
FROM component c
LEFT JOIN cogeomordesc gm
    ON c.cokey = gm.cokey AND gm.rvindicator = 'Yes'
LEFT JOIN cosurfmorphgc ssmn
    ON gm.cogeomdesckey = ssmn.cogeomdesckey
LEFT JOIN cosurfmorphss ssh
    ON gm.cogeomdesckey = ssh.cogeomdesckey
-- ... rest of your existing JOINs ...
```

> `rvindicator = 'Yes'` selects the representative geomorphic description.
> If a component has multiple geomorphic descriptions, use the one with
> `rvindicator = 'Yes'`; if none, fall back to the first row.

---

## Testing Checklist

- [ ] `crosswalk_landscape_class("alluvial fan")` returns `"fans"`
- [ ] `crosswalk_landscape_class("hill slope")` returns `"hills_mountains"`
- [ ] `crosswalk_landscape_class(None)` returns `None` (not `"other"`)
- [ ] `landscape_gowers_distance("alluvial fan", "fan remnant")` returns `0.0`
- [ ] `landscape_gowers_distance("hill slope", "drainageway")` returns `1.0`
- [ ] `landscape_gowers_distance(None, "fan remnant")` returns `None`
- [ ] API endpoint accepts `landscape_type: null` and excludes the variable gracefully
- [ ] `landscape_crosswalk_mode` defaults to `"base"` when not provided
- [ ] Gower's score changes predictably when `landscape_type` is provided vs. omitted
- [ ] Existing tests for other Gower's variables are unchanged (no regressions)

---

## Files to Create / Modify

| Action | File |
|---|---|
| **Add** | `soilid/utils/landscape_crosswalk.py` (from attachment) |
| **Modify** | request model (add `landscape_type`, `landscape_crosswalk_mode`) |
| **Modify** | Gower's distance function (add landscape variable block) |
| **Modify** | Component SDA SQL (add geomorphic field columns if missing) |
| **Modify** | Response schema (optionally add `landscape_match` metadata) |
| **Add** | Unit tests for `landscape_crosswalk.py` |

---

## Do NOT Change

- The rule order or regex patterns in `landscape_crosswalk.py` — they are
  validated against 524 AIM/QC field points and calibrated for agreement rates.
- Any existing Gower's variables, weights, or normalisation logic.
- The `"other"` catch-all — both sides mapping to `"other"` is intentionally
  treated as a match (distance = 0); document this if it surfaces in QA.
