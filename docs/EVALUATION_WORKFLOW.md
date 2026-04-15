# SoilID Ecological Site Match Evaluation: Complete Workflow

**Document version:** April 13, 2026  
**Scope:** End-to-end description of all raw data, processing steps, and statistical techniques used to evaluate whether SoilID's Confidence Index (CI) predicts correct ecological site identification.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Raw Data Sources](#2-raw-data-sources)
3. [Stage 1 — SDA Enrichment (R)](#3-stage-1--sda-enrichment-r)
4. [Stage 2 — Input CSV Preparation](#4-stage-2--input-csv-preparation)
5. [Stage 3 — SoilID Pipeline Execution](#5-stage-3--soilid-pipeline-execution)
6. [Stage 4 — Confidence Index Reconstruction](#6-stage-4--confidence-index-reconstruction)
7. [Stage 5 — Statistical Evaluation](#7-stage-5--statistical-evaluation)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [File Inventory](#9-file-inventory)
10. [Independence and Validity Considerations](#10-independence-and-validity-considerations)
11. [Known Limitations](#11-known-limitations)

---

## 1. Overview

The evaluation asks a single question: **does SoilID's Confidence Index meaningfully predict whether the algorithm identifies the correct ecological site at an AIM field plot?**

The approach is a prospective construct validation. The CI formula and thresholds are defined from NCSS (National Cooperative Soil Survey) mapping standards — they are not fit or optimized against the outcome data. AIM field ecological site observations are independent of SSURGO. The evaluation tests whether those independently constructed CI values sort plots by their observed match rate.

The pipeline has five stages:

```
Raw XLSX + SDA queries → Enriched input CSV → SoilID pipeline run
    → CI reconstruction → Statistical tests
```

---

## 2. Raw Data Sources

### 2.1 AIM Field Plot Data

**File:** `Data/aim_data/Supplementary_data1.xlsx` (Sheet1)

The primary input. Each row is one AIM (Assessment, Inventory, and Monitoring) field plot. Key columns:

| Column | Description |
|---|---|
| `PrimaryKey` | Unique plot identifier |
| `Longitude_NAD83` / `Latitude_NAD83` | NAD83 decimal-degree coordinates |
| `EcolSite_AIM` | Field-recorded ecological site ID (pre-QC) |
| `EcolSite_QC` | QC-reviewed ecological site ID (used as ground truth) |
| `Soil Series_AIM` / `Soil_Series_QC` | Field-recorded and QC-reviewed soil series names |
| `AIM_LandscapeType_BEFORE_QC` / `QC_LandscapeType` | Landscape position labels |
| `Slope`, `Elevation`, `Aspect` | Terrain inputs for SoilID |
| `SlopeShapeVertical`, `SlopeShapeHorizontal` | Profile and plan curvature |
| `LandscapeType` | Landscape classification for rank weighting |

**Source for ground truth:** `EcolSite_QC` is the reference ecological site. This column represents the QC reviewer's final determination. A match is scored as `1` when SoilID's top-ranked component's ecological site (`ecoclassid` from SDA `coecoclass`) equals the QC reference after stripping leading `R`/`F` prefixes and normalizing case.

### 2.2 Soil Horizon Data

**File:** `soil_id/tests/us/Data/aim_horizons.csv` (path via `HORIZONS_CSV` constant in test module)

Per-horizon measurements for each plot, joined to plots by `PrimaryKey`. Columns include:
- `HorizonDepthUpper`, `HorizonDepthLower` — depth interval (cm)
- `Texture` — abbreviated USDA texture class (translated via `TEXTURE_ABBREV_MAP`)
- `RockFragments` — fractional rock fragment volume
- `Hue`, `Value`, `Chroma` — Munsell soil color notation

The horizon data feeds two SoilID inputs: the texture/depth profile for soil scoring and Munsell-to-CIELAB color conversion for color-matching scoring.

### 2.3 SSURGO / SDA (Soil Data Access)

All soil survey attributes are queried live from USDA's Soil Data Access (SDA) REST API during both the R enrichment stage and the Python SoilID pipeline. No local SSURGO database is required. The relevant tables accessed are:

| SDA table | Fields extracted |
|---|---|
| `mapunit` | `mukey`, `musym`, `muname`, `mukind`, `mlrasymbol`, `invesintens` |
| `legend` | `areasymbol`, `areaname`, `projectscale` |
| `component` | `compname`, `comppct_r`, `cokey` |
| `coecoclass` | `ecoclassid`, `ecoclassname`, `ecoclasstypename`, `ecositestatus` |
| `cogeomordesc` | Geomorphic description fields |
| `cosurfmorphgc` / `cosurfmorphss` | Geomorphic position and slope shape |

### 2.4 Multiplicity Lookup

**File:** `Data/aim_data/compname_mlra_ecosite_multiplicity.csv`

A pre-built lookup joining dominant SSURGO component names to the number of distinct ecological sites they are correlated to within each MLRA. Generated separately from SDA `coecoclass`. Key columns: `compname_norm`, `mlrasymbol`, `n_ecosites`.

This lookup is used as a fallback when `multiplicity_score` is absent from the run-results CSV.

---

## 3. Stage 1 — SDA Enrichment (R)

**Script:** `scripts/query_soil_survey_order.R`  
**Runtime:** ~5–15 minutes (API rate-limited)  
**Language:** R 4.3.3 with `soilDB`, `dplyr`, `readxl`

### 3.1 Purpose

The R script queries SDA to populate soil survey quality metadata for every plot — fields that are not part of the SoilID Python pipeline's normal output but are needed to compute the full five-component CI.

### 3.2 Input

`Data/aim_data/Supplementary_data1.xlsx` Sheet1 — 524 rows (523 NV + 1 Idaho plot).

### 3.3 Processing

**Point geometry:** Each plot's NAD83 coordinate is formatted as WKT (`point(lon lat)`) and sent to SDA's `SDA_Get_Mukey_from_intersection_with_WktWgs84` function to identify the intersecting map unit.

**Chunk queries:** Points are batched in groups of 100 (configurable via `chunk_size`) to stay within SDA's query limits. Failed chunks fall back to NA rows with a warning.

**Primary SQL query** (per chunk): A multi-table join retrieves from the intersecting map unit:
- `mlrasymbol` — the MLRA code
- `invesintens` — the soil survey mapping investigation intensity (Order 2/3/4/5)
- `mukind` — map unit kind (consociation, complex, association, etc.)
- `dominant_compname`, `dominant_comppct_r` — the highest-`comppct_r` component
- `second_comppct_r` — the second component's percentage (for gap calculation)
- Ecological site ID from `coecoclass` for the dominant component
- Geomorphic attributes from `cogeomordesc`, `cosurfmorphgc`, `cosurfmorphss`

**Component buffer query** (per chunk): A second, separate SQL query expands the spatial search to a 1,000 m buffer around each point. This is structurally distinct from the primary point-intersection query and serves a different purpose.

The buffer geometry is constructed in SDA using SQL Server spatial methods:

```sql
geography::STGeomFromText(pts.wkt, 4326).STBuffer(1000).ToString()
```

This returns all map units whose polygons intersect the 1,000 m circle. The query then joins all `component` rows from those map units, producing a flat list of every named soil series that exists within ~1 km of the plot, along with each component's:
- `compname`, `comppct_r`, `cokey` — component identity and coverage
- `ecoclassid` (via `coecoclass`) — ecological site correlation for that component
- Geomorphic attributes (via `cogeomordesc`, `cosurfmorphgc`, `cosurfmorphss`)
- `mapunit_source` — a flag indicating whether the component's map unit is the **home map unit** (the one the point falls inside) or an **adjacent map unit** (only intersected by the buffer)

The buffer results feed three downstream uses:

1. **AIM series matching (`aim_series_match`):** For each plot, the field-recorded AIM soil series (`Soil Series_AIM`) is looked up in the buffer component list. This identifies whether the field-recorded series appears in the home map unit or only in an adjacent one, and retrieves its SSURGO-matched component name, coverage percentage, landscape class, and ecological site. A `aim_series_selection_changed` flag is set when the landscape-informed selection differs from a name-only lookup.

2. **QC series matching (`qc_series_match`):** Same process using the QC-reviewed series name (`Soil_Series_QC`).

3. **Ecosite buffer matching (`aim_ecosite_buffer_match`, `qc_ecosite_buffer_match`):** The AIM and QC ecological site IDs are searched against `ecoclassid` values in the buffer component list, flagging whether the expected ecosite appears anywhere within 1 km — regardless of whether it is the dominant component at the point.

These matches are used to populate the `aim_series_component_name`, `qc_series_component_name`, landscape crosswalk columns, and `series_selection_sensitivity_summary.csv` in the output. Specifically, `series_selection_sensitivity_summary.csv` reports the percentage of plots where the landscape-constrained series selection differs from the name-only selection — a diagnostic for how often landscape position disambiguates series with the same name in multiple map unit types.

**Multiplicity calculation:** After the SDA queries, the script loads `compname_mlra_ecosite_multiplicity.csv` and joins `n_ecosites_dominant` (the number of distinct ecological sites the dominant series is correlated to in that MLRA) to each point.

### 3.4 Scoring Functions Applied in R

```r
score_order <- function(x) {
  case_when(
    x == "Order 2" ~ 100,
    x == "Order 3" ~  80,
    x == "Order 4" ~  55,
    x == "Order 5" ~  35,
    x == "NoData"  ~  30,
    TRUE           ~  45   # unrecognized values
  )
}

score_mukind <- function(x) {
  lx <- tolower(ifelse(is.na(x), "", x))
  case_when(
    grepl("consociation",     lx) ~ 100,
    grepl("association",      lx) ~  70,
    grepl("complex",          lx) ~  45,
    grepl("undifferentiated", lx) ~  40,
    lx == "nodata" | lx == ""    ~  30,
    TRUE                          ~  55
  )
}
```

`multiplicity_score` is applied separately:
- `n_ecosites == 1` → 100 (single correlated ecosite: high confidence)  
- `n_ecosites >= 2` → 30 (multiple correlated ecosites: penalized)  
- `NA` → 50 (neutral, missing data)

### 3.5 Outputs

| File | Description |
|---|---|
| `outputs/soil_survey_order/points_with_soil_survey_order.csv` | Per-point SDA attributes + `order_score`, `mukind_score`, CI components |
| `Data/aim_data/study_plot_characteristics.csv` | Same 524-row file written to the pipeline input directory |
| `outputs/soil_survey_order/soil_survey_order_summary.csv` | Aggregate summary by survey order |
| `outputs/soil_survey_order/point_landscape_ecosite_comparison.csv` | Landscape crosswalk comparison table |

---

## 4. Stage 2 — Input CSV Preparation

**Script:** Manual filtering step (command-line `pandas`)  
**Purpose:** Restrict the analysis to NV-only plots and merge R-derived CI columns.

### 4.1 NV Filtering

The 524-row `study_plot_characteristics.csv` produced by R contains 523 NV plots and 1 Idaho plot (`NV_NV-Winnemucca-DO-2021_WMDO2021-005`). The Idaho plot is excluded because the study is scoped to Nevada Great Basin conditions. The NV-only subset is saved as:

**File:** `Data/aim_data/study_plot_characteristics_nv.csv` (523 rows)

This file retains all 73 original columns from `Supplementary_data1.xlsx` plus the R-derived columns:
- `mlrasymbol` — native SDA MLRA code (100% populated)
- `n_ecosites_dominant` — number of correlated ecosites for the dominant series (522/523 populated)
- `multiplicity_score` — scored multiplicity value (all 523 populated)

### 4.2 Environment Configuration

**File:** `.env` (repo root)

```
DATA_PATH=Data
```

This variable is read by `soil_id/config.py` to locate the Munsell RGB-LAB reference CSV (`Data/LandPKS_munsell_rgb_lab.csv`). Without it, color scoring silently fails for all plots. A stale parent-directory `.env` pointing to a Windows path was overriding this — the repo-level `.env` takes precedence.

---

## 5. Stage 3 — SoilID Pipeline Execution

**Script:** `scripts/run_all_aim_examples.py`  
**Command:** `python3 scripts/run_all_aim_examples.py --plot-csv Data/aim_data/study_plot_characteristics_nv.csv`  
**Runtime:** ~2–4 hours for 523 plots (live SDA mode)

### 5.1 Row Validation

Before running, each plot row is validated for the presence of required columns: `Slope`, `Elevation`, `Aspect`, `SlopeShapeVertical`, `SlopeShapeHorizontal`, `LandscapeType`. Rows missing any of these are skipped. Rows must also have a matching entry in the horizons CSV.

### 5.2 Per-Plot Processing Loop

For each valid plot row:

**a. Horizon data construction**

Horizons are loaded from the horizons CSV, filtered to the plot's `PrimaryKey`, sorted by `HorizonDepthUpper`, and assembled into lists:
- `soilHorizon` — USDA texture class strings (abbreviated codes translated to full names)
- `topDepth` / `bottomDepth` — integer depth boundaries (cm)
- `rfvDepth` — rock fragment volume as a percent-range string (e.g. `"5-6%"`)
- `lab_Color` — CIELAB color values converted per horizon via Munsell lookup

**b. Color conversion**

Munsell notation (`Hue`, `Value`, `Chroma`) is converted to CIE-Lab using `soil_id.color.munsell2rgb` and `find_closest_rgb_in_reference`. Requires the Munsell reference CSV to be accessible via `DATA_PATH`. Failures return `None` per horizon (color scoring still proceeds, reduced accuracy).

**c. SoilID list call**

`soil_id.us_soil.list_soils(lon, lat, sim=False)` is called with the plot's NAD83 coordinates. This queries SDA for all soil components within the map unit intersecting the point and within a surrounding buffer. Returns a `SoilListOutputData` object containing:
- `map_unit_component_data_csv` — component-level SSURGO attributes
- `soil_list_json` — SoilID's candidate component list with ranks

If `list_soils` fails to return a valid object (e.g., coordinates outside SSURGO coverage), the plot is recorded as `status="skipped"`.

**d. Component metadata extraction**

`_build_component_metadata()` parses the list output to build a dictionary keyed by `cokey` (and fallback `name::compname`), storing each component's:
- `ecological_site` — `ecoclassid_update` or `ecoclassid` from `coecoclass`, normalized (leading R/F stripped, uppercased)
- `landscape_class` — derived from `build_sda_landscape_label()` crosswalk of geomorphic attributes

**e. Baseline ranking (no terrain)**

`soil_id.us_soil.rank_soils()` is called with texture, depth, rock fragment, and color inputs but **without** terrain parameters (`pAspect`, `pSlopeShapeVert`, `pSlopeShapeHoriz`, `pLandscape` all omitted). This is the data-only baseline that ranks components purely on soil property similarity.

**f. Terrain-augmented ranking**

`rank_soils()` is called again with the same soil inputs **plus** all terrain parameters. This is the terrain-augmented model that additionally weights components by landscape position compatibility.

**g. Match evaluation**

For both the baseline and terrain results:
1. The top-ranked component (`rank_data_loc == "1"`) is identified.
2. Its associated ecological site is looked up from the component metadata dict.
3. The predicted site is compared to `EcolSite_QC` (QC ground truth) after normalization.
4. Two boolean match columns are written per result:
   - `baseline_qc_ecological_site_match`
   - `terrain_qc_ecological_site_match`

**h. Plot metadata pass-through**

CI-related columns from the input CSV are written verbatim to the results row:
- `mlrasymbol`, `confidence_index`, `uncertainty_class`, `uncertainty_reason`
- `dominant_comppct_r`, `second_comppct_r`, `component_gap`
- `n_ecosites_dominant`, `multiplicity_score`

### 5.3 Output

**File:** `Data/aim_data/study_plot_characteristics_nv_run_results_<TIMESTAMP>Z.csv`

One row per plot. Key columns:

| Column | Source | Description |
|---|---|---|
| `PrimaryKey` | Input | Plot identifier |
| `status` | Pipeline | `"passed"` or `"skipped"` |
| `baseline_qc_ecological_site_match` | Pipeline | 1/0/None — baseline model vs QC truth |
| `terrain_qc_ecological_site_match` | Pipeline | 1/0/None — terrain model vs QC truth |
| `confidence_index` | SDA (via R) | Raw CI value from enrichment step |
| `uncertainty_class` | SDA (via R) | Three-level CI class |
| `uncertainty_reason` | SDA (via R) | Primary reason for class assignment |
| `dominant_comppct_r` | SDA | Top component percentage (%) |
| `second_comppct_r` | SDA | Second component percentage (%) |
| `component_gap` | SDA | `dominant_comppct_r - second_comppct_r` |
| `mlrasymbol` | SDA (native) | MLRA code |
| `multiplicity_score` | R enrichment | Ecosite multiplicity score |
| `n_ecosites_dominant` | R enrichment | Number of correlated ecosites |

The April 13, 2026 canonical run:
- **522 passed, 0 failed** (1 skipped — outside SSURGO coverage at −117.46, 39.35)
- All CI columns populated from native SDA values, no lookup approximations

---

## 6. Stage 4 — Confidence Index Reconstruction

**Script:** `scripts/_analyze_ci_revised.py`  
**Purpose:** Recalculate CI using the revised step-function formula and reconstruct `order_score` / `mukind_score` from categorical labels (these fields are not written to the run-results CSV by the pipeline).

### 6.1 Score Reconstruction

Because `order_score` and `mukind_score` are computed in R but not propagated through the Python pipeline CSV, the analysis script approximates them from the categorical `uncertainty_class` and `uncertainty_reason` columns:

```python
def approx_order_score(row):
    if row["uncertainty_reason"] == "Lower-intensity mapping order":
        return 45   # Order 4/5 blend
    elif row["uncertainty_class"] == "Low uncertainty (high confidence)":
        return 92   # predominantly Order 2
    else:
        return 80   # Order 3 dominant

def approx_mukind_score(row):
    r = str(row["uncertainty_reason"]).lower()
    if "complex" in r or "undiff" in r:
        return 42   # complex/undifferentiated
    elif row["uncertainty_class"] == "Low uncertainty (high confidence)":
        return 100  # consociation
    else:
        return 85   # moderate confidence map unit kind
```

### 6.2 Step-Function Score Recalculation

New `dominant_score` (step function replacing the old linear interpolation):

| `dominant_comppct_r` | Score |
|---|---|
| NA | 50 (neutral) |
| < 50% | 20 |
| 50–79% | 55 |
| ≥ 80% | 100 |

New `gap_score`:

| `component_gap` | Score |
|---|---|
| NA | 50 (neutral) |
| < 20 pp | 30 |
| 20–39 pp | 60 |
| ≥ 40 pp | 90 |

### 6.3 CI Formula

$$\text{CI} = 0.20 \cdot \text{order\_score} + 0.10 \cdot \text{mukind\_score} + 0.35 \cdot \text{dominant\_score} + 0.20 \cdot \text{gap\_score} + 0.15 \cdot \text{multiplicity\_score}$$

### 6.4 Class Thresholds

| CI value | Class |
|---|---|
| ≥ 78 | Low uncertainty (high confidence) |
| 55–77 | Moderate uncertainty |
| < 55 | High uncertainty |

Uncertainty reason logic:
1. If `uncertainty_reason == "Lower-intensity mapping order"` → preserved from original
2. If reason contains `"complex"` or `"undiff"` → `"Complex/undifferentiated map unit"`
3. If `dominant_comppct_r < 80` → `"Weak dominant component"`
4. If `component_gap < 20` → `"Top components have similar proportion"`
5. Otherwise → `"Stronger map unit confidence profile"`

### 6.5 MLRA Enrichment Fallback

If `mlrasymbol` is absent or all-null in the run-results CSV, the script reads it from `study_plot_characteristics_nv.csv` and normalizes the format (e.g., `28BY` → `28b`, `026X` → `26`) using a regex extractor.

If `multiplicity_score` is absent or all-null, it is looked up from `compname_mlra_ecosite_multiplicity.csv` by `(compname_norm, mlrasymbol)`.

---

## 7. Stage 5 — Statistical Evaluation

**Scripts:** `scripts/_analyze_ci_revised.py`, `scripts/_analyze_ci_mlra_clustered.py`

### 7.1 Technique 1 — Class Distribution and Shift

**What:** Tabulates the count of plots in each uncertainty class (Low/Moderate/High) under both the old and new CI formula.

**Why:** Confirms that formula changes do not wholesale reclassify the dataset, and that any distribution shifts come from real changes in scoring logic rather than data artifacts.

**Output:** Old-vs-new counts per class; old-vs-new CI summary statistics (min, Q1, median, Q3, max).

---

### 7.2 Technique 2 — Primary Match Rate Table with Wilson Confidence Intervals

**What:** For each CI class, computes the baseline and terrain match rates. Wilson score intervals are computed at 95% confidence level using `scipy.stats.binomtest`.

$$\text{Wilson CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \quad z = 1.96$$

**Why:** Wilson intervals are preferred over Wald intervals for proportions, particularly when $n$ is moderate and $\hat{p}$ is near 0 or 1. They remain valid even for small class sizes.

**Key test:** Do the Wilson intervals for Low and High uncertainty classes overlap? Non-overlap confirms a real effect beyond sampling uncertainty.

---

### 7.3 Technique 3 — Chi-Square Test of Independence

**What:** Tests whether the distribution of match/no-match outcomes differs significantly across three CI classes.

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

**Contingency table structure:**

|  | Low | Moderate | High |
|---|---|---|---|
| No match | $O_{11}$ | $O_{12}$ | $O_{13}$ |
| Match | $O_{21}$ | $O_{22}$ | $O_{23}$ |

**Implementation:** `scipy.stats.chi2_contingency` with Yates correction disabled (sufficient cell counts).

**Interpretation:** A significant result (p < 0.05) rejects the null hypothesis that CI class and match outcome are independent. Degrees of freedom = 2.

---

### 7.4 Technique 4 — Spearman Rank Correlation

**What:** Computes Spearman's $\rho$ between individual CI scores and binary match outcomes across all 523 plots.

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}$$

**Why:** Unlike Pearson correlation, Spearman's $\rho$ requires only monotone ordering (not linearity), which is appropriate for a CI that uses step functions and a binary outcome. It is also robust to tied CI values.

**Implementation:** `scipy.stats.spearmanr`; computed separately for old CI (pipeline-stored) and new CI (step-function recalculated).

---

### 7.5 Technique 5 — Per-MLRA Breakdown with Wilson Intervals

**What:** Repeats the three-class match rate table stratified by MLRA (Major Land Resource Area). Computes Wilson 95% CI for each MLRA × class cell.

**Why:** Soil survey quality varies systematically by MLRA — MLRAs differ in survey age, cartographic scale, and density of ecological site correlations. MLRA-stratified analysis tests whether the CI gradient is consistent across geographically distinct survey areas, or is driven by one MLRA.

**Flags:** MLRAs with `mean_base_rate > 0.15` (see §7.8) are flagged because their apparent match rates are partially driven by ecosite concentration rather than CI performance.

---

### 7.6 Technique 6 — CI Decile Calibration

**What:** Divides the CI range into up to 10 equal-frequency quantile bins and computes the mean match rate in each bin.

**Implementation:** `pandas.qcut` with `duplicates="drop"` to handle tied CI values. The number of usable bins is reduced when there are many tied values (which occur at step-function boundary values like CI = 54.2).

**Why:** Calibration plots ask whether the CI score is a well-calibrated probability — does a CI of 60 correspond to roughly 60% match rate? Perfect monotone calibration with no dips across deciles would be ideal; the current data shows the correct overall direction with some within-range variation.

---

### 7.7 Technique 7 — MLRA Cluster-Robust Logistic Regression

**Script:** `scripts/_analyze_ci_mlra_clustered.py`

**What:** Fits a logistic regression of match outcome on continuous CI score, with standard errors clustered by MLRA using the sandwich variance estimator.

$$\logit P(\text{match}) = \alpha + \beta \cdot \text{CI}$$

**Why clustering:** Plots within the same MLRA are not independent observations — they share the same soil survey context, ecological region, and CI distribution. Ignoring this cluster structure would underestimate standard errors and overstate statistical significance. The sandwich estimator (Huber-White) produces valid inference even when the within-cluster correlation structure is misspecified.

**Implementation:** `statsmodels.formula.api.logit` with `cov_type="cluster"`, `cov_kwds={"groups": mlrasymbol}`.

**Interpretation:**
- $\beta$ is the log-odds change per 1-point CI increase
- Exponentiated: odds ratio $= e^\beta$ per point; $e^{10\beta}$ per 10-point increase
- $p < 0.001$ confirms the CI signal is not explained by within-MLRA noise

**Model 1:** Cluster-robust (primary inferential result)  
**Model 2:** MLRA fixed-effects (dummy variables for each MLRA) — controls for between-MLRA baseline differences; confirms the CI effect is not manufactured by MLRA composition

---

### 7.8 Technique 8 — Ecosite Base-Rate Covariate Model

**What:** Adds an ecosite concentration covariate to the logistic regression to test whether the CI coefficient is inflated by "base-rate luck."

**Ecosite base-rate formula:**

$$\text{base\_rate}_i = \frac{|\{j : \text{MLRA}_j = \text{MLRA}_i,\ \text{EcolSite}_j = \text{EcolSite}_i\}|}{|\{j : \text{MLRA}_j = \text{MLRA}_i\}|}$$

This measures the fraction of plots in the same MLRA that expect the same ecological site. A high base-rate means SoilID could match that plot by simply predicting the MLRA's most common ecosite — without CI doing any real work.

**Model 3:** `match ~ confidence_index + ecosite_base_rate` (cluster-robust)

**Attenuation ratio:**

$$\text{attenuation} = \frac{\hat{\beta}_{\text{Model1}} - \hat{\beta}_{\text{Model3}}}{\hat{\beta}_{\text{Model1}}}$$

A low attenuation (< 10%) indicates the CI coefficient is not being driven by ecosite concentration.

**Per-MLRA table:** Reports mean and maximum base-rate for each MLRA to identify which MLRAs have structurally inflated match rates.

---

### 7.9 Technique 9 — Anomaly Group Tracking

**What:** Identifies plots that satisfy a specific cross-classification condition that should shift between analysis runs — in this case, plots that the old formula labeled `Moderate uncertainty + "Stronger map unit confidence profile"` but whose raw `dominant_comppct_r` value would put them in a different reason category under the new threshold.

**Why:** This check confirms that the formula revision is behaving as intended and that no systematic mislabeling persists in the new results.

---

## 8. Data Flow Diagram

```
Supplementary_data1.xlsx (524 pts)
          │
          ▼
  [R: query_soil_survey_order.R]
  SDA queries (100-pt chunks)
  ├── invesintens (survey order)
  ├── mukind
  ├── mlrasymbol
  ├── dominant/second comppct_r
  └── multiplicity via compname lookup
          │
          ▼
  study_plot_characteristics.csv (524 rows)
  + study_plot_characteristics_nv.csv (523 rows, filtered)
          │
          ▼
  [Python: run_all_aim_examples.py]
  ├── load horizons → texture / depth / rfv / color
  ├── list_soils(lon, lat) → SDA candidate components
  ├── rank_soils(soil only)     → baseline top-ranked component
  ├── rank_soils(soil + terrain) → terrain top-ranked component
  └── compare top component's ecoclassid vs EcolSite_QC
          │
          ▼
  study_plot_characteristics_nv_run_results_<TS>Z.csv
  (522 passed / 1 skipped)
          │
          ▼
  [Python: _analyze_ci_revised.py]
  ├── reconstruct order_score / mukind_score from labels
  ├── apply step-function dominant_score / gap_score
  ├── compute new CI
  ├── assign new uncertainty_class / reason
  └── run statistical tests (§7.1–7.9)
          │
          ▼
  [Python: _analyze_ci_mlra_clustered.py]
  ├── merge mlrasymbol / ecosite_base_rate
  ├── fit cluster-robust logit (Model 1)
  ├── fit MLRA fixed-effects logit (Model 2)
  └── fit base-rate covariate model (Model 3)
```

---

## 9. File Inventory

### Input Files

| File | Description |
|---|---|
| `Data/aim_data/Supplementary_data1.xlsx` | Raw AIM field plot workbook (Sheet1) |
| `Data/aim_data/compname_mlra_ecosite_multiplicity.csv` | Pre-built multiplicity lookup by (compname, mlrasymbol) |
| `Data/LandPKS_munsell_rgb_lab.csv` | Munsell to RGB/Lab reference table for color scoring |

### Intermediate Files

| File | Description |
|---|---|
| `Data/aim_data/study_plot_characteristics.csv` | 524-row R output (all plots, includes R-derived CI columns) |
| `Data/aim_data/study_plot_characteristics_nv.csv` | 523-row NV-only input for pipeline run |
| `outputs/soil_survey_order/points_with_soil_survey_order.csv` | Per-point SDA attributes with `order_score`, `mukind_score` |

### Results Files

| File | Description |
|---|---|
| `Data/aim_data/study_plot_characteristics_nv_run_results_20260413T214320Z.csv` | Canonical NV run results (522 passed) |

### Analysis Scripts

| Script | Language | Purpose |
|---|---|---|
| `scripts/query_soil_survey_order.R` | R 4.3.3 | SDA enrichment, CI component scoring |
| `scripts/run_all_aim_examples.py` | Python 3.12 | SoilID pipeline runner |
| `scripts/_analyze_ci_revised.py` | Python 3.12 | CI reconstruction + statistical tests (§7.1–7.6, 7.9) |
| `scripts/_analyze_ci_mlra_clustered.py` | Python 3.12 | Logistic regression models (§7.7–7.8) |

### Configuration

| File | Purpose |
|---|---|
| `.env` (repo root) | `DATA_PATH=Data` — enables color scoring |

---

## 10. Independence and Validity Considerations

### Design independence

The CI formula and its weights, thresholds, and scoring breakpoints were derived from NCSS soil survey standards — not from regression or optimization against the AIM match outcome data. This rules out the primary form of overfitting that would invalidate an evaluation.

### Outcome independence

AIM field ecological site IDs (`EcolSite_QC`) are recorded in the field by independent range conservationists. They are not derived from SSURGO or influenced by the SoilID algorithm. The ground-truth outcome is genuinely exogenous.

### Structural coupling

CI and SoilID's predicted ecological site both draw from the same SSURGO map unit. CI's `dominant_score` and `gap_score` describe the top component that SoilID will rank first; SoilID then reports that same component's `ecoclassid`. This is a form of internal consistency rather than circularity — CI correctly describes how certain SSURGO is about that prediction. The evaluation is therefore best characterized as **construct validation** (does CI correctly summarize SSURGO quality?) rather than **external predictive validation** (does CI predict an outcome unrelated to SSURGO?).

### Temporal independence

The SDA queries used to compute CI inputs (R script) and the SDA queries used during `list_soils` (pipeline) both query the live SDA database. If SSURGO data changes between the R enrichment run and the pipeline run, a small number of plots could have inconsistent CI inputs vs. candidate lists. In practice this risk is low over a single-day run window.

---

## 11. Known Limitations

1. **NV-only scope:** All 523 plots are in Nevada (Great Basin). Results may not generalize to different ecological regions with different SSURGO survey coverage and ecological site correlation patterns.

2. **order\_score and mukind\_score not stored in pipeline output:** The Python pipeline does not write `order_score` or `mukind_score` to the run-results CSV. The analysis script reconstructs them from categorical labels (`uncertainty_class` and `uncertainty_reason`). The approximation is conservative and consistent but not exact.

3. **Area-weighted multiplicity not available:** The `multiplicity_score` uses a binary rule — 1 correlated ecosite → 100, 2+ → 30 — treating all correlations equally regardless of area proportion. A dominant series correlated to one primary ecosite (90% of area) and one marginal ecosite (10% of area) receives the same penalty score as a 50/50 split.

4. **1 pipeline failure:** Plot at (−117.46, 39.35) has no SSURGO coverage. It contributes to class counts from the R-enrichment step (its CI inputs exist) but is excluded from all match-rate calculations.

5. **Small per-MLRA Low-class cells:** Low-uncertainty cells within individual MLRAs range from n=1 to n=17. These point estimates carry very wide Wilson intervals and should not be interpreted independently.

6. **SDA API dependency:** All SDA queries are live. Network failures or SDA downtime will silently produce NA rows for affected chunks (logged as warnings). The April 13 run had 0 such failures.
