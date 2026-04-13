# AIM/QC Matching Evaluation: Revised Approach and Recent Changes

## Purpose
This document summarizes the recent implementation changes and defines the revised approach for evaluating AIM/QC soil series, ecological site, and landscape matching.

## What Changed

### 1. Confidence Index (CI) logic was revised in the SDA uncertainty pipeline
Updated file: `scripts/query_soil_survey_order.R`

Key updates:
- Added `mlrasymbol` and `dominant_compname` to SDA output.
- Updated dominant component scoring to NCSS-aligned breakpoints:
  - `dom < 50 -> 20`
  - `50 <= dom < 80 -> 55`
  - `dom >= 80 -> 100`
- Updated component gap scoring breakpoints:
  - `gap < 20 -> 30`
  - `20 <= gap < 40 -> 60`
  - `gap >= 40 -> 90`
- Added ecosite multiplicity as an explicit CI component:
  - Join on `(dominant_compname_norm, mlrasymbol)`
  - `n_ecosites_dominant >= 2 -> multiplicity_score = 30`
  - `n_ecosites_dominant == 1 -> multiplicity_score = 100`
  - missing lookup -> `multiplicity_score = 50`
- Revised CI weights:
  - `order_score: 0.20`
  - `mukind_score: 0.10`
  - `dominant_score: 0.35`
  - `gap_score: 0.20`
  - `multiplicity_score: 0.15`
- Updated uncertainty reason threshold:
  - weak dominant component now uses `< 80`.
- Added output fields:
  - `mlrasymbol`
  - `dominant_compname`
  - `n_ecosites_dominant`
  - `multiplicity_score`

### 2. Multiplicity lookup builder was added
New file: `scripts/build_ecosite_multiplicity_lookup.py`

What it does:
- Reads all `Data/aim_data/*_compname_ecosite_raw_pairs.csv` files.
- Normalizes component names and ecosite IDs.
- Computes distinct ecosite counts by `(compname_norm, mlrasymbol)`.
- Writes:
  - `Data/aim_data/compname_mlra_ecosite_multiplicity.csv`

Observed run result:
- Output created with `33,051` rows.

### 3. AIM run output metadata was extended
Updated file: `scripts/run_all_aim_examples.py`

`_extract_plot_metadata()` now carries through:
- `mlrasymbol`
- `n_ecosites_dominant`
- `multiplicity_score`

This allows downstream analysis scripts to consume uncertainty structure directly from run outputs when present.

### 4. Revised reporting script was upgraded
Updated file: `scripts/_analyze_ci_revised.py`

Enhancements:
- Updated reconstructed CI logic to match revised scoring approach.
- Supports multiplicity-aware CI reconstruction when `multiplicity_score` exists.
- Adds primary 3-class reporting with Wilson 95% confidence intervals.
- Adds chi-square test for association of match outcome vs uncertainty class.
- Adds per-MLRA class breakdown (when `mlrasymbol` exists in the run-results CSV).

### 5. Clustered MLRA inference script was added and hardened
New file: `scripts/_analyze_ci_mlra_clustered.py`

What it does:
- Fits cluster-robust logistic model:
  - `match ~ confidence_index`, clustered by MLRA.
- Fits MLRA fixed-effects model:
  - `match ~ confidence_index + C(mlrasymbol)`.
- Reports coefficient, robust SE, p-value, and odds ratio.

Compatibility and robustness fixes included:
- Uses `fit(cov_type='cluster', cov_kwds=...)` for broad `statsmodels` compatibility.
- If run-results CSV lacks `mlrasymbol`, script can merge MLRA from
  `Data/aim_data/study_plot_characteristics.csv` using `PrimaryKey`.
- Suppresses misleading ICC/variance proxy output when fixed-effects model does not converge.

### 6. Dependency updates
Updated files:
- `requirements/base.in`
- `requirements.txt`

Added:
- `statsmodels`

## Revised AIM/QC Evaluation Framework

### A. Data generation and enrichment
1. Build ecosite multiplicity lookup:
   - `python scripts/build_ecosite_multiplicity_lookup.py`
2. Generate/refresh uncertainty fields and CI inputs:
   - `Rscript scripts/query_soil_survey_order.R`
3. Run AIM/QC matching batch:
   - `python scripts/run_all_aim_examples.py --plot-csv Data/aim_data/study_plot_characteristics.csv --output-dir Data/aim_data --list-source live`

### B. Primary reporting outputs
1. Run revised CI summary analysis:
   - `python scripts/_analyze_ci_revised.py`
2. Interpret primary tables in this order:
   - 3-class Wilson CI table (primary performance summary)
   - Chi-square association test (class separability)
   - Class-by-reason diagnostics
   - Calibration and rank correlation as secondary diagnostics

### C. MLRA-aware inference
1. Run clustered analysis:
   - `python scripts/_analyze_ci_mlra_clustered.py`
2. Focus on:
   - Cluster-robust `confidence_index` coefficient sign and significance
   - Odds ratio per +1 CI point
   - Fixed-effects coefficient agreement directionally

## Current Known Operational Notes

1. Older run-results files may not include `mlrasymbol`.
   - The clustered script includes fallback merge logic using `PrimaryKey`.
2. Full end-to-end reruns depend on local data paths and available reference files.
3. Fixed-effects logistic models may produce convergence warnings in sparse/high-cardinality MLRA partitions.
   - In that case, cluster-robust model is still the primary inferential result.

## Recommended Interpretation Standard

Use this hierarchy for decisions:
1. **Primary:** 3-class Wilson CI match table + chi-square test.
2. **Primary inferential:** MLRA cluster-robust logistic coefficient on CI.
3. **Secondary diagnostics:** calibration bins, reason-stratified breakdowns, Spearman rank correlation.

This structure reduces over-reliance on a single rank-correlation metric and makes the evaluation more robust to MLRA clustering and uncertainty-class sample imbalance.

## Files Touched in This Revision
- `scripts/query_soil_survey_order.R`
- `scripts/build_ecosite_multiplicity_lookup.py`
- `scripts/run_all_aim_examples.py`
- `scripts/_analyze_ci_revised.py`
- `scripts/_analyze_ci_mlra_clustered.py`
- `requirements/base.in`
- `requirements.txt`
