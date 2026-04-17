# Soil Map Information Uncertainty and Ecological Site Match Accuracy

## Objective
This report summarizes methods and results from a reproducible workflow designed to evaluate one primary question:

How strongly does soil map information uncertainty affect ecological site match accuracy?

The evaluation combines expert QAQC of AIM inputs, SDA enrichment, automated Soil ID runs, and statistical testing. The assessment framework integrates three lines of evidence: uncertainty-class patterns, confidence index relationships with match outcomes, and MLRA-aware regression results.

## Study Design Overview
The analysis was run through an eight-stage pipeline:

1. Apply expert QAQC corrections to soil series names at the row (PrimaryKey) level.
2. Enrich points with Soil Data Access (SDA) outputs and uncertainty diagnostics.
3. Build filtered AIM horizon products and derive restrictive-layer/bedrock indicators.
4. Run automated Soil ID predictions for each study point.
5. Evaluate uncertainty classes and confidence index behavior, with confidence index used as one supporting metric.
6. Fit MLRA-aware regression models to test robustness of uncertainty effects.
7. Fit CI component-family logistic models (Stage 7) to compare `ci_only`, `components`, `order_mukind`, `dom_gap`, and `interaction` formulations.
8. Export tables and summaries for interpretation.

Pipeline entry point used:
- `Data/aim_data/R_evaluation/scripts/run_master_series_processing.R`

## Data Inputs
Primary inputs:
- QAQC workbook and correction table for series-name updates.
- SDA outputs linked to study points.
- AIM national PlotCharacteristics and SoilHorizon records.
- Soil ID runtime outputs from enriched study plot records.

Scale of analysis from the completed run:
- Study points analyzed: 524
- Soil ID result rows analyzed: 524
- Unique MLRAs in model-based analyses: 10

## Methods

### 1. QAQC correction method
Manual QAQC edits were applied by PrimaryKey row matching rather than global typo substitution. This preserves context-specific corrections where similar string values may require different replacements at different sites.

Observed QAQC update magnitude:
- 151 corrected cells total
- 74 AIM field corrections
- 77 QC field corrections

QAQC correction subtype breakdown (applied edits):
- 75 typo/content corrections
- 72 formatting-only corrections
- 4 missing-to-filled corrections

Subtype definitions used for reporting:
- Formatting-only: old/new values become equivalent after case-folding and removal of punctuation/spacing separators.
- Typo/content: old/new values remain different after normalization and represent true string/content change.
- Missing-to-filled: blank/NA-like entries replaced with a populated soil series value.

### 2. Uncertainty-focused analysis variables
The uncertainty evaluation used:
- Categorical uncertainty class (low, moderate, high)
- Confidence index
- Binary ecological site match outcome

The confidence index is a composite score that summarizes the strength and consistency of map and profile evidence supporting a candidate ecological site match, with higher values indicating stronger support.

### 3. Statistical tests
Three layers of inference were used:

1. Class-level association:
- Chi-square test of uncertainty class versus match outcome.

2. Rank association:
- Spearman correlation between confidence index and match outcome.

3. MLRA-aware predictive models:
- Cluster-robust logistic regression: match ~ confidence index, clustered by MLRA.
- MLRA fixed-effects logistic regression: match ~ confidence index + factor(mlra).
- Base-rate adjusted logistic model: match ~ confidence index + ecosite base rate.

4. CI component-family comparison (Stage 7):
- Logistic model family comparison using AIC/BIC, McFadden pseudo-R², Brier, and logloss.
- Robust coefficient tables for component terms.
- Permutation and drop-one importance diagnostics.
- Collinearity diagnostics (VIF).

These models were used to determine whether confidence index effects persist after accounting for regional structure and baseline ecological-site prevalence.

## Results

### 1. Data completeness and run stability
- Full pipeline completed successfully (exit code 0).
- Soil ID runtime summary: total=524, passed=523, failed=0.
- All 524 study PrimaryKeys were matched in filtered AIM plot/horizon outputs.

Interpretation:
The dataset and pipeline were sufficiently complete to support uncertainty-to-accuracy inference across nearly the full study footprint.

### 2. Uncertainty class distribution
Rows analyzed in current CI stage: 524

Class counts:
- Low uncertainty (high confidence): 59
- Moderate uncertainty: 185
- High uncertainty: 280

Interpretation:
The sample is weighted toward moderate/high uncertainty conditions, which is important context for interpreting effect size and practical utility.

### 3. Class-level association with match accuracy
- Chi-square p-value for uncertainty class versus match outcome: 9.52728e-05

Interpretation:
Match accuracy differs significantly across uncertainty classes, indicating that map-information uncertainty categories carry meaningful predictive information about ecological site match success.

### 4. Confidence index as one line of evidence
Spearman association between the current confidence index and match:
- Current CI: r=0.2044, p=2.37732e-06

Interpretation:
Confidence index is positively associated with match accuracy and supports uncertainty-informed ranking of likely ecological site matches.

### 5. MLRA-clustered and adjusted model results

Cluster-robust model (match ~ CI):
- beta(CI)=+0.03728
- robust SE=0.01316
- p=0.0046043
- OR per +1 CI point=1.0380 (95% CI 1.0116 to 1.0651)

MLRA fixed-effects model:
- beta(CI)=+0.04385
- p=1.98052e-05
- OR per +1 CI point=1.0448 (95% CI 1.0240 to 1.0661)

Base-rate adjusted model:
- beta(CI)=+0.03933
- robust SE=0.01131
- p=0.000509811
- CI beta attenuation after adding base rate: -5.5%
- ecosite base-rate coefficient positive, not statistically significant (p=0.105872)

Interpretation:
The confidence index remains a significant positive predictor of match accuracy across all model formulations. The small attenuation after base-rate adjustment suggests CI captures information beyond simple regional base prevalence.

### 6. Stage 7 CI component-model results (updated)

Stage 7 was run on:
- `Data/aim_data/R_evaluation/outputs/aim_qc/study_plot_characteristics_enriched_run_results_20260414T163452Z.csv`

Scope:
- Rows used: 524
- Unique MLRAs: 10

Model-family comparison summary:
- Best AIC: `components` (AIC = 700.50)
- Best predictive error (logloss/Brier): `interaction` and `components` were nearly identical, with `interaction` marginally lowest logloss.
- Weakest family: `order_mukind` (highest AIC/logloss, lowest pseudo-R²).

Top predictors in the full component model:
- `dom_score_new` (largest degradation when permuted or dropped)
- `multiplicity_score` (second largest degradation)

Diagnostic stability:
- Max VIF among component predictors: 2.71 (low collinearity)

Interpretation:
- CI decomposition confirms that dominance clarity and ecosite multiplicity ambiguity are the strongest structural uncertainty drivers.
- Full component models improve fit over `ci_only` in this updated run, but gains are modest in absolute prediction error terms.
- These results support using Stage 7 as a structural interpretation layer rather than replacing the primary CI-based inferential models.

## Main Finding Relative to the Objective
From the perspective of assessing soil map information uncertainty effects on ecological site match accuracy:

1. Uncertainty is not neutral; it is significantly associated with match outcomes.
2. Confidence index provides meaningful additional evidence about match accuracy.
3. Confidence effects persist when accounting for MLRA clustering and base-rate structure.
4. Stage 7 decomposition shows the largest uncertainty signal comes from dominant-component clarity (`dom_score_new`) and local ecosite multiplicity (`multiplicity_score`).

Together, these results support a multi-evidence interpretation in which soil map uncertainty patterns, confidence index behavior, and MLRA-aware model results are used jointly to assess ecological site matching outcomes.

## Practical Implications
- Expert QAQC should be retained because it materially improves categorical soil inputs used in matching.
- Automated Soil ID can then be used at scale with high operational reliability.
- Reporting the current confidence index alongside other uncertainty diagnostics provides a scientifically useful signal for interpretation and prioritization.
- Stage 7 component outputs should be used to explain *why* confidence is high/low at a site, especially where dominant-component strength and ecosite multiplicity disagree.

## Limitations
- One run-time warning indicated localized data/service availability constraints for at least one location.
- Dependency warnings (for example, future/deprecation warnings) were non-fatal but indicate maintenance work is advisable.
- Class imbalance toward moderate/high uncertainty should be considered when generalizing to different landscapes.

## Reproducibility and Key Artifacts
Primary script:
- `Data/aim_data/R_evaluation/scripts/run_master_series_processing.R`

Key output locations:
- Soil survey / uncertainty summaries: `Data/aim_data/R_evaluation/outputs/soil_survey_order`
- AIM filtered horizon outputs: `Data/aim_data/R_evaluation/data/AIM_Data`
- Soil ID run and CI/model outputs: `Data/aim_data/R_evaluation/outputs/aim_qc`

Representative run-results file used for downstream stages:
- `Data/aim_data/R_evaluation/outputs/aim_qc/study_plot_characteristics_enriched_run_results_20260414T163452Z.csv`
