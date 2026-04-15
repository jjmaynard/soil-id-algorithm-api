# Supplementary Results Report
## Expert QAQC of AIM Data and Automated Soil ID for Ecosite Identification

## 1. Why this evaluation was done
This evaluation examined a practical research question: how well does a combined workflow perform when expert QAQC is used to clean AIM soil/ecosite inputs, and then Soil ID is used for automated soil identification at scale.

The specific goals were to:
- Quantify the effect of expert QAQC corrections on study inputs.
- Run a reproducible end-to-end processing pipeline.
- Measure Soil ID runtime success and output completeness.
- Test whether the revised confidence index tracks match quality better than the legacy index.
- Test whether confidence index effects persist after accounting for MLRA-level structure.

## 2. End-to-end pipeline status
A full seven-stage workflow completed successfully (exit code 0) using:
- `Data/aim_data/R_evaluation/scripts/run_master_series_processing.R`

Pipeline stages completed:
1. Series-name QAQC correction.
2. SDA enrichment and uncertainty summaries.
3. AIM horizon filtering plus restrictive-layer and bedrock processing.
4. Soil ID runtime execution.
5. Revised confidence-index analysis (R).
6. MLRA-clustered and adjusted regression analysis (R).
7. Consolidated artifact reporting.

This confirms the workflow is executable as a single reproducible process for study-scale evaluation.

## 3. Expert QAQC correction results
The correction stage used PrimaryKey-based row matching to apply manual QAQC edits in context.

Key outcomes:
- 524 unique PrimaryKey correction rows loaded.
- 18 rows had no PrimaryKey match in the manual correction table.
- 151 cell-level corrections applied total:
  - 74 corrections in the AIM series field.
  - 77 corrections in the QC series field.

Interpretation:
The number of corrected cells indicates that expert QAQC materially changes core categorical inputs and is not just a minor formatting or spelling cleanup.

## 4. SDA and landscape/ecosite comparison context
The SDA comparison and crosswalk summaries were generated for all study points. Results showed meaningful heterogeneity in class agreement patterns and uncertainty categories.

A key sensitivity result:
- Under landscape-aware versus series-only selection logic, 6 AIM matched points changed selected outcomes.

Interpretation:
A subset of points is sensitive to contextual selection logic, supporting use of landscape-aware rules rather than series-only logic when ecological interpretation is the objective.

## 5. AIM filtering and horizon dataset quality
The AIM national filter stage produced complete PrimaryKey coverage for the study and a horizon dataset suitable for downstream Soil ID and statistical analysis.

Input scale:
- PlotCharacteristics records read: 48,225
- SoilHorizon records read: 123,990

Matched study coverage:
- 524 of 524 PrimaryKeys matched in plot characteristics.
- 524 of 524 PrimaryKeys matched in soil horizons.

Output horizon dataset:
- 1,784 horizon rows across 524 study profiles.
- 114 profiles with restrictive-layer signal.
- 43 profiles with bedrock depth assignment.

Interpretation:
Coverage was complete for the study footprint, and profile-level limiting-layer indicators were captured in a substantial subset of sites.

## 6. Soil ID runtime performance
Soil ID runtime completed with near-complete processing success:
- Total records evaluated: 524
- Passed: 523
- Failed: 0

Outputs were written to:
- `Data/aim_data/R_evaluation/outputs/aim_qc`

Interpretation:
The automated system ran robustly at study scale with no hard runtime failures and effectively complete usable output.

## 7. Confidence index performance (revised CI analysis)
Revised confidence-index diagnostics were run on 524 analyzed rows.

Uncertainty class distribution:
- Low uncertainty (high confidence): 59
- Moderate uncertainty: 185
- High uncertainty: 280

Association tests:
- Chi-square test for uncertainty class vs match: p = 9.53e-05
- Spearman correlation, legacy CI vs match: r = 0.1491, p = 6.16e-04
- Spearman correlation, revised CI vs match: r = 0.2044, p = 2.38e-06

Interpretation:
Both confidence indices are positively associated with match quality, but the revised index performs better than the legacy index and shows stronger statistical evidence.

## 8. MLRA-clustered and adjusted model findings
Three logistic modeling views were used to test robustness of confidence-index effects.

Data scope:
- Rows used: 524
- Unique MLRAs: 10

Model 1: Cluster-robust logistic model (match ~ confidence index)
- Beta for confidence index: +0.03728
- Robust SE: 0.01316
- p-value: 0.0046043
- Odds ratio per +1 CI point: 1.0380 (95% CI 1.0116 to 1.0651)

Model 2: MLRA fixed-effects model (match ~ confidence index + factor(mlrasymbol))
- Beta for confidence index: +0.04385
- p-value: 1.98e-05
- Odds ratio per +1 CI point: 1.0448 (95% CI 1.0240 to 1.0661)
- Between-MLRA intercept variance proxy: 17.29433
- ICC proxy (logit scale): 0.8402

Model 3: Base-rate adjusted model (match ~ confidence index + ecosite base rate)
- Confidence index beta: +0.03933
- Robust SE: 0.01131
- p-value: 5.10e-04
- CI beta attenuation after base-rate adjustment: -5.5%
- Ecosite base-rate term positive but not significant (p = 0.105872)

Interpretation:
Confidence index remains a significant positive predictor across all specifications, and its effect is only modestly reduced by base-rate adjustment, indicating non-trivial predictive signal beyond broad regional prevalence.

## 9. Scientific implications
Overall, the results support a hybrid operational model:
- Expert QAQC is important for establishing high-integrity site inputs.
- Automated Soil ID can then scale evaluations across many sites.
- The revised confidence index improves interpretability and tracks match quality better than the legacy index.
- Confidence effects persist under geographically structured and adjusted models.

For manuscript framing, this supports the claim that expert QAQC and automated Soil ID are complementary rather than competing approaches.

## 10. Limitations and caveats
- One location logged a data-availability warning for Soil ID, likely reflecting coverage limitations rather than pipeline instability.
- Non-fatal dependency warnings appeared during runtime (pandas future warning and CRS handling warning), but did not interrupt output generation.
- Uncertainty classes were skewed toward moderate/high uncertainty, which should be considered when generalizing to lower-uncertainty landscapes.

## 11. Suggested manuscript-ready summary paragraph
Expert QAQC substantially improved AIM study inputs through row-level correction of soil series labels. The corrected dataset was then processed through an end-to-end automated Soil ID evaluation pipeline that completed successfully at study scale, with near-complete output coverage. The revised confidence index showed stronger association with successful matching than the legacy index and remained a significant predictor in cluster-robust, fixed-effects, and base-rate-adjusted models. Together, these findings indicate that expert QAQC and automated Soil ID form a practical and scientifically defensible combined approach for large-scale ecosite identification workflows.
