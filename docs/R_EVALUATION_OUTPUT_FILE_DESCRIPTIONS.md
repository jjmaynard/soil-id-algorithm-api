# R Evaluation Pipeline Output File Descriptions

This document describes files produced by running Data/aim_data/R_evaluation/scripts/run_master_series_processing.R.

## Pipeline Output Overview

### Stage 1: Series-name correction

| File | Description |
|---|---|
| Data/aim_data/R_evaluation/data/Supplementary_data1_corrected.xlsx | Corrected version of Supplementary_data1.xlsx after standardized series-name cleanup. |
| Data/aim_data/R_evaluation/data/Supplementary_data1_corrected_run_<UTC>.xlsx | Fallback corrected workbook path if primary output path is unavailable. |

### Stage 2: Soil survey order and uncertainty enrichment

Output directory: Data/aim_data/R_evaluation/outputs/soil_survey_order

| File | Description |
|---|---|
| points_with_soil_survey_order.csv | Plot-level dataset with soil survey order fields appended. |
| points_with_soil_uncertainty.csv | Plot-level dataset with uncertainty class and reason fields appended. |
| soil_survey_order_summary.csv | Overall frequency summary of soil survey order classes. |
| soil_survey_order_by_mukind_summary.csv | Soil survey order summary grouped by mapunit kind. |
| soil_survey_order_by_projectscale_summary.csv | Soil survey order summary grouped by project scale. |
| soil_uncertainty_class_summary.csv | Summary counts and percentages by uncertainty class. |
| point_landscape_ecosite_comparison.csv | Wide plot-level comparison table joining AIM, QC, and SDA ecosystem/landscape fields. |
| landscape_ecosite_match_summary.csv | Match-rate summary for ecological site and landscape comparisons. |
| landscape_crosswalk_fuzzy_match_summary.csv | Summary for fuzzy matching against landscape crosswalk labels. |
| landscape_crosswalk_fuzzy_strict_summary.csv | Strict fuzzy-match summary variant. |
| landscape_crosswalk_fuzzy_loose_summary.csv | Loose fuzzy-match summary variant. |
| series_selection_sensitivity_summary.csv | Sensitivity summary for alternate component-series selection behavior. |

Also updated in data folder:

| File | Description |
|---|---|
| Data/aim_data/R_evaluation/data/study_plot_characteristics.csv | Canonical study plot table used by later pipeline stages. |

### Stage 3: AIM horizon filtering

| File | Description |
|---|---|
| Data/aim_data/R_evaluation/data/AIM_Data/study_soil_horizons.csv | Filtered horizon table used by SoilID runtime, including layer quality handling. |
| Data/aim_data/R_evaluation/data/AIM_Data/aim_filter_diagnostics.csv | Diagnostics per plot for filtering/exclusion decisions. |
| Data/aim_data/R_evaluation/data/AIM_Data/aim_filter_summary.txt | Text summary of filter counts and key outcomes. |

### Stage 4: SoilID runtime outputs

Output directory: Data/aim_data/R_evaluation/outputs/aim_qc

Per run timestamp:

| Pattern | Description |
|---|---|
| study_plot_characteristics_enriched_run_results_<TIMESTAMP>.csv | Core per-plot runtime result table (expected vs baseline/terrain predictions, matches, CI inputs). |
| study_plot_characteristics_enriched_run_summary_<TIMESTAMP>.txt | Human-readable run summary. |
| study_plot_characteristics_enriched_run_summary_<TIMESTAMP>.json | Machine-readable run summary and run metadata. |
| study_plot_characteristics_enriched_run_results_latest.csv | Convenience copy of most recently generated run-results CSV. |

### Stage 5: CI revised analysis outputs

Per run timestamp:

| Pattern | Description |
|---|---|
| *_ci_revised_rows.csv | Per-plot rows with recalculated confidence index, uncertainty class/reason, and helper features. |
| *_ci_revised_class_summary.csv | Counts by revised uncertainty class. |
| *_ci_revised_reason_summary.csv | Counts by revised uncertainty reason. |
| *_ci_revised_wilson.csv | Wilson confidence intervals for match rates by class (baseline and terrain). |
| *_ci_revised_calibration.csv | Calibration table by CI decile. |
| *_ci_revised_summary.txt | Statistical summary (chi-square and Spearman correlations) and output manifest. |

### Stage 6: MLRA clustered analysis outputs

Per run timestamp:

| Pattern | Description |
|---|---|
| *_mlra_cluster_models.csv | Model coefficients/results for MLRA-clustered analysis. |
| *_mlra_base_rate_by_mlra.csv | Baseline rates summarized by MLRA. |
| *_mlra_cluster_summary.txt | Text summary of MLRA-cluster analysis. |

### Stage 7: CI component model outputs

Per run timestamp:

| Pattern | Description |
|---|---|
| *_ci_component_models.csv | Fitted component-level model outputs. |
| *_ci_component_importance.csv | Variable importance table. |
| *_ci_component_collinearity.csv | Collinearity diagnostics among predictors. |
| *_ci_component_model_comparison.csv | Comparison of component-model variants. |
| *_ci_component_summary.txt | Text summary of component-model findings. |
| study_plot_characteristics_enriched_run_results_latest_ci_component_*.csv/txt | Convenience copies for latest component-model outputs. |

### Stage 8/9: Concise plot-level comparison output

| File Pattern | Description |
|---|---|
| study_plot_characteristics_enriched_plot_level_matches_<TIMESTAMP>.csv | Compact plot-level table containing AIM and QC expected series/ecosite versus terrain series/ecosite and TRUE/FALSE match flags. |

---

## Column Descriptions: study_plot_characteristics_enriched_run_results_20260414T031040Z_ci_revised_rows.csv

Source file:
Data/aim_data/R_evaluation/outputs/aim_qc/study_plot_characteristics_enriched_run_results_20260414T031040Z_ci_revised_rows.csv

This file is the Stage 5 per-row CI-revised output. It contains original Stage 4 result columns plus derived CI-revision fields.

| Column | Description |
|---|---|
| PrimaryKey | Unique plot identifier used as the primary join key. |
| source | Runtime processing source label (for example API/local mode context). |
| status | Runtime status for the plot (success/failure class). |
| error | Error message text if runtime evaluation failed for this plot. |
| mlrasymbol | Normalized MLRA symbol used in analysis joins and scoring context. |
| confidence_index | Original confidence index from Stage 4 output. |
| uncertainty_class | Original uncertainty class from Stage 4 output. |
| uncertainty_reason | Original uncertainty reason from Stage 4 output. |
| dominant_comppct_r | Dominant component percentage used in confidence scoring. |
| second_comppct_r | Second-ranked component percentage. |
| component_gap | Difference between dominant and second component percentages. |
| n_ecosites_dominant | Number of ecosites associated with dominant series/component context. |
| multiplicity_score | Original multiplicity score input to confidence framework. |
| expected_soil_series | Expected soil series reference value used for evaluation. |
| expected_ecological_site | Expected ecological site reference value used for evaluation. |
| expected_landscape_type | Expected landscape type reference value used for evaluation. |
| expected_landscape_class | Expected landscape class reference value used for evaluation. |
| expected_rank_baseline | Rank position of expected series in baseline ranking output. |
| expected_rank_terrain | Rank position of expected series in terrain-aware ranking output. |
| expected_component_id_baseline | Component identifier tied to expected baseline selection. |
| expected_component_id_terrain | Component identifier tied to expected terrain selection. |
| expected_sda_ecological_site | SDA ecological site attached to expected component context. |
| expected_sda_landscape_type | SDA landscape type attached to expected component context. |
| expected_sda_landscape_class | SDA landscape class attached to expected component context. |
| baseline_soil_series | Baseline selected soil series (without terrain adjustment). |
| baseline_ecological_site | Baseline selected ecological site. |
| baseline_landscape_class | Baseline selected landscape class. |
| terrain_aim_soil_series | Terrain-aware selected soil series for AIM pathway. |
| terrain_aim_ecological_site | Terrain-aware selected ecological site for AIM pathway. |
| terrain_aim_landscape_class | Terrain-aware selected landscape class for AIM pathway. |
| baseline_component_id | Baseline selected component identifier. |
| terrain_aim_component_id | Terrain-aware AIM selected component identifier. |
| baseline_aim_soil_series_match | Baseline AIM soil-series match flag versus expected AIM value. |
| baseline_aim_ecological_site_match | Baseline AIM ecological-site match flag versus expected AIM value. |
| baseline_aim_landscape_class_match | Baseline AIM landscape-class match flag versus expected AIM value. |
| terrain_aim_soil_series_match | Terrain AIM soil-series match flag versus expected AIM value. |
| terrain_aim_ecological_site_match | Terrain AIM ecological-site match flag versus expected AIM value. |
| terrain_aim_landscape_class_match | Terrain AIM landscape-class match flag versus expected AIM value. |
| top_changed | Indicator that top-ranked selection changed between baseline and terrain pathways. |
| aim_expected_soil_series | AIM expected soil series (normalized evaluation input). |
| aim_expected_ecological_site | AIM expected ecological site (normalized evaluation input). |
| aim_expected_landscape_class | AIM expected landscape class (normalized evaluation input). |
| qc_expected_soil_series | QC expected soil series (normalized evaluation input). |
| qc_expected_ecological_site | QC expected ecological site (normalized evaluation input). |
| qc_expected_landscape_class | QC expected landscape class (normalized evaluation input). |
| landscape_class_qc_changed | Indicator that QC process changed landscape class from AIM entry. |
| any_qc_changed | Indicator that QC changed one or more expected target attributes. |
| baseline_qc_soil_series_match | Baseline soil-series match flag versus QC expected series. |
| baseline_qc_ecological_site_match | Baseline ecological-site match flag versus QC expected site. |
| baseline_qc_landscape_class_match | Baseline landscape-class match flag versus QC expected class. |
| terrain_qc_soil_series | Terrain-aware selected soil series for QC pathway. |
| terrain_qc_ecological_site | Terrain-aware selected ecological site for QC pathway. |
| terrain_qc_landscape_class | Terrain-aware selected landscape class for QC pathway. |
| terrain_qc_component_id | Terrain-aware QC selected component identifier. |
| terrain_qc_soil_series_match | Terrain QC soil-series match flag versus QC expected series. |
| terrain_qc_ecological_site_match | Terrain QC ecological-site match flag versus QC expected site. |
| terrain_qc_landscape_class_match | Terrain QC landscape-class match flag versus QC expected class. |
| aim_qc_soil_series_match | Indicator that AIM expected series equals QC expected series. |
| aim_qc_ecological_site_match | Indicator that AIM expected ecological site equals QC expected ecological site. |
| aim_qc_landscape_class_match | Indicator that AIM expected landscape class equals QC expected landscape class. |
| join_key | Join key created for stable joins across runtime and plot metadata tables. |
| compname_norm | Normalized component/series name used for multiplicity and lookup joins. |
| dom | Numeric cast of dominant_comppct_r used in revised CI scoring. |
| gap | Numeric cast of component_gap used in revised CI scoring. |
| ci_old | Numeric cast of original confidence_index. |
| order_score | Revised CI subscore based on map-order/uncertainty context. |
| mukind_score | Revised CI subscore reflecting mapunit complexity/undifferentiated conditions. |
| dom_score_new | Revised CI subscore from dominant component strength thresholds. |
| gap_score_new | Revised CI subscore from dominant-versus-second component gap thresholds. |
| multiplicity_score_new | Revised CI subscore for ecosite multiplicity behavior. |
| ci_new | Recomputed confidence index from weighted subscore model. |
| uc_new | Recomputed uncertainty class derived from ci_new thresholds. |
| reason_new | Recomputed uncertainty reason derived from revised rule set. |
| match | Binary baseline QC ecological-site match indicator used in revised statistics. |
| match_t | Binary terrain QC ecological-site match indicator used in revised statistics. |
| ci_new_decile | Decile bin index for ci_new used in calibration analysis. |

---

## Notes

- The outputs/aim_qc directory can also contain additional post-processing artifacts (for example lookup validation reports or API-call payload exports) that are not direct run_master_series_processing.R stage outputs.
- The latest run convenience files are useful for dashboards, but timestamped files should be used for reproducible analysis snapshots.
