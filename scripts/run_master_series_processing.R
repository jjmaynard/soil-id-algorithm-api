#!/usr/bin/env Rscript
# Master pipeline for series-name correction + SDA enrichment.
#
# Steps:
# 1) Run Data/aim_data/R_evaluation/scripts/correct_series_names.R to produce
#    Data/aim_data/R_evaluation/data/Supplementary_data1_corrected.xlsx.
# 2) Run Data/aim_data/R_evaluation/scripts/query_soil_survey_order.R with INPUT_XLSX pointing to the
#    corrected workbook.
#
# Inputs:
# - Data/aim_data/R_evaluation/data/Supplementary_data1.xlsx
# - Data/aim_data/R_evaluation/data/manual_qc_soil_series_names.csv (must include PrimaryKey,
#   aim_series_component_name, qc_series_component_name)
#
# Primary outputs from step 1:
# - Data/aim_data/R_evaluation/data/Supplementary_data1_corrected.xlsx
#   (or a timestamped fallback: Supplementary_data1_corrected_run_<UTC>.xlsx)
#
# Primary outputs from step 2 (written by query_soil_survey_order.R):
# - Data/aim_data/R_evaluation/data/study_plot_characteristics.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/points_with_soil_survey_order.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/points_with_soil_uncertainty.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/soil_survey_order_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/soil_survey_order_by_mukind_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/soil_survey_order_by_projectscale_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/soil_uncertainty_class_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/point_landscape_ecosite_comparison.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/landscape_ecosite_match_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/landscape_crosswalk_fuzzy_match_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/landscape_crosswalk_fuzzy_strict_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/landscape_crosswalk_fuzzy_loose_summary.csv
# - Data/aim_data/R_evaluation/outputs/soil_survey_order/series_selection_sensitivity_summary.csv

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) == 0) {
  stop("Unable to determine script path from commandArgs().")
}

script_path <- normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = TRUE)
script_dir <- dirname(script_path)
r_eval_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = TRUE)

correct_script <- file.path(script_dir, "correct_series_names.R")
query_script <- file.path(script_dir, "query_soil_survey_order.R")
corrected_xlsx <- file.path(r_eval_dir, "data", "Supplementary_data1_corrected.xlsx")
alt_corrected_xlsx <- file.path(
  r_eval_dir,
  "data",
  sprintf("Supplementary_data1_corrected_run_%s.xlsx", format(Sys.time(), "%Y%m%dT%H%M%SZ", tz = "UTC"))
)

if (!file.exists(correct_script)) stop("Missing script: ", correct_script)
if (!file.exists(query_script)) stop("Missing script: ", query_script)

message("[1/3] Running series-name correction script...")
res_correct <- system2(
  "Rscript",
  shQuote(correct_script),
  env = c(sprintf("OUT_XLSX=%s", corrected_xlsx))
)

effective_corrected_xlsx <- corrected_xlsx
if (res_correct != 0) {
  message("  Primary corrected output path failed. Retrying with alternate output path...")
  res_correct_retry <- system2(
    "Rscript",
    shQuote(correct_script),
    env = c(sprintf("OUT_XLSX=%s", alt_corrected_xlsx))
  )
  if (res_correct_retry != 0) {
    stop("correct_series_names.R failed on both primary and alternate output paths")
  }
  effective_corrected_xlsx <- alt_corrected_xlsx
}

if (!file.exists(effective_corrected_xlsx)) {
  stop("Expected corrected workbook not found: ", effective_corrected_xlsx)
}

message("[2/3] Running SDA query script with corrected workbook...")
old_input_xlsx <- Sys.getenv("INPUT_XLSX", unset = "")
on.exit({
  Sys.setenv(INPUT_XLSX = old_input_xlsx)
}, add = TRUE)
Sys.setenv(INPUT_XLSX = corrected_xlsx)
Sys.setenv(INPUT_XLSX = effective_corrected_xlsx)

res_query <- system2("Rscript", shQuote(query_script))
if (res_query != 0) {
  stop("query_soil_survey_order.R failed with exit code ", res_query)
}

message("[3/3] Done.")
message("  Corrected workbook: ", effective_corrected_xlsx)
message("  Updated study plots CSV: ", file.path(r_eval_dir, "data", "study_plot_characteristics.csv"))
message("  SDA outputs directory: ", file.path(r_eval_dir, "outputs", "soil_survey_order"))
message("  Key SDA outputs:")
message("    - points_with_soil_survey_order.csv")
message("    - points_with_soil_uncertainty.csv")
message("    - soil_survey_order_summary.csv")
message("    - soil_survey_order_by_mukind_summary.csv")
message("    - soil_survey_order_by_projectscale_summary.csv")
message("    - soil_uncertainty_class_summary.csv")
message("    - point_landscape_ecosite_comparison.csv")
message("    - landscape_ecosite_match_summary.csv")
message("    - landscape_crosswalk_fuzzy_match_summary.csv")
message("    - landscape_crosswalk_fuzzy_strict_summary.csv")
message("    - landscape_crosswalk_fuzzy_loose_summary.csv")
message("    - series_selection_sensitivity_summary.csv")
