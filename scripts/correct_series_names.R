#!/usr/bin/env Rscript
# =============================================================================
# Data/aim_data/R_evaluation/scripts/correct_series_names.R
#
# Corrects "Soil Series_AIM" and "Soil_Series_QC" columns of
# Supplementary_data1.xlsx (Sheet1) using manual_qc_soil_series_names.csv.
#
# Design:
#   - The ORIGINAL file is NEVER modified.
#   - Only the cells requiring correction are updated via targeted cell writes;
#     all other data, formatting, formulas, and sheets are preserved via
#     openxlsx::loadWorkbook.
#   - Corrections are applied by PrimaryKey row lookup, not by global value
#     replacement. This avoids ambiguity for names that appear in multiple
#     contexts.
#
# Usage:
#   Rscript Data/aim_data/R_evaluation/scripts/correct_series_names.R
#
# Inputs:
#   Data/aim_data/R_evaluation/data/Supplementary_data1.xlsx        (original, read-only)
#   Data/aim_data/R_evaluation/data/manual_qc_soil_series_names.csv (correction reference)
#
# Output:
#   Data/aim_data/R_evaluation/data/Supplementary_data1_corrected.xlsx
# =============================================================================

# ── 0. Dependencies ────────────────────────────────────────────────────────────
needed   <- c("readxl", "openxlsx", "dplyr", "readr", "purrr")
missing  <- needed[!sapply(needed, requireNamespace, quietly = TRUE)]
if (length(missing)) {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(lapply(needed, library, character.only = TRUE))

# ── 1. Paths ───────────────────────────────────────────────────────────────────
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) == 0) {
  stop("Unable to determine script path from commandArgs().")
}

script_path <- normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/", mustWork = TRUE)
script_dir <- dirname(script_path)
r_eval_dir <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = TRUE)
DATA_DIR <- file.path(r_eval_dir, "data")

EXCEL_IN  <- file.path(DATA_DIR, "Supplementary_data1.xlsx")
CSV_REF   <- file.path(DATA_DIR, "manual_qc_soil_series_names.csv")

excel_out_env <- Sys.getenv("OUT_XLSX", unset = "")
if (nzchar(excel_out_env)) {
  EXCEL_OUT <- ifelse(
    grepl("^(/|[A-Za-z]:)", excel_out_env),
    excel_out_env,
    file.path(r_eval_dir, excel_out_env)
  )
} else {
  EXCEL_OUT <- file.path(DATA_DIR, "Supplementary_data1_corrected.xlsx")
}

if (!file.exists(EXCEL_IN))  stop("Input Excel not found: ", EXCEL_IN)
if (!file.exists(CSV_REF))   stop("Reference CSV not found: ", CSV_REF)

# ── 2. Load reference and build lookup tables ──────────────────────────────────
message("\n[1/4] Loading reference corrections from CSV ...")

# readr handles the UTF-8 BOM silently
ref <- read_csv(CSV_REF, show_col_types = FALSE)

required_ref_cols <- c("PrimaryKey", "aim_series_component_name", "qc_series_component_name")
missing_ref_cols <- setdiff(required_ref_cols, names(ref))
if (length(missing_ref_cols) > 0) {
  stop("Reference CSV is missing required columns: ", paste(missing_ref_cols, collapse = ", "))
}

ref_by_pk <- ref %>%
  transmute(
    PrimaryKey = trimws(as.character(PrimaryKey)),
    aim_series_component_name = as.character(aim_series_component_name),
    qc_series_component_name = as.character(qc_series_component_name)
  ) %>%
  filter(!is.na(PrimaryKey), PrimaryKey != "")

dup_pk <- ref_by_pk %>%
  count(PrimaryKey, name = "n") %>%
  filter(n > 1)

if (nrow(dup_pk) > 0) {
  message(sprintf(
    "  [WARNING] Found %d duplicate PrimaryKey entries in reference CSV. Using last occurrence.",
    nrow(dup_pk)
  ))
}

ref_by_pk <- ref_by_pk %>%
  mutate(row_id = row_number()) %>%
  group_by(PrimaryKey) %>%
  slice_tail(n = 1) %>%
  ungroup() %>%
  select(-row_id)

message(sprintf("  Loaded %d unique PrimaryKey correction rows", nrow(ref_by_pk)))

# ── 3. Identify which cells need updating ──────────────────────────────────────
message("\n[2/4] Reading Sheet1 and identifying cells to update ...")

# Use readxl so column names with spaces are preserved correctly
AIM_COL <- "Soil Series_AIM"   # has a space (not an underscore) in the Excel file
QC_COL  <- "Soil_Series_QC"

s1 <- read_excel(EXCEL_IN, sheet = 1)

if (!AIM_COL %in% names(s1)) stop("Column '", AIM_COL, "' not found in Sheet1.")
if (!QC_COL  %in% names(s1)) stop("Column '", QC_COL,  "' not found in Sheet1.")
if (!"PrimaryKey" %in% names(s1)) stop("Column 'PrimaryKey' not found in Sheet1.")

# Column indices (1-based) needed for targeted openxlsx cell writes
col_idx     <- setNames(seq_along(names(s1)), names(s1))
aim_col_idx <- col_idx[[AIM_COL]]
qc_col_idx  <- col_idx[[QC_COL]]

# Join correction values by PrimaryKey so updates are row-specific
s1_joined <- s1 %>%
  mutate(PrimaryKey = trimws(as.character(PrimaryKey))) %>%
  left_join(ref_by_pk, by = "PrimaryKey")

aim_new <- ifelse(
  !is.na(s1_joined$aim_series_component_name) & trimws(s1_joined$aim_series_component_name) != "",
  s1_joined$aim_series_component_name,
  as.character(s1[[AIM_COL]])
)

qc_new <- ifelse(
  !is.na(s1_joined$qc_series_component_name) & trimws(s1_joined$qc_series_component_name) != "",
  s1_joined$qc_series_component_name,
  as.character(s1[[QC_COL]])
)

# Rows where the value is actually different
aim_changed <- which(!is.na(aim_new) & aim_new != as.character(s1[[AIM_COL]]))
qc_changed  <- which(!is.na(qc_new)  & qc_new  != as.character(s1[[QC_COL]]))

n_rows_without_ref <- sum(is.na(s1_joined$aim_series_component_name) & is.na(s1_joined$qc_series_component_name))
message(sprintf("  Rows without PrimaryKey match in reference CSV: %d", n_rows_without_ref))

message(sprintf("  Cells to update — AIM: %d,  QC: %d",
                length(aim_changed), length(qc_changed)))

if (length(aim_changed) > 0) {
  message("  AIM corrections:")
  walk(aim_changed, ~ message(sprintf(
    "    Excel row %4d:  %-35s ->  %s",
    .x + 1L,
    paste0("'", s1[[AIM_COL]][.x], "'"),
    aim_new[.x]
  )))
}

if (length(qc_changed) > 0) {
  message("  QC corrections:")
  walk(qc_changed, ~ message(sprintf(
    "    Excel row %4d:  %-35s ->  %s",
    .x + 1L,
    paste0("'", s1[[QC_COL]][.x], "'"),
    qc_new[.x]
  )))
}

# ── 4. Load workbook and apply targeted cell updates ──────────────────────────
message("\n[3/4] Loading original workbook (preserving formatting and all sheets) ...")
wb <- loadWorkbook(EXCEL_IN)

message("[4/4] Writing corrected cells and saving output ...")

# Excel row = data row index + 1 (header occupies row 1)
for (i in aim_changed) {
  writeData(wb, sheet = 1,
            x        = as.character(aim_new[i]),
            startRow = i + 1L,
            startCol = aim_col_idx,
            colNames = FALSE)
}

for (i in qc_changed) {
  writeData(wb, sheet = 1,
            x        = as.character(qc_new[i]),
            startRow = i + 1L,
            startCol = qc_col_idx,
            colNames = FALSE)
}

saveWorkbook(wb, EXCEL_OUT, overwrite = TRUE)

message(sprintf(
  "\nDone.\n  Original preserved : %s\n  Corrected output   : %s\n  Changes applied    : %d AIM + %d QC = %d cells",
  EXCEL_IN, EXCEL_OUT,
  length(aim_changed), length(qc_changed),
  length(aim_changed) + length(qc_changed)
))
