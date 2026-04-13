#!/usr/bin/env Rscript

# Query SDA for soil survey mapping order (Order 2/3/4/5) at each point location.
# Input expected from Supplementary_data1.xlsx (Sheet1).

required_packages <- c("readxl", "dplyr", "purrr", "soilDB", "tibble")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(purrr)
  library(soilDB)
  library(tibble)
})

project_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
input_xlsx <- file.path(project_dir, "data", "raw", "Supplementary_data1.xlsx")
if (!file.exists(input_xlsx)) {
  input_xlsx <- file.path(project_dir, "Data", "aim_data", "Supplementary_data1.xlsx")
}
input_sheet <- "Sheet1"
id_col <- "PrimaryKey"
lon_col <- "Longitude_NAD83"
lat_col <- "Latitude_NAD83"
chunk_size <- 100
buffer_m <- 1000

output_dir <- file.path(project_dir, "outputs", "soil_survey_order")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Also write the key point-level output to Data/aim_data/ for use by the Python pipeline
aim_data_dir <- file.path(project_dir, "Data", "aim_data")
output_study_plots <- file.path(aim_data_dir, "study_plot_characteristics.csv")

output_points <- file.path(output_dir, "points_with_soil_survey_order.csv")
output_uncertainty_points <- file.path(output_dir, "points_with_soil_uncertainty.csv")
output_summary <- file.path(output_dir, "soil_survey_order_summary.csv")
output_mukind_summary <- file.path(output_dir, "soil_survey_order_by_mukind_summary.csv")
output_scale_summary <- file.path(output_dir, "soil_survey_order_by_projectscale_summary.csv")
output_uncertainty_summary <- file.path(output_dir, "soil_uncertainty_class_summary.csv")
output_landscape_comparison <- file.path(output_dir, "point_landscape_ecosite_comparison.csv")
output_landscape_match_summary <- file.path(output_dir, "landscape_ecosite_match_summary.csv")
output_landscape_fuzzy_summary <- file.path(output_dir, "landscape_crosswalk_fuzzy_match_summary.csv")
output_landscape_fuzzy_strict_summary <- file.path(output_dir, "landscape_crosswalk_fuzzy_strict_summary.csv")
output_landscape_fuzzy_loose_summary <- file.path(output_dir, "landscape_crosswalk_fuzzy_loose_summary.csv")
output_selection_sensitivity_summary <- file.path(output_dir, "series_selection_sensitivity_summary.csv")
multiplicity_lookup_path <- file.path(project_dir, "Data", "aim_data", "compname_mlra_ecosite_multiplicity.csv")

comparison_cols <- c(
  "EcositeID_AIM_Before_QC",
  "EcositeID_QC",
  "EcositeID_landpks_ecosite",
  "Soil Series_AIM",
  "Soil_Series_QC",
  "AIM_LandscapeType_BEFORE_QC",
  "QC_LandscapeType",
  "AIM_PlantID",
  "QC_PlantID"
)

if (!file.exists(input_xlsx)) {
  stop(sprintf("Input workbook not found: %s", input_xlsx))
}

raw_df <- read_excel(input_xlsx, sheet = input_sheet)
required_cols <- c(id_col, lon_col, lat_col)
missing_cols <- setdiff(required_cols, names(raw_df))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns in %s: %s", input_sheet, paste(missing_cols, collapse = ", ")))
}

pts <- raw_df %>%
  mutate(point_index = row_number()) %>%
  select(
    point_index,
    all_of(id_col),
    all_of(lon_col),
    all_of(lat_col),
    any_of(comparison_cols)
  ) %>%
  rename(
    point_id = all_of(id_col),
    lon = all_of(lon_col),
    lat = all_of(lat_col)
  ) %>%
  mutate(
    lon = as.numeric(lon),
    lat = as.numeric(lat)
  ) %>%
  filter(!is.na(lon), !is.na(lat)) %>%
  filter(dplyr::between(lon, -180, 180), dplyr::between(lat, -90, 90)) %>%
  mutate(wkt = sprintf("point(%.8f %.8f)", lon, lat))

if (nrow(pts) == 0) {
  stop("No valid point coordinates found after filtering.")
}

message(sprintf("Loaded %s valid points from %s/%s.", nrow(pts), input_xlsx, input_sheet))

build_sql <- function(chunk_df) {
  point_rows <- paste0(
    "SELECT ",
    chunk_df$point_index,
    " AS point_index, '",
    chunk_df$wkt,
    "' AS wkt"
  )

  points_cte <- paste(point_rows, collapse = " UNION ALL ")

  paste0(
    "WITH pts AS (", points_cte, ") ",
    ", comp_ranked AS (",
    "  SELECT mukey, cokey, compname, comppct_r, ",
    "         ROW_NUMBER() OVER (PARTITION BY mukey ORDER BY ISNULL(comppct_r, -1) DESC, cokey) AS rn ",
    "  FROM component",
    "), ecoclass_ranked AS (",
    "  SELECT cokey, ecoclassid, ecoclassname, ecoclasstypename, ecositestatus, ",
    "         ROW_NUMBER() OVER (PARTITION BY cokey ORDER BY coecoclasskey) AS rn ",
    "  FROM coecoclass",
    "), geom_ranked AS (",
    "  SELECT cokey, cogeomdkey, geomftname, geomfname, geomfmod, ",
    "         ROW_NUMBER() OVER (PARTITION BY cokey ORDER BY cogeomdkey) AS rn ",
    "  FROM cogeomordesc",
    ") ",
    "SELECT pts.point_index, mu.mukey, m.musym, m.muname, m.mukind, m.mlrasymbol, m.invesintens AS soil_survey_order, ",
    "       l.areasymbol, l.areaname, l.projectscale, ",
    "       c1.compname AS dominant_compname, c1.comppct_r AS dominant_comppct_r, c2.comppct_r AS second_comppct_r, ",
    "       eco.ecoclassid, eco.ecoclassname, eco.ecoclasstypename, eco.ecositestatus, ",
    "       geo.geomftname, geo.geomfname, geo.geomfmod, gc.geomposmntn, gc.geomposhill, gc.geompostrce, gc.geomposflats, ss.shapeacross, ss.shapedown ",
    "FROM pts ",
    "OUTER APPLY SDA_Get_Mukey_from_intersection_with_WktWgs84(pts.wkt) AS mu ",
    "LEFT JOIN mapunit AS m ON m.mukey = mu.mukey ",
    "LEFT JOIN legend AS l ON l.lkey = m.lkey ",
    "LEFT JOIN comp_ranked AS c1 ON c1.mukey = m.mukey AND c1.rn = 1 ",
    "LEFT JOIN comp_ranked AS c2 ON c2.mukey = m.mukey AND c2.rn = 2 ",
    "LEFT JOIN ecoclass_ranked AS eco ON eco.cokey = c1.cokey AND eco.rn = 1 ",
    "LEFT JOIN geom_ranked AS geo ON geo.cokey = c1.cokey AND geo.rn = 1 ",
    "LEFT JOIN cosurfmorphgc AS gc ON gc.cogeomdkey = geo.cogeomdkey ",
    "LEFT JOIN cosurfmorphss AS ss ON ss.cogeomdkey = geo.cogeomdkey"
  )
}

query_chunk <- function(chunk_df) {
  sql <- build_sql(chunk_df)
  tryCatch(
    {
      SDA_query(sql)
    },
    error = function(e) {
      warning(sprintf(
        "SDA query failed for chunk starting point_index=%s: %s",
        chunk_df$point_index[1],
        e$message
      ))
      tibble(
        point_index = chunk_df$point_index,
        mukey = NA_character_,
        musym = NA_character_,
        muname = NA_character_,
        mukind = NA_character_,
        mlrasymbol = NA_character_,
        soil_survey_order = NA_character_,
        areasymbol = NA_character_,
        areaname = NA_character_,
        projectscale = NA_character_,
        dominant_compname = NA_character_,
        dominant_comppct_r = NA_real_,
        second_comppct_r = NA_real_,
        ecoclassid = NA_character_,
        ecoclassname = NA_character_,
        ecoclasstypename = NA_character_,
        ecositestatus = NA_character_,
        geomftname = NA_character_,
        geomfname = NA_character_,
        geomfmod = NA_character_,
        geomposmntn = NA_character_,
        geomposhill = NA_character_,
        geompostrce = NA_character_,
        geomposflats = NA_character_,
        shapeacross = NA_character_,
        shapedown = NA_character_
      )
    }
  )
}

build_component_sql <- function(chunk_df) {
  point_rows <- paste0(
    "SELECT ",
    chunk_df$point_index,
    " AS point_index, '",
    chunk_df$wkt,
    "' AS wkt"
  )

  points_cte <- paste(point_rows, collapse = " UNION ALL ")

  paste0(
    "WITH pts AS (", points_cte, ") ",
    ", home_mu AS (",
    "  SELECT pts.point_index, mu.mukey AS home_mukey ",
    "  FROM pts ",
    "  OUTER APPLY SDA_Get_Mukey_from_intersection_with_WktWgs84(pts.wkt) AS mu",
    "), buffer_mu AS (",
    "  SELECT pts.point_index, mu.mukey AS buffer_mukey ",
    "  FROM pts ",
    "  OUTER APPLY SDA_Get_Mukey_from_intersection_with_WktWgs84(",
    "    geography::STGeomFromText(pts.wkt, 4326).STBuffer(", buffer_m, ").ToString()",
    "  ) AS mu",
    "), ecoclass_ranked AS (",
    "  SELECT cokey, ecoclassid, ecoclassname, ecoclasstypename, ecositestatus, ",
    "         ROW_NUMBER() OVER (PARTITION BY cokey ORDER BY coecoclasskey) AS rn ",
    "  FROM coecoclass",
    "), geom_ranked AS (",
    "  SELECT cokey, cogeomdkey, geomftname, geomfname, geomfmod, ",
    "         ROW_NUMBER() OVER (PARTITION BY cokey ORDER BY cogeomdkey) AS rn ",
    "  FROM cogeomordesc",
    ") ",
    "SELECT bm.point_index, hm.home_mukey, bm.buffer_mukey AS component_mukey, c.compname, c.comppct_r, c.cokey, ",
    "       CASE WHEN hm.home_mukey IS NOT NULL AND bm.buffer_mukey = hm.home_mukey THEN 'home_mapunit' ELSE 'adjacent_mapunit' END AS mapunit_source, ",
    "       eco.ecoclassid, eco.ecoclassname, eco.ecoclasstypename, eco.ecositestatus, ",
    "       geo.geomftname, geo.geomfname, geo.geomfmod, gc.geomposmntn, gc.geomposhill, gc.geompostrce, gc.geomposflats, ss.shapeacross, ss.shapedown ",
    "FROM buffer_mu AS bm ",
    "LEFT JOIN home_mu AS hm ON hm.point_index = bm.point_index ",
    "LEFT JOIN component AS c ON c.mukey = bm.buffer_mukey ",
    "LEFT JOIN ecoclass_ranked AS eco ON eco.cokey = c.cokey AND eco.rn = 1 ",
    "LEFT JOIN geom_ranked AS geo ON geo.cokey = c.cokey AND geo.rn = 1 ",
    "LEFT JOIN cosurfmorphgc AS gc ON gc.cogeomdkey = geo.cogeomdkey ",
    "LEFT JOIN cosurfmorphss AS ss ON ss.cogeomdkey = geo.cogeomdkey"
  )
}

query_component_chunk <- function(chunk_df) {
  sql <- build_component_sql(chunk_df)
  tryCatch(
    {
      SDA_query(sql)
    },
    error = function(e) {
      warning(sprintf(
        "SDA component query failed for chunk starting point_index=%s: %s",
        chunk_df$point_index[1],
        e$message
      ))
      tibble(
        point_index = chunk_df$point_index,
        home_mukey = NA_character_,
        component_mukey = NA_character_,
        compname = NA_character_,
        comppct_r = NA_real_,
        cokey = NA_character_,
        mapunit_source = NA_character_,
        ecoclassid = NA_character_,
        ecoclassname = NA_character_,
        ecoclasstypename = NA_character_,
        ecositestatus = NA_character_,
        geomftname = NA_character_,
        geomfname = NA_character_,
        geomfmod = NA_character_,
        geomposmntn = NA_character_,
        geomposhill = NA_character_,
        geompostrce = NA_character_,
        geomposflats = NA_character_,
        shapeacross = NA_character_,
        shapedown = NA_character_
      )
    }
  )
}

normalize_txt <- function(x) {
  out <- trimws(tolower(as.character(x)))
  out[out %in% c("", "na", "null")] <- NA_character_
  out
}

normalize_series <- function(x) {
  out <- normalize_txt(x)
  out <- gsub("[^a-z0-9]", "", out)
  out[out == ""] <- NA_character_
  out
}

normalize_compname <- function(x) {
  out <- normalize_txt(x)
  out <- gsub("\\s+", " ", out)
  out[out == ""] <- NA_character_
  out
}

strip_ecosite_prefix <- function(x) {
  out <- trimws(as.character(x))
  out[out %in% c("", "NA", "N/A", "NULL")] <- NA_character_
  # SDA ecoclassid can include a leading R or F that is absent in AIM/QC IDs.
  out <- gsub("^[RF]+", "", toupper(out))
  out
}

normalize_ecosite_id <- function(x) {
  out <- strip_ecosite_prefix(x)
  # Keep only the first listed value if multiple ecosite IDs are provided.
  out <- gsub("[,;|].*$", "", out)
  out <- gsub("[^A-Z0-9]", "", out)
  out[out == ""] <- NA_character_
  out
}

contains_match <- function(x, y) {
  x_norm <- normalize_txt(x)
  y_norm <- normalize_txt(y)
  mapply(
    function(a, b) {
      if (is.na(a) || is.na(b)) {
        return(NA)
      }
      grepl(a, b, fixed = TRUE)
    },
    x_norm,
    y_norm
  )
}

match_component_by_series <- function(point_df, component_df, series_col, landscape_col, prefix) {
  series_tbl <- point_df %>%
    transmute(
      point_index,
      series_value = .data[[series_col]],
      series_norm = normalize_series(.data[[series_col]]),
      landscape_value = .data[[landscape_col]],
      landscape_class = crosswalk_landscape_class(.data[[landscape_col]], mode = "base")
    )

  comp_tbl <- component_df %>%
    mutate(
      compname_norm = normalize_series(compname),
      ecoclassid_display = strip_ecosite_prefix(ecoclassid),
      sda_ecosite_label = trimws(paste(ecoclassid_display, ecoclassname, sep = " | ")),
      sda_landscape_label = trimws(paste(
        geomftname, geomfname, geomfmod,
        geomposmntn, geomposhill, geompostrce, geomposflats,
        shapeacross, shapedown
      )),
      sda_landscape_class = crosswalk_landscape_class(sda_landscape_label, mode = "base")
    )

  candidates <- series_tbl %>%
    inner_join(comp_tbl, by = "point_index") %>%
    filter(!is.na(series_norm), !is.na(compname_norm)) %>%
    mutate(
      contains_either = mapply(
        function(a, b) {
          grepl(a, b, fixed = TRUE) || grepl(b, a, fixed = TRUE)
        },
        series_norm,
        compname_norm
      ),
      landscape_class_match = dplyr::case_when(
        is.na(landscape_class) | is.na(sda_landscape_class) ~ FALSE,
        TRUE ~ landscape_class == sda_landscape_class
      ),
      match_score = dplyr::case_when(
        compname_norm == series_norm ~ 3,
        startsWith(compname_norm, series_norm) | startsWith(series_norm, compname_norm) ~ 2,
        contains_either ~ 1,
        TRUE ~ 0
      )
    ) %>%
    filter(match_score > 0)

  matched_landscape <- candidates %>%
    group_by(point_index) %>%
    arrange(
      desc(match_score),
      desc(landscape_class_match),
      desc(mapunit_source == "home_mapunit"),
      desc(comppct_r),
      .by_group = TRUE
    ) %>%
    slice(1) %>%
    ungroup()

  matched_series_only <- candidates %>%
    group_by(point_index) %>%
    arrange(
      desc(match_score),
      desc(mapunit_source == "home_mapunit"),
      desc(comppct_r),
      .by_group = TRUE
    ) %>%
    slice(1) %>%
    ungroup() %>%
    transmute(
      point_index,
      series_only_component_mukey = component_mukey,
      series_only_component_name = compname,
      series_only_component_mapunit_source = mapunit_source,
      series_only_component_match_score = match_score
    )

  matched <- matched_landscape %>%
    left_join(matched_series_only, by = "point_index") %>%
    transmute(
      point_index,
      !!paste0(prefix, "_series_value") := series_value,
      !!paste0(prefix, "_landscape_value") := landscape_value,
      !!paste0(prefix, "_landscape_class") := landscape_class,
      !!paste0(prefix, "_home_mukey") := home_mukey,
      !!paste0(prefix, "_component_mukey") := component_mukey,
      !!paste0(prefix, "_component_name") := compname,
      !!paste0(prefix, "_component_comppct_r") := as.numeric(comppct_r),
      !!paste0(prefix, "_component_mapunit_source") := mapunit_source,
      !!paste0(prefix, "_component_match_score") := match_score,
      !!paste0(prefix, "_component_landscape_class") := sda_landscape_class,
      !!paste0(prefix, "_component_landscape_class_match") := landscape_class_match,
      !!paste0(prefix, "_series_only_component_mukey") := series_only_component_mukey,
      !!paste0(prefix, "_series_only_component_name") := series_only_component_name,
      !!paste0(prefix, "_series_only_component_mapunit_source") := series_only_component_mapunit_source,
      !!paste0(prefix, "_series_only_component_match_score") := series_only_component_match_score,
      !!paste0(prefix, "_selection_changed") := dplyr::case_when(
        is.na(series_only_component_name) ~ NA,
        TRUE ~ (component_mukey != series_only_component_mukey) |
          (compname != series_only_component_name) |
          (mapunit_source != series_only_component_mapunit_source)
      ),
      !!paste0(prefix, "_sda_ecosite_label") := sda_ecosite_label,
      !!paste0(prefix, "_sda_landscape_label") := sda_landscape_label
    )

  matched
}

match_component_by_ecosite <- function(point_df, component_df, ecosite_col, prefix) {
  ecosite_tbl <- point_df %>%
    transmute(
      point_index,
      ecosite_value = .data[[ecosite_col]],
      ecosite_norm = normalize_ecosite_id(.data[[ecosite_col]])
    )

  comp_tbl <- component_df %>%
    mutate(
      ecoclassid_display = strip_ecosite_prefix(ecoclassid),
      ecoclassid_norm = normalize_ecosite_id(ecoclassid)
    ) %>%
    select(
      point_index,
      home_mukey,
      component_mukey,
      comppct_r,
      mapunit_source,
      ecoclassid_display,
      ecoclassname,
      ecoclassid_norm
    )

  matched <- ecosite_tbl %>%
    inner_join(comp_tbl, by = "point_index") %>%
    filter(!is.na(ecosite_norm), !is.na(ecoclassid_norm)) %>%
    mutate(
      contains_either = mapply(
        function(a, b) {
          grepl(a, b, fixed = TRUE) || grepl(b, a, fixed = TRUE)
        },
        ecosite_norm,
        ecoclassid_norm
      ),
      match_score = dplyr::case_when(
        ecoclassid_norm == ecosite_norm ~ 3,
        startsWith(ecoclassid_norm, ecosite_norm) | startsWith(ecosite_norm, ecoclassid_norm) ~ 2,
        contains_either ~ 1,
        TRUE ~ 0
      )
    ) %>%
    filter(match_score > 0) %>%
    group_by(point_index) %>%
    arrange(
      desc(match_score),
      desc(mapunit_source == "home_mapunit"),
      desc(as.numeric(comppct_r)),
      .by_group = TRUE
    ) %>%
    slice(1) %>%
    ungroup() %>%
    transmute(
      point_index,
      !!paste0(prefix, "_input_ecosite_id") := ecosite_value,
      !!paste0(prefix, "_matched_ecoclassid") := ecoclassid_display,
      !!paste0(prefix, "_matched_ecoclassname") := ecoclassname,
      !!paste0(prefix, "_matched_component_mukey") := component_mukey,
      !!paste0(prefix, "_matched_component_comppct_r") := as.numeric(comppct_r),
      !!paste0(prefix, "_matched_mapunit_source") := mapunit_source,
      !!paste0(prefix, "_buffer_match") := TRUE
    )

  ecosite_tbl %>%
    transmute(
      point_index,
      !!paste0(prefix, "_input_ecosite_id") := ecosite_value
    ) %>%
    left_join(matched, by = c("point_index", paste0(prefix, "_input_ecosite_id"))) %>%
    mutate(
      !!paste0(prefix, "_buffer_match") := dplyr::coalesce(.data[[paste0(prefix, "_buffer_match")]], FALSE)
    )
}

crosswalk_landscape_class <- function(x, mode = "base") {
  lx <- normalize_txt(x)

  if (mode == "strict") {
    return(dplyr::case_when(
      is.na(lx) ~ NA_character_,
      grepl("^playa$|^flat_plain$|lake plain|valley floor", lx) ~ "flats_plains",
      grepl("alluvial fan|fan remnant", lx) ~ "fans",
      grepl("^hill_mountain$|mountain ridge|mountains?$", lx) ~ "hills_mountains",
      grepl("terrace|plateau|mesa", lx) ~ "terraces_plateaus",
      grepl("drainageway|channel|swale|wash", lx) ~ "drainages",
      grepl("dune|sand sheet", lx) ~ "dunes_sands",
      grepl("rock outcrop|badland|cliff", lx) ~ "rocklands",
      TRUE ~ "other"
    ))
  }

  if (mode == "loose") {
    return(dplyr::case_when(
      is.na(lx) ~ NA_character_,
      grepl("playa|flat|plain|basin|valley|lake|bottom|depression", lx) ~ "flats_plains",
      grepl("alluvial|fan|apron|toeslope|footslope", lx) ~ "fans",
      grepl("hill|mountain|ridge|escarpment|slope|upland|summit", lx) ~ "hills_mountains",
      grepl("terrace|plateau|mesa|bench|tableland", lx) ~ "terraces_plateaus",
      grepl("drainage|channel|swale|wash|draw|arroyo|floodplain", lx) ~ "drainages",
      grepl("dune|sand|aeolian", lx) ~ "dunes_sands",
      grepl("rock|outcrop|badland|cliff|talus", lx) ~ "rocklands",
      TRUE ~ "other"
    ))
  }

  dplyr::case_when(
    is.na(lx) ~ NA_character_,
    grepl("playa|flat|plain|basin|valley floor|lake plain|bottom", lx) ~ "flats_plains",
    grepl("alluvial fan|alluvialfan|fan remnant|fan", lx) ~ "fans",
    grepl("hill|mountain|ridge|escarpment|slope", lx) ~ "hills_mountains",
    grepl("terrace|plateau|mesa|bench", lx) ~ "terraces_plateaus",
    grepl("drainageway|channel|swale|wash|draw", lx) ~ "drainages",
    grepl("dune|sand sheet|sandy", lx) ~ "dunes_sands",
    grepl("rock outcrop|badland|cliff", lx) ~ "rocklands",
    TRUE ~ "other"
  )
}

score_order <- function(x) {
  dplyr::case_when(
    x == "Order 2" ~ 100,
    x == "Order 3" ~ 80,
    x == "Order 4" ~ 55,
    x == "Order 5" ~ 35,
    x == "NoData" ~ 30,
    TRUE ~ 45
  )
}

score_mukind <- function(x) {
  lx <- tolower(ifelse(is.na(x), "", x))
  dplyr::case_when(
    grepl("consociation", lx) ~ 100,
    grepl("association", lx) ~ 70,
    grepl("complex", lx) ~ 45,
    grepl("undifferentiated", lx) ~ 40,
    lx == "nodata" | lx == "" ~ 30,
    TRUE ~ 55
  )
}

chunks <- split(pts, ceiling(seq_len(nrow(pts)) / chunk_size))
results <- purrr::map_dfr(chunks, query_chunk)
component_results <- purrr::map_dfr(chunks, query_component_chunk)

multiplicity_lookup <- if (file.exists(multiplicity_lookup_path)) {
  read.csv(multiplicity_lookup_path, stringsAsFactors = FALSE) %>%
    transmute(
      compname_norm = normalize_compname(compname_norm),
      mlrasymbol = normalize_txt(mlrasymbol),
      n_ecosites = as.integer(n_ecosites)
    ) %>%
    distinct(compname_norm, mlrasymbol, .keep_all = TRUE)
} else {
  warning(sprintf("Multiplicity lookup not found: %s", multiplicity_lookup_path))
  tibble(
    compname_norm = character(),
    mlrasymbol = character(),
    n_ecosites = integer()
  )
}

aim_series_match <- match_component_by_series(
  point_df = pts,
  component_df = component_results,
  series_col = "Soil Series_AIM",
  landscape_col = "AIM_LandscapeType_BEFORE_QC",
  prefix = "aim_series"
)

qc_series_match <- match_component_by_series(
  point_df = pts,
  component_df = component_results,
  series_col = "Soil_Series_QC",
  landscape_col = "QC_LandscapeType",
  prefix = "qc_series"
)

aim_ecosite_buffer_match <- match_component_by_ecosite(
  point_df = pts,
  component_df = component_results,
  ecosite_col = "EcositeID_AIM_Before_QC",
  prefix = "aim_ecosite"
)

qc_ecosite_buffer_match <- match_component_by_ecosite(
  point_df = pts,
  component_df = component_results,
  ecosite_col = "EcositeID_QC",
  prefix = "qc_ecosite"
)

point_results <- pts %>%
  select(point_index, point_id, lon, lat, any_of(comparison_cols)) %>%
  left_join(results, by = "point_index") %>%
  left_join(aim_series_match, by = "point_index") %>%
  left_join(qc_series_match, by = "point_index") %>%
  left_join(aim_ecosite_buffer_match, by = "point_index") %>%
  left_join(qc_ecosite_buffer_match, by = "point_index") %>%
  mutate(
    aim_series_series_value = dplyr::coalesce(aim_series_series_value, .data[["Soil Series_AIM"]]),
    qc_series_series_value = dplyr::coalesce(qc_series_series_value, .data[["Soil_Series_QC"]]),
    ecoclassid_dominant = strip_ecosite_prefix(ecoclassid),
    soil_survey_order = ifelse(is.na(soil_survey_order), "NoData", soil_survey_order),
    mukind = ifelse(is.na(mukind), "NoData", mukind),
    projectscale = ifelse(is.na(projectscale), "NoData", projectscale),
    mlrasymbol = normalize_txt(mlrasymbol),
    dominant_comppct_r = as.numeric(dominant_comppct_r),
    second_comppct_r = as.numeric(second_comppct_r),
    dominant_compname_norm = normalize_compname(dominant_compname),
    component_gap = dominant_comppct_r - second_comppct_r,
    order_score = score_order(soil_survey_order),
    mukind_score = score_mukind(mukind),
    # NCSS Soil Survey Manual Part 627: >=80% dominant component is preferred consociation quality.
    dominant_score = dplyr::case_when(
      is.na(dominant_comppct_r) ~ 50,
      dominant_comppct_r < 50   ~ 20,
      dominant_comppct_r < 80   ~ 55,
      TRUE                       ~ 100
    ),
    gap_score = dplyr::case_when(
      is.na(component_gap) ~ 50,
      component_gap < 20   ~ 30,
      component_gap < 40   ~ 60,
      TRUE                  ~ 90
    )
  ) %>%
  left_join(multiplicity_lookup, by = c("dominant_compname_norm" = "compname_norm", "mlrasymbol" = "mlrasymbol")) %>%
  mutate(
    n_ecosites_dominant = as.integer(n_ecosites),
    multiplicity_score = dplyr::case_when(
      is.na(n_ecosites_dominant) ~ 50,
      n_ecosites_dominant >= 2 ~ 30,
      n_ecosites_dominant == 1 ~ 100,
      TRUE ~ 50
    ),
    confidence_index = round(
      (0.20 * order_score) +
      (0.10 * mukind_score) +
      (0.35 * dominant_score) +
      (0.20 * gap_score) +
      (0.15 * multiplicity_score),
      1
    ),
    uncertainty_class = dplyr::case_when(
      confidence_index >= 78 ~ "Low uncertainty (high confidence)",
      confidence_index >= 55 ~ "Moderate uncertainty",
      TRUE ~ "High uncertainty"
    ),
    uncertainty_reason = dplyr::case_when(
      soil_survey_order %in% c("Order 4", "Order 5", "NoData") ~ "Lower-intensity mapping order",
      grepl("complex|undifferentiated", tolower(mukind)) ~ "Complex/undifferentiated map unit",
      !is.na(dominant_comppct_r) & dominant_comppct_r < 80 ~ "Weak dominant component",
      !is.na(component_gap) & component_gap < 20 ~ "Top components have similar proportion",
      TRUE ~ "Stronger map unit confidence profile"
    ),
    sda_landscape_label_dominant = trimws(paste(geomftname, geomfname, geomfmod)),
    sda_ecosite_label_dominant = trimws(paste(ecoclassid_dominant, ecoclassname, sep = " | ")),
    sda_landscape_label_aim_series = aim_series_sda_landscape_label,
    sda_landscape_label_qc_series = qc_series_sda_landscape_label,
    file_landscape_aim_class = crosswalk_landscape_class(AIM_LandscapeType_BEFORE_QC, mode = "base"),
    file_landscape_qc_class = crosswalk_landscape_class(QC_LandscapeType, mode = "base"),
    sda_landscape_class_aim_series = crosswalk_landscape_class(sda_landscape_label_aim_series, mode = "base"),
    sda_landscape_class_qc_series = crosswalk_landscape_class(sda_landscape_label_qc_series, mode = "base"),
    file_landscape_aim_class_strict = crosswalk_landscape_class(AIM_LandscapeType_BEFORE_QC, mode = "strict"),
    file_landscape_qc_class_strict = crosswalk_landscape_class(QC_LandscapeType, mode = "strict"),
    sda_landscape_class_aim_series_strict = crosswalk_landscape_class(sda_landscape_label_aim_series, mode = "strict"),
    sda_landscape_class_qc_series_strict = crosswalk_landscape_class(sda_landscape_label_qc_series, mode = "strict"),
    file_landscape_aim_class_loose = crosswalk_landscape_class(AIM_LandscapeType_BEFORE_QC, mode = "loose"),
    file_landscape_qc_class_loose = crosswalk_landscape_class(QC_LandscapeType, mode = "loose"),
    sda_landscape_class_aim_series_loose = crosswalk_landscape_class(sda_landscape_label_aim_series, mode = "loose"),
    sda_landscape_class_qc_series_loose = crosswalk_landscape_class(sda_landscape_label_qc_series, mode = "loose"),
    match_ecosite_aim_dominant = contains_match(EcositeID_AIM_Before_QC, sda_ecosite_label_dominant),
    match_ecosite_qc_dominant = contains_match(EcositeID_QC, sda_ecosite_label_dominant),
    match_ecosite_landpks = contains_match(EcositeID_landpks_ecosite, sda_ecosite_label_dominant),
    match_landscape_aim = contains_match(AIM_LandscapeType_BEFORE_QC, sda_landscape_label_aim_series),
    match_landscape_qc = contains_match(QC_LandscapeType, sda_landscape_label_qc_series),
    fuzzy_match_landscape_aim = dplyr::case_when(
      is.na(file_landscape_aim_class) | is.na(sda_landscape_class_aim_series) ~ NA,
      TRUE ~ file_landscape_aim_class == sda_landscape_class_aim_series
    ),
    fuzzy_match_landscape_qc = dplyr::case_when(
      is.na(file_landscape_qc_class) | is.na(sda_landscape_class_qc_series) ~ NA,
      TRUE ~ file_landscape_qc_class == sda_landscape_class_qc_series
    ),
    fuzzy_match_landscape_aim_strict = dplyr::case_when(
      is.na(file_landscape_aim_class_strict) | is.na(sda_landscape_class_aim_series_strict) ~ NA,
      TRUE ~ file_landscape_aim_class_strict == sda_landscape_class_aim_series_strict
    ),
    fuzzy_match_landscape_qc_strict = dplyr::case_when(
      is.na(file_landscape_qc_class_strict) | is.na(sda_landscape_class_qc_series_strict) ~ NA,
      TRUE ~ file_landscape_qc_class_strict == sda_landscape_class_qc_series_strict
    ),
    fuzzy_match_landscape_aim_loose = dplyr::case_when(
      is.na(file_landscape_aim_class_loose) | is.na(sda_landscape_class_aim_series_loose) ~ NA,
      TRUE ~ file_landscape_aim_class_loose == sda_landscape_class_aim_series_loose
    ),
    fuzzy_match_landscape_qc_loose = dplyr::case_when(
      is.na(file_landscape_qc_class_loose) | is.na(sda_landscape_class_qc_series_loose) ~ NA,
      TRUE ~ file_landscape_qc_class_loose == sda_landscape_class_qc_series_loose
    )
  ) %>%
  select(
    point_id, lon, lat,
    any_of(comparison_cols),
    mukey, musym, muname, mukind, mlrasymbol,
    soil_survey_order,
    areasymbol, areaname, projectscale,
    dominant_compname,
    ecoclassid_dominant, ecoclassname, ecoclasstypename, ecositestatus,
    geomftname, geomfname, geomfmod, geomposmntn, geomposhill, geompostrce, geomposflats, shapeacross, shapedown,
    sda_ecosite_label_dominant, sda_landscape_label_dominant,
    aim_ecosite_input_ecosite_id, aim_ecosite_matched_ecoclassid, aim_ecosite_matched_ecoclassname,
    aim_ecosite_matched_component_mukey, aim_ecosite_matched_component_comppct_r, aim_ecosite_matched_mapunit_source, aim_ecosite_buffer_match,
    qc_ecosite_input_ecosite_id, qc_ecosite_matched_ecoclassid, qc_ecosite_matched_ecoclassname,
    qc_ecosite_matched_component_mukey, qc_ecosite_matched_component_comppct_r, qc_ecosite_matched_mapunit_source, qc_ecosite_buffer_match,
    aim_series_series_value, aim_series_landscape_value, aim_series_landscape_class, aim_series_home_mukey, aim_series_component_mukey, aim_series_component_name, aim_series_component_comppct_r, aim_series_component_mapunit_source, aim_series_component_match_score, aim_series_component_landscape_class, aim_series_component_landscape_class_match,
    aim_series_series_only_component_mukey, aim_series_series_only_component_name, aim_series_series_only_component_mapunit_source, aim_series_series_only_component_match_score, aim_series_selection_changed,
    aim_series_sda_ecosite_label, aim_series_sda_landscape_label,
    qc_series_series_value, qc_series_landscape_value, qc_series_landscape_class, qc_series_home_mukey, qc_series_component_mukey, qc_series_component_name, qc_series_component_comppct_r, qc_series_component_mapunit_source, qc_series_component_match_score, qc_series_component_landscape_class, qc_series_component_landscape_class_match,
    qc_series_series_only_component_mukey, qc_series_series_only_component_name, qc_series_series_only_component_mapunit_source, qc_series_series_only_component_match_score, qc_series_selection_changed,
    qc_series_sda_ecosite_label, qc_series_sda_landscape_label,
    sda_landscape_label_aim_series, sda_landscape_label_qc_series,
    file_landscape_aim_class, file_landscape_qc_class, sda_landscape_class_aim_series, sda_landscape_class_qc_series,
    file_landscape_aim_class_strict, file_landscape_qc_class_strict, sda_landscape_class_aim_series_strict, sda_landscape_class_qc_series_strict,
    file_landscape_aim_class_loose, file_landscape_qc_class_loose, sda_landscape_class_aim_series_loose, sda_landscape_class_qc_series_loose,
    match_ecosite_aim_dominant, match_ecosite_qc_dominant, match_ecosite_landpks, match_landscape_aim, match_landscape_qc,
    fuzzy_match_landscape_aim, fuzzy_match_landscape_qc,
    fuzzy_match_landscape_aim_strict, fuzzy_match_landscape_qc_strict,
    fuzzy_match_landscape_aim_loose, fuzzy_match_landscape_qc_loose,
    dominant_comppct_r, second_comppct_r, component_gap,
    n_ecosites_dominant, multiplicity_score,
    confidence_index, uncertainty_class, uncertainty_reason
  )

summary_tbl <- point_results %>%
  count(soil_survey_order, name = "n_points", sort = TRUE) %>%
  mutate(
    pct_points = round(100 * n_points / sum(n_points), 2)
  )

mukind_summary_tbl <- point_results %>%
  count(soil_survey_order, mukind, name = "n_points", sort = TRUE) %>%
  mutate(
    pct_points = round(100 * n_points / sum(n_points), 2)
  )

scale_summary_tbl <- point_results %>%
  count(soil_survey_order, projectscale, name = "n_points", sort = TRUE) %>%
  mutate(
    pct_points = round(100 * n_points / sum(n_points), 2)
  )

uncertainty_summary_tbl <- point_results %>%
  count(uncertainty_class, uncertainty_reason, name = "n_points", sort = TRUE) %>%
  mutate(
    pct_points = round(100 * n_points / sum(n_points), 2)
  )

landscape_match_summary_tbl <- point_results %>%
  summarise(
    n_points = n(),
    n_with_sda_ecosite_dominant = sum(!is.na(normalize_txt(sda_ecosite_label_dominant))),
    n_with_sda_landscape_aim_series = sum(!is.na(normalize_txt(sda_landscape_label_aim_series))),
    n_with_sda_landscape_qc_series = sum(!is.na(normalize_txt(sda_landscape_label_qc_series))),
    n_match_ecosite_aim_dominant = sum(match_ecosite_aim_dominant %in% TRUE, na.rm = TRUE),
    n_match_ecosite_qc_dominant = sum(match_ecosite_qc_dominant %in% TRUE, na.rm = TRUE),
    n_spatial_buffer_match_ecosite_aim = sum(aim_ecosite_buffer_match %in% TRUE, na.rm = TRUE),
    n_spatial_buffer_match_ecosite_qc = sum(qc_ecosite_buffer_match %in% TRUE, na.rm = TRUE),
    n_match_ecosite_landpks = sum(match_ecosite_landpks %in% TRUE, na.rm = TRUE),
    n_match_landscape_aim = sum(match_landscape_aim %in% TRUE, na.rm = TRUE),
    n_match_landscape_qc = sum(match_landscape_qc %in% TRUE, na.rm = TRUE),
    n_fuzzy_match_landscape_aim = sum(fuzzy_match_landscape_aim %in% TRUE, na.rm = TRUE),
    n_fuzzy_match_landscape_qc = sum(fuzzy_match_landscape_qc %in% TRUE, na.rm = TRUE),
    n_fuzzy_match_landscape_aim_strict = sum(fuzzy_match_landscape_aim_strict %in% TRUE, na.rm = TRUE),
    n_fuzzy_match_landscape_qc_strict = sum(fuzzy_match_landscape_qc_strict %in% TRUE, na.rm = TRUE),
    n_fuzzy_match_landscape_aim_loose = sum(fuzzy_match_landscape_aim_loose %in% TRUE, na.rm = TRUE),
    n_fuzzy_match_landscape_qc_loose = sum(fuzzy_match_landscape_qc_loose %in% TRUE, na.rm = TRUE)
  ) %>%
  mutate(
    pct_with_sda_landscape_aim_series = round(100 * n_with_sda_landscape_aim_series / n_points, 2),
    pct_with_sda_landscape_qc_series = round(100 * n_with_sda_landscape_qc_series / n_points, 2),
    pct_match_ecosite_aim_dominant = round(100 * n_match_ecosite_aim_dominant / n_points, 2),
    pct_match_ecosite_qc_dominant = round(100 * n_match_ecosite_qc_dominant / n_points, 2),
    pct_spatial_buffer_match_ecosite_aim = round(100 * n_spatial_buffer_match_ecosite_aim / n_points, 2),
    pct_spatial_buffer_match_ecosite_qc = round(100 * n_spatial_buffer_match_ecosite_qc / n_points, 2),
    pct_match_ecosite_landpks = round(100 * n_match_ecosite_landpks / n_points, 2),
    pct_match_landscape_aim = round(100 * n_match_landscape_aim / n_points, 2),
    pct_match_landscape_qc = round(100 * n_match_landscape_qc / n_points, 2),
    pct_fuzzy_match_landscape_aim = round(100 * n_fuzzy_match_landscape_aim / n_points, 2),
    pct_fuzzy_match_landscape_qc = round(100 * n_fuzzy_match_landscape_qc / n_points, 2),
    pct_fuzzy_match_landscape_aim_strict = round(100 * n_fuzzy_match_landscape_aim_strict / n_points, 2),
    pct_fuzzy_match_landscape_qc_strict = round(100 * n_fuzzy_match_landscape_qc_strict / n_points, 2),
    pct_fuzzy_match_landscape_aim_loose = round(100 * n_fuzzy_match_landscape_aim_loose / n_points, 2),
    pct_fuzzy_match_landscape_qc_loose = round(100 * n_fuzzy_match_landscape_qc_loose / n_points, 2),
    pct_fuzzy_match_landscape_aim_given_sda = round(100 * n_fuzzy_match_landscape_aim / n_with_sda_landscape_aim_series, 2),
    pct_fuzzy_match_landscape_qc_given_sda = round(100 * n_fuzzy_match_landscape_qc / n_with_sda_landscape_qc_series, 2),
    pct_fuzzy_match_landscape_aim_strict_given_sda = round(100 * n_fuzzy_match_landscape_aim_strict / n_with_sda_landscape_aim_series, 2),
    pct_fuzzy_match_landscape_qc_strict_given_sda = round(100 * n_fuzzy_match_landscape_qc_strict / n_with_sda_landscape_qc_series, 2),
    pct_fuzzy_match_landscape_aim_loose_given_sda = round(100 * n_fuzzy_match_landscape_aim_loose / n_with_sda_landscape_aim_series, 2),
    pct_fuzzy_match_landscape_qc_loose_given_sda = round(100 * n_fuzzy_match_landscape_qc_loose / n_with_sda_landscape_qc_series, 2)
  )

landscape_fuzzy_summary_tbl <- point_results %>%
  filter(!is.na(file_landscape_qc_class), !is.na(sda_landscape_class_qc_series)) %>%
  count(file_landscape_qc_class, sda_landscape_class_qc_series, name = "n_points", sort = TRUE) %>%
  mutate(
    pct_points = round(100 * n_points / sum(n_points), 2)
  )

landscape_fuzzy_strict_summary_tbl <- point_results %>%
  filter(!is.na(file_landscape_qc_class_strict), !is.na(sda_landscape_class_qc_series_strict)) %>%
  count(file_landscape_qc_class_strict, sda_landscape_class_qc_series_strict, name = "n_points", sort = TRUE) %>%
  mutate(
    pct_points = round(100 * n_points / sum(n_points), 2)
  )

landscape_fuzzy_loose_summary_tbl <- point_results %>%
  filter(!is.na(file_landscape_qc_class_loose), !is.na(sda_landscape_class_qc_series_loose)) %>%
  count(file_landscape_qc_class_loose, sda_landscape_class_qc_series_loose, name = "n_points", sort = TRUE) %>%
  mutate(
    pct_points = round(100 * n_points / sum(n_points), 2)
  )

selection_sensitivity_summary_tbl <- point_results %>%
  summarise(
    n_points = n(),
    n_aim_matched = sum(!is.na(aim_series_component_name)),
    n_qc_matched = sum(!is.na(qc_series_component_name)),
    n_aim_selection_changed = sum(aim_series_selection_changed %in% TRUE, na.rm = TRUE),
    n_qc_selection_changed = sum(qc_series_selection_changed %in% TRUE, na.rm = TRUE)
  ) %>%
  mutate(
    pct_aim_selection_changed_of_matched = round(100 * n_aim_selection_changed / n_aim_matched, 2),
    pct_qc_selection_changed_of_matched = round(100 * n_qc_selection_changed / n_qc_matched, 2)
  )

write.csv(point_results, output_points, row.names = FALSE)
write.csv(point_results, output_uncertainty_points, row.names = FALSE)

# Write to Data/aim_data/ for direct use by run_all_aim_examples.py
tryCatch(
  write.csv(point_results, output_study_plots, row.names = FALSE),
  error = function(e) warning(sprintf("Could not write study_plot_characteristics.csv: %s", e$message))
)

write.csv(summary_tbl, output_summary, row.names = FALSE)
write.csv(mukind_summary_tbl, output_mukind_summary, row.names = FALSE)
write.csv(scale_summary_tbl, output_scale_summary, row.names = FALSE)
write.csv(uncertainty_summary_tbl, output_uncertainty_summary, row.names = FALSE)
write.csv(point_results, output_landscape_comparison, row.names = FALSE)
write.csv(landscape_match_summary_tbl, output_landscape_match_summary, row.names = FALSE)
write.csv(landscape_fuzzy_summary_tbl, output_landscape_fuzzy_summary, row.names = FALSE)
write.csv(landscape_fuzzy_strict_summary_tbl, output_landscape_fuzzy_strict_summary, row.names = FALSE)
write.csv(landscape_fuzzy_loose_summary_tbl, output_landscape_fuzzy_loose_summary, row.names = FALSE)
write.csv(selection_sensitivity_summary_tbl, output_selection_sensitivity_summary, row.names = FALSE)

message(sprintf("Wrote point-level results: %s", output_points))
message(sprintf("Wrote uncertainty point-level results: %s", output_uncertainty_points))
message(sprintf("Wrote summary table: %s", output_summary))
message(sprintf("Wrote mukind cross-summary: %s", output_mukind_summary))
message(sprintf("Wrote projectscale cross-summary: %s", output_scale_summary))
message(sprintf("Wrote uncertainty summary: %s", output_uncertainty_summary))
message(sprintf("Wrote landscape/ecosite comparison: %s", output_landscape_comparison))
message(sprintf("Wrote landscape/ecosite match summary: %s", output_landscape_match_summary))
message(sprintf("Wrote landscape fuzzy crosswalk summary: %s", output_landscape_fuzzy_summary))
message(sprintf("Wrote landscape strict fuzzy summary: %s", output_landscape_fuzzy_strict_summary))
message(sprintf("Wrote landscape loose fuzzy summary: %s", output_landscape_fuzzy_loose_summary))
message(sprintf("Wrote series-selection sensitivity summary: %s", output_selection_sensitivity_summary))
message("Summary of point distribution by soil survey order:")
print(summary_tbl)
message("Summary of point distribution by soil survey order and mukind:")
print(mukind_summary_tbl)
message("Summary of point distribution by soil survey order and projectscale:")
print(scale_summary_tbl)
message("Summary of uncertainty classes and primary drivers:")
print(uncertainty_summary_tbl)
message("Summary of SDA vs file landscape/ecosite matching:")
print(landscape_match_summary_tbl)
message("Summary of fuzzy crosswalk landscape class pairing:")
print(landscape_fuzzy_summary_tbl)
message("Summary of strict fuzzy crosswalk landscape class pairing:")
print(landscape_fuzzy_strict_summary_tbl)
message("Summary of loose fuzzy crosswalk landscape class pairing:")
print(landscape_fuzzy_loose_summary_tbl)
message("Summary of selection sensitivity (landscape-aware vs series-only):")
print(selection_sensitivity_summary_tbl)
