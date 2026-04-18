import math

import pandas as pd


def _is_nan(value):
    """Return True if value is a float NaN or pandas NA."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


_RANK_SCORE_COL = {
    "rank_data_loc": "Score_Data_Loc",
    "rank_data": "Score_Data",
    "rank_loc": "cond_prob",
}
_RANK_FIELD_COL = {
    "rank_data_loc": "Rank_Data_Loc",
    "rank_data": "Rank_Data",
    "rank_loc": "Rank_Loc",
}


def finalize_rank_output(
    D_final_loc: pd.DataFrame,
    location: str,
    horz_feature_sims: dict = None,
    site_feature_detail: dict = None,
    rank_method: str = "rank_data_loc",
):
    """Build soilRank output sorted by the chosen rank method.

    rank_method options:
        "rank_data_loc" (default) – combined data + location score
        "rank_data"               – data (horizon + site) score only
        "rank_loc"                – location (spatial probability) score only
    """
    # Calculate minimum rank values per compname_grp for each rank field
    df_copy = D_final_loc.copy()

    # Find rows with minimum ranks and their corresponding scores for each group
    def get_min_values(group: pd.DataFrame):
        min_data_loc_idx = group["Rank_Data_Loc"].idxmin()
        min_data_idx = group["Rank_Data"].idxmin()
        min_loc_idx = group["Rank_Loc"].idxmin()

        return pd.Series(
            {
                "Rank_Data_Loc_grp": group["Rank_Data_Loc"].min(),
                "Rank_Data_grp": group["Rank_Data"].min(),
                "Rank_Loc_grp": group["Rank_Loc"].min(),
                "Score_Data_Loc_grp": group.loc[min_data_loc_idx, "Score_Data_Loc"],
                "Score_Data_grp": group.loc[min_data_idx, "Score_Data"],
                "Score_Loc_grp": group.loc[min_loc_idx, "cond_prob"],
            }
        )

    min_values = df_copy.groupby("compname_grp").apply(get_min_values).reset_index()

    # Merge minimum values back to original data
    df_copy = df_copy.merge(min_values, on="compname_grp")

    # Sort by the chosen rank method so soilRank order reflects it.
    score_col = _RANK_SCORE_COL.get(rank_method, "Score_Data_Loc")
    rank_field = _RANK_FIELD_COL.get(rank_method, "Rank_Data_Loc")
    sort_cols = (
        ["soilID_rank_final", score_col, "compname"]
        if "soilID_rank_final" in df_copy.columns
        else [score_col, "compname"]
    )
    sort_asc = [False] * (len(sort_cols) - 1) + [True]
    df_copy = df_copy.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    Rank = [
        {
            "name": row.compname.capitalize(),
            "component": row.compname_grp.capitalize(),
            "componentID": row.cokey,
            "score_data_loc": (
                None if row.missing_status == "Location data only" else round(row.Score_Data_Loc, 3)
            ),
            "score_data_loc_group": (
                None
                if row.missing_status == "Location data only"
                else round(row.Score_Data_Loc_grp, 3)
            ),
            "rank_data_loc": (
                None if row.missing_status == "Location data only" else row.Rank_Data_Loc
            ),
            "rank_data_loc_group": (
                None if row.missing_status == "Location data only" else row.Rank_Data_Loc_grp
            ),
            "score_data": (
                None if row.missing_status == "Location data only" else round(row.Score_Data, 3)
            ),
            "score_data_group": (
                None if row.missing_status == "Location data only" else round(row.Score_Data_grp, 3)
            ),
            "rank_data": None if row.missing_status == "Location data only" else row.Rank_Data,
            "rank_data_group": None
            if row.missing_status == "Location data only"
            else row.Rank_Data_grp,
            "score_loc": round(row.cond_prob, 3),
            "score_loc_group": round(row.Score_Loc_grp, 3),
            "rank_loc": row.Rank_Loc,
            "rank_loc_group": row.Rank_Loc_grp,
            "componentData": row.missing_status,
            "not_displayed": (
                row.Rank_Loc == "Not Displayed"
                if row.missing_status == "Location data only"
                else getattr(row, rank_field) == "Not Displayed"
            ),
            "ecoclassid": getattr(row, "ecoclassid", None) if not _is_nan(getattr(row, "ecoclassid", None)) and getattr(row, "ecoclassid", None) != "" else None,
            "ecoclassname": getattr(row, "ecoclassname", None) if not _is_nan(getattr(row, "ecoclassname", None)) and getattr(row, "ecoclassname", None) != "" else None,
            # --- Diagnostic: component-level scores and per-feature breakdown ---
            "score_data_horz": (
                None
                if row.missing_status in ("Location data only", "Site data only")
                else (None if _is_nan(float(row.D_horz)) else round(float(row.D_horz), 3))
            ),
            "score_data_site": (
                None
                if row.missing_status == "Location data only"
                else (None if _is_nan(float(row.D_site)) else round(float(row.D_site) / 0.5, 3))
            ),
            "horizon_match": (
                horz_feature_sims.get(row.compname) if horz_feature_sims else None
            ),
            "site_match": (
                site_feature_detail.get(row.compname) if site_feature_detail else None
            ),
        }
        for _, row in df_copy.iterrows()
    ]

    output_data = {
        "metadata": {
            "location": location,
            "model": "v2",
            "rank_method": rank_method,
        },
        "soilRank": Rank,
    }

    return output_data
