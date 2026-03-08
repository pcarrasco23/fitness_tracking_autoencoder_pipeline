import pandas as pd

from .feature_analysis import calculate_feature_errors_lstm


# Generate a detailed chart showing user_id, date, and feature contributions
# for all validation user records
def generate_detailed_validation_analysis(df, feature_names):
    # Filter only valid users
    validation_df = df[df["is_training"] == 0].copy()

    # Sort by user_id and date
    validation_df = validation_df.sort_values(["user_id", "date"])

    # Create columns list
    error_suffix = "_error"
    total_error_col = "total_error"

    # Select relevant columns
    output_columns = ["user_id", "date", total_error_col]

    # Add individual feature errors
    for feature in feature_names:
        feature_error_col = f"{feature}{error_suffix}"
        if feature_error_col in validation_df.columns:
            output_columns.append(feature_error_col)

    # Create the detailed analysis dataframe
    detailed_analysis = validation_df[output_columns].copy()

    # Rename columns for clarity
    rename_dict = {total_error_col: "total_reconstruction_error"}
    for feature in feature_names:
        old_col = f"{feature}{error_suffix}"
        new_col = f"{feature}_contribution"
        if old_col in detailed_analysis.columns:
            rename_dict[old_col] = new_col

    detailed_analysis = detailed_analysis.rename(columns=rename_dict)

    # Add a column showing top contributing feature for each record
    feature_contribution_cols = [
        f"{feature}_contribution"
        for feature in feature_names
        if f"{feature}_contribution" in detailed_analysis.columns
    ]

    if feature_contribution_cols:
        detailed_analysis["top_contributing_feature"] = detailed_analysis[
            feature_contribution_cols
        ].idxmax(axis=1)
        detailed_analysis["top_contributing_feature"] = detailed_analysis[
            "top_contributing_feature"
        ].str.replace("_contribution", "")
        detailed_analysis["top_feature_error"] = detailed_analysis[
            feature_contribution_cols
        ].max(axis=1)

    return detailed_analysis


def generate_detailed_validation_analysis_lstm(
    model, X_windows, y_windows, user_ids_windows, window_end_dates, feature_names
):
    """
    Per-window detailed analysis for validation users, mirroring
    generate_detailed_validation_analysis but for LSTM sliding windows.
    Each row represents one 7-day window, labelled by its end date.
    """
    # Run inference on all validation windows
    validation_mask = y_windows == 0
    val_windows = X_windows[validation_mask]
    val_user_ids = user_ids_windows[validation_mask]
    val_end_dates = window_end_dates[validation_mask]

    errors = calculate_feature_errors_lstm(model, val_windows, feature_names)

    # Build output dataframe
    result = pd.DataFrame(
        {
            "user_id": val_user_ids,
            "window_end_date": val_end_dates,
            "total_reconstruction_error": errors["total_error"].values,
        }
    )

    # Add per-feature contributions
    contribution_cols = []
    for feature in feature_names:
        col = f"{feature}_contribution"
        result[col] = errors[feature].values
        contribution_cols.append(col)

    # Top contributing feature per window
    result["top_contributing_feature"] = result[contribution_cols].idxmax(axis=1).str.replace(
        "_contribution", "", regex=False
    )
    result["top_feature_error"] = result[contribution_cols].max(axis=1)

    return result.sort_values(["user_id", "window_end_date"])
