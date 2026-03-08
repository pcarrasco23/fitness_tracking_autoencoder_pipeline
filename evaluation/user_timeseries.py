import pandas as pd

from .feature_analysis import calculate_feature_errors, calculate_feature_errors_lstm


# Analyze time series for each user
def analyze_user_timeseries(
    model, df, X_normalized, feature_names, scaler, user_health_df
):
    """
    Analyze reconstruction errors over time for each user
    (Date is now an input feature)
    """
    results = []

    for user_id in df["user_id"].unique():
        user_data = df[df["user_id"] == user_id].copy()

        # Get user's health status
        is_training = user_health_df[user_health_df["user_id"] == user_id][
            "is_training"
        ].values[0]
        health_score = user_health_df[user_health_df["user_id"] == user_id][
            "health_score"
        ].values[0]

        # Extract features
        user_indices = df[df["user_id"] == user_id].index
        user_features_normalized = X_normalized[user_indices]

        # Get first and last date for this user
        first_date = user_data["date"].min()
        last_date = user_data["date"].max()

        # Calculate errors
        errors = calculate_feature_errors(
            model, user_features_normalized, feature_names
        )

        # Add to user data
        user_data["total_error"] = errors["total_error"].values

        for feature in feature_names:
            user_data[f"{feature}_error"] = errors[feature].values

        # Calculate statistics
        avg_total_error = errors["total_error"].mean()
        max_total_error = errors["total_error"].max()
        std_total_error = errors["total_error"].std()

        # Find most problematic features
        feature_errors = errors[feature_names].mean()
        top_3_features = feature_errors.nlargest(3)

        results.append(
            {
                "user_id": user_id,
                "is_training": is_training,
                "health_score": health_score,
                "first_date": first_date,
                "last_date": last_date,
                "avg_reconstruction_error": avg_total_error,
                "max_reconstruction_error": max_total_error,
                "std_reconstruction_error": std_total_error,
                "top_problem_feature_1": top_3_features.index[0],
                "top_problem_error_1": top_3_features.values[0],
                "top_problem_feature_2": top_3_features.index[1],
                "top_problem_error_2": top_3_features.values[1],
                "top_problem_feature_3": top_3_features.index[2],
                "top_problem_error_3": top_3_features.values[2],
                "num_days": len(user_data),
            }
        )

    return pd.DataFrame(results)


def analyze_user_timeseries_lstm(
    model, X_windows, user_ids_windows, feature_names, user_health_df
):
    """
    Aggregate per-window LSTM reconstruction errors up to per-user statistics.
    Mirrors the output structure of analyze_user_timeseries for easy comparison.
    """
    results = []

    for user_id in user_health_df["user_id"].unique():
        mask = user_ids_windows == user_id
        if not mask.any():
            continue

        user_windows = X_windows[mask]
        errors = calculate_feature_errors_lstm(model, user_windows, feature_names)

        is_training = user_health_df[user_health_df["user_id"] == user_id][
            "is_training"
        ].values[0]
        health_score = user_health_df[user_health_df["user_id"] == user_id][
            "health_score"
        ].values[0]

        avg_total_error = errors["total_error"].mean()
        max_total_error = errors["total_error"].max()
        std_total_error = errors["total_error"].std()

        feature_errors = errors[feature_names].mean()
        top_3_features = feature_errors.nlargest(3)

        results.append(
            {
                "user_id": user_id,
                "is_training": is_training,
                "health_score": health_score,
                "avg_reconstruction_error": avg_total_error,
                "max_reconstruction_error": max_total_error,
                "std_reconstruction_error": std_total_error,
                "top_problem_feature_1": top_3_features.index[0],
                "top_problem_error_1": top_3_features.values[0],
                "top_problem_feature_2": top_3_features.index[1],
                "top_problem_error_2": top_3_features.values[1],
                "top_problem_feature_3": top_3_features.index[2],
                "top_problem_error_3": top_3_features.values[2],
                "num_windows": int(mask.sum()),
            }
        )

    return pd.DataFrame(results)
