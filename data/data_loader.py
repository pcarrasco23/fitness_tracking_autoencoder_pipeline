import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .health_scores import calculate_user_health_scores
from .synthetic_healthy_users import generate_synthetic_healthy_users


# Load time series data and calculate per-user health labels
def load_timeseries_data(filepath, health_decay_rate=0.01):

    df = pd.read_csv(filepath)

    df.head(2).to_csv("sample_input_data.csv", index=False)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by user and date
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Generate 200 synthetic healthy users and append them before scoring,
    # so the health score calculation sees them as part of the population
    df = generate_synthetic_healthy_users(df, num_synthetic_users=200)

    # Calculate user-level health scores
    user_health_df = calculate_user_health_scores(df, decay_rate=health_decay_rate)

    # Merge health labels back to original dataframe
    df = df.merge(
        user_health_df[["user_id", "is_training", "health_score"]],
        on="user_id",
        how="left",
    )

    # Feature columns (health metrics only)
    feature_cols = [
        "steps",
        "heart_rate_avg",
        "sleep_hours",
        "calories_burned",
        "exercise_minutes",
        "stress_level",
        "weight_kg",
        "bmi",
    ]

    if "gender" in df.columns:
        df["gender_encoded"] = df["gender"].map(
            {"M": 0, "F": 1, "Male": 0, "Female": 1}
        )
        feature_cols.append("gender_encoded")

    # Extract features
    X = df[feature_cols].values
    y = df["is_training"].values  # 1 = training row, 0 = validation row

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y, scaler, df, feature_cols, user_health_df


def build_lstm_windows(df, X_normalized, seq_len=7):
    """
    Build sliding windows of seq_len consecutive days per user for LSTM training.

    For each user with N days of data, produces (N - seq_len + 1) windows.
    Users with fewer than seq_len days are skipped.

    Returns:
        X_windows: np.ndarray of shape (num_windows, seq_len, num_features)
        y_windows: np.ndarray of shape (num_windows,) — 1=training, 0=validation
    """
    windows = []
    labels = []
    user_ids = []
    end_dates = []

    for user_id, user_group in df.groupby("user_id"):
        pos = user_group.index.values  # positional row indices (df has RangeIndex)
        user_X = X_normalized[pos]  # (num_days, num_features), already date-sorted
        user_label = user_group["is_training"].iloc[0]
        user_dates = user_group["date"].values

        num_days = len(user_X)
        if num_days < seq_len:
            continue

        for i in range(num_days - seq_len + 1):
            windows.append(user_X[i : i + seq_len])
            labels.append(user_label)
            user_ids.append(user_id)
            end_dates.append(user_dates[i + seq_len - 1])

    X_windows = np.array(windows, dtype=np.float32)  # (N, seq_len, features)
    y_windows = np.array(labels, dtype=np.int64)  # (N,)
    user_ids_windows = np.array(user_ids)  # (N,)
    window_end_dates = np.array(end_dates)  # (N,)

    return X_windows, y_windows, user_ids_windows, window_end_dates
