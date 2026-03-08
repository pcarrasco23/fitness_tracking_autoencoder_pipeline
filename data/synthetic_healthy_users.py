import numpy as np
import pandas as pd

from .health_criteria import (
    steps_range,
    heart_rate_range,
    sleep_range,
    exercise_range,
    stress_range,
    bmi_range,
    weight_kg_range,
    calories_burned_range,
)


def calculate_healthy_steps(age):
    """Return a random step count in the healthy range for the given age."""
    lo, hi = steps_range(age)
    return np.random.randint(lo, hi + 1)


def calculate_healthy_sleep_hours(age):
    """Return a random sleep duration in the healthy range for the given age."""
    lo, hi = sleep_range(age)
    return round(np.random.uniform(lo, hi), 1)


def calculate_healthy_exercise_minutes(age):
    """Return a random daily exercise duration in the healthy range for the given age."""
    lo, hi = exercise_range(age)
    return np.random.randint(lo, hi + 1)


def calculate_healthy_heart_rate(age):
    """Return a random resting heart rate in the healthy range for the given age."""
    lo, hi = heart_rate_range(age)
    return round(np.random.uniform(lo, hi), 1)


def calculate_healthy_stress_level(age):
    """Return a random stress level in the healthy range for the given age (scale 1-10)."""
    lo, hi = stress_range(age)
    return round(np.random.uniform(lo, hi), 1)


def calculate_healthy_bmi(age):
    """Return a random BMI in the healthy range for the given age."""
    lo, hi = bmi_range(age)
    return round(np.random.uniform(lo, hi), 1)


def calculate_healthy_weight_kg(age):
    """Return a random weight in the healthy range for the given age."""
    lo, hi = weight_kg_range(age)
    return round(np.random.uniform(lo, hi), 1)


def calculate_healthy_calories_burned(age):
    """Return a random daily calories burned in the healthy range for the given age."""
    lo, hi = calories_burned_range(age)
    return np.random.randint(lo, hi + 1)


# Generate synthetic healthy users by making metrics fall within healthy ranges
#
# Parameters:
# - df: original dataframe
# - num_synthetic_users: number of synthetic healthy users to create
# - seed: random seed for reproducibility
def generate_synthetic_healthy_users(df, num_synthetic_users, seed=42):
    np.random.seed(seed)

    # Get all unique users
    all_users = df["user_id"].unique()

    # Find the maximum user_id to create new unique IDs
    max_user_id = df["user_id"].max()
    if isinstance(max_user_id, str):
        new_user_ids = [f"SYNTH_{i:04d}" for i in range(num_synthetic_users)]
    else:
        new_user_ids = range(max_user_id + 1, max_user_id + num_synthetic_users + 1)

    synthetic_data = []

    print(f"Generating {num_synthetic_users} synthetic healthy users...")

    for new_user_id in new_user_ids:
        # Randomly select any user as template
        template_user_id = np.random.choice(all_users)
        template_data = df[df["user_id"] == template_user_id].copy()

        age = template_data["age"].iloc[0] if "age" in template_data.columns else None

        synthetic_user_data = template_data.copy()
        synthetic_user_data["user_id"] = new_user_id

        # Modify all 8 metrics to ensure synthetic users fall within healthy ranges
        metrics_to_modify = np.random.choice(
            [
                "steps",
                "sleep_hours",
                "exercise_minutes",
                "heart_rate_avg",
                "stress_level",
                "bmi",
                "weight_kg",
                "calories_burned",
            ],
            size=8,
            replace=False,
        )

        # Set each selected metric to a value within the healthy range
        for metric in metrics_to_modify:
            if metric == "steps":
                synthetic_user_data["steps"] = calculate_healthy_steps(age)

            elif metric == "sleep_hours":
                synthetic_user_data["sleep_hours"] = calculate_healthy_sleep_hours(age)

            elif metric == "exercise_minutes":
                synthetic_user_data["exercise_minutes"] = (
                    calculate_healthy_exercise_minutes(age)
                )

            elif metric == "heart_rate_avg":
                synthetic_user_data["heart_rate_avg"] = calculate_healthy_heart_rate(
                    age
                )

            elif metric == "stress_level":
                synthetic_user_data["stress_level"] = calculate_healthy_stress_level(
                    age
                )

            elif metric == "bmi":
                synthetic_user_data["bmi"] = calculate_healthy_bmi(age)

            elif metric == "weight_kg":
                synthetic_user_data["weight_kg"] = calculate_healthy_weight_kg(age)

            elif metric == "calories_burned":
                synthetic_user_data["calories_burned"] = (
                    calculate_healthy_calories_burned(age)
                )

        # Add small random noise (±3%) to make data more realistic
        numeric_cols = [
            "steps",
            "heart_rate_avg",
            "sleep_hours",
            "calories_burned",
            "exercise_minutes",
            "stress_level",
            "weight_kg",
            "bmi",
        ]

        for col in numeric_cols:
            if col in synthetic_user_data.columns:
                noise = np.random.uniform(0.97, 1.03, size=len(synthetic_user_data))
                synthetic_user_data[col] = np.maximum(
                    synthetic_user_data[col] * noise, 0
                )

        synthetic_data.append(synthetic_user_data)

    synthetic_df = pd.concat(synthetic_data, ignore_index=True)

    print(f"✓ Generated {num_synthetic_users} synthetic healthy users")
    if "gender" in synthetic_df.columns:
        print(
            f"✓ Gender column preserved: {synthetic_df['gender'].nunique()} unique values"
        )
    if "age" in synthetic_df.columns:
        print(
            f"✓ Age column preserved: range {synthetic_df['age'].min()}-{synthetic_df['age'].max()}"
        )
    if "date" in synthetic_df.columns:
        print(
            f"✓ Date column preserved: {synthetic_df['date'].min()} to {synthetic_df['date'].max()}"
        )

    combined_df = pd.concat([df, synthetic_df], ignore_index=True)

    print(f"✓ Original dataset size: {len(df)}")
    print(f"✓ New dataset size: {len(combined_df)}")

    return combined_df
