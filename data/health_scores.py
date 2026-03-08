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


def calculate_user_health_scores(df, decay_rate=0.03, noise_tolerance=0.03):
    user_health_scores = []

    # Get the most recent date in the dataset
    max_date = df["date"].max()

    for user_id, user_data in df.groupby("user_id"):
        user_data = user_data.sort_values("date").copy()

        # Get user's age (assuming age is constant for each user)
        age = user_data["age"].iloc[0] if "age" in user_data.columns else None

        # Calculate days from most recent date (0 = most recent)
        user_data["days_ago"] = (max_date - user_data["date"]).dt.days

        # Calculate exponential weights (more recent = higher weight)
        user_data["weight"] = np.exp(-decay_rate * user_data["days_ago"])
        user_data["weight"] = user_data["weight"] / user_data["weight"].sum()

        # Calculate weighted averages
        avg_steps = (user_data["steps"] * user_data["weight"]).sum()
        avg_heart_rate = (user_data["heart_rate_avg"] * user_data["weight"]).sum()
        avg_sleep = (user_data["sleep_hours"] * user_data["weight"]).sum()
        avg_exercise = (user_data["exercise_minutes"] * user_data["weight"]).sum()
        avg_stress = (user_data["stress_level"] * user_data["weight"]).sum()
        avg_bmi = (user_data["bmi"] * user_data["weight"]).sum()
        avg_weight_kg = (user_data["weight_kg"] * user_data["weight"]).sum()
        avg_calories = (user_data["calories_burned"] * user_data["weight"]).sum()

        # Score each metric using shared healthy ranges (1 point per passing criterion)
        # Ranges are widened by noise_tolerance on each side to match the ±noise
        # applied during synthetic user generation
        health_score = 0

        steps_min, steps_max = steps_range(age)
        if steps_min * (1 - noise_tolerance) <= avg_steps <= steps_max * (1 + noise_tolerance):
            health_score += 1

        hr_min, hr_max = heart_rate_range(age)
        if hr_min * (1 - noise_tolerance) <= avg_heart_rate <= hr_max * (1 + noise_tolerance):
            health_score += 1

        sleep_min, sleep_max = sleep_range(age)
        if sleep_min * (1 - noise_tolerance) <= avg_sleep <= sleep_max * (1 + noise_tolerance):
            health_score += 1

        exercise_min, _ = exercise_range(age)
        if avg_exercise >= exercise_min * (1 - noise_tolerance):
            health_score += 1

        _, stress_max = stress_range(age)
        if avg_stress <= stress_max * (1 + noise_tolerance):
            health_score += 1

        bmi_min, bmi_max = bmi_range(age)
        if bmi_min * (1 - noise_tolerance) <= avg_bmi <= bmi_max * (1 + noise_tolerance):
            health_score += 1

        weight_min, weight_max = weight_kg_range(age)
        if weight_min * (1 - noise_tolerance) <= avg_weight_kg <= weight_max * (1 + noise_tolerance):
            health_score += 1

        cal_min, cal_max = calories_burned_range(age)
        if cal_min * (1 - noise_tolerance) <= avg_calories <= cal_max * (1 + noise_tolerance):
            health_score += 1

        # A user is healthy only if all 8 criteria are met
        is_training = 1 if health_score == 8 else 0

        user_health_scores.append(
            {
                "user_id": user_id,
                "age": age,
                "health_score": health_score,
                "is_training": is_training,
                "avg_steps": avg_steps,
                "avg_heart_rate": avg_heart_rate,
                "avg_sleep": avg_sleep,
                "avg_exercise": avg_exercise,
                "avg_stress": avg_stress,
                "avg_bmi": avg_bmi,
            }
        )

    return pd.DataFrame(user_health_scores)
