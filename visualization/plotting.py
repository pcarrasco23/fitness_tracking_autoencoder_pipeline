import numpy as np
import matplotlib.pyplot as plt


def plot_user_timeseries_errors(df, user_ids):
    """
    Plot reconstruction errors over time for specific users
    Shows both weighted and unweighted for validation users
    """
    n_users = len(user_ids)
    fig, axes = plt.subplots(n_users, 1, figsize=(14, 4 * n_users))

    if n_users == 1:
        axes = [axes]

    for idx, user_id in enumerate(user_ids):
        user_data = df[df["user_id"] == user_id].sort_values("date")

        # Plot error
        axes[idx].plot(
            user_data["date"],
            user_data["total_error"],
            marker="o",
            linestyle="-",
            markersize=3,
            label="Error",
            alpha=0.7,
        )

        axes[idx].set_xlabel("Date")
        axes[idx].set_ylabel("Reconstruction Error")

        health_status = (
            "Training" if user_data["is_training"].iloc[0] == 1 else "Validation"
        )
        axes[idx].set_title(
            f"User {user_id} - {health_status} - Reconstruction Error Over Time"
        )
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# Plot comparison of training vs validation users over time
def plot_training_vs_validation_timeseries(df):
    """
    Plot average reconstruction errors for training vs validation users over time
    """
    # Group by date and health status for unweighted errors
    daily_errors = (
        df.groupby(["date", "is_training"])["total_error"].mean().reset_index()
    )

    plt.figure(figsize=(14, 5))

    for is_training in [0, 1]:
        data = daily_errors[daily_errors["is_training"] == is_training]
        label = "Training" if is_training == 1 else "Validation"
        plt.plot(data["date"], data["total_error"], label=label, linewidth=2, alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Average Reconstruction Error")
    plt.title("Average Reconstruction Error: Training vs Validation Users")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# Visualize feature contributions
def visualize_feature_contributions(feature_contribution):
    """
    Visualize feature contributions to unhealthy detection
    (Simplified - no temporal weighting)
    """
    plt.figure(figsize=(14, 6))

    # Plot 1: Error comparison (training vs validation)
    plt.subplot(1, 2, 1)
    x = np.arange(len(feature_contribution))
    width = 0.35

    plt.bar(
        x - width / 2,
        feature_contribution["training_error"],
        width,
        label="Training",
        alpha=0.7,
        color="green",
    )
    plt.bar(
        x + width / 2,
        feature_contribution["validation_error"],
        width,
        label="Validation",
        alpha=0.7,
        color="red",
    )

    plt.xticks(x, feature_contribution["feature"], rotation=45, ha="right")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error:\nTraining vs Validation Users")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    # Plot 2: Error difference
    plt.subplot(1, 2, 2)
    colors = [
        "red" if x > 0 else "green" for x in feature_contribution["error_difference"]
    ]
    plt.barh(
        feature_contribution["feature"],
        feature_contribution["error_difference"],
        color=colors,
        alpha=0.7,
    )
    plt.xlabel("Error Difference (Validation - Training)")
    plt.title("Feature Contribution to\nValidation Detection")
    plt.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()
