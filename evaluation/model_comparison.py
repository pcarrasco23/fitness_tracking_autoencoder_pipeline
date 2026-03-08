import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .feature_analysis import (
    calculate_feature_errors,
    calculate_feature_errors_lstm,
    analyze_features,
    analyze_features_lstm,
)


def compare_models(
    standard_model,
    lstm_model,
    training_data,
    validation_data,
    feature_names,
    lstm_training_data,
    lstm_validation_data,
):
    """
    Compare performance of standard autoencoder vs LSTM autoencoder.

    Args:
        training_data:        2D array (N, features) for the standard autoencoder
        validation_data:      2D array (N, features) for the standard autoencoder
        lstm_training_data:   3D array (N, seq_len, features) for the LSTM
        lstm_validation_data: 3D array (N, seq_len, features) for the LSTM
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: Standard Autoencoder vs LSTM Autoencoder")
    print("=" * 80)

    # Calculate errors for both models
    print("\nCalculating reconstruction errors for Standard Autoencoder...")
    standard_training_errors = calculate_feature_errors(
        standard_model, training_data, feature_names
    )
    standard_validation_errors = calculate_feature_errors(
        standard_model, validation_data, feature_names
    )

    print("Calculating reconstruction errors for LSTM Autoencoder...")
    lstm_training_errors = calculate_feature_errors_lstm(
        lstm_model, lstm_training_data, feature_names
    )
    lstm_validation_errors = calculate_feature_errors_lstm(
        lstm_model, lstm_validation_data, feature_names
    )

    # Calculate average total errors
    standard_training_avg = standard_training_errors["total_error"].mean()
    standard_validation_avg = standard_validation_errors["total_error"].mean()
    lstm_training_avg = lstm_training_errors["total_error"].mean()
    lstm_validation_avg = lstm_validation_errors["total_error"].mean()

    # Calculate separation (how well each model distinguishes healthy from unhealthy)
    standard_separation = standard_validation_avg / (standard_training_avg + 1e-8)
    lstm_separation = lstm_validation_avg / (lstm_training_avg + 1e-8)

    comparison_results = pd.DataFrame(
        {
            "Model": ["Standard Autoencoder", "LSTM Autoencoder"],
            "Training Avg Error": [standard_training_avg, lstm_training_avg],
            "Validation Avg Error": [standard_validation_avg, lstm_validation_avg],
            "Separation Ratio": [standard_separation, lstm_separation],
            "Error Increase (%)": [
                (standard_validation_avg / standard_training_avg - 1) * 100,
                (lstm_validation_avg / lstm_training_avg - 1) * 100,
            ],
        }
    )

    print("\n" + "=" * 80)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 80)
    print(comparison_results.to_string(index=False))

    # Feature-level comparison
    standard_feature_contrib, _, _ = analyze_features(
        standard_model, training_data, validation_data, feature_names
    )
    lstm_feature_contrib, _, _ = analyze_features_lstm(
        lstm_model, lstm_training_data, lstm_validation_data, feature_names
    )

    # Merge feature contributions
    feature_comparison = pd.merge(
        standard_feature_contrib[["feature", "error_difference"]].rename(
            columns={"error_difference": "standard_error_diff"}
        ),
        lstm_feature_contrib[["feature", "error_difference"]].rename(
            columns={"error_difference": "lstm_error_diff"}
        ),
        on="feature",
    )

    print("\n" + "=" * 80)
    print("FEATURE-LEVEL COMPARISON (Error Difference: Unhealthy - Healthy)")
    print("=" * 80)
    print(feature_comparison.to_string(index=False))

    # Visualize comparison
    visualize_model_comparison(comparison_results, feature_comparison)

    return comparison_results, feature_comparison


def visualize_model_comparison(comparison_results, feature_comparison):
    """
    Visualize comparison between Standard and LSTM Autoencoders
    """
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Overall Error Comparison
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(comparison_results))
    width = 0.35

    ax1.bar(
        x - width / 2,
        comparison_results["Training Avg Error"],
        width,
        label="Training",
        alpha=0.7,
        color="green",
    )
    ax1.bar(
        x + width / 2,
        comparison_results["Validation Avg Error"],
        width,
        label="Validation",
        alpha=0.7,
        color="red",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_results["Model"], rotation=15, ha="right")
    ax1.set_ylabel("Average Reconstruction Error")
    ax1.set_title("Overall Error: Training vs Validation")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Separation Ratio
    ax2 = plt.subplot(2, 2, 2)
    colors = ["#1f77b4", "#ff7f0e"]
    ax2.bar(
        comparison_results["Model"],
        comparison_results["Separation Ratio"],
        color=colors,
        alpha=0.7,
    )
    ax2.set_ylabel("Separation Ratio (Validation/Training)")
    ax2.set_title("Model Separation Performance\n(Higher = Better at Distinguishing)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=15)

    # Plot 3: Feature-level comparison
    ax3 = plt.subplot(2, 2, 3)
    x = np.arange(len(feature_comparison))
    width = 0.35

    ax3.bar(
        x - width / 2,
        feature_comparison["standard_error_diff"],
        width,
        label="Standard AE",
        alpha=0.7,
        color="#1f77b4",
    )
    ax3.bar(
        x + width / 2,
        feature_comparison["lstm_error_diff"],
        width,
        label="LSTM AE",
        alpha=0.7,
        color="#ff7f0e",
    )

    ax3.set_xticks(x)
    ax3.set_xticklabels(feature_comparison["feature"], rotation=45, ha="right")
    ax3.set_ylabel("Error Difference (Unhealthy - Healthy)")
    ax3.set_title("Feature Contributions by Model")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Error Increase Percentage
    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(
        comparison_results["Model"],
        comparison_results["Error Increase (%)"],
        color=colors,
        alpha=0.7,
    )
    ax4.set_ylabel("Error Increase (%)")
    ax4.set_title("Percentage Error Increase\n(Unhealthy vs Healthy)")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()
