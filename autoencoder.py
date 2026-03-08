# Standard library
import os
import pickle

# Third-party
import pandas as pd
import torch
import matplotlib.pyplot as plt
import kagglehub
from torch.utils.data import DataLoader

# Local modules
from training.dataset import TrackingDataset, LSTMWindowDataset
from training.models import Autoencoder, LSTMAutoencoder
from data.data_loader import load_timeseries_data, build_lstm_windows
from training.training import train_autoencoder, train_lstm_autoencoder
from evaluation.feature_analysis import (
    calculate_feature_errors,
    calculate_feature_errors_lstm,
    analyze_features,
    analyze_features_lstm,
)
from evaluation.user_timeseries import analyze_user_timeseries, analyze_user_timeseries_lstm
from visualization.plotting import (
    plot_user_timeseries_errors,
    plot_training_vs_validation_timeseries,
    visualize_feature_contributions,
)
from evaluation.validation_analysis import (
    generate_detailed_validation_analysis,
    generate_detailed_validation_analysis_lstm,
)
from evaluation.model_comparison import compare_models


# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 32
ENCODING_DIM = 16
LEARNING_RATE = 0.001
EPOCHS = 100

# LSTM-specific hyperparameters — tuned separately from the standard AE
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 1
LSTM_LEARNING_RATE = 0.0005
LSTM_EPOCHS = 100

AE_PTH = "healthy_autoencoder.pth"
LSTM_PTH = "lstm_healthy_autoencoder.pth"
SCALER_PKL = "scaler.pkl"


# =============================================================================
# Data loading
# =============================================================================

# Download the Kaggle dataset
waqasishtiaq_fitness_path = kagglehub.dataset_download("waqasishtiaq/fitness")
file_path = waqasishtiaq_fitness_path + "/health_fitness_tracking_365days.csv"

print("Loading time series data...")
X, y, scaler, df, feature_names, user_health_df = load_timeseries_data(file_path)

max_date = df["date"].max()

print(f"\nDataset Info:")
print(f"Total records: {len(df)}")
print(f"Total users: {df['user_id'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nUser Health Distribution:")
print(user_health_df["is_training"].value_counts())

# Split records into training and validation subsets
training_data = X[y == 1]
validation_data = X[y == 0]

print(f"\nTraining records: {len(training_data)}")
print(f"Validation records: {len(validation_data)}")

# Build a DataLoader using only healthy records — the autoencoder learns normal patterns
training_dataset = TrackingDataset(training_data)
training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)


# =============================================================================
# Standard Autoencoder — training and error analysis
# =============================================================================

input_dim = X.shape[1]
model = Autoencoder(input_dim, ENCODING_DIM)

if os.path.exists(AE_PTH):
    print(f"\nFound existing model '{AE_PTH}' — skipping training, loading weights...")
    model.load_state_dict(torch.load(AE_PTH, weights_only=True))
    model.eval()
    losses = []
else:
    print("\nTraining autoencoder on training users' time series data...")
    losses = train_autoencoder(model, training_loader, EPOCHS, LEARNING_RATE)
    torch.save(model.state_dict(), AE_PTH)
    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved model to '{AE_PTH}' and scaler to '{SCALER_PKL}'")

# Calculate per-feature reconstruction errors across the full dataset
print("\nCalculating reconstruction errors...")
all_errors = calculate_feature_errors(model, X, feature_names)

# Attach errors back to the dataframe for downstream analysis and export
df["total_error"] = all_errors["total_error"].values
for feature in feature_names:
    df[f"{feature}_error"] = all_errors[feature].values

# Compare how each feature contributes to the training/validation separation
print("\nAnalyzing feature contributions...")
feature_contribution, training_errors, validation_errors = analyze_features(
    model, training_data, validation_data, feature_names
)

print("\n" + "=" * 80)
print("FEATURE CONTRIBUTIONS TO VALIDATION DETECTION")
print("=" * 80)
print(feature_contribution.to_string(index=False))

# Summarise per-user reconstruction error statistics
print("\nAnalyzing per-user time series...")
user_analysis = analyze_user_timeseries(
    model, df, X, feature_names, scaler, user_health_df
)
user_analysis_sorted = user_analysis.sort_values(
    "avg_reconstruction_error", ascending=False
)

print("\n" + "=" * 80)
print("TOP 10 USERS WITH HIGHEST AVERAGE RECONSTRUCTION ERROR")
print("=" * 80)
print(user_analysis_sorted.head(10).to_string(index=False))

# Save user time series analysis to CSV
user_analysis_sorted.to_csv("user_timeseries_analysis.csv", index=False)
print("\nSaved user time series analysis to user_timeseries_analysis.csv")

print("\n" + "=" * 80)
print("VALIDATION USERS ANALYSIS")
print("=" * 80)
validation_users = user_analysis[user_analysis["is_training"] == 0].sort_values(
    "avg_reconstruction_error", ascending=False
)

# Save validation users analysis to CSV
validation_users.to_csv("validation_users_analysis.csv", index=False)
print("Saved validation users analysis to validation_users_analysis.csv")

print(
    validation_users[
        [
            "user_id",
            "first_date",
            "last_date",
            "health_score",
            "avg_reconstruction_error",
            "max_reconstruction_error",
            "std_reconstruction_error",
            "top_problem_feature_1",
            "top_problem_error_1",
            "num_days",
        ]
    ].to_string(index=False)
)


# =============================================================================
# Visualizations — Standard Autoencoder
# =============================================================================

print("\nGenerating visualizations...")
visualize_feature_contributions(feature_contribution)

# Sample two training and two validation users for time-series plots
sample_training_users = (
    user_health_df[user_health_df["is_training"] == 1]["user_id"].head(2).tolist()
)
sample_validation_users = (
    user_health_df[user_health_df["is_training"] == 0]["user_id"].head(2).tolist()
)
sample_users = sample_training_users + sample_validation_users

plot_user_timeseries_errors(df, sample_users)
plot_training_vs_validation_timeseries(df)


# =============================================================================
# Detailed validation user report
# =============================================================================

print("\nGenerating detailed validation user analysis...")
detailed_validation = generate_detailed_validation_analysis(df, feature_names)

print("\n" + "=" * 80)
print("DETAILED VALIDATION USERS ANALYSIS - ALL RECORDS WITH DATES")
print("=" * 80)
print(f"Total validation records: {len(detailed_validation)}")
detailed_validation.to_csv("validation_detailed_analysis.csv", index=False)
print("Saved detailed validation analysis to validation_detailed_analysis.csv")
print("\nShowing first 50 records:")
print(detailed_validation.head(50).to_string(index=False))


training_avg_error = df[df["is_training"] == 1]["total_error"].mean()
validation_avg_error = df[df["is_training"] == 0]["total_error"].mean()

print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
print(f"Average reconstruction error (training users): {training_avg_error:.4f}")
print(f"Average reconstruction error (validation users): {validation_avg_error:.4f}")
print(
    f"Error increase for validation: {(validation_avg_error/training_avg_error - 1)*100:.2f}%"
)

print("\nAnalysis complete!")


# =============================================================================
# LSTM Autoencoder — build 7-day sliding windows
# =============================================================================

LSTM_SEQ_LEN = 7

print(f"\nBuilding {LSTM_SEQ_LEN}-day sliding windows for LSTM...")
X_windows, y_windows, user_ids_windows, window_end_dates = build_lstm_windows(df, X, seq_len=LSTM_SEQ_LEN)

lstm_training_data = X_windows[y_windows == 1]
lstm_validation_data = X_windows[y_windows == 0]

print(f"Total windows: {len(X_windows)}")
print(f"Training windows: {len(lstm_training_data)}")
print(f"Validation windows: {len(lstm_validation_data)}")

lstm_training_dataset = LSTMWindowDataset(lstm_training_data)
lstm_training_loader = DataLoader(
    lstm_training_dataset, batch_size=BATCH_SIZE, shuffle=True
)


# =============================================================================
# LSTM Autoencoder — training and error analysis
# =============================================================================

lstm_model = LSTMAutoencoder(
    input_dim=input_dim,
    seq_len=LSTM_SEQ_LEN,
    hidden_dim=LSTM_HIDDEN_DIM,
    encoding_dim=ENCODING_DIM,
    num_layers=LSTM_NUM_LAYERS,
)

if os.path.exists(LSTM_PTH):
    print(f"\nFound existing model '{LSTM_PTH}' — skipping training, loading weights...")
    lstm_model.load_state_dict(torch.load(LSTM_PTH, weights_only=True))
    lstm_model.eval()
    lstm_losses = []
else:
    print("\n" + "=" * 80)
    print("TRAINING LSTM AUTOENCODER")
    print("=" * 80)
    print("\nTraining LSTM autoencoder on training users' 7-day windows...")
    lstm_losses = train_lstm_autoencoder(
        lstm_model, lstm_training_loader, LSTM_EPOCHS, LSTM_LEARNING_RATE
    )
    torch.save(lstm_model.state_dict(), LSTM_PTH)
    print(f"Saved LSTM model to '{LSTM_PTH}'")

# Calculate per-feature reconstruction errors on all windows
print("\nCalculating LSTM reconstruction errors...")
lstm_all_errors = calculate_feature_errors_lstm(lstm_model, X_windows, feature_names)

# Identify which features drive the LSTM's validation detection
print("\nAnalyzing LSTM feature contributions...")
lstm_feature_contribution, lstm_training_errors, lstm_validation_errors = (
    analyze_features_lstm(
        lstm_model, lstm_training_data, lstm_validation_data, feature_names
    )
)

print("\n" + "=" * 80)
print("LSTM FEATURE CONTRIBUTIONS TO VALIDATION DETECTION")
print("=" * 80)
print(lstm_feature_contribution.to_string(index=False))

# Per-user LSTM reconstruction error analysis
print("\nAnalyzing per-user LSTM time series...")
lstm_user_analysis = analyze_user_timeseries_lstm(
    lstm_model, X_windows, user_ids_windows, feature_names, user_health_df
)
lstm_user_analysis_sorted = lstm_user_analysis.sort_values(
    "avg_reconstruction_error", ascending=False
)

print("\n" + "=" * 80)
print("LSTM — TOP 10 USERS WITH HIGHEST AVERAGE RECONSTRUCTION ERROR")
print("=" * 80)
print(lstm_user_analysis_sorted.head(10).to_string(index=False))

lstm_user_analysis_sorted.to_csv("lstm_user_timeseries_analysis.csv", index=False)
print("\nSaved LSTM user analysis to lstm_user_timeseries_analysis.csv")

lstm_validation_users = lstm_user_analysis[
    lstm_user_analysis["is_training"] == 0
].sort_values("avg_reconstruction_error", ascending=False)

lstm_validation_users.to_csv("lstm_validation_users_analysis.csv", index=False)
print("Saved LSTM validation user analysis to lstm_validation_users_analysis.csv")

# Detailed per-window analysis for validation users
print("\nGenerating detailed LSTM validation analysis...")
lstm_detailed_validation = generate_detailed_validation_analysis_lstm(
    lstm_model, X_windows, y_windows, user_ids_windows, window_end_dates, feature_names
)
lstm_detailed_validation.to_csv("lstm_validation_detailed_analysis.csv", index=False)
print(f"Total validation windows: {len(lstm_detailed_validation)}")
print("Saved detailed LSTM validation analysis to lstm_validation_detailed_analysis.csv")
print("\nShowing first 50 records:")
print(lstm_detailed_validation.head(50).to_string(index=False))

lstm_training_avg_error = lstm_all_errors[y_windows == 1]["total_error"].mean()
lstm_validation_avg_error = lstm_all_errors[y_windows == 0]["total_error"].mean()

print("\n" + "=" * 80)
print("LSTM AUTOENCODER STATISTICS")
print("=" * 80)
print(f"Average reconstruction error (training users): {lstm_training_avg_error:.4f}")
print(
    f"Average reconstruction error (validation users): {lstm_validation_avg_error:.4f}"
)
print(
    f"Error increase for validation: {(lstm_validation_avg_error/lstm_training_avg_error - 1)*100:.2f}%"
)

# =============================================================================
# Visualizations — LSTM Autoencoder
# =============================================================================

print("\nGenerating visualizations...")
visualize_feature_contributions(lstm_feature_contribution)

# Build a window-level DataFrame for LSTM plots (one row per sliding window)
lstm_plot_df = pd.DataFrame({
    "user_id": user_ids_windows,
    "date": window_end_dates,
    "is_training": y_windows,
    "total_error": lstm_all_errors["total_error"].values,
})

# Sample two training and two validation users for time-series plots
sample_training_users = (
    user_health_df[user_health_df["is_training"] == 1]["user_id"].head(2).tolist()
)
sample_validation_users = (
    user_health_df[user_health_df["is_training"] == 0]["user_id"].head(2).tolist()
)
sample_users = sample_training_users + sample_validation_users

plot_user_timeseries_errors(lstm_plot_df, sample_users)
plot_training_vs_validation_timeseries(lstm_plot_df)

print("\n" + "=" * 80)
print("LSTM AUTOENCODER ANALYSIS COMPLETE!")
print("=" * 80)

# =============================================================================
# Compare Standard AE vs LSTM AE
# =============================================================================

# Training loss curves (only shown when both models were freshly trained)
if losses and lstm_losses:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Standard Autoencoder", linewidth=2)
    plt.plot(lstm_losses, label="LSTM Autoencoder", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Standard Autoencoder", linewidth=2)
    plt.plot(lstm_losses, label="LSTM Autoencoder", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.yscale("log")
    plt.title("Training Loss Comparison (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Side-by-side comparison of Standard AE vs LSTM AE
comparison_results, feature_comparison = compare_models(
    model,
    lstm_model,
    training_data,
    validation_data,
    feature_names,
    lstm_training_data=lstm_training_data,
    lstm_validation_data=lstm_validation_data,
)
