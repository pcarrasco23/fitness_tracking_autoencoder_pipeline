import torch
import pandas as pd


def calculate_feature_errors(model, data, feature_names):
    model.eval()

    with torch.no_grad():
        data_tensor = torch.FloatTensor(data)
        reconstructed = model(data_tensor)

        # Calculate squared error for each feature
        feature_errors = (data_tensor - reconstructed) ** 2
        feature_errors = feature_errors.numpy()

    error_df = pd.DataFrame(feature_errors, columns=feature_names)
    error_df["total_error"] = feature_errors.sum(axis=1)

    return error_df


def calculate_feature_errors_lstm(model, data, feature_names):
    """
    Calculate per-window reconstruction errors for the LSTM autoencoder.

    Args:
        data: np.ndarray of shape (N, seq_len, num_features)

    Returns:
        DataFrame with one row per window; columns are feature names + total_error.
        Feature errors are averaged over the seq_len dimension so units match
        the standard autoencoder output.
    """
    model.eval()

    with torch.no_grad():
        # data is already (N, seq_len, num_features)
        data_tensor = torch.FloatTensor(data)
        reconstructed = model(data_tensor)  # (N, seq_len, num_features)

        # Squared error per timestep, then average over seq_len → (N, num_features)
        feature_errors = ((data_tensor - reconstructed) ** 2).mean(dim=1).numpy()

    error_df = pd.DataFrame(feature_errors, columns=feature_names)
    error_df["total_error"] = feature_errors.sum(axis=1)

    return error_df


def analyze_features(model, training_data, validation_data, feature_names):
    """
    Compare feature reconstruction errors between training and validation users
    """
    training_errors = calculate_feature_errors(model, training_data, feature_names)
    validation_errors = calculate_feature_errors(model, validation_data, feature_names)

    training_mean = training_errors[feature_names].mean()
    validation_mean = validation_errors[feature_names].mean()
    error_difference = validation_mean - training_mean

    feature_contribution = pd.DataFrame(
        {
            "feature": feature_names,
            "training_error": training_mean.values,
            "validation_error": validation_mean.values,
            "error_difference": error_difference.values,
            "relative_increase": (
                (validation_mean - training_mean) / (training_mean + 1e-8) * 100
            ).values,
        }
    ).sort_values("error_difference", ascending=False)

    return feature_contribution, training_errors, validation_errors


def analyze_features_lstm(model, training_data, validation_data, feature_names):
    """
    Compare feature reconstruction errors between training and validation users for LSTM
    """
    training_errors = calculate_feature_errors_lstm(model, training_data, feature_names)
    validation_errors = calculate_feature_errors_lstm(
        model, validation_data, feature_names
    )

    training_mean = training_errors[feature_names].mean()
    validation_mean = validation_errors[feature_names].mean()
    error_difference = validation_mean - training_mean

    feature_contribution = pd.DataFrame(
        {
            "feature": feature_names,
            "training_error": training_mean.values,
            "validation_error": validation_mean.values,
            "error_difference": error_difference.values,
            "relative_increase": (
                (validation_mean - training_mean) / (training_mean + 1e-8) * 100
            ).values,
        }
    ).sort_values("error_difference", ascending=False)

    return feature_contribution, training_errors, validation_errors
