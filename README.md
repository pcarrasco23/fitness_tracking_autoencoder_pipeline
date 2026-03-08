# Smartwatch Autoencoder — Health Anomaly Detection

A health anomaly detection system that uses autoencoder neural networks trained on smartwatch data to identify users whose health patterns deviate from healthy norms. Two architectures are implemented and compared: a standard feedforward autoencoder and an LSTM-based sequence autoencoder.

---

## Overview

Because labelled unhealthy data is scarce, both models are trained exclusively on **healthy** users (real + synthetic). At inference time, the reconstruction error for any user measures how much their data deviates from the healthy baseline the model has learned. High reconstruction error signals an anomaly.

The pipeline produces per-user and per-day CSV reports identifying which users deviate from healthy norms and which specific health metrics are driving the deviation.

---

## Architecture

### Standard Autoencoder
Operates on individual daily records (one row = one day).

```
Input (9 features) → 32 → 16 → Bottleneck (16) → 16 → 32 → Output (9 features)
```

### LSTM Autoencoder
Operates on 7-day sliding windows, capturing temporal patterns.

```
Input (7 days × 9 features)
  → LSTM Encoder → Latent vector (16)
  → LSTM Decoder → Reconstructed (7 days × 9 features)
```

---

## Project Structure

```
smartwatch_autoencoder/
├── autoencoder.py                  # Main entry point — orchestrates the full pipeline
│
├── data/
│   ├── data_loader.py              # Load CSV, normalize features, build LSTM windows
│   ├── health_criteria.py          # Age-adjusted healthy ranges (WHO/CDC/AHA/NSF guidelines)
│   ├── health_scores.py            # Per-user health scoring with exponential decay weighting
│   └── synthetic_healthy_users.py  # Generate 200 synthetic healthy users (±3% noise)
│
├── training/
│   ├── models.py                   # Autoencoder and LSTMAutoencoder model definitions
│   ├── dataset.py                  # PyTorch Dataset wrappers for both model types
│   └── training.py                 # Training loops with gradient clipping for LSTM
│
├── evaluation/
│   ├── feature_analysis.py         # Per-feature reconstruction error calculation
│   ├── user_timeseries.py          # Per-user error aggregation and top problem features
│   ├── validation_analysis.py      # Detailed per-record/per-window validation reports
│   └── model_comparison.py         # Side-by-side Standard AE vs LSTM AE comparison
│
├── visualization/
│   └── plotting.py                 # Time-series error plots, feature contribution charts
│
└── generate_pptx.py                # Generate PowerPoint presentations from results
```

---

## Features

- **9 health metrics**: steps, heart rate, sleep hours, calories burned, exercise minutes, stress level, weight, BMI, gender
- **Age-adjusted healthy ranges** based on WHO, CDC, AHA, NSF, and Mifflin-St Jeor guidelines
- **200 synthetic healthy users** generated to augment the training set
- **Model caching**: if `.pth` files exist, models are loaded rather than retrained
- **Per-feature anomaly attribution**: identifies which metrics drive each user's anomaly score
- **Dual model evaluation**: both architectures produce equivalent per-user CSV reports

---

## Configuration

| Constant | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 32 | DataLoader batch size |
| `ENCODING_DIM` | 16 | Bottleneck latent dimension (both models) |
| `LEARNING_RATE` | 0.001 | Standard AE learning rate |
| `EPOCHS` | 100 | Standard AE training epochs |
| `LSTM_HIDDEN_DIM` | 64 | LSTM hidden state size |
| `LSTM_NUM_LAYERS` | 1 | Stacked LSTM layers |
| `LSTM_LEARNING_RATE` | 0.0005 | LSTM learning rate |
| `LSTM_EPOCHS` | 100 | LSTM training epochs |
| `LSTM_SEQ_LEN` | 7 | Sliding window length (days) |

---

## Usage

```bash
python autoencoder.py
```

On the first run, both models are trained and weights are saved. On subsequent runs, the saved weights are loaded and only the evaluation phase runs.

### Prerequisites

```bash
pip install torch pandas scikit-learn matplotlib kagglehub python-pptx
```

The dataset is downloaded automatically via `kagglehub` (`waqasishtiaq/fitness`).

---

## Output Files

### Model Artifacts
| File | Description |
|---|---|
| `healthy_autoencoder.pth` | Trained standard autoencoder weights |
| `lstm_healthy_autoencoder.pth` | Trained LSTM autoencoder weights |
| `scaler.pkl` | Fitted StandardScaler for feature normalization |

### Standard Autoencoder Reports
| File | Description |
|---|---|
| `user_timeseries_analysis.csv` | Per-user summary: avg/max/std error, top 3 problem features |
| `validation_users_analysis.csv` | Same, filtered to validation (anomalous) users only |
| `validation_detailed_analysis.csv` | Per-day breakdown with feature-level error contributions |

### LSTM Autoencoder Reports
| File | Description |
|---|---|
| `lstm_user_timeseries_analysis.csv` | Per-user summary (averaged across 7-day windows) |
| `lstm_validation_users_analysis.csv` | Same, filtered to validation users only |
| `lstm_validation_detailed_analysis.csv` | Per-window breakdown with feature-level error contributions |

---

## How to Interpret Results

**`avg_reconstruction_error`** is the primary anomaly severity signal. Users whose patterns are unfamiliar to the model (trained only on healthy data) will produce high reconstruction errors.

**`top_problem_feature_1/2/3`** identify which health metrics are driving the anomaly, enabling targeted intervention:

| Feature | High error suggests |
|---|---|
| `heart_rate_avg` | Cardiovascular concern |
| `sleep_hours` | Sleep disorder or poor sleep hygiene |
| `stress_level` | Mental health or burnout risk |
| `steps` / `exercise_minutes` | Sedentary behaviour |
| `bmi` / `weight_kg` | Metabolic or nutritional concern |
| `calories_burned` | Metabolic abnormality |

Use the per-day detail CSVs to identify *when* a user's anomaly score began rising — useful for correlating with specific events or periods of deterioration.

---

## Why Autoencoders vs Traditional Analytics

Traditional rule-based analytics detects deviations in individual metrics. Autoencoders detect deviations in the **joint pattern** of all metrics simultaneously — a user whose heart rate, sleep, and stress are all slightly elevated (but none cross a threshold individually) will produce a high reconstruction error because the overall combination is unfamiliar to the model.
