[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20688881.svg)](https://doi.org/10.5281/zenodo.20688881)

# Unsupervised Anomaly Detection for Industrial IoT Sensors

Master thesis project — Frankfurt School of Finance and Management

Implements and compares **13 anomaly detection methods** on temperature sensor data from the Testo Saveris IoT platform. The benchmark uses a synthetic injection framework with six physically grounded anomaly types to produce quantitative, reproducible results in an unsupervised setting (no real-world fault labels available).

## Methods

| # | Method | Category |
|---|--------|----------|
| 1 | Z-Score | Statistical |
| 2 | Moving Average | Statistical |
| 3 | STL Decomposition | Statistical |
| 4 | Isolation Forest | ML |
| 5 | Local Outlier Factor | ML |
| 6 | One-Class SVM | ML |
| 7 | Simple Autoencoder | Deep Learning |
| 8 | LSTM Autoencoder | Deep Learning |
| 9 | USAD | Deep Learning |
| 10 | Prophet | Bayesian / Forecasting |
| 11 | Domain Rules | Rule-Based |
| 12 | Chronos (amazon/chronos-t5-small) | Foundation Model |
| 13 | Ensemble | Combined |

## Benchmark

Six anomaly types are injected into a held-out copy of the data after all models are trained:

- **Point spike** — single-timestep ±σ excursion
- **Contextual** — normal values pasted at the wrong time of day
- **Collective** — sustained level shift
- **Drift** — linear ramp over 12 hours
- **Compressor failure** — exponential warm-up (Newton's law of cooling)
- **Door left open** — fast rise + exponential recovery

Evaluation uses Average Precision (AUC-PR) and F1@calib as primary metrics, both at the global (max-aggregated) and per-sensor level.

## Usage

Designed to run on **Google Colab**.

1. Add your Testo Saveris API token as a Colab Secret named `TESTO_API_TOKEN`
2. Open the notebook in Colab
3. Run all cells top-to-bottom

Individual sections can be re-run independently once `final_df` is loaded. Set `use_cached_data: True` in `CONFIG` to skip the API call after the first run.

## Dependencies

Installed automatically by the first two cells:

```
pip install pytorch_lightning chronos-forecasting
```

All other dependencies (TensorFlow, PyTorch, scikit-learn, statsmodels, prophet) are pre-installed on Colab.

## Data

Temperature readings from the Testo Saveris IoT platform. The notebook fetches data via the Saveris REST API for a configurable date range (default: 90 days). Raw data can be cached to Google Drive for offline re-runs.
