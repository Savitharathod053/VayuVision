"""
scripts/generate_xgboost_predictions.py

Regenerates data/processed/xgboost_predictions.csv using the CURRENT production model
(models/pm25_xgboost_model.pkl = V3 extended) on the chronological 20% holdout
from master_dataset_extended.csv.

This feeds the /model-trust endpoint with real V3 residuals for the Model Trust page.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH  = Path("models/pm25_xgboost_model.pkl")
DATA_PATH   = Path("data/processed/master_dataset_extended.csv")
OUTPUT_PATH = Path("data/processed/xgboost_predictions.csv")

FEATURES = [
    "pm25_value","pm25_lag_1","pm25_lag_3","pm25_lag_6","pm25_lag_12","pm25_lag_24",
    "pm25_roll_3","pm25_roll_6","pm25_roll_12","pm25_roll_24",
    "temperature_2m","relative_humidity_2m","wind_speed_10m","wind_sin","wind_cos",
    "surface_pressure","precipitation","boundary_layer_height",
    "fire_count_punjab","fire_count_haryana","fire_count_up","fire_count_delhi",
    "hour","day","month","day_of_week","latitude","longitude",
]
TARGET = "target_pm25_1h"

def main():
    print("=" * 65)
    print("VayuDrishti: Regenerating xgboost_predictions.csv (V3 model)")
    print("=" * 65)

    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}  ({MODEL_PATH.stat().st_size:,} bytes)")

    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["dt", "station_name"]).reset_index(drop=True)
    df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
    print(f"Total clean rows: {len(df_clean):,}")

    # Chronological 20% holdout (same split used in trust evaluation)
    n_test = int(len(df_clean) * 0.20)
    test_df = df_clean.iloc[-n_test:].copy()
    print(f"Test rows: {len(test_df):,}")
    print(f"Test range: {test_df['dt'].min()}  →  {test_df['dt'].max()}")

    X_test = test_df[FEATURES]
    y_pred = np.maximum(0.0, model.predict(X_test))
    y_actual = test_df[TARGET].values

    out_df = pd.DataFrame({
        "station":       test_df["station_name"].values,
        "timestamp":     test_df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "actual_pm25":   np.round(y_actual, 2),
        "predicted_pm25": np.round(y_pred, 2),
    })

    overall_mae  = float(np.mean(np.abs(y_actual - y_pred)))
    overall_rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    overall_r2 = float(1.0 - ss_res / ss_tot)

    # Per-range breakdown
    masks = {
        "0-100":   y_actual < 100,
        "100-200": (y_actual >= 100) & (y_actual < 200),
        "200-300": (y_actual >= 200) & (y_actual < 300),
        "300+":    y_actual >= 300,
    }

    print(f"\n=== V3 Production Model — 20% Holdout Evaluation ===")
    print(f"  Total test samples : {len(y_actual):,}")
    print(f"  Overall MAE        : {overall_mae:.4f}")
    print(f"  Overall RMSE       : {overall_rmse:.4f}")
    print(f"  Overall R²         : {overall_r2:.4f}")
    print(f"\n  Per-range breakdown:")
    print(f"  {'Range':<12} {'N':>7}  {'MAE':>8}  {'RMSE':>8}")
    print(f"  {'-'*42}")
    for rng, mask in masks.items():
        n = mask.sum()
        if n > 0:
            r_mae  = float(np.mean(np.abs(y_actual[mask] - y_pred[mask])))
            r_rmse = float(np.sqrt(np.mean((y_actual[mask] - y_pred[mask])**2)))
            print(f"  {rng:<12} {n:>7,}  {r_mae:>8.4f}  {r_rmse:>8.4f}")
        else:
            print(f"  {rng:<12} {n:>7,}  {'N/A':>8}  {'N/A':>8}")

    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(out_df):,} rows → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
