"""
scripts/recompute_uncertainty_buckets.py

Recomputes uncertainty_buckets.csv using the PRODUCTION V3 model's actual residuals
on a chronological 20% holdout from master_dataset_extended.csv.

Bucket decision rules (minimum 30 samples for a reliable CI):
  - 0-100:   always included
  - 100-200: always included
  - 200-300: include if >= 30 samples
  - 300+:    include if >= 30 samples, else note as insufficient

CI = 10th to 90th percentile of (actual - predicted) residuals per bucket.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH   = Path("models/pm25_xgboost_model.pkl")
DATA_PATH    = Path("data/processed/master_dataset_extended.csv")
OUTPUT_PATH  = Path("data/processed/uncertainty_buckets.csv")
MIN_SAMPLES  = 30   # minimum to compute a reliable bucket CI

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
    print("VayuDrishti: Recomputing Uncertainty Buckets (V3 model)")
    print("=" * 65)

    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}  ({MODEL_PATH.stat().st_size:,} bytes)")

    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["dt", "station_name"]).reset_index(drop=True)
    df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
    print(f"Total clean rows: {len(df_clean):,}")

    # Chronological 20% test split (same as training evaluation)
    n_test = int(len(df_clean) * 0.20)
    test_df = df_clean.iloc[-n_test:].copy()
    print(f"Test rows (last 20%): {len(test_df):,}")
    print(f"Test date range: {test_df['dt'].min()}  →  {test_df['dt'].max()}")

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].values
    y_pred = np.maximum(0.0, model.predict(X_test))
    residuals = y_test - y_pred  # positive = under-prediction, negative = over-prediction

    buckets_def = [
        ("0-100",   (y_test <  100)),
        ("100-200", (y_test >= 100) & (y_test < 200)),
        ("200-300", (y_test >= 200) & (y_test < 300)),
        ("300-999", (y_test >= 300)),
    ]

    rows = []
    print("\nBucket analysis:")
    print(f"  {'Bucket':<12} {'N':>6}  {'p10 resid':>10}  {'p90 resid':>10}  {'Include?'}")
    print("  " + "-" * 60)

    for bucket_name, mask in buckets_def:
        n = mask.sum()
        if n >= MIN_SAMPLES:
            r = residuals[mask]
            p10 = float(np.percentile(r, 10))
            p90 = float(np.percentile(r, 90))
            include = True
            note = "✓ included"
        else:
            p10 = p90 = np.nan
            include = False
            note = f"✗ only {n} samples (need {MIN_SAMPLES})"

        print(f"  {bucket_name:<12} {n:>6}  {p10:>10.2f}  {p90:>10.2f}  {note}")

        if include:
            rows.append({
                "bucket": bucket_name,
                "lower_error": round(p10, 4),
                "upper_error": round(p90, 4),
                "samples": int(n),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(rows)} buckets → {OUTPUT_PATH}")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()
