"""
scripts/winter_holdout_validation.py

Honest winter-holdout validation for VayuDrishti PM2.5 forecasting.

Protocol:
  - Hold out: January 2026 (entirely unseen by the validation model)
  - Train on: all other data (Feb 2025 - Sep 2026, excluding Jan 2026)
  - Validation model uses IDENTICAL features + hyperparameters as production V3
  - Saved to: models/pm25_xgboost_validation_jan_holdout.pkl
  - Does NOT touch or replace the production model (pm25_xgboost_model_v3_extended.pkl)

Three-way comparison on Jan 2026 holdout:
  1. Persistence baseline (current t predicts t+1)
  2. V1 original model (trained only on 90-day summer data, never saw winter)
  3. This validation model (saw most of winter but NOT January specifically)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH      = Path("data/processed/master_dataset_extended.csv")
V1_MODEL_PATH  = Path("models/pm25_xgboost_model.pkl")          # original summer-only model
V3_MODEL_PATH  = Path("models/pm25_xgboost_model_v3_extended.pkl")  # production - DO NOT OVERWRITE
HOLDOUT_MODEL  = Path("models/pm25_xgboost_validation_jan_holdout.pkl")

# ---------------------------------------------------------------------------
# Feature list — EXACTLY matching V3 production model
# ---------------------------------------------------------------------------
FEATURES = [
    "pm25_value",
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24",
    "pm25_roll_3",
    "pm25_roll_6",
    "pm25_roll_12",
    "pm25_roll_24",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_sin",
    "wind_cos",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",
    "fire_count_punjab",
    "fire_count_haryana",
    "fire_count_up",
    "fire_count_delhi",
    "hour",
    "day",
    "month",
    "day_of_week",
    "latitude",
    "longitude",
]

TARGET = "target_pm25_1h"

# ---------------------------------------------------------------------------
# V3-identical hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1,
    enable_categorical=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Return overall + per-range MAE/RMSE/R² with sample counts."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    masks = {
        "0_100":   y_true < 100,
        "100_200": (y_true >= 100) & (y_true < 200),
        "200_300": (y_true >= 200) & (y_true < 300),
        "300_plus": y_true >= 300,
    }

    result = {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}
    for name, mask in masks.items():
        n = mask.sum()
        result[f"n_{name}"] = int(n)
        if n > 0:
            result[f"MAE_{name}"] = mean_absolute_error(y_true[mask], y_pred[mask])
        else:
            result[f"MAE_{name}"] = np.nan

    return result


def safe_predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, model.predict(X))


def print_banner(msg: str):
    print("\n" + "=" * 80)
    print(f"  {msg}")
    print("=" * 80)


def print_results_table(metrics_list: list[dict]):
    """Pretty-print the three-way comparison table."""
    hdr = (
        f"{'Model':<45} {'MAE':>7} {'RMSE':>8} {'R²':>7} "
        f"{'MAE<100':>9} {'MAE100-200':>11} {'MAE200-300':>11} {'MAE300+':>9}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for m in metrics_list:
        def fmt(v): return f"{v:.2f}" if not np.isnan(v) else "  N/A"
        row = (
            f"{m['label']:<45} {m['MAE']:>7.2f} {m['RMSE']:>8.2f} {m['R2']:>7.4f} "
            f"{fmt(m['MAE_0_100']):>9} {fmt(m['MAE_100_200']):>11} "
            f"{fmt(m['MAE_200_300']):>11} {fmt(m['MAE_300_plus']):>9}"
        )
        print(row)


def print_sample_counts(metrics_list: list[dict]):
    """Show January holdout sample distribution."""
    ref = metrics_list[0]  # same holdout for all models
    total = ref['n_0_100'] + ref['n_100_200'] + ref['n_200_300'] + ref['n_300_plus']
    print(f"\n  January 2026 holdout sample distribution (total: {total:,}):")
    for rng, key in [("0–100   (Good/Moderate)", "n_0_100"),
                     ("100–200 (Poor)         ", "n_100_200"),
                     ("200–300 (Very Poor)    ", "n_200_300"),
                     ("300+    (Severe/Hazard)", "n_300_plus")]:
        n = ref[key]
        pct = 100.0 * n / total if total else 0
        print(f"    PM2.5 {rng}: {n:>6,}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print_banner("VayuDrishti — January 2026 Winter Holdout Validation")

    # 1. Load dataset
    print(f"\n[1] Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["dt", "station_name"]).reset_index(drop=True)
    print(f"    Total rows   : {len(df):,}")
    print(f"    Date range   : {df['dt'].min()}  →  {df['dt'].max()}")

    # 2. Drop rows where features or target are NaN
    required = FEATURES + [TARGET, "pm25_value"]
    df_clean = df.dropna(subset=required).copy()
    print(f"    After dropna : {len(df_clean):,} rows")

    # 3. Define January 2026 holdout mask
    jan_mask = (df_clean["dt"].dt.year == 2026) & (df_clean["dt"].dt.month == 1)
    train_df = df_clean[~jan_mask].copy()
    holdout_df = df_clean[jan_mask].copy()

    print(f"\n[2] Split:")
    print(f"    Training rows (all except Jan 2026) : {len(train_df):,}")
    print(f"    January 2026 holdout rows           : {len(holdout_df):,}")
    print(f"    Holdout Jan PM2.5  mean={holdout_df['pm25_value'].mean():.1f}  "
          f"max={holdout_df['pm25_value'].max():.1f}")

    X_train  = train_df[FEATURES]
    y_train  = train_df[TARGET].values
    X_holdout = holdout_df[FEATURES]
    y_holdout = holdout_df[TARGET].values

    # 4. Sample weights (same scheme as V3: 1 + pm25/100, capped at 5)
    sample_weights = np.clip(1.0 + (y_train / 100.0), 1.0, 5.0)
    print(f"\n    Sample weight stats: min={sample_weights.min():.2f}  "
          f"mean={sample_weights.mean():.2f}  max={sample_weights.max():.2f}")

    # 5. Persistence baseline — use current pm25_value as prediction
    print("\n[3] Evaluating Persistence baseline ...")
    preds_persistence = holdout_df["pm25_value"].values
    metrics_pers = compute_metrics(y_holdout, preds_persistence,
                                   label="1. Persistence  (t → t+1)")

    # 6. V1 original model (summer-only, never saw winter)
    print("[4] Evaluating V1 original model (summer-only) ...")
    if not V1_MODEL_PATH.exists():
        raise FileNotFoundError(f"V1 model not found at: {V1_MODEL_PATH}")
    m_v1 = joblib.load(V1_MODEL_PATH)
    # V1 uses the same feature list
    preds_v1 = safe_predict(m_v1, X_holdout)
    metrics_v1 = compute_metrics(y_holdout, preds_v1,
                                  label="2. V1 Original  (summer-only)")

    # 7. Train validation holdout model
    print("[5] Training January-holdout validation model (same params as V3) ...")
    m_holdout = xgb.XGBRegressor(**XGB_PARAMS)
    m_holdout.fit(X_train, y_train, sample_weight=sample_weights,
                  verbose=False)
    preds_holdout = safe_predict(m_holdout, X_holdout)
    metrics_holdout = compute_metrics(y_holdout, preds_holdout,
                                      label="3. Holdout model (excl. Jan 2026)")

    # 8. Save holdout model
    HOLDOUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_holdout, HOLDOUT_MODEL)
    print(f"    ✓ Saved holdout model to: {HOLDOUT_MODEL}")
    print(f"    ✓ Production V3 at {V3_MODEL_PATH} — UNTOUCHED")

    # 9. Print results
    print_banner("Full Three-Way Comparison — January 2026 Holdout")
    all_metrics = [metrics_pers, metrics_v1, metrics_holdout]
    print_results_table(all_metrics)
    print_sample_counts(all_metrics)

    # 10. Summary interpretation
    print_banner("Key Takeaways")
    delta_mae_v1   = metrics_pers["MAE"] - metrics_v1["MAE"]
    delta_mae_val  = metrics_pers["MAE"] - metrics_holdout["MAE"]
    delta_300_v1   = (metrics_pers["MAE_300_plus"] - metrics_v1["MAE_300_plus"]
                      if not np.isnan(metrics_v1["MAE_300_plus"]) else float('nan'))
    delta_300_val  = (metrics_pers["MAE_300_plus"] - metrics_holdout["MAE_300_plus"]
                      if not np.isnan(metrics_holdout["MAE_300_plus"]) else float('nan'))

    print(f"  Persistence MAE        : {metrics_pers['MAE']:.2f}")
    print(f"  V1 (summer-only) MAE   : {metrics_v1['MAE']:.2f}  "
          f"(Δ vs persistence: {delta_mae_v1:+.2f})")
    print(f"  Holdout model MAE      : {metrics_holdout['MAE']:.2f}  "
          f"(Δ vs persistence: {delta_mae_val:+.2f})")
    print(f"  -- Severe 300+ range --")
    print(f"  V1 MAE (300+)          : {metrics_v1['MAE_300_plus']:.2f}  "
          f"(Δ vs persistence: {delta_300_v1:+.2f})"
          if not np.isnan(metrics_v1['MAE_300_plus']) else "  V1 MAE (300+): N/A")
    print(f"  Holdout MAE (300+)     : {metrics_holdout['MAE_300_plus']:.2f}  "
          f"(Δ vs persistence: {delta_300_val:+.2f})"
          if not np.isnan(metrics_holdout['MAE_300_plus']) else "  Holdout MAE (300+): N/A")
    print(f"\n  Winter-training improvement over V1 at 300+: "
          f"{(metrics_v1['MAE_300_plus'] - metrics_holdout['MAE_300_plus']):.2f} MAE points"
          if (not np.isnan(metrics_v1['MAE_300_plus']) and
              not np.isnan(metrics_holdout['MAE_300_plus'])) else "")
    print()


if __name__ == "__main__":
    main()
