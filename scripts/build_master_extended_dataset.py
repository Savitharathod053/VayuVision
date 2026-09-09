"""
scripts/build_master_extended_dataset.py

Builds the clean, expanded multi-season master modeling dataset for VayuDrishti
spanning the full 18.5-month confirmed historical window (2025-02-18 to 2026-09-08).

Quality Controls & Standards Applied:
  1. Station Whitelist: Keeps exclusively official government network stations
     (matching DPCC, CPCB, UPPCB, HSPCB, IMD, IITM, RSPCB).
     Excludes non-government/community/calibration monitors.
  2. Sentinel Filter: Removes 999.99 hardware clipping ceilings and unphysical spikes.
  3. Continuous Spatial Resampling: Hourly averages per station merged with Open-Meteo & FIRMS.
  4. Feature Engineering: Autoregressive lags, rolling statistics, wind vectors, and time features.

Outputs:
  data/processed/master_dataset_extended.csv
"""

import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OPENAQ_CSV = RAW_DIR / "openaq_extended_historical_pm25.csv"
WEATHER_CSV = RAW_DIR / "openmeteo_extended_historical_delhi.csv"
FIRMS_CSV = RAW_DIR / "firms_extended_historical.csv"
AEROSOL_CSV = RAW_DIR / "aerosol_nrt.csv"
OUTPUT_CSV = PROCESSED_DIR / "master_dataset_extended.csv"

# Official Government Regulatory Network Agencies
GOV_AGENCIES = ["DPCC", "CPCB", "UPPCB", "HSPCB", "IMD", "IITM", "RSPCB"]
GOV_STATION_REGEX = re.compile(r".+,\s*.+\s*-\s*(" + "|".join(GOV_AGENCIES) + r")$")


def is_official_gov_station(station_name: str) -> bool:
    return bool(GOV_STATION_REGEX.match(str(station_name).strip()))


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("VayuDrishti: Building Clean 18.5-Month Official Government Master Dataset")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Load Extended OpenAQ PM2.5, Filter Stations & Clean Sentinels
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading & Filtering Extended OpenAQ PM2.5...")
    if not OPENAQ_CSV.exists():
        raise FileNotFoundError(f"Missing required file: {OPENAQ_CSV}")

    df_aq = pd.read_csv(OPENAQ_CSV)
    raw_total = len(df_aq)
    initial_stations = df_aq["station_name"].nunique()
    print(f"  - Total raw records in source: {raw_total:,} across {initial_stations} stations")

    # 1A. Filter out non-government stations
    gov_mask = df_aq["station_name"].apply(is_official_gov_station)
    excluded_stations = df_aq.loc[~gov_mask, "station_name"].unique().tolist()
    df_aq = df_aq[gov_mask].copy()
    gov_stations_count = df_aq["station_name"].nunique()
    print(f"  - Filtered to {gov_stations_count} official government network stations.")
    print(f"  - Excluded non-government/calibration stations ({len(excluded_stations)}): {excluded_stations}")
    print(f"  - Raw records after station filtering: {len(df_aq):,}")

    # 1B. Remove sentinel/hardware-clipping values (exact 999.99, > 1000, < 0)
    sentinel_mask = (df_aq["pm25_value"] == 999.99) | (df_aq["pm25_value"] > 1000.0) | (df_aq["pm25_value"] < 0)
    sentinel_count = sentinel_mask.sum()
    df_aq = df_aq[~sentinel_mask].copy()
    print(f"  - Removed {sentinel_count:,} sentinel / clipping values (999.99 or >1000 µg/m³).")
    print(f"  - Clean raw records for aggregation: {len(df_aq):,}")

    # 1C. Convert to IST and floor to hourly intervals
    df_aq["dt_ist"] = pd.to_datetime(df_aq["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df_aq["timestamp_hour"] = df_aq["dt_ist"].dt.floor("h")

    # Hourly mean per station
    df_hourly_aq = (
        df_aq.groupby(["station_name", "city", "latitude", "longitude", "timestamp_hour"], as_index=False)
        .agg({"pm25_value": "mean"})
    )
    df_hourly_aq["time_str"] = df_hourly_aq["timestamp_hour"].dt.strftime("%Y-%m-%d %H:00:00")
    print(f"  - Resampled to hourly station records: {len(df_hourly_aq):,}")

    # -------------------------------------------------------------------------
    # Step 2: Load Extended Open-Meteo Weather & Merge
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading & Merging Extended Open-Meteo Weather Data...")
    if not WEATHER_CSV.exists():
        raise FileNotFoundError(f"Missing required file: {WEATHER_CSV}")

    df_weather = pd.read_csv(WEATHER_CSV)
    print(f"  - Loaded weather hourly rows: {len(df_weather):,}")

    df_weather["time_str"] = pd.to_datetime(df_weather["time"]).dt.strftime("%Y-%m-%d %H:00:00")

    weather_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
        "precipitation",
        "boundary_layer_height"
    ]

    df_weather[weather_cols] = df_weather[weather_cols].interpolate(method="linear", limit=2)
    df_weather[weather_cols] = df_weather[weather_cols].ffill(limit=2)

    wind_rad = np.radians(df_weather["wind_direction_10m"].astype(float))
    df_weather["wind_sin"] = np.sin(wind_rad)
    df_weather["wind_cos"] = np.cos(wind_rad)

    weather_merge_cols = weather_cols + ["wind_sin", "wind_cos"]

    df_merged = pd.merge(
        df_hourly_aq,
        df_weather[["time_str"] + weather_merge_cols],
        on="time_str",
        how="left"
    )
    print(f"  - Merged AQ + Weather rows: {len(df_merged):,}")

    # -------------------------------------------------------------------------
    # Step 3: Load Extended NASA FIRMS Active Fire Detections
    # -------------------------------------------------------------------------
    print("\n[Step 3] Loading & Merging Extended NASA FIRMS Fire Counts...")
    fire_daily = pd.DataFrame(columns=["date_str", "fire_count_punjab", "fire_count_haryana", "fire_count_up", "fire_count_delhi"])

    if FIRMS_CSV.exists():
        df_firms = pd.read_csv(FIRMS_CSV, low_memory=False)
        print(f"  - Loaded active fire detections: {len(df_firms):,}")

        if not df_firms.empty and "acq_date" in df_firms.columns and "state" in df_firms.columns:
            state_piv = (
                df_firms.groupby(["acq_date", "state"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            state_piv = state_piv.rename(columns={"acq_date": "date_str"})
            
            state_piv["fire_count_punjab"] = state_piv.get("Punjab", 0)
            state_piv["fire_count_haryana"] = state_piv.get("Haryana", 0)
            state_piv["fire_count_up"] = state_piv.get("Uttar Pradesh", 0)
            state_piv["fire_count_delhi"] = state_piv.get("Delhi", 0)
            
            fire_daily = state_piv[["date_str", "fire_count_punjab", "fire_count_haryana", "fire_count_up", "fire_count_delhi"]]

    df_merged["date_str"] = df_merged["timestamp_hour"].dt.strftime("%Y-%m-%d")
    df_merged = pd.merge(df_merged, fire_daily, on="date_str", how="left")

    fire_cols = ["fire_count_punjab", "fire_count_haryana", "fire_count_up", "fire_count_delhi"]
    df_merged[fire_cols] = df_merged[fire_cols].fillna(0).astype(int)

    # -------------------------------------------------------------------------
    # Step 4: Check NASA LANCE Aerosol AOD History
    # -------------------------------------------------------------------------
    print("\n[Step 4] Checking NASA LANCE Aerosol AOD History...")
    if AEROSOL_CSV.exists():
        df_aero = pd.read_csv(AEROSOL_CSV)
        print(f"  - Found {len(df_aero)} records in aerosol_nrt.csv (collection recently initiated).")
        print("  - AOD feature reserved for future retraining once continuous historical stream accumulates.")

    # -------------------------------------------------------------------------
    # Step 5: Temporal Features
    # -------------------------------------------------------------------------
    print("\n[Step 5] Adding Temporal Calendar Features...")
    dt_series = df_merged["timestamp_hour"]
    df_merged["hour"] = dt_series.dt.hour
    df_merged["day"] = dt_series.dt.day
    df_merged["month"] = dt_series.dt.month
    df_merged["day_of_week"] = dt_series.dt.dayofweek

    # -------------------------------------------------------------------------
    # Step 6: Engineer PM2.5 Lags, Rolling Means & Target per Station
    # -------------------------------------------------------------------------
    print("\n[Step 6] Engineering Station-Level Autoregressive Lags & Rolling Features...")
    df_merged = df_merged.sort_values(by=["station_name", "timestamp_hour"]).reset_index(drop=True)

    station_groups = []
    for station_name, group in df_merged.groupby("station_name"):
        g = group.copy().sort_values(by="timestamp_hour")
        
        # Lags: t-1, t-3, t-6, t-12, t-24
        g["pm25_lag_1"] = g["pm25_value"].shift(1)
        g["pm25_lag_3"] = g["pm25_value"].shift(3)
        g["pm25_lag_6"] = g["pm25_value"].shift(6)
        g["pm25_lag_12"] = g["pm25_value"].shift(12)
        g["pm25_lag_24"] = g["pm25_value"].shift(24)

        # Rolling means on past observations (shifted by 1 to prevent target leakage)
        g["pm25_roll_3"] = g["pm25_value"].shift(1).rolling(3, min_periods=3).mean()
        g["pm25_roll_6"] = g["pm25_value"].shift(1).rolling(6, min_periods=6).mean()
        g["pm25_roll_12"] = g["pm25_value"].shift(1).rolling(12, min_periods=12).mean()
        g["pm25_roll_24"] = g["pm25_value"].shift(1).rolling(24, min_periods=24).mean()

        # Target: 1-hour ahead PM2.5 (t+1)
        g["target_pm25_1h"] = g["pm25_value"].shift(-1)

        station_groups.append(g)

    df_engineered = pd.concat(station_groups, ignore_index=True)

    # -------------------------------------------------------------------------
    # Step 7: Handle Missing Values & Format Columns
    # -------------------------------------------------------------------------
    print("\n[Step 7] Handling Missing Values & Validating Quality...")
    required_features = [
        "pm25_value",
        "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
        "pm25_roll_3", "pm25_roll_6", "pm25_roll_12", "pm25_roll_24",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_sin", "wind_cos",
        "surface_pressure", "precipitation", "boundary_layer_height",
        "target_pm25_1h"
    ]

    initial_len = len(df_engineered)
    df_clean = df_engineered.dropna(subset=required_features).copy()
    dropped_count = initial_len - len(df_clean)
    print(f"  - Dropped {dropped_count:,} warmup/boundary rows (first 24h lag warmup + last step target).")
    print(f"  - Final clean modeling records: {len(df_clean):,}")

    df_clean = df_clean.rename(columns={"time_str": "timestamp"})

    final_columns = [
        "station_name",
        "city",
        "latitude",
        "longitude",
        "timestamp",
        "pm25_value",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
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
        "pm25_lag_1",
        "pm25_lag_3",
        "pm25_lag_6",
        "pm25_lag_12",
        "pm25_lag_24",
        "pm25_roll_3",
        "pm25_roll_6",
        "pm25_roll_12",
        "pm25_roll_24",
        "wind_sin",
        "wind_cos",
        "target_pm25_1h"
    ]

    df_final = df_clean[final_columns].sort_values(by=["timestamp", "station_name"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step 8: Save Processed Dataset
    # -------------------------------------------------------------------------
    print(f"\n[Step 8] Saving clean master dataset to: {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"  - Successfully saved {len(df_final):,} rows ({OUTPUT_CSV.stat().st_size / (1024*1024):.2f} MB).")

    # -------------------------------------------------------------------------
    # Step 9: Diagnostics, Seasonal & Station Audit
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXTENDED MASTER DATASET VERIFICATION & AUDIT REPORT:")
    print("=" * 80)
    print(f"Total Modeling Rows:     {len(df_final):,}")
    print(f"Total Columns:           {len(df_final.columns)}")
    print(f"Unique Stations:         {df_final['station_name'].nunique()} (All 100% verified official government)")
    print(f"Date Range Covered:      {df_final['timestamp'].min()} -> {df_final['timestamp'].max()}")

    # Verify all stations match expected pattern
    all_match = all(is_official_gov_station(s) for s in df_final["station_name"].unique())
    print(f"All Station Names Valid: {all_match} (0 non-government stations)")

    # Seasonal breakdown
    df_final["dt"] = pd.to_datetime(df_final["timestamp"])
    winter_mask = (df_final["dt"] >= "2025-11-01") & (df_final["dt"] <= "2026-02-28 23:59:59")
    autumn_fire_mask = (df_final["dt"] >= "2025-10-01") & (df_final["dt"] <= "2025-11-30 23:59:59")
    summer_mask = (df_final["dt"] >= "2025-04-01") & (df_final["dt"] <= "2025-06-30 23:59:59")
    monsoon_mask = (df_final["dt"] >= "2025-07-01") & (df_final["dt"] <= "2025-09-30 23:59:59")

    winter_rows = winter_mask.sum()
    winter_pm_mean = df_final.loc[winter_mask, "pm25_value"].mean()
    winter_pm_max = df_final.loc[winter_mask, "pm25_value"].max()
    winter_pm_p99 = df_final.loc[winter_mask, "pm25_value"].quantile(0.99)

    print("\nSeasonal Distribution & Corrected Max PM2.5:")
    print(f"  * Winter 2025-2026 (Nov 1 - Feb 28) : {winter_rows:,} records")
    print(f"      - Mean PM2.5: {winter_pm_mean:.2f} µg/m³")
    print(f"      - Max PM2.5:  {winter_pm_max:.2f} µg/m³ (Corrected, no 999.99 sentinels)")
    print(f"      - 99th %ile:  {winter_pm_p99:.2f} µg/m³")
    print(f"  * Autumn Stubble Peak (Oct 1 - Nov 30): {autumn_fire_mask.sum():,} records")
    print(f"  * Summer 2025 (Apr 1 - Jun 30)        : {summer_mask.sum():,} records (Mean: {df_final.loc[summer_mask, 'pm25_value'].mean():.2f} µg/m³)")
    print(f"  * Monsoon 2025 (Jul 1 - Sep 30)       : {monsoon_mask.sum():,} records (Mean: {df_final.loc[monsoon_mask, 'pm25_value'].mean():.2f} µg/m³)")

    print("\nAgency Distribution:")
    agency_counts = {}
    for agency in GOV_AGENCIES:
        count = sum(1 for s in df_final["station_name"].unique() if s.endswith(f"- {agency}"))
        agency_counts[agency] = count
        print(f"  - {agency:<6}: {count:>2} stations")

    print("\n" + "=" * 80)
    print("Master dataset extended rebuild complete and verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
