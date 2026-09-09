"""
scripts/fetch_openmeteo_extended_historical.py

Fetches the full 18.5-month historical hourly weather data for Delhi (NCR)
from Open-Meteo Historical Archive API (2025-02-18 to present) and saves
as JSON and clean DataFrame CSV.

Outputs:
  - JSON: data/raw/openmeteo_extended_historical_delhi.json
  - CSV : data/raw/openmeteo_extended_historical_delhi.csv

Usage:
    python scripts/fetch_openmeteo_extended_historical.py
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
LAT = 28.6139
LON = 77.2090
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = date(2025, 2, 18)

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",
]

OUTPUT_DIR = Path("data/raw")
JSON_OUT = OUTPUT_DIR / "openmeteo_extended_historical_delhi.json"
CSV_OUT = OUTPUT_DIR / "openmeteo_extended_historical_delhi.csv"
# ─────────────────────────────────────────────────────────────────────────────


def fetch_data(start: date, end: date) -> dict:
    """Make the API request and return the parsed JSON response."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }

    print(f"Requesting Open-Meteo historical archive for Delhi ({LAT}, {LON})...")
    print(f"Date range: {start.isoformat()} -> {end.isoformat()}")

    try:
        resp = requests.get(BASE_URL, params=params, timeout=40)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the Open-Meteo API. Check your internet connection.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: The request to Open-Meteo timed out.")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"Error: Unexpected network error — {exc}")
        sys.exit(1)

    print(f"HTTP Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"Error: API returned status {resp.status_code}.")
        try:
            err = resp.json()
            print(f"API message: {err.get('reason', resp.text[:300])}")
        except Exception:
            print(f"Raw response: {resp.text[:300]}")
        sys.exit(1)

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("Error: Could not parse API response as JSON.")
        sys.exit(1)

    if "hourly" not in data:
        print("Error: 'hourly' key missing from API response.")
        sys.exit(1)

    return data


def build_dataframe(data: dict) -> pd.DataFrame:
    """Convert the Open-Meteo hourly dict into a clean, indexed DataFrame."""
    hourly = data["hourly"]

    missing = [v for v in ["time"] + HOURLY_VARS if v not in hourly]
    if missing:
        print(f"Warning: Expected columns missing from response: {missing}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today_date = date.today()
    print("=" * 80)
    print("Open-Meteo Extended Historical Weather Ingestion (Feb 2025 -> Present)")
    print("=" * 80)

    # 1. Fetch data
    data = fetch_data(START_DATE, today_date)

    # 2. Save raw JSON
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Raw JSON saved  -> {JSON_OUT}")

    # 3. Build & Save CSV
    df = build_dataframe(data)
    df.to_csv(CSV_OUT)
    print(f"Clean CSV saved -> {CSV_OUT}\n")

    # 4. Summary & Verification
    print("=" * 80)
    print("OPEN-METEO EXTENDED DATASET SUMMARY")
    print("=" * 80)
    print(f"Total Rows (Hours)      : {len(df):,}")
    print(f"Earliest Timestamp      : {df.index.min()}")
    print(f"Latest Timestamp        : {df.index.max()}")
    print(f"Total Days Covered      : {(df.index.max() - df.index.min()).days + 1} days")
    print(f"Columns ({len(df.columns)})         : {list(df.columns)}")
    print(f"Missing Values per Col  :\n{df.isnull().sum().to_string()}")
    print(f"File size on disk       : {CSV_OUT.stat().st_size / 1024:.1f} KB")
    print("=" * 80)


if __name__ == "__main__":
    main()
