"""
scripts/fetch_openaq_extended_historical.py

Fetches the full ~18.5-month historical PM2.5 dataset (Feb 18, 2025 to present)
for active Delhi NCR monitoring stations from OpenAQ API v3.

Uses a monthly date-window slicing approach to prevent HTTP 408 timeouts on
deep offset pagination, includes robust rate-limit backoff, and features full
incremental resumability via a checkpoint file.

Outputs:
  - CSV: data/raw/openaq_extended_historical_pm25.csv
    Columns: station_name, city, latitude, longitude, timestamp, pm25_value
  - Checkpoint: data/raw/.openaq_extended_checkpoint.json
"""

import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import calendar
import pandas as pd
import requests
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://api.openaq.org/v3"
OUTPUT_DIR = Path("data/raw")
OUTPUT_CSV = OUTPUT_DIR / "openaq_extended_historical_pm25.csv"
CHECKPOINT_FILE = OUTPUT_DIR / ".openaq_extended_checkpoint.json"
LOCATIONS_CACHE = OUTPUT_DIR / "openaq_locations_extended.json"

START_DATE = date(2025, 2, 18)
REQUEST_DELAY_SEC = 0.35  # delay between successive API calls
MAX_RETRIES = 4


def get_headers():
    load_dotenv()
    api_key = os.getenv("OPENAQ_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENAQ_API_KEY is missing from .env.", flush=True)
        sys.exit(1)
    return {
        "Accept": "application/json",
        "User-Agent": "VayuDrishti-Extended/1.0",
        "X-API-Key": api_key
    }


def generate_monthly_windows(start_date, end_date):
    """
    Generate list of (start_iso, end_iso, label) tuples sliced by month.
    """
    windows = []
    curr_start = start_date
    
    while curr_start <= end_date:
        # Last day of current month
        _, last_day_num = calendar.monthrange(curr_start.year, curr_start.month)
        month_end = date(curr_start.year, curr_start.month, last_day_num)
        
        curr_end = min(month_end, end_date)
        
        start_iso = f"{curr_start.isoformat()}T00:00:00Z"
        end_iso = f"{curr_end.isoformat()}T23:59:59Z"
        label = f"{curr_start.strftime('%Y-%m')} ({curr_start.strftime('%d %b %Y')} -> {curr_end.strftime('%d %b %Y')})"
        
        windows.append((start_iso, end_iso, label))
        
        # Advance to first day of next month
        curr_start = month_end + timedelta(days=1)
        
    return windows


def load_active_stations(headers):
    """
    Retrieve active Delhi NCR stations from /v3/locations or cache.
    """
    data = None
    if LOCATIONS_CACHE.exists():
        try:
            with open(LOCATIONS_CACHE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

    if not data:
        print("Querying Delhi NCR stations from OpenAQ /v3/locations...", flush=True)
        try:
            r = requests.get(
                f"{BASE_URL}/locations",
                headers=headers,
                params={"bbox": "76.8,28.2,77.6,28.9", "limit": 100},
                timeout=30
            )
            if r.status_code == 200:
                data = r.json().get("results", [])
                with open(LOCATIONS_CACHE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                print(f"Error fetching locations: HTTP {r.status_code}", flush=True)
                return []
        except Exception as e:
            print(f"Exception fetching locations: {e}", flush=True)
            return []

    now_utc = datetime.now(timezone.utc)
    active_stations = []

    for loc in data:
        # Check staleness based on datetimeLast
        dt_last_obj = loc.get("datetimeLast") or {}
        dt_last_str = dt_last_obj.get("utc") if isinstance(dt_last_obj, dict) else dt_last_obj
        
        is_active = False
        if dt_last_str:
            try:
                dt_last = datetime.fromisoformat(dt_last_str.replace("Z", "+00:00"))
                if (now_utc - dt_last).days <= 60:
                    is_active = True
            except Exception:
                is_active = True
        else:
            is_active = True

        if not is_active:
            continue

        coords = loc.get("coordinates") or {}
        sensors = loc.get("sensors") or []
        pm25_sensor_ids = []
        for s in sensors:
            p_obj = s.get("parameter") or {}
            p_name = p_obj.get("name") or s.get("name") or ""
            if "pm25" in str(p_name).lower() or p_name == "pm25":
                pm25_sensor_ids.append(s.get("id"))

        # Sort descending so newest active sensors (e.g. ID ~12M+) are queried first
        pm25_sensor_ids.sort(reverse=True)

        if pm25_sensor_ids:
            active_stations.append({
                "id": loc.get("id"),
                "name": loc.get("name", "Unknown Station"),
                "city": loc.get("locality") or "Delhi NCR",
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "pm25_sensors": pm25_sensor_ids
            })

    return active_stations


def load_checkpoint():
    """
    Load completed tasks from checkpoint JSON.
    """
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_checkpoint(completed_set):
    """
    Save completed tasks to checkpoint JSON.
    """
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(list(completed_set), f)
    except Exception as e:
        print(f"  [Checkpoint warning] Failed to write checkpoint: {e}", flush=True)


def fetch_sensor_month_records(sensor_id, start_iso, end_iso, headers):
    """
    Fetch all measurements for a sensor within a single monthly window.
    Handles pagination (1-3 pages) and rate-limiting / retry logic.
    """
    records = []
    page = 1
    max_pages = 8  # Safety cap per month (~8,000 readings max)

    while page <= max_pages:
        url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
        params = {
            "datetime_from": start_iso,
            "datetime_to": end_iso,
            "limit": 1000,
            "page": page
        }

        resp = None
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY_SEC)
                resp = requests.get(url, headers=headers, params=params, timeout=25)
                if resp.status_code == 200:
                    break
                elif resp.status_code == 429:
                    backoff = (attempt + 1) * 5
                    print(f"    [Rate limit 429] Sensor {sensor_id} page {page}: waiting {backoff}s...", flush=True)
                    time.sleep(backoff)
                elif resp.status_code in (500, 502, 503, 504):
                    backoff = (attempt + 1) * 3
                    print(f"    [Server error {resp.status_code}] Sensor {sensor_id} page {page}: retrying in {backoff}s...", flush=True)
                    time.sleep(backoff)
                elif resp.status_code == 408:
                    print(f"    [Timeout 408] Sensor {sensor_id} page {page}: retrying in 4s...", flush=True)
                    time.sleep(4)
                else:
                    print(f"    [HTTP {resp.status_code}] Sensor {sensor_id} page {page}: {resp.text[:120]}", flush=True)
                    break
            except requests.exceptions.RequestException as e:
                time.sleep(2)
                if attempt == MAX_RETRIES - 1:
                    print(f"    [Network error] Sensor {sensor_id} page {page}: {e}", flush=True)

        if not resp or resp.status_code != 200:
            break

        try:
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                val = r.get("value")
                p_period = r.get("period") or {}
                dt_from = p_period.get("datetimeFrom") or {}
                ts = dt_from.get("utc") if isinstance(dt_from, dict) else None
                if not ts:
                    ts = r.get("datetime", {}).get("utc") if isinstance(r.get("datetime"), dict) else r.get("datetime")
                
                coords = r.get("coordinates") or {}

                if ts and val is not None:
                    records.append({
                        "timestamp": ts,
                        "value": val,
                        "lat": coords.get("latitude"),
                        "lon": coords.get("longitude")
                    })

            if len(results) < 1000:
                # Finished this month
                break

            page += 1

        except Exception as e:
            print(f"    [Parse error] Sensor {sensor_id} page {page}: {e}", flush=True)
            break

    return records


def append_records_to_csv(records, station_meta):
    """
    Append fetched records incrementally to the output CSV file.
    """
    if not records:
        return

    st_name = station_meta["name"]
    st_city = station_meta["city"]
    st_lat = station_meta["latitude"]
    st_lon = station_meta["longitude"]

    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["station_name", "city", "latitude", "longitude", "timestamp", "pm25_value"])
        
        for r in records:
            lat = r["lat"] if r["lat"] is not None else st_lat
            lon = r["lon"] if r["lon"] is not None else st_lon
            writer.writerow([
                st_name,
                st_city,
                lat,
                lon,
                r["timestamp"],
                r["value"]
            ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = get_headers()

    today_date = date.today()
    windows = generate_monthly_windows(START_DATE, today_date)

    print("=" * 80, flush=True)
    print("OpenAQ Extended Historical PM2.5 Ingest (Feb 2025 -> Present)", flush=True)
    print("=" * 80, flush=True)
    print(f"Target date range       : {START_DATE.isoformat()} -> {today_date.isoformat()} ({len(windows)} monthly windows)", flush=True)
    print(f"Output CSV destination  : {OUTPUT_CSV}", flush=True)
    print(f"Checkpoint file         : {CHECKPOINT_FILE}\n", flush=True)

    stations = load_active_stations(headers)
    print(f"Active stations identified with PM2.5 sensors: {len(stations)}", flush=True)

    if not stations:
        print("No active stations found.", flush=True)
        sys.exit(0)

    completed_tasks = load_checkpoint()
    print(f"Loaded {len(completed_tasks):,} completed month-sensor tasks from checkpoint.\n", flush=True)

    total_records_session = 0
    start_time = time.time()

    for idx, st in enumerate(stations, 1):
        st_name = st["name"]
        st_id = st["id"]
        pm25_sensors = st["pm25_sensors"]
        primary_sensor = pm25_sensors[0]

        print(f"[{idx}/{len(stations)}] Station: {st_name} (ID: {st_id}, Primary PM2.5 Sensor: {primary_sensor})", flush=True)

        station_rows = 0

        for start_iso, end_iso, label in windows:
            task_key = f"{st_id}_{primary_sensor}_{start_iso[:10]}"
            if task_key in completed_tasks:
                continue

            # Fetch records for this monthly window
            records = fetch_sensor_month_records(primary_sensor, start_iso, end_iso, headers)
            
            if records:
                append_records_to_csv(records, st)
                station_rows += len(records)
                total_records_session += len(records)

            # Mark task as completed and persist checkpoint
            completed_tasks.add(task_key)
            save_checkpoint(completed_tasks)

        if station_rows > 0:
            print(f"    -> Added {station_rows:,} measurements for this station", flush=True)
        else:
            print(f"    -> All monthly windows already up-to-date or 0 new records", flush=True)

    elapsed_min = (time.time() - start_time) / 60.0

    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY OF EXTENDED HISTORICAL INGESTION", flush=True)
    print("=" * 80, flush=True)
    print(f"Elapsed Time                : {elapsed_min:.1f} minutes", flush=True)
    print(f"Total stations processed    : {len(stations)}", flush=True)

    if OUTPUT_CSV.exists():
        print(f"Analyzing generated dataset {OUTPUT_CSV}...", flush=True)
        try:
            # Read CSV and summarize
            df = pd.read_csv(OUTPUT_CSV)
            total_rows = len(df)
            unique_stations = df["station_name"].nunique()
            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()

            # Winter 2025-2026 row count
            w_mask = (df["timestamp"] >= "2025-11-01T00:00:00Z") & (df["timestamp"] <= "2026-02-28T23:59:59Z")
            winter_rows = w_mask.sum()

            print(f"Total rows in dataset       : {total_rows:,}")
            print(f"Unique stations covered     : {unique_stations}")
            print(f"Earliest timestamp          : {min_ts}")
            print(f"Latest timestamp            : {max_ts}")
            print(f"Winter 2025-2026 row count  : {winter_rows:,} measurements (Nov 1, 2025 -> Feb 28, 2026)")
            print(f"File size on disk           : {OUTPUT_CSV.stat().st_size / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"Error analyzing output CSV: {e}", flush=True)
    else:
        print("Warning: Output CSV file was not created.", flush=True)

    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
