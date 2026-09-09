"""
scripts/fetch_firms_extended_historical.py

Fetches NASA FIRMS active fire detections (VIIRS SNPP SP & NRT) from 2025-02-18
to present over Northwest India (Punjab, Haryana, Delhi NCR, Uttar Pradesh)
using 5-day batched API queries.

Enriches detections with state-level classifications and saves incrementally
with full resumability.

Outputs:
  - CSV: data/raw/firms_extended_historical.csv
  - Checkpoint: data/raw/.firms_extended_checkpoint.json
"""

import io
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Bounding box: west, south, east, north
BBOX = "73,27,84,33"
SOURCES = ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"]
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

START_DATE = date(2025, 2, 18)
OUTPUT_DIR = Path("data/raw")
OUTPUT_CSV = OUTPUT_DIR / "firms_extended_historical.csv"
CHECKPOINT_FILE = OUTPUT_DIR / ".firms_extended_checkpoint.json"

CHUNK_DAYS = 5
REQUEST_DELAY_SEC = 0.35
MAX_RETRIES = 3


def classify_state(lat, lon):
    """
    Classify geographic coordinates into Indian states / regions
    using approximate bounding boxes.
    """
    # 1. Delhi NCR
    if 28.38 <= lat <= 28.92 and 76.82 <= lon <= 77.40:
        return "Delhi"

    # 2. Punjab
    if 29.50 <= lat <= 32.55 and 73.80 <= lon <= 76.95:
        return "Punjab"

    # 3. Haryana
    if 27.65 <= lat <= 30.95 and 74.40 <= lon <= 77.60:
        return "Haryana"

    # 4. Uttar Pradesh
    if 23.80 <= lat <= 30.50 and 77.10 <= lon <= 84.65:
        return "Uttar Pradesh"

    return "Other"


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_checkpoint(completed_set):
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(list(completed_set), f)
    except Exception as e:
        print(f"  [Checkpoint warning] {e}", flush=True)


def fetch_firms_chunk(map_key, source, start_dt, days=5):
    """
    Fetch 5-day chunk of fire detections from NASA FIRMS Area API.
    """
    url = f"{BASE_URL}/{map_key}/{source}/{BBOX}/{days}/{start_dt.isoformat()}"
    
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY_SEC)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                text = resp.text.strip()
                if "Invalid MAP_KEY" in text:
                    print("Error: NASA FIRMS rejected MAP_KEY.", flush=True)
                    return None
                return text
            elif resp.status_code == 429:
                wait_t = (attempt + 1) * 5
                print(f"    [Rate limit 429] Waiting {wait_t}s...", flush=True)
                time.sleep(wait_t)
            else:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            time.sleep(2)
            if attempt == MAX_RETRIES - 1:
                print(f"    [Network error] {start_dt}: {e}", flush=True)

    return None


def generate_5day_chunks(start_dt, end_dt):
    chunks = []
    curr = start_dt
    while curr <= end_dt:
        days = min(CHUNK_DAYS, (end_dt - curr).days + 1)
        chunks.append((curr, days))
        curr += timedelta(days=days)
    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv()

    map_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        print("Error: FIRMS_MAP_KEY is missing from .env.", flush=True)
        sys.exit(1)

    today = date.today()
    chunks = generate_5day_chunks(START_DATE, today)

    print("=" * 80, flush=True)
    print("NASA FIRMS Extended Historical Fire Ingestion (Feb 2025 -> Present)", flush=True)
    print("=" * 80, flush=True)
    print(f"Target date range       : {START_DATE.isoformat()} -> {today.isoformat()} ({len(chunks)} 5-day chunks)", flush=True)
    print(f"Target Bounding Box     : {BBOX} (Punjab, Haryana, Delhi NCR, UP)", flush=True)
    print(f"Output CSV destination  : {OUTPUT_CSV}", flush=True)
    print(f"Checkpoint file         : {CHECKPOINT_FILE}\n", flush=True)

    completed_chunks = load_checkpoint()
    print(f"Loaded {len(completed_chunks)} completed chunk-source tasks from checkpoint.\n", flush=True)

    all_dfs = []
    
    # If CSV exists and has data, read it to continue
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            all_dfs.append(existing_df)
            print(f"Loaded {len(existing_df):,} existing records from previous runs.\n", flush=True)
        except Exception:
            pass

    total_new_rows = 0
    start_time = time.time()

    for idx, (chunk_start, days) in enumerate(chunks, 1):
        chunk_end = chunk_start + timedelta(days=days - 1)
        
        chunk_dfs = []

        for src in SOURCES:
            task_key = f"{src}_{chunk_start.isoformat()}_{days}"
            if task_key in completed_chunks:
                continue

            csv_text = fetch_firms_chunk(map_key, src, chunk_start, days)
            if csv_text:
                lines = csv_text.split('\n')
                if len(lines) > 1 and lines[0].startswith("latitude,longitude"):
                    try:
                        df_c = pd.read_csv(io.StringIO(csv_text))
                        if not df_c.empty:
                            chunk_dfs.append(df_c)
                    except Exception as e:
                        print(f"    [CSV parse error] {src} {chunk_start}: {e}", flush=True)

            completed_chunks.add(task_key)
            save_checkpoint(completed_chunks)

        if chunk_dfs:
            combined_chunk = pd.concat(chunk_dfs, ignore_index=True)
            # Add state column
            combined_chunk["state"] = combined_chunk.apply(
                lambda r: classify_state(r["latitude"], r["longitude"]), axis=1
            )
            all_dfs.append(combined_chunk)
            total_new_rows += len(combined_chunk)
            print(f"[{idx}/{len(chunks)}] {chunk_start.isoformat()} -> {chunk_end.isoformat()} : +{len(combined_chunk):,} fires", flush=True)
        else:
            print(f"[{idx}/{len(chunks)}] {chunk_start.isoformat()} -> {chunk_end.isoformat()} : 0 fires", flush=True)

    # Consolidate and deduplicate all records
    if all_dfs:
        print("\nConsolidating and deduplicating dataset...", flush=True)
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        dedup_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in final_df.columns]
        if dedup_cols:
            initial_count = len(final_df)
            final_df = final_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
            print(f"Deduplication removed {initial_count - len(final_df):,} duplicate detections.", flush=True)

        final_df["acq_date"] = pd.to_datetime(final_df["acq_date"]).dt.strftime("%Y-%m-%d")
        final_df = final_df.sort_values(by=["acq_date", "acq_time"]).reset_index(drop=True)

        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Final extended FIRMS dataset saved -> {OUTPUT_CSV}\n", flush=True)

        # Summary Metrics
        elapsed_min = (time.time() - start_time) / 60.0
        min_date = final_df["acq_date"].min()
        max_date = final_df["acq_date"].max()
        state_counts = final_df["state"].value_counts().to_dict()

        # Winter / Stubble Season Count (Oct 1 - Nov 30, 2025)
        stubble_mask = (final_df["acq_date"] >= "2025-10-01") & (final_df["acq_date"] <= "2025-11-30")
        stubble_count = stubble_mask.sum()

        print("=" * 80, flush=True)
        print("NASA FIRMS EXTENDED HISTORICAL DATASET SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"Elapsed Time                : {elapsed_min:.1f} minutes", flush=True)
        print(f"Total Fire Detections       : {len(final_df):,}", flush=True)
        print(f"Date Range Covered          : {min_date} -> {max_date}", flush=True)
        print(f"Total Days Covered          : {(pd.to_datetime(max_date) - pd.to_datetime(min_date)).days + 1} days", flush=True)
        print(f"Peak Stubble Burning Period : {stubble_count:,} fires (Oct 1 -> Nov 30, 2025)", flush=True)
        print(f"Breakdown by State/Region   : {state_counts}", flush=True)
        print(f"File Size on Disk           : {OUTPUT_CSV.stat().st_size / (1024*1024):.2f} MB", flush=True)
        print("=" * 80, flush=True)
    else:
        print("No fire records collected.", flush=True)


if __name__ == "__main__":
    main()
