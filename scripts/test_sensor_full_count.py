"""
scripts/test_sensor_full_count.py

Fetches the total row count and measurement density for the active PM2.5 sensor
from 2025-02-18 to present across the 5 stations.
"""

import json
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAQ_API_KEY", "").strip()

headers = {
    "Accept": "application/json",
    "User-Agent": "VayuDrishti-DepthTest/1.0",
    "X-API-Key": api_key
}

BASE_URL = "https://api.openaq.org/v3"

active_sensors = [
    {"name": "Alipur, Delhi - DPCC", "loc_id": 6932, "sensor_id": 12235698},
    {"name": "Anand Vihar, New Delhi - DPCC", "loc_id": 235, "sensor_id": 12235610},
    {"name": "R K Puram, Delhi - DPCC", "loc_id": 17, "sensor_id": 12234787},
    {"name": "Punjabi Bagh, Delhi - DPCC", "loc_id": 50, "sensor_id": 12234796},
    {"name": "Bawana, Delhi - DPCC", "loc_id": 8472, "sensor_id": 12235294},
]

print("Measuring total historical rows and hourly coverage for active PM2.5 sensors (Feb 2025 -> Sep 2026)...")

for item in active_sensors:
    s_id = item["sensor_id"]
    st_name = item["name"]
    
    # Query sensor measurements with pagination
    page = 1
    total_rows = 0
    all_ts = []
    
    # Let's count by fetching pages or checking meta
    # OpenAQ v3 measurements page
    while page <= 25:
        params = {
            "datetime_from": "2025-02-01T00:00:00Z",
            "datetime_to": "2026-09-08T23:59:59Z",
            "limit": 1000,
            "page": page
        }
        time.sleep(0.3)
        r = requests.get(f"{BASE_URL}/sensors/{s_id}/measurements", headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            print(f"  Error sensor {s_id} page {page}: {r.status_code}")
            break
        res = r.json().get("results", [])
        if not res:
            break
        total_rows += len(res)
        for r_item in res:
            p_p = r_item.get("period") or {}
            dt_f = p_p.get("datetimeFrom") or {}
            ts = dt_f.get("utc") if isinstance(dt_f, dict) else None
            if not ts:
                ts = r_item.get("datetime", {}).get("utc") if isinstance(r_item.get("datetime"), dict) else r_item.get("datetime")
            if ts:
                all_ts.append(ts)
        if len(res) < 1000:
            break
        page += 1
        
    all_ts.sort()
    earliest = all_ts[0] if all_ts else "None"
    latest = all_ts[-1] if all_ts else "None"
    print(f"Station: {st_name}")
    print(f"  Sensor ID: {s_id}")
    print(f"  Total Measurements: {total_rows:,} rows")
    print(f"  Earliest Timestamp: {earliest}")
    print(f"  Latest Timestamp  : {latest}")
    print(f"  Pages queried     : {page}")
    print("-" * 60)
