"""
scripts/test_openaq_depth.py

Comprehensive test to verify the exact historical depth of OpenAQ API v3
for 5 representative Delhi NCR stations.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAQ_API_KEY", "").strip()

if not api_key:
    print("Error: OPENAQ_API_KEY missing.", flush=True)
    sys.exit(1)

headers = {
    "Accept": "application/json",
    "User-Agent": "VayuDrishti-DepthTest/1.0",
    "X-API-Key": api_key
}

BASE_URL = "https://api.openaq.org/v3"

def query_api(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                print(f"    [Rate limit 429] Waiting 5s...", flush=True)
                time.sleep(5)
            else:
                print(f"    [API Error {r.status_code}] {url} -> {r.text[:200]}", flush=True)
                return None
        except Exception as e:
            print(f"    [Request Exception] {e}", flush=True)
            time.sleep(2)
    return None

print("=" * 80, flush=True)
print("OPENAQ HISTORICAL DEPTH VERIFICATION FOR DELHI NCR", flush=True)
print("=" * 80, flush=True)

# 1. Fetch Delhi locations
print("1. Fetching Delhi NCR locations from /v3/locations...", flush=True)
loc_data = query_api(f"{BASE_URL}/locations", {"bbox": "76.8,28.2,77.6,28.9", "limit": 100})
if not loc_data:
    print("Failed to fetch locations.", flush=True)
    sys.exit(1)

all_locations = loc_data.get("results", [])
print(f"Total locations found in Delhi NCR bbox: {len(all_locations)}", flush=True)

# Define 5 key target stations
target_keys = ["Alipur", "Anand Vihar", "R K Puram", "Punjabi Bagh", "Bawana"]
selected_stations = []

for key in target_keys:
    for loc in all_locations:
        name = loc.get("name", "")
        if key.lower() in name.lower():
            selected_stations.append(loc)
            break

print(f"\nSelected 5 representative test stations:", flush=True)
for i, s in enumerate(selected_stations, 1):
    print(f"  {i}. {s.get('name')} (ID: {s.get('id')})", flush=True)

# 2. For each station, inspect sensors and test measurements
now_utc = datetime.now(timezone.utc)
# Test windows
windows = [
    ("Full 3-Year Window (3 years ago -> Now)", (now_utc - timedelta(days=3*365)).strftime("%Y-%m-%dT00:00:00Z"), now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")),
    ("Winter 2023-2024 (Nov 2023 - Feb 2024)", "2023-11-01T00:00:00Z", "2024-02-29T23:59:59Z"),
    ("Winter 2024-2025 (Nov 2024 - Feb 2025)", "2024-11-01T00:00:00Z", "2025-02-28T23:59:59Z"),
    ("Winter 2025-2026 (Nov 2025 - Feb 2026)", "2025-11-01T00:00:00Z", "2026-02-28T23:59:59Z"),
    ("Last 90 Days", (now_utc - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00Z"), now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")),
    ("Wide Open (No datetime_from, or 2018-01-01)", "2018-01-01T00:00:00Z", now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")),
]

results_summary = []

for st in selected_stations:
    loc_id = st.get("id")
    st_name = st.get("name")
    print(f"\n{'='*80}", flush=True)
    print(f"STATION: {st_name} (Location ID: {loc_id})", flush=True)
    print(f"{'='*80}", flush=True)
    
    # Also fetch /locations/{id} to get full sensor list and datetimeFirst/datetimeLast
    time.sleep(0.5)
    loc_detail = query_api(f"{BASE_URL}/locations/{loc_id}")
    if loc_detail and loc_detail.get("results"):
        loc_info = loc_detail["results"][0]
        loc_dt_first = loc_info.get("datetimeFirst")
        loc_dt_last = loc_info.get("datetimeLast")
        print(f"Location metadata -> datetimeFirst: {loc_dt_first}, datetimeLast: {loc_dt_last}", flush=True)
        sensors_list = loc_info.get("sensors", [])
    else:
        sensors_list = st.get("sensors", [])
    
    # Filter for PM2.5 sensors
    pm25_sensors = []
    for s in sensors_list:
        p_obj = s.get("parameter") or {}
        p_name = p_obj.get("name") or s.get("name") or ""
        if "pm25" in str(p_name).lower() or p_name == "pm25":
            pm25_sensors.append(s)
            
    print(f"Found {len(pm25_sensors)} PM2.5 sensor(s):", flush=True)
    
    station_entry = {
        "station_name": st_name,
        "location_id": loc_id,
        "sensors": []
    }
    
    for s in pm25_sensors:
        s_id = s.get("id")
        time.sleep(0.5)
        s_detail = query_api(f"{BASE_URL}/sensors/{s_id}")
        s_meta = s_detail.get("results", [{}])[0] if s_detail and s_detail.get("results") else s
        
        s_dt_first = s_meta.get("datetimeFirst")
        s_dt_last = s_meta.get("datetimeLast")
        s_summary = s_meta.get("summary")
        print(f"\n  * Sensor ID {s_id}:", flush=True)
        print(f"    - Metadata datetimeFirst: {s_dt_first}", flush=True)
        print(f"    - Metadata datetimeLast : {s_dt_last}", flush=True)
        print(f"    - Metadata summary: {s_summary}", flush=True)
        
        sensor_test_info = {
            "sensor_id": s_id,
            "meta_datetime_first": s_dt_first,
            "meta_datetime_last": s_dt_last,
            "windows_tested": {}
        }
        
        # Test each window
        for win_name, dt_start, dt_end in windows:
            time.sleep(0.4)
            params = {
                "datetime_from": dt_start,
                "datetime_to": dt_end,
                "limit": 1000,
                "page": 1
            }
            m_res = query_api(f"{BASE_URL}/sensors/{s_id}/measurements", params)
            if not m_res:
                print(f"    [{win_name}] -> Failed query", flush=True)
                continue
            
            results = m_res.get("results", [])
            meta = m_res.get("meta", {})
            found_count = meta.get("found")
            
            ts_list = []
            for r_item in results:
                p_p = r_item.get("period") or {}
                dt_f = p_p.get("datetimeFrom") or {}
                ts = dt_f.get("utc") if isinstance(dt_f, dict) else None
                if not ts:
                    ts = r_item.get("datetime", {}).get("utc") if isinstance(r_item.get("datetime"), dict) else r_item.get("datetime")
                if ts:
                    ts_list.append(ts)
            
            if ts_list:
                ts_list.sort()
                earliest = ts_list[0]
                latest = ts_list[-1]
                print(f"    [{win_name}] -> Found {len(results)} rows (meta.found: {found_count}) | Range: {earliest} to {latest}", flush=True)
                sensor_test_info["windows_tested"][win_name] = {
                    "count_page1": len(results),
                    "meta_found": found_count,
                    "earliest": earliest,
                    "latest": latest
                }
            else:
                print(f"    [{win_name}] -> 0 rows returned (meta.found: {found_count})", flush=True)
                sensor_test_info["windows_tested"][win_name] = {
                    "count_page1": 0,
                    "meta_found": found_count,
                    "earliest": None,
                    "latest": None
                }
                
        station_entry["sensors"].append(sensor_test_info)
        
    results_summary.append(station_entry)

# Save test results to json for reference
with open("data/raw/openaq_depth_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "="*80, flush=True)
print("TEST COMPLETED. SUMMARY FILE SAVED TO data/raw/openaq_depth_test_results.json", flush=True)
print("="*80, flush=True)
