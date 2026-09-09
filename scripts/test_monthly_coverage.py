"""
scripts/test_monthly_coverage.py

Tests monthly measurement availability for active PM2.5 sensors across 5 Delhi stations
from Feb 2025 to Sep 2026, avoiding deep pagination 408 timeouts.
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

stations = [
    {"name": "Alipur, Delhi - DPCC", "loc_id": 6932, "sensor_id": 12235698, "legacy_sensor": 19935},
    {"name": "Anand Vihar, New Delhi - DPCC", "loc_id": 235, "sensor_id": 12235610, "legacy_sensor": 384},
    {"name": "R K Puram, Delhi - DPCC", "loc_id": 17, "sensor_id": 12234787, "legacy_sensor": 35},
    {"name": "Punjabi Bagh, Delhi - DPCC", "loc_id": 50, "sensor_id": 12234796, "legacy_sensor": 396},
    {"name": "Bawana, Delhi - DPCC", "loc_id": 8472, "sensor_id": 12235294, "legacy_sensor": 36391},
]

# Months to test for active sensor
months = [
    ("Feb 2025", "2025-02-01T00:00:00Z", "2025-02-28T23:59:59Z"),
    ("Winter peak (Nov 2025)", "2025-11-01T00:00:00Z", "2025-11-30T23:59:59Z"),
    ("Winter peak (Dec 2025)", "2025-12-01T00:00:00Z", "2025-12-31T23:59:59Z"),
    ("Winter peak (Jan 2026)", "2026-01-01T00:00:00Z", "2026-01-31T23:59:59Z"),
    ("Summer (May 2026)", "2026-05-01T00:00:00Z", "2026-05-31T23:59:59Z"),
    ("Recent (Aug 2026)", "2026-08-01T00:00:00Z", "2026-08-31T23:59:59Z"),
]

print("="*80, flush=True)
print("TESTING MONTHLY PM2.5 DATA DENSITY PER STATION (ACTIVE SENSORS)", flush=True)
print("="*80, flush=True)

station_stats = []

for st in stations:
    s_id = st["sensor_id"]
    name = st["name"]
    print(f"\nStation: {name} (Active Sensor: {s_id})", flush=True)
    
    month_counts = {}
    for m_label, dt_start, dt_end in months:
        time.sleep(0.35)
        # Fetch page 1
        r = requests.get(
            f"{BASE_URL}/sensors/{s_id}/measurements",
            headers=headers,
            params={"datetime_from": dt_start, "datetime_to": dt_end, "limit": 1000, "page": 1},
            timeout=20
        )
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            meta = data.get("meta", {})
            found = meta.get("found")
            print(f"  [{m_label}] -> Found {len(results)} rows (meta.found: {found})", flush=True)
            month_counts[m_label] = {"page1_count": len(results), "meta_found": found}
        else:
            print(f"  [{m_label}] -> HTTP {r.status_code}", flush=True)
            month_counts[m_label] = {"error": r.status_code}
            
    station_stats.append({
        "name": name,
        "active_sensor_id": s_id,
        "legacy_sensor_id": st["legacy_sensor"],
        "monthly_density": month_counts
    })

print("\n" + "="*80, flush=True)
print("ALL MONTHLY DENSITY CHECKS COMPLETED.", flush=True)
print("="*80, flush=True)
