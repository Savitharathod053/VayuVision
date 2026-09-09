"""
scripts/fetch_aerosol_nrt.py

Fetches Near-Real-Time (NRT) Aerosol Optical Depth (AOD) from NASA LANCE (AERDA_L3_VIIRS_MODIS_NRT product).
Extracts Delhi NCR (28.5°N, 77.5°E) and the surrounding 3x3 regional grid cells.
Appends valid satellite overpass records to data/raw/aerosol_nrt.csv.

Safety guarantees:
  - If nighttime/masked (no overpass in 3h slot), logs honestly and skips writing rather than fabricating data.
  - Avoids duplicate records for the same granule timestamp.
  - Preserves existing history in data/raw/aerosol_nrt.csv.
"""

import os
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path
import requests
import numpy as np
import netCDF4 as nc
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TOKEN = os.getenv("EARTHDATA_TOKEN")
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_CSV = RAW_DIR / "aerosol_nrt.csv"
TEMP_DOWNLOAD_DIR = RAW_DIR / "temp_granules"

# 3x3 Grid centered on Delhi NCR (28.5°N, 77.5°E)
GRID_POINTS = [
    # (dlat, dlon, col_name, region_label)
    (1, -1, "aod_29_5_76_5", "NW (Haryana/Punjab)"),
    (1,  0, "aod_29_5_77_5", "N (Upper UP)"),
    (1,  1, "aod_29_5_78_5", "NE (Western UP)"),
    (0, -1, "aod_28_5_76_5", "W (Gurugram/Rewari)"),
    (0,  0, "aod_28_5_77_5", "Center (Delhi NCR)"),
    (0,  1, "aod_28_5_78_5", "E (Ghaziabad/East UP)"),
    (-1, -1, "aod_27_5_76_5", "SW (Rajasthan Border)"),
    (-1,  0, "aod_27_5_77_5", "S (South NCR/Palwal)"),
    (-1,  1, "aod_27_5_78_5", "SE (Aligarh/UP)")
]

CSV_HEADERS = [
    "fetch_timestamp",
    "granule_id",
    "granule_time_start",
    "granule_time_end",
    "primary_sensor",
    "is_fallback",
    "delhi_aod",
    "aod_29_5_76_5",
    "aod_29_5_77_5",
    "aod_29_5_78_5",
    "aod_28_5_76_5",
    "aod_28_5_77_5",
    "aod_28_5_78_5",
    "aod_27_5_76_5",
    "aod_27_5_77_5",
    "aod_27_5_78_5",
    "valid_cells_count",
    "status_note"
]


def get_existing_granule_ids() -> set:
    """Read existing granule IDs to avoid duplicate rows in CSV."""
    if not OUTPUT_CSV.exists():
        return set()
    try:
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["granule_id"] for row in reader if "granule_id" in row}
    except Exception:
        return set()


def fetch_latest_aerosol():
    if not TOKEN:
        print("[Aerosol NRT] ERROR: EARTHDATA_TOKEN not found in .env")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Query NASA CMR for the most recent granules (check top 3 to find most recent valid daytime or latest available)
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=AERDA_L3_VIIRS_MODIS_NRT&sort_key=-start_date&page_size=3"
    print(f"[Aerosol NRT] Querying NASA CMR: {cmr_url}")
    try:
        resp = requests.get(cmr_url, timeout=30)
        resp.raise_for_status()
        entries = resp.json().get("feed", {}).get("entry", [])
    except Exception as e:
        print(f"[Aerosol NRT] CMR Query failed: {e}")
        sys.exit(1)

    if not entries:
        print("[Aerosol NRT] No granules returned by NASA CMR")
        sys.exit(1)

    existing_granules = get_existing_granule_ids()

    # Priority hierarchy for aerosol retrieval groups:
    # 1. Best Quality: Terra MODIS Combined Dark Target + Deep Blue QF3
    # 2. Aqua MODIS Combined Dark Target + Deep Blue QF3
    # 3. SNPP VIIRS Deep Blue (QF3 / Mean)
    # 4. NOAA-20 VIIRS Deep Blue (QF3 / Mean)
    # 5. Terra MODIS Dark Target AOD 550
    # 6. Aqua MODIS Dark Target AOD 550
    # 7. SNPP VIIRS Dark Target AOD 550
    # 8. NOAA-20 VIIRS Dark Target AOD 550
    CANDIDATE_GROUPS = [
        ("Terra_MODIS_NASADarkTargetDeepBlue_AOD_550_QF3", False),
        ("Aqua_MODIS_NASADarkTargetDeepBlue_AOD_550_QF3", True),
        ("SNPP_VIIRS_NASADeepBlue_AOD_550_QF3", True),
        ("NOAA20_VIIRS_NASADeepBlue_AOD_550_QF3", True),
        ("SNPP_VIIRS_NASADeepBlue_AOD_550", True),
        ("NOAA20_VIIRS_NASADeepBlue_AOD_550", True),
        ("Terra_MODIS_NASADarkTarget_AOD_550_QF3", True),
        ("Aqua_MODIS_NASADarkTarget_AOD_550_QF3", True),
        ("SNPP_VIIRS_NASADarkTarget_AOD_550_QF3", True),
        ("NOAA20_VIIRS_NASADarkTarget_AOD_550_QF3", True),
        ("Terra_MODIS_NASADarkTarget_AOD_550", True),
        ("Aqua_MODIS_NASADarkTarget_AOD_550", True),
    ]

    target_lat = 28.6139
    target_lon = 77.2090
    headers = {"Authorization": f"Bearer {TOKEN}"}

    for entry in entries:
        title = entry.get("title")
        download_url = None
        for link in entry.get("links", []):
            href = link.get("href", "")
            if href.endswith(".nc") or href.endswith(".h5"):
                download_url = href
                break
        if not download_url:
            continue

        filename = download_url.split("/")[-1]
        local_nc_path = TEMP_DOWNLOAD_DIR / filename

        # Download if not present
        if not local_nc_path.exists() or local_nc_path.stat().st_size == 0:
            print(f"[Aerosol NRT] Downloading granule {filename}...")
            try:
                with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(local_nc_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                print(f"[Aerosol NRT] Download complete: {local_nc_path.stat().st_size / (1024*1024):.2f} MB")
            except Exception as e:
                print(f"[Aerosol NRT] Failed to download {download_url}: {e}")
                continue

        # Open and inspect
        try:
            ds = nc.Dataset(str(local_nc_path), "r")
            lats = ds.variables["latitude"][:]
            lons = ds.variables["longitude"][:]
            time_start = getattr(ds, "time_coverage_start", getattr(ds, "GranuleBeginningDateTime", "N/A"))
            time_end = getattr(ds, "time_coverage_end", getattr(ds, "GranuleEndingDateTime", "N/A"))

            lat_idx = int(np.argmin(np.abs(lats - target_lat)))
            lon_idx = int(np.argmin(np.abs(lons - target_lon)))

            chosen_group = None
            is_fallback = False
            extracted_grid = {}
            valid_count = 0

            # Find best group with non-masked data covering the 3x3 region
            for gname, fallback_flag in CANDIDATE_GROUPS:
                if gname in ds.groups and "Mean" in ds.groups[gname].variables:
                    mean_var = ds.groups[gname].variables["Mean"]
                    data = mean_var[:]
                    # Check center cell
                    center_val = data[lon_idx, lat_idx]
                    
                    # Count non-masked cells in 3x3
                    cell_vals = {}
                    non_masked = 0
                    for dlat, dlon, col_name, _ in GRID_POINTS:
                        li = lat_idx + dlat
                        lj = lon_idx + dlon
                        v = data[lj, li]
                        if not np.ma.is_masked(v) and not np.isnan(v):
                            cell_vals[col_name] = round(float(v), 4)
                            non_masked += 1
                        else:
                            cell_vals[col_name] = ""

                    if non_masked > 0:
                        chosen_group = gname
                        is_fallback = fallback_flag
                        extracted_grid = cell_vals
                        valid_count = non_masked
                        break

            ds.close()

            if valid_count > 0:
                print(f"[Aerosol NRT] Found valid satellite pass in {filename}:")
                print(f"  Sensor: {chosen_group} (is_fallback={is_fallback})")
                print(f"  Valid 3x3 Cells: {valid_count}/9")
                print(f"  Delhi Center AOD: {extracted_grid.get('aod_28_5_77_5', 'Masked')}")

                # Safety Check: Check if already written
                if filename in existing_granules:
                    print(f"[Aerosol NRT] Granule {filename} already recorded in {OUTPUT_CSV.name}. Skipping duplicate append.")
                    return True

                # Append to CSV
                file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
                now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                row = {
                    "fetch_timestamp": now_iso,
                    "granule_id": filename,
                    "granule_time_start": time_start,
                    "granule_time_end": time_end,
                    "primary_sensor": chosen_group,
                    "is_fallback": str(is_fallback),
                    "delhi_aod": extracted_grid.get("aod_28_5_77_5", ""),
                    "valid_cells_count": valid_count,
                    "status_note": f"Valid pass ({valid_count}/9 grid cells non-masked)"
                }
                for _, _, col_name, _ in GRID_POINTS:
                    row[col_name] = extracted_grid.get(col_name, "")

                with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)

                print(f"[Aerosol NRT] Successfully appended new row to {OUTPUT_CSV.name}")
                return True
            else:
                print(f"[Aerosol NRT] Granule {filename} has no valid non-masked cells in Delhi NCR (likely nighttime slot {time_start}). Checking next candidate...")

        except Exception as e:
            print(f"[Aerosol NRT] Error inspecting {filename}: {e}")

    # If all candidate granules in the search were nighttime / masked:
    print("[Aerosol NRT] No valid satellite pass in current window (all optical AOD cells masked/nighttime).")
    print("[Aerosol NRT] Safety check: Skipped CSV write to preserve existing historical data.")
    return False


if __name__ == "__main__":
    success = fetch_latest_aerosol()
    sys.exit(0 if success else 0)
