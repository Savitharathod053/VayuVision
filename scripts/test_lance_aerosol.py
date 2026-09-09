import os
import sys
import requests
import numpy as np
from dotenv import load_dotenv
import netCDF4 as nc
import h5py

load_dotenv()

TOKEN = os.getenv("EARTHDATA_TOKEN")
if not TOKEN:
    print("ERROR: EARTHDATA_TOKEN not found in .env")
    sys.exit(1)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Fetch latest granules from NASA CMR
print("=" * 70)
print("1. Querying NASA CMR for recent AERDA_L3_VIIRS_MODIS_NRT granules...")
cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=AERDA_L3_VIIRS_MODIS_NRT&sort_key=-start_date&page_size=5"
resp = requests.get(cmr_url, timeout=30)
resp.raise_for_status()

entries = resp.json().get("feed", {}).get("entry", [])
if not entries:
    print("ERROR: No granules found in CMR")
    sys.exit(1)

print(f"Found {len(entries)} recent granules.")
target_entry = entries[0]
granule_title = target_entry.get("title")
download_url = None

for link in target_entry.get("links", []):
    href = link.get("href", "")
    if href.endswith(".nc") or href.endswith(".hdf") or href.endswith(".h5"):
        download_url = href
        break

if not download_url:
    for link in target_entry.get("links", []):
        href = link.get("href", "")
        if href.startswith("http"):
            download_url = href
            break

print(f"Selected Granule: {granule_title}")
print(f"Download URL: {download_url}")

# 2. Download file
filename = download_url.split("/")[-1]
dest_path = os.path.join(OUTPUT_DIR, filename)

print("\n" + "=" * 70)
print(f"2. Downloading {filename} using Bearer token...")
headers = {"Authorization": f"Bearer {TOKEN}"}

if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
    with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        print(f"Content Length: {total_size / (1024*1024):.2f} MB")
        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
    print(f"Downloaded successfully: {dest_path} ({os.path.getsize(dest_path)} bytes)")
else:
    print(f"File already exists locally: {dest_path} ({os.path.getsize(dest_path)} bytes)")

# 3. Inspect File Format & Structure
print("\n" + "=" * 70)
print("3. Inspecting internal structure of downloaded file...")

try:
    ds = nc.Dataset(dest_path, "r")
    print(f"Format detected via netCDF4: {ds.disk_format} ({ds.data_model})")
    print(f"Global Attributes ({len(ds.ncattrs())}):")
    for attr in list(ds.ncattrs())[:8]:
        print(f"  {attr}: {getattr(ds, attr)}")
    
    print(f"\nDimensions ({len(ds.dimensions)}):")
    for dim_name, dim in ds.dimensions.items():
        print(f"  {dim_name}: size = {len(dim)}")
        
    print(f"\nGroups ({len(ds.groups)}):")
    for grp_name in ds.groups:
        print(f"  Group: {grp_name}")

    print(f"\nTop-level & Group Variables:")
    all_vars = {}
    
    def explore_group(grp, prefix=""):
        for var_name, var in grp.variables.items():
            full_name = f"{prefix}/{var_name}" if prefix else var_name
            all_vars[full_name] = var
            if any(k in var_name.lower() for k in ["aod", "optical", "lat", "lon", "time", "aerosol", "dark_target"]):
                dims = str(var.dimensions)
                shape = str(var.shape)
                dtype = str(var.dtype)
                scale = getattr(var, "scale_factor", 1.0)
                units = getattr(var, "units", "N/A")
                print(f"  * {full_name} | dims={dims} | shape={shape} | dtype={dtype} | units={units} | scale={scale}")
        for child_name, child_grp in grp.groups.items():
            explore_group(child_grp, prefix=f"{prefix}/{child_name}" if prefix else child_name)

    explore_group(ds)
    
    # Also test opening with h5py to confirm HDF5 compatibility
    try:
        with h5py.File(dest_path, "r") as hf:
            print(f"\nh5py Verification: Successfully opened as HDF5 container (keys={list(hf.keys())[:6]})")
    except Exception as he:
        print(f"h5py test: {he}")
    
    # 4. Identify Grid and Delhi NCR AOD
    print("\n" + "=" * 70)
    print("4. Extracting Delhi NCR (28.6139°N, 77.2090°E) Grid Cell & AOD...")
    
    lat_var = None
    lon_var = None
    for name, var in all_vars.items():
        base_name = name.split("/")[-1].lower()
        if base_name in ["latitude", "lat"]:
            lat_var = var
        elif base_name in ["longitude", "lon"]:
            lon_var = var

    if lat_var is not None and lon_var is not None:
        lats = lat_var[:]
        lons = lon_var[:]
        print(f"Latitude range: min={np.min(lats):.2f}, max={np.max(lats):.2f}, shape={lats.shape}")
        print(f"Longitude range: min={np.min(lons):.2f}, max={np.max(lons):.2f}, shape={lons.shape}")
        
        target_lat = 28.6139
        target_lon = 77.2090
        
        if lats.ndim == 1 and lons.ndim == 1:
            lat_idx = int(np.argmin(np.abs(lats - target_lat)))
            lon_idx = int(np.argmin(np.abs(lons - target_lon)))
            cell_lat = float(lats[lat_idx])
            cell_lon = float(lons[lon_idx])
            print(f"Delhi Target: ({target_lat}, {target_lon}) -> Nearest Cell Index: [lat_idx={lat_idx}, lon_idx={lon_idx}] -> Grid Center: ({cell_lat:.3f}°N, {cell_lon:.3f}°E)")
            
            print("\nAOD / Aerosol Values for Delhi NCR Cell:")
            for name, var in all_vars.items():
                base_name = name.split("/")[-1].lower()
                if any(k in base_name for k in ["aod", "optical", "aerosol"]):
                    data = var[:]
                    try:
                        if data.ndim == 2:
                            val = data[lat_idx, lon_idx]
                        elif data.ndim == 3:
                            val = data[0, lat_idx, lon_idx]
                        elif data.ndim == 4:
                            val = data[0, 0, lat_idx, lon_idx]
                        else:
                            val = "N/A (dim mismatch)"
                            
                        if np.ma.is_masked(val) or (hasattr(var, '_FillValue') and val == var._FillValue):
                            val_str = "MASKED / FILL_VALUE"
                        else:
                            val_str = f"{float(val):.4f}"
                        print(f"  - {name}: {val_str} (units: {getattr(var, 'units', 'N/A')})")
                    except Exception as ex:
                        print(f"  - {name}: Error extracting value ({ex})")
                        
    ds.close()

except Exception as e:
    print(f"Error inspecting file: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Inspection complete.")
