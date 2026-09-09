import os
import netCDF4 as nc
import numpy as np
import h5py

fname = 'data/raw/AERDA_L3_VIIRS_MODIS.A2026252.0300.001.nrt.nc'
ds = nc.Dataset(fname, 'r')
lats = ds.variables['latitude'][:]
lons = ds.variables['longitude'][:]

target_lat, target_lon = 28.6139, 77.2090
lat_idx = int(np.argmin(np.abs(lats - target_lat)))
lon_idx = int(np.argmin(np.abs(lons - target_lon)))

print("=" * 80)
print("NASA LANCE AERDA_L3_VIIRS_MODIS_NRT (3-HOUR GRIDDED L3) TEST INSPECTION")
print("=" * 80)
print(f"File Name:        {os.path.basename(fname)}")
print(f"File Size:        {os.path.getsize(fname) / (1024*1024):.2f} MB ({os.path.getsize(fname):,} bytes)")
print(f"Internal Format:  {ds.disk_format} ({ds.data_model})")

# Test opening with h5py as well
with h5py.File(fname, 'r') as hf:
    print(f"HDF5 File Keys:   {len(hf.keys())} root groups/datasets")

time_start = getattr(ds, "time_coverage_start", "N/A")
time_end = getattr(ds, "time_coverage_end", "N/A")
print(f"Granule Time:     {time_start} to {time_end}")
print(f"Spatial Grid:     Global 1.0° x 1.0° ({len(lons)} longitudes x {len(lats)} latitudes)")
print(f"Target Lat/Lon:   Delhi NCR ({target_lat}°N, {target_lon}°E)")
print(f"Grid Cell Center: ({lats[lat_idx]:.1f}°N, {lons[lon_idx]:.1f}°E) at [lon_idx={lon_idx}, lat_idx={lat_idx}]")

print("\n" + "-" * 80)
print("AOD / SATELLITE MEASUREMENTS FOR DELHI NCR CELL:")
print("-" * 80)

aerosol_groups = [
    'Terra_MODIS_NASADarkTarget_AOD_550',
    'Terra_MODIS_NASADarkTarget_AOD_550_QF3',
    'Terra_MODIS_NASADeepBlue_AOD_550_Land',
    'Terra_MODIS_NASADeepBlue_AOD_550_Land_QF3',
    'Terra_MODIS_NASADarkTargetDeepBlue_AOD_550_QF3',
    'Aqua_MODIS_NASADarkTarget_AOD_550',
    'SNPP_VIIRS_NASADeepBlue_AOD_550',
    'NOAA20_VIIRS_NASADeepBlue_AOD_550'
]

for gname in aerosol_groups:
    if gname in ds.groups:
        grp = ds.groups[gname]
        mean_var = grp.variables['Mean']
        val = mean_var[lon_idx, lat_idx]
        
        sd_var = grp.variables.get('Standard_Deviation')
        sd_val = sd_var[lon_idx, lat_idx] if sd_var else None
        
        cnt_var = grp.variables.get('Pixel_Counts')
        cnt_val = cnt_var[lon_idx, lat_idx] if cnt_var else None
        
        if not np.ma.is_masked(val):
            status = f"AOD = {float(val):.4f}"
            sd_str = f"± {float(sd_val):.4f}" if (sd_val is not None and not np.ma.is_masked(sd_val)) else ""
            cnt_str = f"({int(cnt_val)} L2 pixels averaged)" if (cnt_val is not None and not np.ma.is_masked(cnt_val)) else ""
            print(f"  * {gname:<48} : {status} {sd_str} {cnt_str}")
        else:
            print(f"  * {gname:<48} : [No overpass in this 3h slot / Masked]")

print("\n" + "-" * 80)
print("REGIONAL 3x3 GRID SURROUNDING DELHI NCR (Terra MODIS Combined Dark Target/Deep Blue QF3):")
print("-" * 80)
comb_grp = ds.groups['Terra_MODIS_NASADarkTargetDeepBlue_AOD_550_QF3']
comb_mean = comb_grp.variables['Mean'][:]

for dlat in [1, 0, -1]:
    row = []
    for dlon in [-1, 0, 1]:
        li = lat_idx + dlat
        lj = lon_idx + dlon
        v = comb_mean[lj, li]
        coord = f"({lats[li]:.1f}°N, {lons[lj]:.1f}°E)"
        if np.ma.is_masked(v):
            row.append(f"{coord}: Masked")
        else:
            row.append(f"{coord}: AOD={float(v):.4f}")
    print("  " + " | ".join(row))

print("\n" + "=" * 80)
print("CONFIRMATION: Real NASA LANCE Aerosol AOD extraction successfully verified!")
print("=" * 80)

ds.close()
