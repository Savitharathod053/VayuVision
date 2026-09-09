"""
Analyze and clean master_dataset_extended.csv:
1. List all unique station names
2. Flag non-government stations
3. Check for sentinel values (999.99 etc.)
4. Rebuild clean dataset with only confirmed gov stations
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/master_dataset_extended.csv')
print(f"=== CURRENT DATASET: {df.shape[0]:,} rows x {df.shape[1]} cols ===\n")

all_stations = sorted(df['station_name'].dropna().unique().tolist())
print(f"Total unique stations: {len(all_stations)}\n")

# -----------------------------------------------------------------------
# Define the 63 EXPECTED government-network stations
# (DPCC / CPCB / UPPCB / HSPCB / IMD / RSPCB)
# -----------------------------------------------------------------------
OFFICIAL_GOV_AGENCIES = {'DPCC', 'CPCB', 'UPPCB', 'HSPCB', 'IMD', 'RSPCB'}

def is_official_gov_station(name):
    """Returns True if the station name ends with '- AGENCY' from the official list."""
    if not isinstance(name, str):
        return False
    parts = name.strip().split(' - ')
    if len(parts) >= 2:
        agency = parts[-1].strip().upper()
        return agency in OFFICIAL_GOV_AGENCIES
    return False

print("=== STATION CLASSIFICATION ===")
gov_stations = []
non_gov_stations = []

for s in all_stations:
    if is_official_gov_station(s):
        gov_stations.append(s)
    else:
        non_gov_stations.append(s)
        
print(f"\n✅ OFFICIAL GOV STATIONS ({len(gov_stations)}):")
for s in gov_stations:
    rows = df[df['station_name'] == s]
    print(f"   {s}  [{len(rows):,} rows]")

print(f"\n❌ NON-GOV / SUSPECT STATIONS ({len(non_gov_stations)}) — WILL BE REMOVED:")
for s in non_gov_stations:
    rows = df[df['station_name'] == s]
    pm = rows['pm25_value'].dropna()
    pm_min = pm.min() if len(pm) > 0 else float('nan')
    pm_max = pm.max() if len(pm) > 0 else float('nan')
    pm_mean = pm.mean() if len(pm) > 0 else float('nan')
    sentinel_count = (pm >= 900).sum() if len(pm) > 0 else 0
    print(f"   '{s}'")
    print(f"      Rows: {len(rows):,}  |  PM2.5: {pm_min:.1f}–{pm_max:.1f} (mean {pm_mean:.1f})  |  Sentinels(>=900): {sentinel_count}")

# -----------------------------------------------------------------------
# Sentinel Value Check ACROSS ALL stations
# -----------------------------------------------------------------------
print("\n=== SENTINEL VALUE CHECK (pm25_value >= 900) ===")
sentinel_mask = df['pm25_value'] >= 900
total_sentinels = sentinel_mask.sum()
print(f"Total rows with pm25 >= 900: {total_sentinels:,}")
if total_sentinels > 0:
    by_station = df[sentinel_mask].groupby('station_name')['pm25_value'].agg(['count', 'min', 'max'])
    print(by_station.to_string())

print("\n=== WINTER MAX PM2.5 (BEFORE CLEANING) ===")
winter_df = df[df['month'].isin([10, 11, 12, 1, 2])]
print(f"  Max PM2.5 (Oct-Feb): {winter_df['pm25_value'].max():.2f}")
print(f"  Values >= 999: {(winter_df['pm25_value'] >= 999).sum():,}")
print(f"  Values >= 500: {(winter_df['pm25_value'] >= 500).sum():,}")

# -----------------------------------------------------------------------
# REBUILD CLEAN DATASET
# -----------------------------------------------------------------------
print("\n=== REBUILDING CLEAN DATASET ===")

# Step 1: Keep only official government stations
df_clean = df[df['station_name'].isin(gov_stations)].copy()
print(f"After removing non-gov stations: {len(df_clean):,} rows (removed {len(df) - len(df_clean):,})")

# Step 2: Null out sentinel values (>= 999) in pm25_value and all derived columns
sentinel_pm_mask = df_clean['pm25_value'] >= 999
if sentinel_pm_mask.sum() > 0:
    print(f"Nulling {sentinel_pm_mask.sum():,} sentinel PM2.5 values (>=999)...")
    pm_cols = [c for c in df_clean.columns if 'pm25' in c]
    for col in pm_cols:
        df_clean.loc[df_clean['pm25_value'] >= 999, col] = np.nan
        
# Step 3: Also check for suspicious sentinel-like values in other ranges
# (e.g., exactly 999.99, or values >= 500 that are statistically implausible)
# For PM2.5 in Delhi: physically plausible max is ~999 µg/m³ in extreme events
# but realistic cap is ~900 µg/m³ — we already covered >=999

# Check for 999.99 exactly
mask_999_99 = df_clean['pm25_value'] == 999.99
if mask_999_99.sum() > 0:
    print(f"Found {mask_999_99.sum()} rows with exactly 999.99 — nulling...")
    pm_cols = [c for c in df_clean.columns if 'pm25' in c]
    for col in pm_cols:
        df_clean.loc[mask_999_99, col] = np.nan

# Step 4: Drop rows where pm25_value is NaN (these are useless for training)
rows_before_dropna = len(df_clean)
df_clean = df_clean.dropna(subset=['pm25_value'])
print(f"After dropping NaN pm25 rows: {len(df_clean):,} rows (removed {rows_before_dropna - len(df_clean):,} NaN rows)")

# Step 5: Verify final station list
final_stations = sorted(df_clean['station_name'].unique().tolist())
print(f"\nFinal station count: {len(final_stations)}")
all_gov = all(is_official_gov_station(s) for s in final_stations)
print(f"All remaining stations are official gov format: {all_gov}")

# Step 6: Stats
print("\n=== FINAL DATASET STATS ===")
print(f"  Total rows: {len(df_clean):,}")
print(f"  Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
winter_clean = df_clean[df_clean['month'].isin([10, 11, 12, 1, 2])]
print(f"  Winter rows (Oct-Feb): {len(winter_clean):,}")
print(f"  Max PM2.5 (winter, after clean): {winter_clean['pm25_value'].max():.2f}")
print(f"  Mean PM2.5 (winter, after clean): {winter_clean['pm25_value'].mean():.2f}")
print(f"  Max PM2.5 (all year, after clean): {df_clean['pm25_value'].max():.2f}")
print(f"  Values >= 999 remaining: {(df_clean['pm25_value'] >= 999).sum():,}")

print("\n=== FINAL STATION LIST ===")
for s in final_stations:
    count = (df_clean['station_name'] == s).sum()
    print(f"  {s}  [{count:,} rows]")

# Save clean dataset
out_path = 'data/processed/master_dataset_extended.csv'
df_clean.to_csv(out_path, index=False)
print(f"\n✅ Saved clean dataset → {out_path}")
print(f"   {len(df_clean):,} rows × {len(df_clean.columns)} columns")
