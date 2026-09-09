"""
Second-pass cleanup of master_dataset_extended.csv.

Science-based thresholds:
- PM2.5 >= 900 µg/m³: ALWAYS a sentinel/saturation artifact — null out
  (Delhi CPCB Horiba instruments saturate at 900-1000; real record is ~800 in Nov 2023 Diwali)
- PM2.5 >= 500 in summer months (Mar-Sep): instrument error — null out
  (Delhi can hit 500+ only in severe Nov-Dec smog, physically impossible in summer)
- PM2.5 == exactly 0.0 with suspicious context: leave as-is (could be clean-air day)
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/master_dataset_extended.csv')
print(f"INPUT: {len(df):,} rows, {df['station_name'].nunique()} stations")
print(f"  Values >= 900: {(df['pm25_value'] >= 900).sum()}")
print(f"  Values >= 500: {(df['pm25_value'] >= 500).sum()}")
print()

pm_cols = [c for c in df.columns if 'pm25' in c]

# ----------------------------------------------------------------
# Rule 1: Null ALL pm25 columns where pm25_value >= 900 (saturation)
# ----------------------------------------------------------------
mask_sat = df['pm25_value'] >= 900
n_sat = mask_sat.sum()
for col in pm_cols:
    df.loc[mask_sat, col] = np.nan
print(f"Nulled {n_sat} rows with pm25_value >= 900 (instrument saturation)")

# ----------------------------------------------------------------
# Rule 2: Null where pm25_value >= 500 AND month in Mar-Sep (summer)
# These are physically impossible for Delhi summer pollution events
# ----------------------------------------------------------------
summer_months = [3, 4, 5, 6, 7, 8, 9]
mask_summer_high = (df['pm25_value'] >= 500) & (df['month'].isin(summer_months))
n_summer = mask_summer_high.sum()
for col in pm_cols:
    df.loc[mask_summer_high, col] = np.nan
print(f"Nulled {n_summer} summer rows with pm25_value >= 500 (physically impossible)")

# ----------------------------------------------------------------
# Drop rows where pm25_value is now NaN (unusable for training)
# ----------------------------------------------------------------
rows_before = len(df)
df = df.dropna(subset=['pm25_value'])
rows_dropped = rows_before - len(df)
print(f"Dropped {rows_dropped} NaN pm25 rows => {len(df):,} rows remaining")

# ----------------------------------------------------------------
# Final verification
# ----------------------------------------------------------------
print()
print("=== FINAL DATASET VERIFICATION ===")
print(f"  Total rows: {len(df):,}")
print(f"  Unique stations: {df['station_name'].nunique()}")
print(f"  All stations official gov format: {all(s.split(' - ')[-1].strip().upper() in {'DPCC','CPCB','UPPCB','HSPCB','IMD','RSPCB'} for s in df['station_name'].unique())}")
print(f"  Values >= 999 remaining: {(df['pm25_value'] >= 999).sum()}")
print(f"  Values >= 900 remaining: {(df['pm25_value'] >= 900).sum()}")
print(f"  Values >= 500 remaining: {(df['pm25_value'] >= 500).sum()}")
print()

winter = df[df['month'].isin([10, 11, 12, 1, 2])]
summer = df[df['month'].isin([3, 4, 5, 6, 7, 8, 9])]
print(f"  Winter (Oct-Feb) rows: {len(winter):,}")
print(f"  Winter max PM2.5: {winter['pm25_value'].max():.2f}")
print(f"  Winter mean PM2.5: {winter['pm25_value'].mean():.2f}")
print(f"  Winter >= 500 remaining: {(winter['pm25_value'] >= 500).sum()}")
print()
print(f"  Summer (Mar-Sep) rows: {len(summer):,}")
print(f"  Summer max PM2.5: {summer['pm25_value'].max():.2f}")
print(f"  Summer mean PM2.5: {summer['pm25_value'].mean():.2f}")
print(f"  Summer >= 500 remaining (should be 0): {(summer['pm25_value'] >= 500).sum()}")
print()
print(f"  Overall max PM2.5: {df['pm25_value'].max():.2f}")
print(f"  Overall mean PM2.5: {df['pm25_value'].mean():.2f}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ----------------------------------------------------------------
# Station summary
# ----------------------------------------------------------------
print()
print("=== STATION SUMMARY (58 official gov stations) ===")
for stn in sorted(df['station_name'].unique()):
    cnt = (df['station_name'] == stn).sum()
    pm_max = df.loc[df['station_name'] == stn, 'pm25_value'].max()
    pm_mean = df.loc[df['station_name'] == stn, 'pm25_value'].mean()
    print(f"  {stn}  [{cnt:,} rows | max {pm_max:.0f} | mean {pm_mean:.0f}]")

# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------
out_path = 'data/processed/master_dataset_extended.csv'
df.to_csv(out_path, index=False)
print()
print(f"SAVED: {out_path}")
print(f"  {len(df):,} rows x {len(df.columns)} columns")
