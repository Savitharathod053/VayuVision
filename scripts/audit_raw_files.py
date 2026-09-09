"""Audit raw data files for integrity."""
import pandas as pd

print('=== data/raw/openaq_raw.csv (LIVE SNAPSHOT) ===')
df = pd.read_csv('data/raw/openaq_raw.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
locs = sorted(df['location'].dropna().unique().tolist())
print(f'Unique stations (location col): {len(locs)}')
for s in locs[:15]:
    print(f'  {s}')
if len(locs) > 15:
    print(f'  ... and {len(locs)-15} more')
params = df['parameter'].dropna().unique().tolist()
print(f'Parameters: {params}')
dt_col = 'datetime' if 'datetime' in df.columns else df.columns[df.columns.str.contains('time', case=False)][0]
print(f'Date range ({dt_col}): {df[dt_col].min()} to {df[dt_col].max()}')
pm25 = df[df['parameter'] == 'pm25']
print(f'PM2.5 rows: {len(pm25)}  |  value range: {pm25["value"].min():.1f} - {pm25["value"].max():.1f}')
print()
print('Sample rows (first 5):')
print(df[['location', 'parameter', 'value', 'unit', dt_col]].head(5).to_string())
print()

print('=== data/raw/openaq_historical_pm25.csv (HISTORICAL) ===')
dh = pd.read_csv('data/raw/openaq_historical_pm25.csv')
print(f'Shape: {dh.shape}')
print(f'Columns: {dh.columns.tolist()}')
print(f'Unique stations: {dh["station_name"].nunique()}')
print(f'Date range: {dh["timestamp"].min()} to {dh["timestamp"].max()}')
print(f'PM2.5 max: {dh["pm25_value"].max():.2f}  |  mean: {dh["pm25_value"].mean():.2f}')
print()
print('Sample rows (first 3):')
print(dh[['station_name', 'timestamp', 'pm25_value', 'latitude', 'longitude']].head(3).to_string())
