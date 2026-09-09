import pandas as pd
import re

df = pd.read_csv('data/processed/master_dataset_extended.csv')
stations = sorted(df['station_name'].unique().tolist())
print(f"Total unique stations in master_dataset_extended.csv: {len(stations)}\n")

gov_agencies = ['DPCC', 'CPCB', 'UPPCB', 'HSPCB', 'IMD', 'IITM', 'RSPCB']
pattern = re.compile(r'.+,\s*.+\s*-\s*(' + '|'.join(gov_agencies) + r')$')

matched = []
flagged = []

for s in stations:
    if pattern.match(s):
        matched.append(s)
    else:
        flagged.append(s)

print("=" * 80)
print(f"1. STATIONS MATCHING OFFICIAL GOV FORMAT ({len(matched)} stations):")
print("=" * 80)
for i, s in enumerate(matched, 1):
    rows = (df['station_name'] == s).sum()
    print(f"  {i:>2}. [GOV] {s:<55} ({rows:,} rows)")

print("\n" + "=" * 80)
print(f"2. FLAGGED STATIONS NOT MATCHING OFFICIAL GOV FORMAT ({len(flagged)} stations):")
print("=" * 80)
for i, s in enumerate(flagged, 1):
    rows = (df['station_name'] == s).sum()
    sub = df[df['station_name'] == s]
    pm_mean = sub['pm25_value'].mean()
    pm_min = sub['pm25_value'].min()
    pm_max = sub['pm25_value'].max()
    print(f"  {i:>2}. [FLAGGED] \"{s}\"")
    print(f"       Rows: {rows:,} | PM2.5 Range: {pm_min:.1f} to {pm_max:.1f} µg/m³ (Mean: {pm_mean:.1f} µg/m³)")
    print(f"       City: {sub['city'].iloc[0]} | Coords: ({sub['latitude'].iloc[0]:.4f}, {sub['longitude'].iloc[0]:.4f})")
