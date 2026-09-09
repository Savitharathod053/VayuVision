"""
Investigate remaining extreme PM2.5 values (>=500) after initial cleanup.
Decide on correct threshold for instrument saturation.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/master_dataset_extended.csv')

print(f"Current dataset: {len(df):,} rows")
print()

# Extreme values
high = df[df['pm25_value'] >= 500].copy()
print(f"Rows with PM2.5 >= 500: {len(high)}")
print()
print("By station:")
by_stn = high.groupby('station_name')['pm25_value'].agg(['count', 'min', 'max', 'mean']).sort_values('count', ascending=False)
print(by_stn.to_string())
print()

print("Top 30 extreme values:")
top30 = high[['station_name', 'timestamp', 'pm25_value', 'month']].sort_values('pm25_value', ascending=False).head(30)
print(top30.to_string())
print()

# Remaining sentinel-range values
print(f"Values >= 900 remaining: {(df['pm25_value'] >= 900).sum()}")
print(f"Values 800-900: {((df['pm25_value'] >= 800) & (df['pm25_value'] < 900)).sum()}")
print(f"Values 700-800: {((df['pm25_value'] >= 700) & (df['pm25_value'] < 800)).sum()}")
print(f"Values 600-700: {((df['pm25_value'] >= 600) & (df['pm25_value'] < 700)).sum()}")
print(f"Values 500-600: {((df['pm25_value'] >= 500) & (df['pm25_value'] < 600)).sum()}")

# Note: Delhi CPCB scientific consensus:
# - Real physical max during severe winter episodes: ~900 µg/m³ (Nov-Dec)
# - Values above 900 = instrument saturation sentinel -> already nulled
# - Values 500-900 in Oct-Feb ARE physically possible (Delhi smog crisis)
# - Values 500-900 in summer (Mar-Sep) are more suspicious
print()
summer = df[(df['month'].isin([3, 4, 5, 6, 7, 8, 9])) & (df['pm25_value'] >= 500)]
print(f"Summer (Mar-Sep) values >= 500: {len(summer)}")
if len(summer) > 0:
    print(summer[['station_name', 'timestamp', 'pm25_value', 'month']].sort_values('pm25_value', ascending=False).head(20).to_string())
