"""
scripts/audit_station_counts.py

Audits per-station row counts in openaq_extended_historical_pm25.csv,
sorts from lowest to highest, and flags any station with < 50% of median row count.
"""

import pandas as pd

df = pd.read_csv("data/raw/openaq_extended_historical_pm25.csv")
n_st = df["station_name"].nunique()

st_counts = df.groupby("station_name").agg(
    total_rows=("pm25_value", "count"),
    min_date=("timestamp", "min"),
    max_date=("timestamp", "max")
).reset_index()

st_counts["min_date_day"] = st_counts["min_date"].str[:10]
st_counts["max_date_day"] = st_counts["max_date"].str[:10]

median_rows = st_counts["total_rows"].median()
threshold_50pct = median_rows * 0.50

st_counts["pct_of_median"] = (st_counts["total_rows"] / median_rows) * 100.0
st_counts["is_flagged"] = st_counts["total_rows"] < threshold_50pct

st_counts = st_counts.sort_values(by="total_rows", ascending=True).reset_index(drop=True)

print("=" * 120)
print(f"OPENAQ EXTENDED DATASET AUDIT: PER-STATION ROW COUNTS (63 STATIONS)")
print("=" * 120)
print(f"Total Stations            : {n_st}")
print(f"Median Station Row Count  : {median_rows:,.0f} rows")
print(f"50% of Median Threshold   : {threshold_50pct:,.0f} rows\n")

print(f"{'#':<3} | {'Station Name':<50} | {'Total Rows':<10} | {'% of Median':<11} | {'Start Date':<10} | {'End Date':<10} | {'Status':<15}")
print("-" * 120)

for idx, r in st_counts.iterrows():
    status = "[FLAGGED < 50%]" if r["is_flagged"] else "OK"
    name = r["station_name"]
    rows = r["total_rows"]
    pct = r["pct_of_median"]
    s_dt = r["min_date_day"]
    e_dt = r["max_date_day"]
    print(f"{idx+1:<3} | {name:<50} | {rows:>10,d} | {pct:>10.1f}% | {s_dt} | {e_dt} | {status}")

flagged_df = st_counts[st_counts["is_flagged"]]
print("\n" + "=" * 120)
print(f"FLAGGED STATIONS SUMMARY (< {threshold_50pct:,.0f} rows): {len(flagged_df)} out of {n_st} stations")
print("=" * 120)

for idx, r in flagged_df.iterrows():
    name = r["station_name"]
    rows = r["total_rows"]
    pct = r["pct_of_median"]
    s_dt = r["min_date_day"]
    e_dt = r["max_date_day"]
    print(f"  * {name:<45}: {rows:>6,d} rows ({pct:5.1f}% of median) | Range: {s_dt} -> {e_dt}")
