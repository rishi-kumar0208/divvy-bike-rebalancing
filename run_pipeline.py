"""
End-to-end pipeline script: builds features, trains LightGBM, reports coverage.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from src.utils import connect_duckdb
from src.features import (
    build_station_day_calendar,
    compute_inventory_bounds,
    add_lag_features,
    add_rolling_features,
)
from src.models import train_lgbm, evaluate_coverage, coverage_summary
from src.rebalancing import run_rebalancing_pipeline

DATA_PATH = "data/raw/divvy.parquet"

FEATURES = [
    'min_start_inventory_prev', 'max_start_inventory_prev',
    'station_capacity_day_prev', 'temperature_prev', 'events_prev',
    'trips_departed_prev', 'trips_arrived_prev',
    'trips_departed_roll7', 'trips_arrived_roll7',
    'temperature_roll7', 'min_start_inventory_roll7',
    'max_start_inventory_roll7', 'station_id',
]
CATEGORICAL_FEATURES = ['station_id', 'events_prev']

print("Step 1/5  Building station×day calendar...")
con = connect_duckdb(DATA_PATH)
df = build_station_day_calendar(con)
print(f"         Calendar shape: {df.shape}")

print("Step 2/5  NA handling and inventory bounds...")
# Forward fill capacity and weather within station group
for col in ['station_capacity_day', 'temperature', 'events']:
    df[col] = df.groupby('station_id')[col].transform(lambda s: s.ffill())

# Fill zero-trip cumulative flow with 0 (already done in build, but guard here)
df['min_cumulative_flow'] = df['min_cumulative_flow'].fillna(0)
df['max_cumulative_flow'] = df['max_cumulative_flow'].fillna(0)

df = compute_inventory_bounds(df)
print(f"         Inversions repaired: {(df['max_start_inventory'] < df['min_start_inventory']).sum()}")

print("Step 3/5  Adding rolling features (on raw columns)...")
df = add_rolling_features(df, train_end_date='2017-09-30')

print("Step 4/5  Adding lag features and splitting train/test...")
df = add_lag_features(df)

# Restrict to stations that appear in test set
test_stations = df.loc[df['trip_date'] >= '2017-10-01', 'station_id'].unique()
df = df[df['station_id'].isin(test_stations)].copy()

# Cast categoricals before model fit
df['station_id']  = df['station_id'].astype('category')
df['events_prev'] = df['events_prev'].astype('category')

train_df = df[df['trip_date'] < '2017-10-01'].copy()
test_df  = df[df['trip_date'] >= '2017-10-01'].copy()
print(f"         Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

# Drop rows where any feature or target is null
train_df = train_df.dropna(subset=FEATURES + ['s_true'])
test_df  = test_df.dropna(subset=FEATURES + ['s_true'])

print("Step 5/5  Training LightGBM and evaluating coverage...")
model = train_lgbm(train_df[FEATURES], train_df['s_true'], CATEGORICAL_FEATURES)

train_df['s_hat'] = model.predict(train_df[FEATURES])
test_df['s_hat']  = model.predict(test_df[FEATURES])

train_df = evaluate_coverage(train_df, pred_col='s_hat')
test_df  = evaluate_coverage(test_df,  pred_col='s_hat')

train_summary = coverage_summary(train_df)
test_summary  = coverage_summary(test_df)

print()
print("=" * 45)
print("           COVERAGE RESULTS")
print("=" * 45)
print(f"{'Metric':<25} {'Train':>8}  {'Test':>8}")
print("-" * 45)
print(f"{'Coverage rate':<25} {train_summary['coverage_rate']:>8.4f}  {test_summary['coverage_rate']:>8.4f}")
print(f"{'Mean efficiency':<25} {train_summary['mean_efficiency']:>8.4f}  {test_summary['mean_efficiency']:>8.4f}")
print(f"{'RMSE':<25} {train_summary['rmse']:>8.4f}  {test_summary['rmse']:>8.4f}")
print("=" * 45)
print(f"Train rows evaluated: {len(train_df):,}")
print(f"Test  rows evaluated: {len(test_df):,}")

print()
print("Step 6/6  Running rebalancing optimization (min-cost flow)...")
# s_hat_r is the rounded prediction used as input to rebalancing
test_df['s_hat_r'] = test_df['s_hat'].round()
test_df, flows_df, costs_df = run_rebalancing_pipeline(test_df, k=8)

print()
print("=" * 45)
print("        POST-REBALANCING (OR) RESULTS")
print("=" * 45)
or_coverage = test_df['covered_or'].mean()
or_efficiency = test_df.loc[test_df['covered_or'] == 1, 'efficiency_or'].mean()
print(f"{'Coverage rate':<25} {or_coverage:>8.4f}")
print(f"{'Mean efficiency':<25} {or_efficiency:>8.4f}")
print("=" * 45)

# Save outputs
import os
os.makedirs("reports", exist_ok=True)
test_df.to_csv("reports/rebalancing_results.csv", index=False)
if not flows_df.empty:
    flows_df.to_csv("reports/flow_log.csv", index=False)
if not costs_df.empty:
    costs_df.to_csv("reports/cost_log.csv", index=False)
print("Saved rebalancing_results.csv, flow_log.csv, cost_log.csv to reports/")
