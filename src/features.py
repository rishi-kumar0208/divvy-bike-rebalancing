"""
Feature engineering functions for the Divvy bike rebalancing pipeline.

Functions
---------
build_station_day_calendar(con)
    Build the full station×day calendar using hybrid progressive densification.
compute_inventory_bounds(df)
    Derive [L, U] inventory bounds from cumulative net flow statistics.
add_lag_features(df)
    Shift selected columns by 1 day within each station group to create _prev features.
add_rolling_features(df)
    Compute 7-day rolling averages for activity and inventory columns with
    a 3-tier fallback strategy for missing values.
"""

import pandas as pd
import numpy as np


def build_station_day_calendar(con) -> pd.DataFrame:
    """
    Build the station×day calendar using hybrid progressive densification.

    Each station receives a row for every calendar day from its own first observed
    date (the minimum of its first appearance as from_station_id in starttime or
    as to_station_id in stoptime) through the dataset-wide maximum date.

    Station capacity is derived as GREATEST(cap_start, cap_end) via a FULL OUTER
    JOIN of departure-side and arrival-side capacity observations per station×day.

    Cumulative net flow uses hourly buckets joined with a FULL OUTER JOIN — a LEFT
    JOIN on departures would silently drop hours where only arrivals occurred.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection with the 'trips' view registered.

    Returns
    -------
    pandas.DataFrame
        One row per (station_id, trip_date) pair with columns:
        station_id, trip_date, min_cumulative_flow, max_cumulative_flow,
        trips_departed, trips_arrived, station_capacity_day,
        latitude_start, longitude_start, temperature, events.
    """
    # --- Hybrid progressive densification: date spines via DuckDB ---
    calendar = con.execute("""
        WITH bounds AS (
            SELECT MIN(starttime::DATE) AS global_min,
                   MAX(stoptime::DATE)  AS global_max
            FROM trips
        ),
        station_dates AS (
            SELECT station_id, MIN(d) AS min_d
            FROM (
                SELECT from_station_id AS station_id, starttime::DATE AS d FROM trips
                UNION ALL
                SELECT to_station_id,                  stoptime::DATE      FROM trips
            ) GROUP BY 1
        ),
        expanded AS (
            SELECT s.station_id, dd.generate_series::DATE AS trip_date
            FROM station_dates s, bounds b,
                 generate_series(s.min_d, b.global_max, INTERVAL 1 DAY) AS dd
        )
        SELECT station_id, trip_date FROM expanded ORDER BY station_id, trip_date
    """).df()
    calendar['trip_date'] = pd.to_datetime(calendar['trip_date'])

    # --- Cumulative net flow using FULL OUTER JOIN on hourly buckets ---
    cum_stats = con.execute("""
        WITH lf_hourly AS (
            SELECT COALESCE(d.station_id, a.station_id)                     AS station_id,
                   COALESCE(d.trip_date,  a.trip_date)                      AS trip_date,
                   COALESCE(d.hour, a.hour)                                 AS hour,
                   COALESCE(a.trips_arrived,  0) - COALESCE(d.trips_departed, 0) AS hourly_net_flow,
                   COALESCE(d.trips_departed, 0)                            AS trips_departed,
                   COALESCE(a.trips_arrived,  0)                            AS trips_arrived
            FROM (
                SELECT from_station_id               AS station_id,
                       starttime::DATE               AS trip_date,
                       EXTRACT(hour FROM starttime)  AS hour,
                       COUNT(*)                      AS trips_departed
                FROM trips GROUP BY 1, 2, 3
            ) d
            FULL OUTER JOIN (
                SELECT to_station_id                 AS station_id,
                       stoptime::DATE                AS trip_date,
                       EXTRACT(hour FROM stoptime)   AS hour,
                       COUNT(*)                      AS trips_arrived
                FROM trips GROUP BY 1, 2, 3
            ) a ON d.station_id = a.station_id
              AND d.trip_date   = a.trip_date
              AND d.hour        = a.hour
        ),
        lf_cum AS (
            SELECT station_id, trip_date, trips_departed, trips_arrived,
                   SUM(hourly_net_flow) OVER (
                       PARTITION BY station_id, trip_date
                       ORDER BY hour
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cumulative_net_flow
            FROM lf_hourly
        )
        SELECT station_id, trip_date,
               MIN(cumulative_net_flow) AS min_cumulative_flow,
               MAX(cumulative_net_flow) AS max_cumulative_flow,
               SUM(trips_departed)      AS trips_departed,
               SUM(trips_arrived)       AS trips_arrived
        FROM lf_cum
        GROUP BY 1, 2
    """).df()
    cum_stats['trip_date'] = pd.to_datetime(cum_stats['trip_date'])

    # --- Capacity: GREATEST(cap_start, cap_end) via FULL OUTER JOIN ---
    cap_df = con.execute("""
        WITH cap_start AS (
            SELECT from_station_id AS station_id,
                   starttime::DATE AS trip_date,
                   MAX(dpcapacity_start) AS cap_s
            FROM trips GROUP BY 1, 2
        ),
        cap_end AS (
            SELECT to_station_id   AS station_id,
                   stoptime::DATE  AS trip_date,
                   MAX(dpcapacity_end) AS cap_e
            FROM trips GROUP BY 1, 2
        )
        SELECT COALESCE(s.station_id, e.station_id)                          AS station_id,
               COALESCE(s.trip_date,  e.trip_date)                           AS trip_date,
               GREATEST(COALESCE(s.cap_s, 0), COALESCE(e.cap_e, 0))::DOUBLE AS cap_obs
        FROM cap_start s
        FULL OUTER JOIN cap_end e
          ON s.station_id = e.station_id AND s.trip_date = e.trip_date
    """).df()
    cap_df['trip_date'] = pd.to_datetime(cap_df['trip_date'])

    # --- Weather: average temperature, mode of events (both departure and arrival sides) ---
    weather_df = con.execute("""
        SELECT COALESCE(from_station_id, to_station_id)    AS station_id,
               COALESCE(starttime::DATE, stoptime::DATE)   AS trip_date,
               AVG(temperature)                            AS temperature,
               MODE() WITHIN GROUP (ORDER BY events)       AS events
        FROM trips
        GROUP BY 1, 2
    """).df()
    weather_df['trip_date'] = pd.to_datetime(weather_df['trip_date'])

    # --- Coordinates: average lat/lon per station from both sides ---
    coords_df = con.execute("""
        SELECT COALESCE(from_station_id, to_station_id)          AS station_id,
               AVG(COALESCE(latitude_start,  latitude_end))      AS latitude_start,
               AVG(COALESCE(longitude_start, longitude_end))     AS longitude_start
        FROM trips
        WHERE COALESCE(latitude_start, latitude_end) IS NOT NULL
        GROUP BY 1
    """).df()

    # --- Join everything onto the calendar ---
    calendar = (
        calendar
        .merge(cum_stats,  on=['station_id', 'trip_date'], how='left')
        .merge(cap_df,     on=['station_id', 'trip_date'], how='left')
        .merge(weather_df, on=['station_id', 'trip_date'], how='left')
        .merge(coords_df,  on='station_id',                how='left')
    )
    calendar.rename(columns={'cap_obs': 'station_capacity_day'}, inplace=True)

    # Zero-trip days: cumulative flow = 0, trip counts = 0
    calendar['min_cumulative_flow'] = calendar['min_cumulative_flow'].fillna(0)
    calendar['max_cumulative_flow'] = calendar['max_cumulative_flow'].fillna(0)
    calendar['trips_departed']      = calendar['trips_departed'].fillna(0).astype(int)
    calendar['trips_arrived']       = calendar['trips_arrived'].fillna(0).astype(int)

    calendar = calendar.sort_values(['station_id', 'trip_date']).reset_index(drop=True)
    return calendar


def compute_inventory_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the feasible starting-inventory interval for each station×day.

    min_start_inventory = clip(-min_cumulative_flow, 0, station_capacity_day)
    max_start_inventory = clip(station_capacity_day - max_cumulative_flow, 0, station_capacity_day)

    If max_start_inventory < min_start_inventory (inversion), set
    min_start_inventory = 0 and max_start_inventory = station_capacity_day.

    Prediction target: s_true = (min_start_inventory + max_start_inventory) / 2

    Parameters
    ----------
    df : pandas.DataFrame
        Calendar DataFrame containing min_cumulative_flow, max_cumulative_flow,
        and station_capacity_day columns.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with min_start_inventory, max_start_inventory, and
        s_true columns added.
    """
    cap = df['station_capacity_day']
    df = df.copy()

    df['min_start_inventory'] = np.clip(-df['min_cumulative_flow'], 0, cap)
    df['max_start_inventory'] = np.clip(cap - df['max_cumulative_flow'], 0, cap)

    # Inversion repair
    inverted = df['max_start_inventory'] < df['min_start_inventory']
    df.loc[inverted, 'min_start_inventory'] = 0
    df.loc[inverted, 'max_start_inventory'] = cap[inverted]

    df['s_true'] = (df['min_start_inventory'] + df['max_start_inventory']) / 2
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create _prev lag features by shifting selected columns by 1 day within each
    station group.

    After shifting, the first row per station has all _prev columns as NaN.
    These rows are dropped entirely — there is no valid prior-day context for a
    station's first day.

    Parameters
    ----------
    df : pandas.DataFrame
        Calendar DataFrame sorted by (station_id, trip_date). Must contain
        min_start_inventory and max_start_inventory columns.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with _prev columns added and first-day rows dropped.
    """
    lag_cols = [
        'min_start_inventory', 'max_start_inventory', 'station_capacity_day',
        'temperature', 'events', 'trips_departed', 'trips_arrived',
    ]
    rename_map = {
        'min_start_inventory': 'min_start_inventory_prev',
        'max_start_inventory': 'max_start_inventory_prev',
        'station_capacity_day': 'station_capacity_day_prev',
        'temperature':          'temperature_prev',
        'events':               'events_prev',
        'trips_departed':       'trips_departed_prev',
        'trips_arrived':        'trips_arrived_prev',
    }

    df = df.sort_values(['station_id', 'trip_date']).copy()
    shifted = (
        df.groupby('station_id')[lag_cols]
        .shift(1)
        .rename(columns=rename_map)
    )
    df = pd.concat([df, shifted], axis=1)

    # Drop first day per station (all _prev cols are NaN)
    df = df.dropna(subset=list(rename_map.values())).reset_index(drop=True)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 7-day rolling averages for activity and inventory columns.

    Rolling features are computed on the raw (unshifted) columns — do not apply
    a shift before computing rolling features.

    Missing values use a 3-tier fallback:
      1. Station-level 7-day rolling mean (primary).
      2. City-wide 7-day rolling mean by date (fallback — all stations are in
         Chicago so city-wide averages are geographically meaningful proxies).
      3. Global dataset mean (last resort).

    trips_departed_roll7 and trips_arrived_roll7 are filled with 0 (no activity
    in the past 7 days means zero is the correct value).

    Parameters
    ----------
    df : pandas.DataFrame
        Calendar DataFrame sorted by (station_id, trip_date). Must contain
        min_start_inventory and max_start_inventory columns.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with _roll7 columns added and NaNs resolved.
    """
    df = df.sort_values(['station_id', 'trip_date']).copy()

    # Source column → destination roll7 column
    roll_cols = {
        'trips_departed':     'trips_departed_roll7',
        'trips_arrived':      'trips_arrived_roll7',
        'temperature':        'temperature_roll7',
        'min_start_inventory': 'min_start_inventory_roll7',
        'max_start_inventory': 'max_start_inventory_roll7',
    }
    # Columns that need the 3-tier fallback (source → destination)
    tiered = {
        'temperature':         'temperature_roll7',
        'min_start_inventory': 'min_start_inventory_roll7',
        'max_start_inventory': 'max_start_inventory_roll7',
    }

    for src_col, dst_col in roll_cols.items():
        df[dst_col] = (
            df.groupby('station_id')[src_col]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )

    # 3-tier fallback for temperature and inventory rolling features
    for src_col, dst_col in tiered.items():
        # Tier 2: city-wide 7-day rolling mean by date
        city_daily = df.groupby('trip_date')[src_col].mean().sort_index()
        city_roll7 = city_daily.rolling(7, min_periods=1).mean()
        city_roll7_mapped = df['trip_date'].map(city_roll7)

        # Tier 3: global dataset mean
        global_mean = df[src_col].mean()

        df[dst_col] = df[dst_col].fillna(city_roll7_mapped)
        df[dst_col] = df[dst_col].fillna(global_mean)

    # Trip rolling features: fill with 0 (no activity = zero)
    df['trips_departed_roll7'] = df['trips_departed_roll7'].fillna(0)
    df['trips_arrived_roll7']  = df['trips_arrived_roll7'].fillna(0)

    return df
