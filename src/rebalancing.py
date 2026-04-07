"""
Rebalancing optimization functions for the Divvy bike rebalancing pipeline.

Functions
---------
haversine_m(lat1, lon1, lat2, lon2)
    Compute the Haversine distance in meters between two geographic coordinates.
build_knn_edges(stations_df, k)
    Build a KNN edge list connecting each station to its k nearest neighbors by
    Haversine distance.
adjust_to_fixed_fleet_int(day_df, fleet_size)
    Clip predicted starting inventories to [0, station_capacity_day] and adjust
    the sum to match the fixed fleet size by randomly adding or removing single
    bikes from eligible stations.
run_rebalancing_pipeline(df_test, k)
    End-to-end rebalancing pipeline: adjust each day to fixed fleet size, solve
    minimum-cost network flow for consecutive day pairs, evaluate post-OR coverage
    and efficiency.
"""

import math
import random
import numpy as np
import pandas as pd
import networkx as nx


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the Haversine distance in meters between two points on the Earth.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point in decimal degrees.
    lat2, lon2 : float
        Latitude and longitude of the second point in decimal degrees.

    Returns
    -------
    float
        Great-circle distance in meters.
    """
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_knn_edges(stations_df: pd.DataFrame, k: int = 8) -> list:
    """
    Build a KNN edge list connecting each station to its k nearest neighbors
    by Haversine distance.

    Edge weights are integer-rounded distances in meters. Edge capacity is set
    to a large constant (10^9) — effectively infinite for the min-cost flow solver.

    Parameters
    ----------
    stations_df : pandas.DataFrame
        DataFrame with columns: station_id, latitude_start, longitude_start.
    k : int, default 8
        Number of nearest neighbors per station.

    Returns
    -------
    list of tuples
        Each tuple is (station_id_a, station_id_b,
        {'capacity': 10^9, 'weight': distance_meters}).
    """
    INF_CAP = 10 ** 9
    ids = stations_df['station_id'].to_numpy()
    lats = stations_df['latitude_start'].to_numpy()
    lons = stations_df['longitude_start'].to_numpy()

    edges = []
    for i in range(len(ids)):
        dists = [
            (int(ids[j]), haversine_m(lats[i], lons[i], lats[j], lons[j]))
            for j in range(len(ids)) if i != j
        ]
        for neighbor_id, dist in sorted(dists, key=lambda x: x[1])[:k]:
            edges.append((
                int(ids[i]),
                neighbor_id,
                {'capacity': INF_CAP, 'weight': int(round(dist))}
            ))
    return edges


def adjust_to_fixed_fleet_int(day_df: pd.DataFrame, fleet_size: int) -> pd.DataFrame:
    """
    Clip predicted starting inventories to [0, station_capacity_day] and adjust
    the sum to match the fixed fleet size.

    Clip s_hat_r to [0, station_capacity_day], then randomly add or remove
    single bikes from eligible stations until the total equals fleet_size.
    Adding: choose randomly from stations below capacity.
    Removing: choose randomly from stations above 0.

    Parameters
    ----------
    day_df : pandas.DataFrame
        One row per station for a single day. Must contain station_id, s_hat_r,
        and station_capacity_day columns.
    fleet_size : int
        Target total number of bikes across all stations.

    Returns
    -------
    pandas.DataFrame
        day_df with s_forecast (clipped) and s_target (fleet-adjusted) columns added.
    """
    day_df = day_df.copy()
    day_df['s_forecast'] = (
        day_df['s_hat_r'].clip(lower=0, upper=day_df['station_capacity_day']).astype(int)
    )
    day_df['s_target'] = day_df['s_forecast'].copy()

    delta = int(fleet_size - day_df['s_target'].sum())

    if delta > 0:
        for _ in range(delta):
            eligible = day_df.index[day_df['s_target'] < day_df['station_capacity_day']]
            if len(eligible) == 0:
                break
            day_df.loc[random.choice(eligible), 's_target'] += 1
    elif delta < 0:
        for _ in range(-delta):
            eligible = day_df.index[day_df['s_target'] > 0]
            if len(eligible) == 0:
                break
            day_df.loc[random.choice(eligible), 's_target'] -= 1

    return day_df


def run_rebalancing_pipeline(df_test: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """
    End-to-end overnight rebalancing pipeline.

    Steps:
      1. Compute fleet_size as the integer sum of s_hat_r on the first test day.
      2. For each test day, clip s_hat_r to [0, station_capacity_day] and adjust
         to fleet_size via adjust_to_fixed_fleet_int.
      3. Build a KNN edge graph (k=8 nearest neighbors by Haversine distance,
         edge capacity=10^9, edge weights in integer meters).
      4. For each consecutive day pair, set node supply/demand as
         s_target_tomorrow - s_target_today and solve minimum-cost network flow
         using NetworkX.
      5. Evaluate post-OR coverage and conditional efficiency using s_target.
         RMSE is not recomputed post-OR.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test-set DataFrame. Must contain: station_id, trip_date, s_hat (raw
        prediction), s_hat_r (rounded), station_capacity_day,
        min_start_inventory, max_start_inventory, s_true,
        latitude_start, longitude_start.
    k : int, default 8
        Number of KNN neighbors for the rebalancing graph.

    Returns
    -------
    pandas.DataFrame
        df_test with s_target, covered_or, and efficiency_or columns added.
    """
    df_test = df_test.copy()
    df_test['trip_date'] = pd.to_datetime(df_test['trip_date'])
    dates = sorted(df_test['trip_date'].unique())

    # Fleet size: integer sum of s_hat_r on the first test day
    fleet_size = int(df_test.loc[df_test['trip_date'] == dates[0], 's_hat_r'].sum())

    # --- Step 2: adjust each day to fixed fleet size ---
    adjusted_days = []
    for dt in dates:
        day_df = df_test[df_test['trip_date'] == dt][
            ['station_id', 's_hat_r', 'station_capacity_day']
        ].copy()
        adj = adjust_to_fixed_fleet_int(day_df, fleet_size)
        adj['trip_date'] = dt
        adjusted_days.append(adj[['station_id', 'trip_date', 's_forecast', 's_target']])

    adj_targets = (
        pd.concat(adjusted_days, ignore_index=True)
        .sort_values(['station_id', 'trip_date'])
        .reset_index(drop=True)
    )

    # --- Step 3: build KNN edge graph ---
    stations_df = (
        df_test[['station_id', 'latitude_start', 'longitude_start']]
        .drop_duplicates(subset='station_id')
        .dropna(subset=['latitude_start', 'longitude_start'])
        .reset_index(drop=True)
    )
    base_edges = build_knn_edges(stations_df, k=k)

    G_base = nx.DiGraph()
    for u, v, attr in base_edges:
        G_base.add_edge(u, v, weight=attr['weight'], capacity=attr['capacity'])

    # --- Step 4: solve min-cost flow for each consecutive day pair ---
    flows_all = []
    costs_all = []

    for i in range(len(dates) - 1):
        d1, d2 = dates[i], dates[i + 1]
        supply_df = adj_targets[adj_targets['trip_date'] == d1].set_index('station_id')['s_target']
        demand_df = adj_targets[adj_targets['trip_date'] == d2].set_index('station_id')['s_target']

        G = G_base.copy()
        net = demand_df.subtract(supply_df, fill_value=0)
        for sid, val in net.items():
            if sid in G:
                G.nodes[sid]['demand'] = int(val)
            else:
                G.add_node(sid, demand=int(val))

        try:
            flow_dict = nx.min_cost_flow(G, demand='demand', capacity='capacity', weight='weight')
            total_cost = nx.cost_of_flow(G, flow_dict, weight='weight')
        except (nx.NetworkXUnfeasible, nx.NetworkXUnbounded):
            continue

        for u, flows in flow_dict.items():
            for v, f in flows.items():
                if f > 0:
                    flows_all.append({'from_station': u, 'to_station': v,
                                      'flow': f, 'trip_date': d1})
        costs_all.append({'trip_date': d1, 'total_cost_meters': total_cost})

    # --- Step 5: merge s_target back and evaluate post-OR coverage ---
    df_test = df_test.merge(
        adj_targets[['station_id', 'trip_date', 's_target']],
        on=['station_id', 'trip_date'],
        how='left',
    )

    s_target = df_test['s_target']
    df_test['covered_or'] = (
        (s_target >= df_test['min_start_inventory']) &
        (s_target <= df_test['max_start_inventory'])
    ).astype(int)

    width = df_test['max_start_inventory'] - df_test['min_start_inventory']
    eff = 1 - (s_target - df_test['s_true']).abs() / width
    df_test['efficiency_or'] = np.where(
        (df_test['covered_or'] == 1) & (width > 0), eff, np.nan
    )

    # Summary print
    cov = df_test['covered_or'].mean()
    eff_mean = df_test.loc[df_test['covered_or'] == 1, 'efficiency_or'].mean()
    print(f"Post-OR coverage : {cov:.4f} ({cov*100:.2f}%)")
    print(f"Post-OR efficiency: {eff_mean:.4f}")
    print(f"Fleet size used  : {fleet_size}")
    if costs_all:
        total = sum(c['total_cost_meters'] for c in costs_all)
        print(f"Total rebalancing cost: {total:,} meter-bikes across {len(costs_all)} nights")

    return df_test, pd.DataFrame(flows_all), pd.DataFrame(costs_all)
