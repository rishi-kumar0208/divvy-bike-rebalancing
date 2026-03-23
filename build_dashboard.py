"""
Generate postOR_station_kpi.csv, postOR_daily_trends.csv, and the
standalone interactive HTML dashboard at reports/figures/dashboard.html.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 1. Load results and compute cumulative metrics
# ---------------------------------------------------------------------------
df = pd.read_csv("reports/rebalancing_results.csv", parse_dates=["trip_date"])

df = df.sort_values(["station_id", "trip_date"]).reset_index(drop=True)

# Expanding coverage per station (skipna=False — covered_or is always 0/1)
df["cumulative_coverage"] = (
    df.groupby("station_id")["covered_or"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

# Expanding efficiency per station (skipna=True by default — NaN rows are uncovered)
df["cumulative_efficiency"] = (
    df.groupby("station_id")["efficiency_or"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

station_kpi = df[["station_id", "trip_date", "cumulative_coverage", "cumulative_efficiency"]].copy()
station_kpi.to_csv("reports/postOR_station_kpi.csv", index=False)

daily_trends = (
    station_kpi
    .groupby("trip_date")
    .agg(
        avg_cumulative_coverage=("cumulative_coverage", "mean"),
        avg_cumulative_efficiency=("cumulative_efficiency", "mean"),
    )
    .reset_index()
)
daily_trends.to_csv("reports/postOR_daily_trends.csv", index=False)

print(f"Saved postOR_station_kpi.csv  ({len(station_kpi):,} rows)")
print(f"Saved postOR_daily_trends.csv ({len(daily_trends):,} rows)")

# ---------------------------------------------------------------------------
# 2. Pre-compute per-frame data
# ---------------------------------------------------------------------------
dates = sorted(station_kpi["trip_date"].unique())
date_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]


def get_frame_data(date):
    day = station_kpi[station_kpi["trip_date"] == date].copy()
    day["cumulative_efficiency"] = day["cumulative_efficiency"].fillna(0.0)
    bottom10_ids = set(day.nsmallest(10, "cumulative_coverage")["station_id"].tolist())
    gray = day[~day["station_id"].isin(bottom10_ids)]
    red  = day[ day["station_id"].isin(bottom10_ids)]
    line = daily_trends[daily_trends["trip_date"] <= date]
    return gray, red, line


# ---------------------------------------------------------------------------
# 3. Build figure — initialize with LAST date so slider default = full data
# ---------------------------------------------------------------------------
gray0, red0, line0 = get_frame_data(dates[-1])

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "<b>Station KPI Scatter</b>",
        "<b>Daily Cumulative Trends</b>",
    ],
    horizontal_spacing=0.12,
)

# Capture subplot title annotations before anything overwrites them
subplot_annotations = list(fig.layout.annotations)

# Trace 0 — gray stations
fig.add_trace(go.Scatter(
    x=gray0["cumulative_efficiency"].tolist(),
    y=gray0["cumulative_coverage"].tolist(),
    mode="markers",
    marker=dict(color="#888888", size=7, opacity=0.65),
    name="Stations",
    text=gray0["station_id"].astype(str).tolist(),
    hovertemplate="Station %{text}<br>Efficiency: %{x:.3f}<br>Coverage: %{y:.3f}<extra></extra>",
), row=1, col=1)

# Trace 1 — bottom-10 red stations
fig.add_trace(go.Scatter(
    x=red0["cumulative_efficiency"].tolist(),
    y=red0["cumulative_coverage"].tolist(),
    mode="markers+text",
    marker=dict(color="#FF4444", size=9),
    text=red0["station_id"].astype(str).tolist(),
    textposition="top center",
    textfont=dict(size=9, color="#FF6666"),
    name="Bottom 10 Coverage",
    hovertemplate="Station %{text}<br>Efficiency: %{x:.3f}<br>Coverage: %{y:.3f}<extra></extra>",
), row=1, col=1)

# Trace 2 — coverage line
fig.add_trace(go.Scatter(
    x=line0["trip_date"].tolist(),
    y=line0["avg_cumulative_coverage"].tolist(),
    mode="lines",
    line=dict(color="#4488FF", width=2.5),
    name="Avg. Cumulative Coverage",
), row=1, col=2)

# Trace 3 — efficiency line
fig.add_trace(go.Scatter(
    x=line0["trip_date"].tolist(),
    y=line0["avg_cumulative_efficiency"].tolist(),
    mode="lines",
    line=dict(color="#FF8800", width=2.5),
    name="Avg. Cumulative Efficiency",
), row=1, col=2)

# ---------------------------------------------------------------------------
# 4. Build frames (one per date)
# ---------------------------------------------------------------------------
def date_annotation(ds):
    return {
        "text": f"<b>Cutoff date: {ds}</b>",
        "x": 0.5, "y": -0.22,
        "xref": "paper", "yref": "paper",
        "showarrow": False,
        "font": {"size": 14, "color": "white"},
    }


frames = []
for date in dates:
    gray, red, line = get_frame_data(date)
    ds = pd.Timestamp(date).strftime("%Y-%m-%d")
    frames.append(go.Frame(
        data=[
            go.Scatter(
                x=gray["cumulative_efficiency"].tolist(),
                y=gray["cumulative_coverage"].tolist(),
                text=gray["station_id"].astype(str).tolist(),
            ),
            go.Scatter(
                x=red["cumulative_efficiency"].tolist(),
                y=red["cumulative_coverage"].tolist(),
                text=red["station_id"].astype(str).tolist(),
            ),
            go.Scatter(
                x=line["trip_date"].tolist(),
                y=line["avg_cumulative_coverage"].tolist(),
            ),
            go.Scatter(
                x=line["trip_date"].tolist(),
                y=line["avg_cumulative_efficiency"].tolist(),
            ),
        ],
        layout=go.Layout(annotations=subplot_annotations + [date_annotation(ds)]),
        name=ds,
        traces=[0, 1, 2, 3],
    ))

fig.frames = frames

# ---------------------------------------------------------------------------
# 5. Slider (starts at last date = full data) + Play/Pause buttons
# ---------------------------------------------------------------------------
slider_steps = []
for i, ds in enumerate(date_strs):
    is_labeled = (i == 0) or (i % 10 == 0) or (i == len(date_strs) - 1)
    short_label = pd.Timestamp(ds).strftime("%b %d") if is_labeled else ""
    slider_steps.append({
        "args": [[ds], {
            "frame": {"duration": 0, "redraw": True},
            "mode": "immediate",
            "transition": {"duration": 0},
        }],
        "label": short_label,
        "method": "animate",
    })

sliders = [{
    "active": len(dates) - 1,
    "steps": slider_steps,
    "x": 0.12,        # leave room for Play button on the left
    "len": 0.83,
    "y": 0.0,
    "pad": {"t": 50, "b": 10},
    "currentvalue": {"visible": False},
    "font": {"color": "#aaaaaa", "size": 9},
    "bgcolor": "#2a2a3e",
    "bordercolor": "#555",
    "tickcolor": "#888",
}]

# Play / Pause buttons — Play always replays from the first frame
updatemenus = [{
    "type": "buttons",
    "showactive": False,
    "x": 0.02,
    "y": 0.01,
    "xanchor": "left",
    "yanchor": "top",
    "bgcolor": "#2a2a3e",
    "bordercolor": "#555",
    "font": {"color": "white", "size": 12},
    "buttons": [
        {
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 400, "redraw": True},
                "fromcurrent": True,    # play from current slider position
                "mode": "immediate",
                "transition": {"duration": 300, "easing": "linear"},
            }],
        },
        {
            "label": "⏸ Pause",
            "method": "animate",
            "args": [[None], {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            }],
        },
    ],
}]

# ---------------------------------------------------------------------------
# 6. Layout
# ---------------------------------------------------------------------------
fig.update_layout(
    template="plotly_dark",
    sliders=sliders,
    updatemenus=updatemenus,
    height=620,
    title=dict(
        text="<b>Post-Rebalancing Station KPI Dashboard</b>",
        x=0.5,
        font=dict(size=20, color="white"),
    ),
    paper_bgcolor="#0f0f1a",
    plot_bgcolor="#0f0f1a",
    font=dict(color="white"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
    margin=dict(t=110, b=130, l=60, r=40),
    annotations=subplot_annotations + [date_annotation(date_strs[-1])],
)

# Fixed 0–1 axes on scatter
fig.update_xaxes(
    range=[0.0, 1.0],
    title_text="Avg. Cumulative Efficiency",
    gridcolor="#2a2a3e",
    zeroline=False,
    row=1, col=1,
)
fig.update_yaxes(
    range=[0.0, 1.0],
    title_text="Avg. Cumulative Coverage",
    gridcolor="#2a2a3e",
    zeroline=False,
    row=1, col=1,
)

# Fixed 0–1 y-axis on line chart; x-axis is dates
fig.update_xaxes(
    title_text="Date",
    gridcolor="#2a2a3e",
    zeroline=False,
    tickformat="%b %d",
    row=1, col=2,
)
fig.update_yaxes(
    range=[0.0, 1.0],
    title_text="Value",
    gridcolor="#2a2a3e",
    zeroline=False,
    row=1, col=2,
)

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
os.makedirs("reports/figures", exist_ok=True)
html_path = "reports/figures/dashboard.html"
fig.write_html(html_path, full_html=True, include_plotlyjs="cdn")
print(f"Dashboard saved → {html_path}")
print(f"Date range : {date_strs[0]}  →  {date_strs[-1]}")
print(f"Stations   : {station_kpi['station_id'].nunique()}")
print(f"File size  : {os.path.getsize(html_path) / 1e6:.1f} MB")
