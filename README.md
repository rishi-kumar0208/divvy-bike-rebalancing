# divvy-bike-rebalancing

Forecasting bike demand and optimizing station rebalancing using time-series modeling and network flow optimization on the Divvy bike sharing dataset.

---

## What This Project Does

Divvy is a bike-sharing system in Chicago. At any given moment, each station has some number of bikes available to rent. If a station runs out of bikes, riders can't rent. If a station fills up completely, riders can't return bikes. To prevent this, operations teams physically move bikes between stations overnight, a process called **rebalancing**.

This project builds a system that:
1. Infers a feasible range of valid starting bike inventory for each station on each day, directly from trip data
2. Trains a model to predict the midpoint of that range
3. Uses those predictions to solve an overnight rebalancing problem, deciding how many bikes to move between which stations to keep the system balanced

---

## The Core Problem: Finding the Right Starting Inventory

Each station has a known dock capacity (`dpcapacity_start`, `dpcapacity_end`), the maximum number of bikes it can hold. What the data does **not** tell us directly is how many bikes are actually occupying those docks at any given moment during the day.

The way to think about it is this: each station has an **inventory curve** for each day. It starts at some unknown value at the beginning of the day, then moves up and down as bikes arrive and depart throughout the day. The shape of that curve (how it moves) is fully determined by the trips we observe in the data.

The only thing we don't know is where the curve **starts**, the y-intercept.

But we can constrain it. There exists a range of valid starting values `[L, U]` such that if the curve starts anywhere in that range, it never dips below 0 or rises above `dpcapacity` at any point during the day. Start too low and the curve goes negative at some point, which is physically impossible. Start too high and it overflows capacity, which is also impossible.

Our goal is to **predict that starting value correctly**, and use those predictions to set each station's inventory to a valid starting point before the day begins through overnight rebalancing.

---

## Dataset

The raw data is the Divvy trip history dataset stored as `divvy.parquet`. Each row is one trip with the following key columns:

| Column | Description |
|--------|-------------|
| `from_station_id`, `starttime` | Which station the bike left from and when |
| `to_station_id`, `stoptime` | Which station the bike arrived at and when |
| `tripduration` | How long the trip took |
| `temperature`, `events` | Weather conditions at the time |
| `usertype`, `gender` | Type of user |
| `dpcapacity_start`, `dpcapacity_end` | Number of docks at the start and end station |
| `latitude_start`, `longitude_start` | Coordinates of the departure station |
| `latitude_end`, `longitude_end` | Coordinates of the arrival station |

---

## Pipeline

The full pipeline runs in 7 stages across 4 notebooks.

---

### Stage 1: Building the Station×Day Calendar
*`notebooks/02_feature_engineering.ipynb`*

Before inferring anything about inventory, we need a complete grid of every station and every day we care about, including days where a station had zero trips. Without this, stations with quiet days would simply have no rows, and we'd have no way to make predictions for those days.

The naive approach would be to take the earliest date in the whole dataset and give every station a row for every day from that date onward. The problem is that some stations didn't exist at the start of the dataset. Giving a station rows from before it opened would create hundreds of fake zero-trip days that never actually happened, which would corrupt the rolling averages and lag features built later.

Instead we use **hybrid progressive densification**:
- For each station, find its own first observed date, either as a `from_station_id` in `starttime` or as a `to_station_id` in `stoptime`, whichever comes first
- Give that station a row for every day from its own first date up to the dataset-wide maximum date

The result is a complete grid with no gaps, where stations only have rows from the day they first appeared.

---

### Stage 2: Computing Cumulative Net Flow
*`notebooks/02_feature_engineering.ipynb`*

To determine the feasible range of starting inventory for a station on a given day, we need to reconstruct how the inventory curve moved throughout that day. We do this by computing the cumulative net flow from trip records in three steps:

**Step 1 - Hourly bucketing:**
All trips are grouped into hourly buckets using `EXTRACT(hour FROM starttime)` and `EXTRACT(hour FROM stoptime)`. This gives us up to 24 slots per station per day, from hour 0 through hour 23.

**Step 2 - Hourly net flow:**
For each hour we compute:
- `trips_arrived`: how many bikes arrived at that station in that hour
- `trips_departed`: how many bikes left from that station in that hour
- `hourly_net_flow = trips_arrived - trips_departed`

**Step 3 - Cumulative sum:**
We compute the running total of `hourly_net_flow` across the 24 hours of the day in order. This gives us the shape of the inventory curve relative to wherever it started, showing how much it moved up or down from its starting point by each hour.

From that cumulative sum we extract:
- `min_cumulative_flow`: the lowest point the curve reached relative to its start
- `max_cumulative_flow`: the highest point the curve reached relative to its start

Days with zero trips get `min_cumulative_flow = 0` and `max_cumulative_flow = 0`, since the curve never moved.

---

### Stage 3: Deriving Inventory Bounds [L, U]
*`notebooks/02_feature_engineering.ipynb`*

Now we use `min_cumulative_flow` and `max_cumulative_flow` to determine the range of valid starting values for the inventory curve. The constraint is that the curve can never go below 0 or above `station_capacity_day` at any point during the day.

**Lower bound L:**
The curve drops by at most `-min_cumulative_flow` from its starting point during the day. If the starting inventory were any lower than this, the curve would go negative at some point, which is physically impossible. So the starting inventory must be at least `-min_cumulative_flow`.

```
L = clip(-min_cumulative_flow, 0, station_capacity_day)
```

**Upper bound U:**
The curve rises by at most `max_cumulative_flow` from its starting point during the day. If the starting inventory were any higher than `station_capacity_day - max_cumulative_flow`, the curve would exceed capacity at some point, which is also impossible. So the starting inventory must be at most `station_capacity_day - max_cumulative_flow`.

```
U = clip(station_capacity_day - max_cumulative_flow, 0, station_capacity_day)
```

**Inversion repair:**
Occasionally after this computation `U < L`, which is a logical contradiction. When that happens we fall back to `L = 0` and `U = station_capacity_day`, treating the full capacity range as feasible rather than discarding the row.

**Prediction target:**
```
s_true = (L + U) / 2
```
The midpoint of the feasible interval is used as the model's prediction target. A prediction is considered **covered** if the rounded predicted value `s_hat_r` falls within the true interval `[L, U]`.

---

### Stage 4: NA Handling and Forward Fills
*`notebooks/02_feature_engineering.ipynb`*

After merging all features onto the station×day calendar, many rows will have NaN values, particularly for days where a station had zero trips. These are filled deliberately rather than dropped, because a day with no trips is real information, not missing data.

**`station_capacity_day` - forward fill within station:**
Capacity is only recorded on days where at least one trip was observed. On zero-trip days it is NaN. We forward fill within each station group; a station's dock count is assumed to stay the same until we observe it change. Physical dock expansions are rare, so this is a safe assumption.

**`min_cumulative_flow` and `max_cumulative_flow` - fill with 0:**
On a zero-trip day the inventory curve never moved. Filling with 0 means `L = 0` and `U = station_capacity_day`. On a day with no trips we have no information to narrow the range, so the full capacity range is treated as feasible.

**`temperature` and `events` - forward fill within station:**
Weather is only recorded on days where trips happened. On zero-trip days these are NaN. We forward fill within each station group; the last observed temperature or weather event is carried forward until a new reading appears. Weather doesn't suddenly become unknown just because nobody rode a bike that day.

**Rolling 7-day averages - 3-tier fallback:**
For `temperature_roll7`, `min_start_inventory_roll7`, and `max_start_inventory_roll7`, if still NaN after the primary rolling calculation:
1. **Station-level 7-day rolling mean** (primary). Uses only that station's own recent history.
2. **City-wide 7-day rolling mean by date** (fallback if station-level is NaN). Uses the average across all stations for that same date as a proxy for what a typical station looked like that week.
3. **Global dataset mean** (last resort if both above are still NaN). This is more reasonable than it might sound; all stations are within the same city, so temperatures across the dataset are geographically close enough that a global mean is a genuinely informative fallback.

**`trips_departed_roll7` and `trips_arrived_roll7` - fill with 0:**
A station with no trips in the past 7 days has a rolling average of 0. Zero activity is the correct value, not missing data.

**Lag features - drop first day per station:**
After shifting each feature column by 1 day within each station group, the very first day of each station's history has no prior day to reference, so all `_prev` columns are NaN for that row. Rather than filling these with anything, those rows are dropped entirely. We cannot make a meaningful prediction for a station's first day since we have no prior context. This results in losing exactly one row per station, which is negligible.

---

### Stage 5: Feature Engineering
*`notebooks/02_feature_engineering.ipynb`*

All features are built from information that would have been available the day before the prediction date. We never use same-day or future information. This is enforced by the `_prev` suffix (yesterday's value) and rolling averages that end the day before.

**Lag features (`_prev`) - yesterday's values:**
- `min_start_inventory_prev`, `max_start_inventory_prev`: yesterday's `L` and `U`
- `station_capacity_day_prev`: yesterday's dock capacity
- `temperature_prev`, `events_prev`: yesterday's weather
- `trips_departed_prev`, `trips_arrived_prev`: yesterday's trip counts

**Rolling 7-day averages (`_roll7`) - recent trend:**
- `trips_departed_roll7`, `trips_arrived_roll7`: average daily trip activity over the past week
- `temperature_roll7`: average temperature over the past week
- `min_start_inventory_roll7`, `max_start_inventory_roll7`: average bounds over the past week

**Station identity:**
- `station_id` as a categorical feature, which lets the model learn station-specific patterns without manual encoding

---

### Stage 6: Model Training and Evaluation
*`notebooks/03_modeling.ipynb`*

**Train/test split:**
- Train: all rows where `trip_date < 2017-10-01`
- Test: all rows where `trip_date >= 2017-10-01`

This is a temporal split; the model is only ever trained on the past and evaluated on the future, reflecting real deployment conditions.

After splitting, the training set is further restricted to only stations that also appear in the test set. This serves two purposes: it avoids training on stations the model will never need to predict, and it prevents categorical encoding issues in LightGBM when a `station_id` appears in train but not test.

**Model:**
A single LightGBM regressor is trained to predict `s_true`, the midpoint of `[L, U]`, for each station×day using the 13 features from Stage 5. LightGBM handles categorical features like `station_id` and `events_prev` natively without one-hot encoding.

**Evaluation:**
The primary metric is **coverage** - what percentage of rounded predictions `s_hat_r` land inside the true interval `[L, U]`. A prediction that is numerically close but outside the interval counts as a miss.

The secondary metric is **conditional efficiency**, which measures, for covered predictions only, how close the prediction was to the midpoint of the interval:
```
efficiency = 1 - |s_hat_r - s_true| / (U - L)
```
Uncovered rows are excluded from this calculation entirely rather than penalized as zero, since efficiency is only meaningful when the prediction is already feasible.

RMSE is also computed but treated as tertiary. A low RMSE does not guarantee coverage, and a prediction inside a wide interval may have higher RMSE but still be operationally correct.

---

### Stage 7: Rebalancing Optimization
*`notebooks/04_rebalancing_optimization.ipynb`*

**The problem:**
The model predicts the ideal starting inventory for each station on each day. The rebalancing step physically moves bikes between stations overnight to get each station as close as possible to its predicted starting value before the day begins.

**Fixed fleet assumption:**
The total number of bikes across all stations is held constant. Rebalancing redistributes bikes; it does not add or remove them from the system. If 3 bikes are added to station 35, 3 bikes must be removed from elsewhere.

**Step 1 - Fleet adjustment:**
For each day we take `s_hat_r` for every station and adjust the values so they sum to exactly the same total fleet size as the first test day. Since we are working with integer bike counts, this is done by randomly adding or removing single bikes from eligible stations until the total matches. Stations cannot go below 0 or above `station_capacity_day`.

**Step 2 - Min-cost flow:**
Now we know the target starting inventory `s_target` for every station tomorrow. We solve a **minimum cost flow** problem to find the cheapest way to move bikes between stations to achieve those targets overnight.

The network is built as follows:
- Each station is a node
- Each station is connected to its 8 nearest neighbors by Haversine distance computed from `latitude_start`, `longitude_start`, which are the edges bikes can travel along
- Edge weights are the distance in meters between stations. Moving bikes further costs more.
- Each station's supply or demand is `s_target_tomorrow - s_target_today`

The solver finds the set of bike movements that satisfies all supply and demand constraints at minimum total distance traveled.

**Evaluation:**
The same coverage and conditional efficiency metrics from Stage 6 are recomputed, but now using `s_target` instead of `s_hat_r`. This tells us whether the rebalanced inventory actually lands inside `[L, U]`, i.e. whether the inventory curve will stay between 0 and capacity throughout the day after rebalancing.

---

## Results

### Model performance (test set: Oct - Nov 2017)

| Metric | Pre-rebalancing (LightGBM) | Post-rebalancing (min-cost flow) |
|--------|---------------------------|----------------------------------|
| Coverage rate | 94.02% | 93.92% |
| Mean conditional efficiency | 90.84% | 90.15% |
| RMSE | 2.015 bikes | N/A |

### Interactive dashboard

An interactive station KPI dashboard is live at:

[**→ View dashboard**](https://rishi-kumar0208.github.io/divvy-bike-rebalancing/reports/figures/dashboard.html)

It shows:
- **Left:** scatter plot of per-station cumulative coverage vs efficiency, with the 10 lowest-coverage stations highlighted in red
- **Right:** daily trend lines for average cumulative coverage and efficiency
- **Slider:** drag to filter both charts to any cutoff date between 2017-10-01 and 2017-11-30

To regenerate the dashboard after a new pipeline run:

```bash
python build_dashboard.py
```

---

## How to Run

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place `divvy.parquet` in `data/raw/`
4. Run notebooks in order: `01_eda.ipynb` -> `02_feature_engineering.ipynb` -> `03_modeling.ipynb` -> `04_rebalancing_optimization.ipynb`
5. Each notebook will prompt you for the path to the parquet file at runtime. Press Enter to use the default `data/raw/divvy.parquet` or type a custom path.
