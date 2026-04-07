"""
Model training and evaluation functions for the Divvy bike rebalancing pipeline.

Functions
---------
train_lgbm(X_train, y_train, categorical_features)
    Train a LightGBM regressor to predict s_true (the midpoint of [L, U]).
evaluate_coverage(df, pred_col)
    Compute per-row coverage and conditional efficiency metrics.
coverage_summary(df)
    Aggregate coverage and efficiency metrics into a summary report.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
               categorical_features: list) -> lgb.LGBMRegressor:
    """
    Train a LightGBM regressor predicting s_true (the midpoint of [L, U]).

    Exact hyperparameters (do not substitute defaults):
        n_estimators=250, learning_rate=0.08, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, random_state=42

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix (13 columns).
    y_train : pandas.Series
        Training target — s_true values.
    categorical_features : list of str
        Columns to treat as categoricals: ['station_id', 'events_prev'].

    Returns
    -------
    lightgbm.LGBMRegressor
        Fitted model.
    """
    model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_features)
    return model


def evaluate_coverage(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """
    Compute per-row coverage and conditional efficiency for a set of predictions.

    covered = 1 if round(pred) >= min_start_inventory and
                   round(pred) <= max_start_inventory else 0

    efficiency = 1 - |round(pred) - s_true| / (max_start_inventory - min_start_inventory)
        Only computed for covered rows; uncovered rows receive NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing min_start_inventory, max_start_inventory, s_true,
        and the prediction column.
    pred_col : str
        Name of the column holding raw (unrounded) predictions.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with 'covered' and 'efficiency' columns added.
    """
    df = df.copy()
    s_hat_r = df[pred_col].round()

    df['covered'] = (
        (s_hat_r >= df['min_start_inventory']) &
        (s_hat_r <= df['max_start_inventory'])
    ).astype(int)

    width = df['max_start_inventory'] - df['min_start_inventory']
    eff = 1 - (s_hat_r - df['s_true']).abs() / width
    df['efficiency'] = np.where((df['covered'] == 1) & (width > 0), eff, np.nan)

    return df


def coverage_summary(df: pd.DataFrame) -> dict:
    """
    Aggregate coverage and conditional efficiency into a summary report.

    RMSE is computed on the unrounded prediction column ('s_hat'), not on
    the rounded 's_hat_r'.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of evaluate_coverage — must contain 'covered', 'efficiency',
        's_true', and 's_hat' columns.

    Returns
    -------
    dict
        Keys: 'coverage_rate', 'mean_efficiency', 'rmse'.
    """
    coverage_rate   = df['covered'].mean()
    mean_efficiency = df.loc[df['covered'] == 1, 'efficiency'].mean()
    rmse = np.sqrt(mean_squared_error(df['s_true'], df['s_hat']))

    return {
        'coverage_rate':   round(float(coverage_rate), 4),
        'mean_efficiency': round(float(mean_efficiency), 4),
        'rmse':            round(float(rmse), 4),
    }
