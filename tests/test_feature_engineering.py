# tests/test_feature_engineering.py

import pandas as pd
import pytest
from src.feature_engineering import add_time_features, add_rolling_features, add_daily_aggregates

def test_add_time_features():
    # Create a small DataFrame with a DateTimeIndex
    idx = pd.date_range('2023-01-01', periods=4, freq='h')
    df = pd.DataFrame({'MT_001': [10, 20, 30, 40]}, index=idx)

    df_new = add_time_features(df)
    assert 'hour' in df_new.columns
    assert 'dayofweek' in df_new.columns
    assert 'month' in df_new.columns

def test_add_rolling_features():
    idx = pd.date_range('2023-01-01', periods=4, freq='h')
    df = pd.DataFrame({'MT_001': [10, 20, 30, 40]}, index=idx)

    df_new = add_rolling_features(df, window='2H')
    assert 'MT_001_rollmean' in df_new.columns
    assert 'MT_001_rollstd' in df_new.columns

def test_add_daily_aggregates():
    idx = pd.date_range('2023-01-01', periods=4, freq='6h')
    df = pd.DataFrame({'MT_001': [10, 20, 30, 40]}, index=idx)

    df_new = add_daily_aggregates(df)
    assert 'MT_001_daily_cumsum' in df_new.columns
