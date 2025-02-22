# src/feature_engineering.py

import pandas as pd

def add_time_features(df):
    """
    Adds basic time-based features to a DataFrame with a DateTimeIndex:
      - hour (0-23)
      - dayofweek (0=Monday, 6=Sunday)
      - month (1-12)
    Returns a new DataFrame with added columns.
    """
    df_with_time = df.copy()
    df_with_time['hour'] = df_with_time.index.hour
    df_with_time['dayofweek'] = df_with_time.index.dayofweek
    df_with_time['month'] = df_with_time.index.month
    return df_with_time

def add_rolling_features(df, window='24h'):
    """
    Adds rolling mean and rolling std for each meter column over a specified time window.
    E.g., window='24h' calculates a 24-hour rolling window if the index is properly spaced.
    Returns a new DataFrame with added columns named like '<col>_rollmean' and '<col>_rollstd'.
    """
    df_rolling = df.copy()
    
    # Identify numeric meter columns, excluding time-based features
    meter_cols = df_rolling.select_dtypes(include='number').columns.difference(['hour','dayofweek','month'])

    # We'll build a dictionary of new rolling columns to avoid repeated insertions
    rolling_data = {}

    for col in meter_cols:
        rollmean = df_rolling[col].rolling(window=window).mean()
        rollstd  = df_rolling[col].rolling(window=window).std()
        
        rolling_data[f'{col}_rollmean'] = rollmean
        rolling_data[f'{col}_rollstd']  = rollstd

    # Concat the new rolling columns in one go
    rolling_df = pd.DataFrame(rolling_data, index=df_rolling.index)
    df_rolling = pd.concat([df_rolling, rolling_df], axis=1)

    return df_rolling

def add_daily_aggregates(df):
    """
    Adds a daily cumulative sum column for each meter, named '<col>_daily_cumsum'.
    Returns a new DataFrame with added columns.
    """
    df_daily = df.copy()
    
    meter_cols = df_daily.select_dtypes(include='number').columns.difference(['hour','dayofweek','month'])

    # We'll store new columns in a dict, then concat
    daily_data = {}

    for col in meter_cols:
        # cumulative sum per day
        daily_cumsum = df_daily[col].groupby(df_daily.index.date).cumsum()
        daily_data[f'{col}_daily_cumsum'] = daily_cumsum

    daily_df = pd.DataFrame(daily_data, index=df_daily.index)
    df_daily = pd.concat([df_daily, daily_df], axis=1)

    return df_daily
