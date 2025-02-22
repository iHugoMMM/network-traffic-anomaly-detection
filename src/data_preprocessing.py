# src/data_preprocessing.py

import pandas as pd
import numpy as np

def remove_outliers_below_threshold(df, threshold=1.0):
    """
    Replace values below a given threshold with NaN (or another placeholder)
    to treat them as outliers or invalid data.
    
    :param df: Pandas DataFrame of meter readings.
    :param threshold: Minimum valid consumption value.
    :return: DataFrame with values below threshold set to NaN.
    """
    df_clean = df.copy()
    df_clean[df_clean < threshold] = np.nan
    return df_clean


def resample_data(df, freq='30min', agg='mean'):
    """
    Resample time-series data to a specified frequency (e.g., 30-minute intervals)
    using a specified aggregation method (mean, sum, etc.).

    :param df: Pandas DataFrame with a DateTime index.
    :param freq: Resampling frequency (e.g., '30T' for 30 minutes).
    :param agg: Aggregation method, e.g. 'mean', 'sum', 'max'.
    :return: Resampled DataFrame.
    """
    if agg == 'mean':
        return df.resample(freq).mean()
    elif agg == 'sum':
        return df.resample(freq).sum()
    elif agg == 'max':
        return df.resample(freq).max()
    # Add more methods as needed
    else:
        raise ValueError(f"Aggregation method '{agg}' not supported.")


def compute_correlations(df):
    """
    Compute the correlation matrix (Pearson by default) for all columns.

    :param df: Pandas DataFrame.
    :return: Correlation matrix as a DataFrame.
    """
    return df.corr()


def get_top_correlations(corr_matrix, top_n=10):
    """
    Given a correlation matrix, find the top N correlated pairs
    (excluding the diagonal and lower triangle).

    :param corr_matrix: DataFrame of correlations (e.g., from df.corr()).
    :param top_n: Number of top correlation pairs to return.
    :return: Series of correlation pairs with highest absolute values.
    """
    # Work with absolute correlation values
    corr_abs = corr_matrix.abs()

    # Mask the upper triangle & diagonal to avoid duplicates
    mask = np.triu(np.ones_like(corr_abs, dtype=bool))
    tri_df = corr_abs.mask(mask)

    # Flatten, drop NaNs, sort by correlation descending
    sorted_corr = tri_df.unstack().dropna().sort_values(ascending=False)
    
    # Return the top N pairs
    return sorted_corr.head(top_n)


def fill_nans_with_previous_day(df, periods_per_day=48):
    """
    Fill NaN values by using the value from the same time on the previous day.
    This is a single-pass approach; if the previous day's value is also NaN,
    the gap remains unfilled.

    :param df: DataFrame with a DateTimeIndex at a consistent frequency (e.g., 30min).
    :param periods_per_day: Number of rows per day (48 for 30-min freq).
    :return: A new DataFrame with NaNs filled where possible using day-lag.
    """
    df_filled = df.copy()

    # Shift the entire DataFrame by 1 day (48 intervals for 30-min data)
    df_shifted = df.shift(periods_per_day)

    # Where df_filled is NaN but df_shifted is not, fill with the shifted value
    mask = df_filled.isna() & df_shifted.notna()
    df_filled[mask] = df_shifted[mask]

    return df_filled


def fill_nans_day_before_after_interpolate(df, periods_per_day=48):
    """
    Fill NaN values in a DataFrame by:
      1) Using the same time slot from the previous day.
      2) Using the same time slot from the next day.
      3) Time-based interpolation (linear).

    :param df: DataFrame with a DateTimeIndex at a consistent frequency (e.g., 30min).
    :param periods_per_day: Number of rows per day (48 for 30-min freq).
    :return: A new DataFrame with NaNs filled wherever possible.
    """
    # 1) Day-Before Fill
    df_filled = df.copy()
    df_shifted_before = df.shift(periods_per_day)
    mask_before = df_filled.isna() & df_shifted_before.notna()
    df_filled[mask_before] = df_shifted_before[mask_before]

    # 2) Day-After Fill
    df_shifted_after = df.shift(-periods_per_day)
    mask_after = df_filled.isna() & df_shifted_after.notna()
    df_filled[mask_after] = df_shifted_after[mask_after]

    # 3) Time-Based Interpolation
    #    This fills any remaining NaNs linearly along the time index.
    df_filled = df_filled.interpolate(method='time')

    return df_filled