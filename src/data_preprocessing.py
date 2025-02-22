# src/data_preprocessing.py

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

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


def find_gaps_in_column(series):
    """
    Find intervals of NaNs in a single time-series (Pandas Series),
    considering only the range between the column's first and last valid data.
    
    Returns a list of (gap_start, gap_end) tuples.
    If the entire series is NaN, returns one interval covering the entire index.
    If there are no gaps, returns an empty list.
    """
    # Identify earliest and latest valid timestamps
    start = series.first_valid_index()
    end = series.last_valid_index()

    # If the entire column is NaN, we define one big "gap" over the entire index
    if start is None or end is None:
        if len(series) == 0:
            return []  # Empty series
        return [(series.index[0], series.index[-1])]  # Entirely NaN

    # Slice only between the column's local min and max valid dates
    sub = series.loc[start:end]

    # Create a mask for NaNs
    mask = sub.isna()

    # Use .diff() on the integer representation to find transitions
    transitions = mask.astype(int).diff()

    # Gap starts where diff == 1, ends where diff == -1
    gap_starts = sub.index[transitions == 1]
    gap_ends   = sub.index[transitions == -1]

    # If the sub starts with NaN
    if len(sub) > 0 and mask.iloc[0]:
        gap_starts = pd.Index([start]).append(gap_starts)
    # If the sub ends with NaN
    if len(sub) > 0 and mask.iloc[-1]:
        gap_ends = gap_ends.append(pd.Index([end]))

    # Pair up each gap_start with the corresponding gap_end
    intervals = list(zip(gap_starts, gap_ends))
    return intervals


def detect_and_plot_gaps(df, random_cols=5, downsample=10):
    """
    1. Randomly selects `random_cols` clients (columns).
    2. Finds the NaN gap intervals for each chosen client.
    3. For each client, creates a row of subplots – one for each gap.
       The total number of columns equals the maximum number of gaps found among the clients.
    4. In each subplot, a zoomed-in view is plotted with 2 days before the gap start
       and 2 days after the gap end.
    5. Downsamples the time series by `downsample` factor to speed up plotting.
    
    :param df: A DataFrame where each column is a client (meter),
               and the index is a DateTimeIndex.
    :param random_cols: Number of columns to sample for inspection.
    :param downsample: Plot only every Nth data point to prevent huge slowdowns.
    """
    # -----------------------------
    # 1) Randomly pick the columns
    # -----------------------------
    all_cols = df.columns.tolist()
    if random_cols > len(all_cols):
        random_cols = len(all_cols)
    chosen_cols = random.sample(all_cols, random_cols)

    # -----------------------------
    # 2) Find gaps for each column
    # -----------------------------
    # Or inline your find_gaps_in_column function
    col_gap_intervals = {}
    for col in chosen_cols:
        intervals = find_gaps_in_column(df[col])
        col_gap_intervals[col] = intervals

    # -----------------------------
    # 3) Determine subplot layout
    # -----------------------------
    n_max = max(len(intervals) for intervals in col_gap_intervals.values())
    if n_max == 0:
        print("no gaps detected for this group")
        return None

    fig, axes = plt.subplots(random_cols, n_max, figsize=(5 * n_max, 3 * random_cols))
    if random_cols == 1:
        axes = np.atleast_2d(axes)  # ensure 2D if only one row
    if n_max == 1:
        axes = np.expand_dims(axes, axis=1)  # ensure 2D if only one column

    # -----------------------------
    # 4) Plot each gap in a subplot
    # -----------------------------
    for row_idx, col in enumerate(chosen_cols):
        series = df[col]
        intervals = col_gap_intervals[col]
        
        for col_idx in range(n_max):
            ax = axes[row_idx, col_idx]
            if col_idx < len(intervals):
                gap_start, gap_end = intervals[col_idx]
                
                # Define zoom window: 2 days before gap_start and 2 days after gap_end
                start_zoom = gap_start - pd.Timedelta(days=2)
                end_zoom = gap_end + pd.Timedelta(days=2)
                
                # Slice the time series
                sub = series.loc[start_zoom:end_zoom]

                # =========== DOWNSAMPLING =========== 
                # Plot every 'downsample' point to avoid huge slowdowns
                sub = sub.iloc[::downsample]

                ax.plot(sub.index, sub, label=col, color='blue')
                ax.axvspan(gap_start, gap_end, color='red', alpha=0.3)
                ax.set_title(f"{col} - Gap {col_idx+1}")
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
            else:
                # Hide unused subplot
                ax.axis('off')

    plt.tight_layout()
    plt.show()


def verify_no_internal_gaps(df):
    """
    Checks if any column has NaNs between its first and last valid data point.
    Prints columns with gaps, or a success message if none.
    """# or inline that code
    
    columns_with_gaps = []
    for col in df.columns:
        intervals = find_gaps_in_column(df[col])
        if len(intervals) > 0:
            columns_with_gaps.append(col)

    if len(columns_with_gaps) == 0:
        print("No internal gaps detected for any client!")
    else:
        print("Columns with internal gaps:")
        for c in columns_with_gaps:
            print("  ", c)

