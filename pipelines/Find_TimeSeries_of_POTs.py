"""
Utilities for extracting ±window time series around POT events.

This module supports two cases:

1. Hourly center variable:
   - center variable is hourly
   - associated variable 1 is hourly
   - associated variable 2 is daily and is mapped to each hour of the day

2. Daily center variable:
   - center variable is daily and is expanded to hourly resolution
   - associated variables 1 and 2 are hourly

The lag columns report the time difference, in hours, between the POT event time
and the time of the maximum associated-variable value within the selected window.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def _read_timeseries(
    path: PathLike,
    time_col: str,
    value_col: str,
    *,
    daily: bool = False,
) -> pd.DataFrame:
    """Read a two-column time series and standardize its timestamp column."""
    df = pd.read_csv(path)

    required = {time_col, value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required column(s): {sorted(missing)}"
        )

    df = df[[time_col, value_col]].copy()
    df = df.rename(columns={time_col: "time"})

    if daily:
        # Normalize daily observations to midnight regardless of whether the
        # input is 'MM/DD/YYYY', ISO format, or already includes a time.
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.normalize()
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Duplicate timestamps can create many-to-many merges. Fail loudly so the
    # source data can be checked rather than silently multiplying rows.
    if df["time"].duplicated().any():
        duplicated = int(df["time"].duplicated().sum())
        raise ValueError(
            f"{path}: found {duplicated} duplicate timestamp(s) in '{time_col}'."
        )

    return df


def _read_pot_events(path: PathLike, pot_time_col: str) -> pd.DataFrame:
    """Read POT-event timestamps and standardize them to a column named 'time'."""
    df = pd.read_csv(path)

    if pot_time_col not in df.columns:
        raise ValueError(
            f"{path}: POT time column '{pot_time_col}' was not found."
        )

    df = df.copy()
    df["time"] = pd.to_datetime(df[pot_time_col], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{path}: no valid POT timestamps were found.")

    return df


def expand_daily_to_hourly(
    df_daily: pd.DataFrame,
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Expand daily values to hourly resolution.

    Each daily value is repeated for 24 timestamps from 00:00 through 23:00.

    Parameters
    ----------
    df_daily
        DataFrame containing one value per day.
    time_col
        Name of the daily date/time column.
    value_col
        Name of the value column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per hour.
    """
    required = {time_col, value_col}
    missing = required.difference(df_daily.columns)
    if missing:
        raise ValueError(
            f"Daily DataFrame is missing required column(s): {sorted(missing)}"
        )

    work = df_daily[[time_col, value_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce").dt.normalize()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[time_col])

    pieces = []
    for t, value in work[[time_col, value_col]].itertuples(index=False, name=None):
        hours = pd.date_range(
            start=t,
            periods=24,
            freq="h",
        )
        pieces.append(
            pd.DataFrame(
                {
                    time_col: hours,
                    value_col: value,
                }
            )
        )

    if not pieces:
        return pd.DataFrame(columns=[time_col, value_col])

    return pd.concat(pieces, ignore_index=True)


def _lag_to_maximum_hours(
    df: pd.DataFrame,
    value_col: str,
    event_time: pd.Timestamp,
) -> float:
    """
    Return hours from event_time to the maximum value in df.

    A negative lag means the maximum occurred before the POT event.
    A positive lag means it occurred after the POT event.
    """
    if df.empty or value_col not in df.columns:
        return np.nan

    sub = df[["time", value_col]].dropna(subset=["time", value_col])
    if sub.empty:
        return np.nan

    max_idx = sub[value_col].idxmax()
    max_time = sub.loc[max_idx, "time"]

    return float((max_time - event_time) / pd.Timedelta(hours=1))


def timeseries_of_pot_center_variable_hourly(
    pot_center_variable_path: PathLike,
    pot_time_col: str,
    original_dataset_center_variable_path: PathLike,
    original_dataset_center_variable_time_col: str,
    original_dataset_center_variable_col_name: str,
    original_dataset_associated_variable_1_path: PathLike,
    original_dataset_associated_variable_1_time_col: str,
    original_dataset_associated_variable_1_col_name: str,
    lag_hour_col_variable_1: str,
    original_dataset_associated_variable_2_path: PathLike,
    original_dataset_associated_variable_2_time_col: str,
    original_dataset_associated_variable_2_col_name: str,
    lag_hour_col_variable_2: str,
    output_path: PathLike,
    window_days: float = 5,
) -> pd.DataFrame:
    """
    Extract windows around POT events when the center variable is hourly.

    The center variable and associated variable 1 are expected to be hourly.
    Associated variable 2 is expected to be daily and is mapped to each hour
    of its corresponding day.

    Lag 1 is calculated from the maximum of associated variable 1.
    Lag 2 is calculated from the maximum DAILY value of associated variable 2.
    """
    pot_df = _read_pot_events(pot_center_variable_path, pot_time_col)

    center_df = _read_timeseries(
        original_dataset_center_variable_path,
        original_dataset_center_variable_time_col,
        original_dataset_center_variable_col_name,
        daily=False,
    )

    assoc1_df = _read_timeseries(
        original_dataset_associated_variable_1_path,
        original_dataset_associated_variable_1_time_col,
        original_dataset_associated_variable_1_col_name,
        daily=False,
    )

    assoc2_daily_df = _read_timeseries(
        original_dataset_associated_variable_2_path,
        original_dataset_associated_variable_2_time_col,
        original_dataset_associated_variable_2_col_name,
        daily=True,
    )

    # Merge the two hourly variables.
    hourly_base = center_df.merge(
        assoc1_df,
        on="time",
        how="inner",
        validate="one_to_one",
    ).sort_values("time")

    # Map each daily associated-variable-2 value to all hourly timestamps
    # belonging to that date.
    hourly_base["date"] = hourly_base["time"].dt.normalize()

    assoc2_daily_map = assoc2_daily_df[
        ["time", original_dataset_associated_variable_2_col_name]
    ].rename(columns={"time": "date"})

    hourly_base = hourly_base.merge(
        assoc2_daily_map,
        on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")

    windows = []
    delta = pd.Timedelta(days=window_days)

    for event_id, row in pot_df.iterrows():
        event_time = row["time"]
        window_start = event_time - delta
        window_end = event_time + delta

        window_df = hourly_base.loc[
            hourly_base["time"].between(window_start, window_end)
        ].copy()

        if window_df.empty:
            continue

        # Associated variable 1 is hourly.
        lag_1 = _lag_to_maximum_hours(
            assoc1_df.loc[
                assoc1_df["time"].between(window_start, window_end)
            ],
            original_dataset_associated_variable_1_col_name,
            event_time,
        )

        # Associated variable 2 is daily. Calculate its lag from the original
        # daily timestamps rather than the repeated hourly representation.
        lag_2 = _lag_to_maximum_hours(
            assoc2_daily_df.loc[
                assoc2_daily_df["time"].between(window_start, window_end)
            ],
            original_dataset_associated_variable_2_col_name,
            event_time,
        )

        window_df["event_id"] = event_id
        window_df["event_time"] = event_time
        window_df[lag_hour_col_variable_1] = lag_1
        window_df[lag_hour_col_variable_2] = lag_2

        windows.append(window_df)

    if not windows:
        result = pd.DataFrame(
            columns=[
                "time",
                original_dataset_center_variable_col_name,
                original_dataset_associated_variable_1_col_name,
                original_dataset_associated_variable_2_col_name,
                "event_id",
                "event_time",
                lag_hour_col_variable_1,
                lag_hour_col_variable_2,
            ]
        )
    else:
        result = (
            pd.concat(windows, ignore_index=True)
            .sort_values(["event_time", "time"])
            .reset_index(drop=True)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    return result


def timeseries_of_pot_center_variable_daily(
    pot_center_variable_path: PathLike,
    pot_time_col: str,
    original_dataset_center_variable_path: PathLike,
    original_dataset_center_variable_time_col: str,
    original_dataset_center_variable_col_name: str,
    original_dataset_associated_variable_1_path: PathLike,
    original_dataset_associated_variable_1_time_col: str,
    original_dataset_associated_variable_1_col_name: str,
    lag_hour_col_variable_1: str,
    original_dataset_associated_variable_2_path: PathLike,
    original_dataset_associated_variable_2_time_col: str,
    original_dataset_associated_variable_2_col_name: str,
    lag_hour_col_variable_2: str,
    output_path: PathLike,
    window_days: float = 5,
) -> pd.DataFrame:
    """
    Extract windows around POT events when the center variable is daily.

    The center variable is expanded to hourly resolution. Associated variables
    1 and 2 are expected to be hourly.

    Lag 1 is calculated from the maximum of associated variable 1.
    Lag 2 is calculated from the maximum of associated variable 2.
    """
    pot_df = _read_pot_events(pot_center_variable_path, pot_time_col)

    center_daily_df = _read_timeseries(
        original_dataset_center_variable_path,
        original_dataset_center_variable_time_col,
        original_dataset_center_variable_col_name,
        daily=True,
    )

    assoc1_df = _read_timeseries(
        original_dataset_associated_variable_1_path,
        original_dataset_associated_variable_1_time_col,
        original_dataset_associated_variable_1_col_name,
        daily=False,
    )

    assoc2_df = _read_timeseries(
        original_dataset_associated_variable_2_path,
        original_dataset_associated_variable_2_time_col,
        original_dataset_associated_variable_2_col_name,
        daily=False,
    )

    center_hourly_df = expand_daily_to_hourly(
        center_daily_df,
        time_col="time",
        value_col=original_dataset_center_variable_col_name,
    )

    # Build one consistent hourly table once, instead of repeatedly merging
    # inside every event loop.
    hourly_base = (
        center_hourly_df
        .merge(
            assoc1_df,
            on="time",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            assoc2_df,
            on="time",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    windows = []
    delta = pd.Timedelta(days=window_days)

    for event_id, row in pot_df.iterrows():
        event_time = row["time"]
        window_start = event_time - delta
        window_end = event_time + delta

        window_df = hourly_base.loc[
            hourly_base["time"].between(window_start, window_end)
        ].copy()

        if window_df.empty:
            continue

        lag_1 = _lag_to_maximum_hours(
            assoc1_df.loc[
                assoc1_df["time"].between(window_start, window_end)
            ],
            original_dataset_associated_variable_1_col_name,
            event_time,
        )

        lag_2 = _lag_to_maximum_hours(
            assoc2_df.loc[
                assoc2_df["time"].between(window_start, window_end)
            ],
            original_dataset_associated_variable_2_col_name,
            event_time,
        )

        window_df["event_id"] = event_id
        window_df["event_time"] = event_time
        window_df[lag_hour_col_variable_1] = lag_1
        window_df[lag_hour_col_variable_2] = lag_2

        windows.append(window_df)

    if not windows:
        result = pd.DataFrame(
            columns=[
                "time",
                original_dataset_center_variable_col_name,
                original_dataset_associated_variable_1_col_name,
                original_dataset_associated_variable_2_col_name,
                "event_id",
                "event_time",
                lag_hour_col_variable_1,
                lag_hour_col_variable_2,
            ]
        )
    else:
        result = (
            pd.concat(windows, ignore_index=True)
            .sort_values(["event_time", "time"])
            .reset_index(drop=True)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    return result


if __name__ == "__main__":
    pass
