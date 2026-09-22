import pandas as pd
import numpy as np
import glob
from datetime import timedelta
import os
from pathlib import Path

def POT_Hourly(center_variable_time, associated_hourly_variable_time, associated_daily_variable_time,
               center_variable_input_column, associated_hourly_variable_input_column, associated_daily_variable_input_column,
               center_variable_input_path, associated_hourly_variable_input_path, associated_daily_variable_input_path,
               center_variable_output_column, associated_hourly_variable_output_column, associated_daily_variable_output_column,
               number_of_events_per_year, thresholds_output_path, pot_output_path, stationary = True):
    
    """
    Extracts Peak-Over-Threshold (POT) events from an hourly center variable and
    identifies the associated maximum values of one hourly variable and one daily
    variable within a ±2.5-day event window.

    Parameters
    ----------
    center_variable_time : str
        Name of the time column in the center-variable dataset.

    associated_hourly_variable_time : str
        Name of the time column in the associated hourly-variable dataset.

    associated_daily_variable_time : str
        Name of the time column in the associated daily-variable dataset.

    center_variable_input_column : str
        Column name of the center variable used to define POT events
        (e.g., RF or NTR).

    associated_hourly_variable_input_column : str
        Column name of the associated hourly variable.

    associated_daily_variable_input_column : str
        Column name of the associated daily variable.

    center_variable_input_path : str
        Path to the center-variable input CSV file.

    associated_hourly_variable_input_path : str
        Path to the associated hourly-variable input CSV file.

    associated_daily_variable_input_path : str
        Path to the associated daily-variable input CSV file.

    center_variable_output_column : str
        Output column name for the extracted POT center-variable values
        (e.g., 'POT_RF' or 'POT_NTR').

    associated_hourly_variable_output_column : str
        Output column name for the associated hourly-variable maximum
        within the event window.

    associated_daily_variable_output_column : str
        Output column name for the associated daily-variable maximum
        within the event window.

    number_of_events_per_year : int
        Number of independent POT events selected per year.

    thresholds_output_path : str
        Path where the threshold summary CSV will be saved.

    pot_output_path : str
        Path where the final POT event table will be saved.

    stationary : bool, optional
        If True, one constant threshold is used for the entire record.
        If False, a separate threshold is estimated for each year.

    Returns
    -------
    merged_df : pandas.DataFrame
        Final POT dataset containing the selected center-variable events and the
        associated hourly and daily maxima.

    df_thresh : pandas.DataFrame
        Threshold information used for stationary or non-stationary POT selection.

    Notes
    -----
    The center variable defines the POT event times. Events are selected from the
    largest values of the center-variable time series while enforcing a declustering
    window of ±2.5 days to avoid selecting multiple peaks from the same event.

    For each selected POT event, the function searches within ±2.5 days around the
    event time and extracts:
    1. the maximum associated hourly-variable value, and
    2. the maximum associated daily-variable value.

    This function is appropriate when the center variable is hourly, such as NTR or RF.
    """
    
    df_center_variable = pd.read_csv(center_variable_input_path, parse_dates=[center_variable_time])
    df_center_variable['time'] = pd.to_datetime(df_center_variable[center_variable_time])
    df_center_variable = df_center_variable.set_index('time')
    
    df_associated_hourly_variable = pd.read_csv(associated_hourly_variable_input_path, parse_dates=[associated_hourly_variable_time])
    df_associated_hourly_variable['time'] = pd.to_datetime(df_associated_hourly_variable[associated_hourly_variable_time])
    df_associated_hourly_variable = df_associated_hourly_variable.set_index('time')
    
    df_associated_daily_variable = pd.read_csv(associated_daily_variable_input_path, parse_dates=[associated_daily_variable_time])    
    df_associated_daily_variable['time'] = pd.to_datetime(df_associated_daily_variable[associated_daily_variable_time])
    df_associated_daily_variable = df_associated_daily_variable.set_index('time')

    # ────────────────────────────────────────────────────────────────────────────────
    # Parameters
    # ────────────────────────────────────────────────────────────────────────────────
    center_variable_series = df_center_variable[center_variable_input_column].dropna()
    decluster_window = timedelta(days=2.5)
    
    # ────────────────────────────────────────────────────────────────────────────────
    # Storage
    # ────────────────────────────────────────────────────────────────────────────────
    all_pot = []
    thresholds = []

    # ────────────────────────────────────────────────────────────────────────────────
    # POT extraction + associated hourly-variable matching
    # ────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # Stationary threshold
    # ─────────────────────────────────────────────
    if stationary:

        # One constant threshold for the whole record
        center_variable_series_sorted = center_variable_series.sort_values(ascending=False)

        selected = []
        selected_values = []

        total_years = center_variable_series.index.year.nunique()
        total_events = number_of_events_per_year * total_years

        for idx, val in center_variable_series_sorted.items():

            # Use full timedelta comparison, not .days
            if any(abs(idx - sel) <= decluster_window for sel in selected):
                continue

            selected.append(idx)
            selected_values.append(val)

            if len(selected) == total_events:
                break

        if len(selected) < total_events:
            raise ValueError(
                "Not enough independent events found to meet the specified number of events per year."
            )

        threshold_value = min(selected_values)

        thresholds.append({
            "Threshold": threshold_value,
            "Number_of_Years": total_years,
            "Target_Total_Events": total_events
        })

        for t, val in zip(selected, selected_values):

            start = t - decluster_window
            end = t + decluster_window

            hourly_window = df_associated_hourly_variable.loc[start:end]

            max_hourly = (
                hourly_window[associated_hourly_variable_input_column].max()
                if not hourly_window.empty
                else np.nan
            )

            all_pot.append({
                "time": t,
                center_variable_output_column: val,
                associated_hourly_variable_output_column: max_hourly
            })

    # ─────────────────────────────────────────────
    # Non-stationary threshold
    # ─────────────────────────────────────────────
    else:

        # Separate threshold for each year
        for year, group in center_variable_series.groupby(center_variable_series.index.year):

            group_sorted = group.sort_values(ascending=False)

            selected = []
            selected_values = []

            for idx, val in group_sorted.items():

                # Use full timedelta comparison
                if any(abs(idx - sel) <= decluster_window for sel in selected):
                    continue

                selected.append(idx)
                selected_values.append(val)

                if len(selected) == number_of_events_per_year:
                    break

            if len(selected) < number_of_events_per_year:
                print(f"Skipping {year}: not enough independent events.")
                continue

            threshold_value = min(selected_values)

            thresholds.append({
                "Year": year,
                "Threshold": threshold_value
            })

            for t, val in zip(selected, selected_values):

                start = t - decluster_window
                end = t + decluster_window

                hourly_window = df_associated_hourly_variable.loc[start:end]

                max_hourly = (
                    hourly_window[associated_hourly_variable_input_column].max()
                    if not hourly_window.empty
                    else np.nan
                )

                all_pot.append({
                    "time": t,
                    center_variable_output_column: val,
                    associated_hourly_variable_output_column: max_hourly
                })

    # ────────────────────────────────────────────────────────────────────────────────
    # Save Thresholds
    # ────────────────────────────────────────────────────────────────────────────────
    df_pot = pd.DataFrame(all_pot).sort_values('time').reset_index(drop=True)
    df_thresh = pd.DataFrame(thresholds)
    df_thresh.to_csv(thresholds_output_path, index=False)
    
    df_pot = df_pot.dropna()
    df_pot = df_pot.drop(columns=['Year'], errors='ignore')
    
    df_pot['time'] = pd.to_datetime(df_pot['time'])
    df_pot = df_pot.set_index('time')
              
    # Ensure dataset is daily (rounded to date)
    df_associated_daily_variable.index = df_associated_daily_variable.index.date  
    
    # Initialize result list
    all_max_series = []

    # ────────────────────────────────────────────────────────────────────────────────
    # associated daily-variable matching
    # ────────────────────────────────────────────────────────────────────────────────
    # Loop through each POT event time (as date)
    for event_time in df_pot.index:
        event_date = event_time.date()
    
        start_date = event_date - timedelta(days=2.5)
        end_date = event_date + timedelta(days=2.5)
    
        # Subset discharge data in that window
        window_data = df_associated_daily_variable.loc[start_date:end_date]
    
        # Get max discharge values and store with event label
        max_rd_series = window_data.max()
        max_rd_series.name = event_date
        all_max_series.append(max_rd_series)
    
    # Combine all into a new DataFrame
    max_df = pd.DataFrame(all_max_series)
    df_pot.index = pd.to_datetime(df_pot.index).round('H')
    
    # Step 1: Create a 'date' column from the POT_df index
    df_pot = df_pot.copy()
    df_pot['date'] = df_pot.index.date  
    df_pot = df_pot.reset_index()
    
    # Step 2: Reset index for max_RD_df and rename its index to 'date'
    max_df = max_df.copy()
    max_df = max_df.reset_index().rename(columns={'index': 'date'})  

    # ────────────────────────────────────────────────────────────────────────────────
    # Save POTs
    # ────────────────────────────────────────────────────────────────────────────────
    
    # Step 3: Merge on 'date'
    merged_df = df_pot.merge(max_df, on='date', how='left')
    merged_df = merged_df.drop(columns=['date'])
    merged_df = merged_df.drop(columns=[associated_daily_variable_time], errors='ignore')
    merged_df = merged_df.dropna()
    
    merged_df = merged_df.rename(columns={associated_daily_variable_input_column: associated_daily_variable_output_column})
    merged_df['time'] = pd.to_datetime(merged_df['time'])
                   
    merged_df.to_csv(pot_output_path, index=False)   
    return merged_df, df_thresh

def POT_Daily(center_variable_time, associated_hourly_variable_1_time, associated_hourly_variable_2_time,
               center_variable_input_column, associated_hourly_variable_1_input_column, associated_hourly_variable_2_input_column,
               center_variable_input_path, associated_hourly_variable_1_input_path, associated_hourly_variable_2_input_path,
               center_variable_output_column, associated_hourly_variable_1_output_column, associated_hourly_variable_2_output_column,
               number_of_events_per_year, thresholds_output_path, pot_output_path, stationary = True):
    """
    Extracts Peak-Over-Threshold (POT) events from a daily center variable and
    identifies the associated maximum values of two hourly variables within a
    ±2.5-day event window.

    Parameters
    ----------
    center_variable_time : str
        Name of the time column in the center-variable dataset.

    associated_hourly_variable_1_time : str
        Name of the time column in the first associated hourly-variable dataset.

    associated_hourly_variable_2_time : str
        Name of the time column in the second associated hourly-variable dataset.

    center_variable_input_column : str
        Column name of the center variable used to define POT events
        (e.g., RD at a selected station).

    associated_hourly_variable_1_input_column : str
        Column name of the first associated hourly variable.

    associated_hourly_variable_2_input_column : str
        Column name of the second associated hourly variable.

    center_variable_input_path : str
        Path to the center-variable input CSV file.

    associated_hourly_variable_1_input_path : str
        Path to the first associated hourly-variable input CSV file.

    associated_hourly_variable_2_input_path : str
        Path to the second associated hourly-variable input CSV file.

    center_variable_output_column : str
        Output column name for the extracted POT center-variable values.

    associated_hourly_variable_1_output_column : str
        Output column name for the maximum first associated hourly variable
        within the event window.

    associated_hourly_variable_2_output_column : str
        Output column name for the maximum second associated hourly variable
        within the event window.

    number_of_events_per_year : int
        Number of independent POT events selected per year.

    thresholds_output_path : str
        Path where the threshold summary CSV will be saved.

    pot_output_path : str
        Path where the final POT event table will be saved.

    stationary : bool, optional
        If True, one constant threshold is used for the entire record.
        If False, a separate threshold is estimated for each year.

    Returns
    -------
    df_pot : pandas.DataFrame
        Final POT dataset containing the selected center-variable events and the
        associated hourly-variable maxima.

    df_thresh : pandas.DataFrame
        Threshold information used for stationary or non-stationary POT selection.

    Notes
    -----
    The center variable defines the POT event times. Events are selected from the
    largest center-variable values while enforcing a ±2.5-day declustering window.

    For each selected POT event, the function searches within ±2.5 days around the
    center-variable event time and extracts the maximum values of two associated
    hourly variables.

    This function is useful when the center variable is daily, such as river
    discharge, and the associated variables are hourly, such as NTR and RF.
    """
    
    df_center_variable = pd.read_csv(center_variable_input_path, parse_dates=[center_variable_time])
    df_center_variable['time'] = pd.to_datetime(df_center_variable[center_variable_time])
    df_center_variable = df_center_variable.set_index('time')
    
    df_associated_hourly_variable_1 = pd.read_csv(associated_hourly_variable_1_input_path, parse_dates=[associated_hourly_variable_1_time])
    df_associated_hourly_variable_1['time'] = pd.to_datetime(df_associated_hourly_variable_1[associated_hourly_variable_1_time])
    df_associated_hourly_variable_1 = df_associated_hourly_variable_1.set_index('time')
    
    df_associated_hourly_variable_2 = pd.read_csv(associated_hourly_variable_2_input_path, parse_dates=[associated_hourly_variable_2_time])
    df_associated_hourly_variable_2['time'] = pd.to_datetime(df_associated_hourly_variable_2[associated_hourly_variable_2_time])
    df_associated_hourly_variable_2 = df_associated_hourly_variable_2.set_index('time')


    # ────────────────────────────────────────────────────────────────────────────────
    # Parameters
    # ────────────────────────────────────────────────────────────────────────────────
    center_variable_series = df_center_variable[center_variable_input_column].dropna()
    decluster_window = timedelta(days=2.5)

    # ────────────────────────────────────────────────────────────────────────────────
    # Storage
    # ────────────────────────────────────────────────────────────────────────────────
    all_pot = []
    thresholds = []

    # ────────────────────────────────────────────────────────────────────────────────
    # POT extraction + associated hourly-variables matching
    # ────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # Stationary threshold
    # ─────────────────────────────────────────────
    if stationary:

        center_variable_series_sorted = center_variable_series.sort_values(ascending=False)

        selected = []
        selected_values = []

        total_years = center_variable_series.index.year.nunique()
        total_events = number_of_events_per_year * total_years

        for idx, val in center_variable_series_sorted.items():

            if any(abs(idx - sel) <= decluster_window for sel in selected):
                continue

            selected.append(idx)
            selected_values.append(val)

            if len(selected) == total_events:
                break

        if len(selected) < total_events:
            raise ValueError(
                "Not enough independent events found to meet the specified number of events per year."
            )

        threshold_value = min(selected_values)

        thresholds.append({
            "Threshold": threshold_value,
            "Number_of_Years": total_years,
            "Target_Total_Events": total_events
        })

        for t, val in zip(selected, selected_values):

            start = t - decluster_window
            end = t + decluster_window

            hourly_variable_1_window = df_associated_hourly_variable_1.loc[start:end]
            hourly_variable_2_window = df_associated_hourly_variable_2.loc[start:end]

            max_hourly_variable_1 = (
                hourly_variable_1_window[associated_hourly_variable_1_input_column].max()
                if not hourly_variable_1_window.empty
                else np.nan
            )

            max_hourly_variable_2 = (
                hourly_variable_2_window[associated_hourly_variable_2_input_column].max()
                if not hourly_variable_2_window.empty
                else np.nan
            )

            all_pot.append({
                "time": t,
                center_variable_output_column: val,
                associated_hourly_variable_1_output_column: max_hourly_variable_1,
                associated_hourly_variable_2_output_column: max_hourly_variable_2
            })

    # ─────────────────────────────────────────────
    # Non-stationary threshold
    # ─────────────────────────────────────────────
    else:

        for year, group in center_variable_series.groupby(center_variable_series.index.year):

            group_sorted = group.sort_values(ascending=False)

            selected = []
            selected_values = []

            for idx, val in group_sorted.items():

                if any(abs(idx - sel) <= decluster_window for sel in selected):
                    continue

                selected.append(idx)
                selected_values.append(val)

                if len(selected) == number_of_events_per_year:
                    break

            if len(selected) < number_of_events_per_year:
                print(f"Skipping {year}: not enough independent events.")
                continue

            threshold_value = min(selected_values)

            thresholds.append({
                "Year": year,
                "Threshold": threshold_value
            })

            for t, val in zip(selected, selected_values):

                start = t - decluster_window
                end = t + decluster_window

                hourly_variable_1_window = df_associated_hourly_variable_1.loc[start:end]
                hourly_variable_2_window = df_associated_hourly_variable_2.loc[start:end]

                max_hourly_variable_1 = (
                    hourly_variable_1_window[associated_hourly_variable_1_input_column].max()
                    if not hourly_variable_1_window.empty
                    else np.nan
                )

                max_hourly_variable_2 = (
                    hourly_variable_2_window[associated_hourly_variable_2_input_column].max()
                    if not hourly_variable_2_window.empty
                    else np.nan
                )

                all_pot.append({
                    "time": t,
                    center_variable_output_column: val,
                    associated_hourly_variable_1_output_column: max_hourly_variable_1,
                    associated_hourly_variable_2_output_column: max_hourly_variable_2
                })

    if len(all_pot) == 0:
        raise ValueError("No POT events were selected.")

    df_pot = pd.DataFrame(all_pot).sort_values("time").reset_index(drop=True)
    df_pot = df_pot.dropna()
    df_thresh = pd.DataFrame(thresholds)

    df_pot.to_csv(pot_output_path, index=False)
    df_thresh.to_csv(thresholds_output_path, index=False)

    return df_pot, df_thresh

def POT_Hourly_all(center_variable_time, associated_hourly_variable_time, associated_second_hourly_variable_time,
                   center_variable_input_column, associated_hourly_variable_input_column, associated_second_hourly_variable_input_column,
                   center_variable_input_path, associated_hourly_variable_input_path, associated_second_hourly_variable_input_path,
                   center_variable_output_column, associated_hourly_variable_output_column, associated_second_hourly_variable_output_column,
                   number_of_events_per_year, thresholds_output_path, pot_output_path, stationary=True):

    """
    Extracts POT events from an hourly center variable and identifies the associated
    maximum values of TWO hourly variables within a ±2.5-day event window.

    positive lag = associated variable peaks after the center variable
    negative lag = associated variable peaks before the center variable
    """
    # ─────────────────────────────────────────────────────────────
    # Read datasets
    # ─────────────────────────────────────────────────────────────
    df_center_variable = pd.read_csv(
        center_variable_input_path,
        parse_dates=[center_variable_time]
    )
    df_center_variable["time"] = pd.to_datetime(df_center_variable[center_variable_time])
    df_center_variable = df_center_variable.set_index("time").sort_index()

    df_associated_hourly_variable = pd.read_csv(
        associated_hourly_variable_input_path,
        parse_dates=[associated_hourly_variable_time]
    )
    df_associated_hourly_variable["time"] = pd.to_datetime(
        df_associated_hourly_variable[associated_hourly_variable_time]
    )
    df_associated_hourly_variable = df_associated_hourly_variable.set_index("time").sort_index()

    df_associated_second_hourly_variable = pd.read_csv(
        associated_second_hourly_variable_input_path,
        parse_dates=[associated_second_hourly_variable_time]
    )
    df_associated_second_hourly_variable["time"] = pd.to_datetime(
        df_associated_second_hourly_variable[associated_second_hourly_variable_time]
    )
    df_associated_second_hourly_variable = df_associated_second_hourly_variable.set_index("time").sort_index()

    # ─────────────────────────────────────────────────────────────
    # Parameters
    # ─────────────────────────────────────────────────────────────
    center_variable_series = df_center_variable[center_variable_input_column].dropna()

    decluster_window = timedelta(days=2.5)
    associated_variable_decluster_window = timedelta(days=2)

    all_pot = []
    thresholds = []

    # ─────────────────────────────────────────────────────────────
    # Helper function: find associated hourly maximum and lag
    # ─────────────────────────────────────────────────────────────
    def get_hourly_associated_max(df, column, event_time, window):
        start = event_time - window
        end = event_time + window

        window_data = df.loc[start:end]
        s = window_data[column].dropna()

        if s.empty:
            return np.nan, np.nan

        max_time = s.idxmax()
        max_value = s.loc[max_time]
        lag_hours = (max_time - event_time).total_seconds() / 3600.0

        return max_value, lag_hours

    # ─────────────────────────────────────────────────────────────
    # Stationary POT threshold
    # ─────────────────────────────────────────────────────────────
    if stationary:

        center_variable_series_sorted = center_variable_series.sort_values(ascending=False)

        selected = []
        selected_values = []

        total_years = center_variable_series.index.year.nunique()
        total_events = number_of_events_per_year * total_years

        for idx, val in center_variable_series_sorted.items():

            if any(abs(idx - sel) <= decluster_window for sel in selected):
                continue

            selected.append(idx)
            selected_values.append(val)

            if len(selected) == total_events:
                break

        if len(selected) < total_events:
            raise ValueError(
                "Not enough independent events found to meet the specified number of events per year."
            )

        threshold_value = min(selected_values)

        thresholds.append({
            "Threshold": threshold_value,
            "Number_of_Years": total_years,
            "Target_Total_Events": total_events
        })

        for t, val in zip(selected, selected_values):

            max_hourly, lag_hourly = get_hourly_associated_max(
                df_associated_hourly_variable,
                associated_hourly_variable_input_column,
                t,
                associated_variable_decluster_window
            )

            max_second_hourly, lag_second_hourly = get_hourly_associated_max(
                df_associated_second_hourly_variable,
                associated_second_hourly_variable_input_column,
                t,
                associated_variable_decluster_window
            )

            all_pot.append({
                "time": t,
                center_variable_output_column: val,
                associated_hourly_variable_output_column: max_hourly,
                associated_second_hourly_variable_output_column: max_second_hourly,
            })

    # ─────────────────────────────────────────────────────────────
    # Non-stationary POT threshold
    # ─────────────────────────────────────────────────────────────
    else:

        for year, group in center_variable_series.groupby(center_variable_series.index.year):

            group_sorted = group.sort_values(ascending=False)

            selected = []
            selected_values = []

            for idx, val in group_sorted.items():

                if any(abs(idx - sel) <= decluster_window for sel in selected):
                    continue

                selected.append(idx)
                selected_values.append(val)

                if len(selected) == number_of_events_per_year:
                    break

            if len(selected) < number_of_events_per_year:
                print(f"Skipping {year}: not enough independent events.")
                continue

            threshold_value = min(selected_values)

            thresholds.append({
                "Year": year,
                "Threshold": threshold_value
            })

            for t, val in zip(selected, selected_values):

                max_hourly, lag_hourly = get_hourly_associated_max(
                    df_associated_hourly_variable,
                    associated_hourly_variable_input_column,
                    t,
                    associated_variable_decluster_window
                )

                max_second_hourly, lag_second_hourly = get_hourly_associated_max(
                    df_associated_second_hourly_variable,
                    associated_second_hourly_variable_input_column,
                    t,
                    associated_variable_decluster_window
                )

                all_pot.append({
                    "time": t,
                    center_variable_output_column: val,
                    associated_hourly_variable_output_column: max_hourly,
                    associated_second_hourly_variable_output_column: max_second_hourly,
                })

    # ─────────────────────────────────────────────────────────────
    # Save outputs
    # ─────────────────────────────────────────────────────────────
    merged_df = pd.DataFrame(all_pot).sort_values("time").reset_index(drop=True)

    df_thresh = pd.DataFrame(thresholds)
    df_thresh.to_csv(thresholds_output_path, index=False)

    # merged_df = merged_df.dropna(
    #     subset=[
    #         center_variable_output_column,
    #         associated_hourly_variable_output_column,
    #         associated_second_hourly_variable_output_column
    #     ]
    # )

    merged_df["time"] = pd.to_datetime(merged_df["time"])
    merged_df.to_csv(pot_output_path, index=False)

    return merged_df, df_thresh