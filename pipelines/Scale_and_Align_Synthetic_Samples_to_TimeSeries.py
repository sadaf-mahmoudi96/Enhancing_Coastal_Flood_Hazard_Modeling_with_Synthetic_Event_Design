
"""
Unified scale-and-align workflow for trivariate NTR/RF/RD design events.


Core logic
----------
1. Read the target design/response samples (Sim_NTR, Sim_RF, Sim_RD).
2. Read the historical POT table and POT-centered time-series table.
3. For each synthetic target, select a historical analogue using inverse-distance
   weighted sampling in the corresponding POT variable.
4. Calculate the scale factor from the historical POT-centered curve.
5. Apply that scale factor to the appropriate original high-resolution time series:
   - NTR: POT-centered NTR curve itself (hourly).
   - RF: scale factor is based on the selected accumulation curve (e.g. acc_24hr);
         the factor is applied to original hourly rainfall. If the original rainfall
         is cumulative, it is converted to mm/hr first.
   - RD: scale factor is based on the POT-centered RD curve; when an original hourly
         discharge series is supplied, that hourly series is scaled.
6. For associated variables, sample a lag from the observed lag pool and shift the
   relative-hour axis by that lag.
7. Align all variables on the center variable at relative hour 0.
8. Save one merged event CSV per target sample and a manifest describing the
   historical analogue, scaling, lags, weights, and return period/year.

The module supports:
- response-based representative events in one CSV;
- design-based samples stored by return period/year;
- one or multiple station series for any variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


@dataclass
class OriginalSeries:
    """Optional original high-resolution series used after a scale factor is derived."""
    path: PathLike
    time_col: str
    value_col: str
    output_col: str

    # Only used for rainfall.
    # "hourly" means values already represent mm during each hour / mm hr-1.
    # "cumulative" means a running accumulation that must be differenced first.
    rainfall_kind: str = "hourly"

    # If cumulative rainfall resets (e.g. forecast cycle), negative differences are
    # interpreted as a reset and the current cumulative value is used for that hour.
    cumulative_reset: bool = True

    # Optional time filter.
    cutoff: Optional[str] = None


@dataclass
class VariableSpec:
    """
    Configuration for one physical variable: NTR, RF, or RD.

    `pot_peak_col` is the historical peak/accumulation column in the POT table.
    `pot_ts_value_col` is the curve used to derive the scale factor.

    For RF, `pot_ts_value_col` should normally be the accumulation diagnostic used
    in the POT analysis (e.g. "acc_24hr"), while `original_series` should point to
    the hourly rainfall used as the boundary condition.

    For RD, `pot_ts_value_col` can be the daily/aggregated discharge curve used by
    the POT analysis, while `original_series` can point to one or more hourly gauge
    or GloFAS series.
    """
    name: str
    target_col: str
    pot_peak_col: str
    pot_ts_value_col: str

    sampling_power: float = 1.8
    min_difference: float = 1e-3

    # Lag column in the POT time-series table. Leave None for the center variable.
    lag_col: Optional[str] = None
    max_abs_lag_hours: Optional[float] = 120.0

    # Optional historical-data availability restriction.
    cutoff: Optional[str] = None

    # Scale-factor bounds. None means unbounded.
    scale_factor_min: Optional[float] = None
    scale_factor_max: Optional[float] = None

    # If empty, the POT-centered curve itself is saved/scaled.
    # If supplied, the scale factor is applied to each original high-resolution series.
    original_series: Sequence[OriginalSeries] = field(default_factory=tuple)

    # Output column when original_series is empty.
    output_col: Optional[str] = None


def _as_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse timestamps and remove timezone information after converting to UTC.
    This avoids tz-aware/tz-naive merge failures.
    """
    out = pd.to_datetime(series, errors="coerce", utc=True)
    return out.dt.tz_convert(None)


def _read_timeseries(path: PathLike, time_col: str, value_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, [time_col, *value_cols], str(path))
    df = df[[time_col, *value_cols]].copy()
    df[time_col] = _parse_datetime(df[time_col])
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="last")
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def cumulative_rainfall_to_mm_per_hr(
    time: pd.Series,
    cumulative_mm: pd.Series,
    reset_on_negative: bool = True,
) -> pd.Series:
    """
    Convert a running cumulative rainfall series (mm) to an hourly rate (mm/hr).

    For regular hourly data this is simply the first difference. For irregular
    timestamps, the accumulation increment is divided by elapsed hours.

    If the cumulative counter resets and `reset_on_negative=True`, a negative
    increment is replaced by the current cumulative value, which treats that row
    as the first accumulation after the reset.
    """
    t = _parse_datetime(pd.Series(time))
    x = pd.to_numeric(pd.Series(cumulative_mm), errors="coerce")

    dt_hr = t.diff().dt.total_seconds() / 3600.0
    dx = x.diff()

    if reset_on_negative:
        reset = dx < 0
        dx.loc[reset] = x.loc[reset]

    rate = dx / dt_hr
    rate.loc[(dt_hr <= 0) | (~np.isfinite(dt_hr))] = np.nan

    # There is no previous observation for the first row.
    if len(rate):
        rate.iloc[0] = np.nan

    return rate


def _load_original_series(spec: OriginalSeries) -> pd.DataFrame:
    df = _read_timeseries(spec.path, spec.time_col, [spec.value_col])

    if spec.cutoff is not None:
        cutoff = pd.Timestamp(spec.cutoff)
        df = df[df[spec.time_col] > cutoff].copy()

    values = df[spec.value_col].copy()

    kind = spec.rainfall_kind.lower()
    if kind not in {"hourly", "cumulative"}:
        raise ValueError(
            f"rainfall_kind for {spec.output_col!r} must be 'hourly' or 'cumulative', "
            f"not {spec.rainfall_kind!r}."
        )

    if kind == "cumulative":
        values = cumulative_rainfall_to_mm_per_hr(
            df[spec.time_col],
            values,
            reset_on_negative=spec.cumulative_reset,
        )

    out = pd.DataFrame(
        {
            "Time": df[spec.time_col].to_numpy(),
            spec.output_col: values.to_numpy(),
        }
    )
    return out.dropna(subset=["Time"]).sort_values("Time")


def inverse_difference_probabilities(
    historical_values: Sequence[float],
    target_value: float,
    power: float = 1.8,
    min_difference: float = 1e-3,
) -> np.ndarray:
    historical = np.asarray(historical_values, dtype=float)
    target_value = float(target_value)

    valid = np.isfinite(historical)
    if not valid.any():
        raise ValueError("Historical peak array contains no finite values.")

    differences = np.abs(historical - target_value)
    differences = np.maximum(differences, min_difference)

    weights = np.zeros(len(historical), dtype=float)
    weights[valid] = 1.0 / np.power(differences[valid], power)

    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not construct valid inverse-distance sampling weights.")

    return weights / total


def _choose_analogue_with_curve(
    pot: pd.DataFrame,
    pot_time_col: str,
    pot_ts: pd.DataFrame,
    event_time_col: str,
    historical_peak_col: str,
    target_value: float,
    rng: np.random.Generator,
    power: float,
    min_difference: float,
    max_tries: int,
) -> tuple[int, pd.DataFrame, float]:
    """
    Select a POT analogue while requiring that a corresponding POT time-series
    curve actually exists.
    """
    peaks = pd.to_numeric(pot[historical_peak_col], errors="coerce").to_numpy()
    probabilities = inverse_difference_probabilities(
        peaks,
        target_value,
        power=power,
        min_difference=min_difference,
    )

    available_times = set(pot_ts[event_time_col].dropna().tolist())
    eligible = np.array([t in available_times for t in pot[pot_time_col]], dtype=bool)
    probabilities = probabilities * eligible.astype(float)

    if probabilities.sum() <= 0:
        raise RuntimeError(
            f"No POT rows in {historical_peak_col!r} have matching curves in the POT "
            f"time-series table."
        )

    probabilities = probabilities / probabilities.sum()

    # With the availability mask, one draw is normally sufficient. Keep max_tries
    # for defensive compatibility with the notebook workflow.
    for _ in range(max_tries):
        idx = int(rng.choice(len(pot), p=probabilities))
        event_time = pot.iloc[idx][pot_time_col]
        curve = pot_ts[pot_ts[event_time_col] == event_time].copy()
        if not curve.empty:
            return idx, curve.sort_values("time"), float(probabilities[idx])

    raise RuntimeError(
        f"Could not find a historical curve for target value {target_value} "
        f"after {max_tries} attempts."
    )


def _bounded_scale_factor(
    target: float,
    historical_reference: float,
    lower: Optional[float],
    upper: Optional[float],
) -> tuple[float, float]:
    if not np.isfinite(historical_reference) or historical_reference == 0:
        raise ValueError(
            f"Cannot scale using historical reference {historical_reference}."
        )

    raw = abs(float(target) / float(historical_reference))
    applied = raw

    if lower is not None:
        applied = max(applied, float(lower))
    if upper is not None:
        applied = min(applied, float(upper))

    return raw, applied


def _sample_lag(
    pot_ts: pd.DataFrame,
    lag_col: Optional[str],
    rng: np.random.Generator,
    max_abs_lag_hours: Optional[float],
) -> float:
    if lag_col is None:
        return 0.0

    _require_columns(pot_ts, [lag_col], "POT time-series")
    pool = pd.to_numeric(pot_ts[lag_col], errors="coerce").dropna().to_numpy(dtype=float)

    if max_abs_lag_hours is not None:
        pool = pool[np.abs(pool) <= float(max_abs_lag_hours)]

    if len(pool) == 0:
        warnings.warn(
            f"No valid lag values remained for {lag_col!r}; using 0 h.",
            RuntimeWarning,
        )
        return 0.0

    return float(rng.choice(pool))


def _midpoint_index(n: int) -> int:
    if n <= 0:
        raise ValueError("Cannot determine midpoint of an empty event.")
    return (n - 1) // 2


def _relative_frame(
    event_df: pd.DataFrame,
    value_cols: Sequence[str],
    lag_hours: float,
    source_prefix: str,
    anchor_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Put a historical event on a common relative-hour axis.

    rel_hour = historical timestamp - selected POT event time + sampled lag

    This is more robust than anchoring at the middle row because:
    - hourly and daily source curves can have different numbers of rows;
    - missing observations do not move the event center;
    - all variables are aligned to the actual historical POT timestamp.
    """
    event_df = event_df.sort_values("Time").reset_index(drop=True)
    _require_columns(event_df, ["Time", *value_cols], source_prefix)

    times = _parse_datetime(event_df["Time"])
    anchor_time = pd.Timestamp(anchor_time)

    rel = (
        (times - anchor_time).dt.total_seconds() / 3600.0
        + float(lag_hours)
    )

    # The workflow is hourly. Round only after computing from true timestamps.
    rel = np.rint(rel).astype("Int64")

    frame = pd.DataFrame({"rel_hour": rel})
    for col in value_cols:
        frame[col] = pd.to_numeric(event_df[col], errors="coerce").to_numpy()

    frame[f"{source_prefix}_Time"] = times.to_numpy()
    frame = frame.dropna(subset=["rel_hour"])
    frame["rel_hour"] = frame["rel_hour"].astype(int)

    # If duplicate timestamps collapse to the same relative hour, average the value
    # and keep the first source timestamp.
    agg = {col: "mean" for col in value_cols}
    agg[f"{source_prefix}_Time"] = "first"

    return (
        frame.groupby("rel_hour", as_index=False)
        .agg(agg)
        .sort_values("rel_hour")
        .reset_index(drop=True)
    )


def _extract_original_window(
    original_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    output_col: str,
    *,
    expected_frequency: str = "1h",
    interpolate_inside: bool = True,
) -> pd.DataFrame:
    """
    Extract the full high-resolution original series over the historical POT window.

    This deliberately does NOT merge only on the POT-curve timestamps. For example,
    if the RD POT curve is daily but the original discharge data are hourly, this
    returns every hourly discharge value between the POT window start and end.

    A regular hourly grid is constructed so different variables/stations can align
    consistently. Internal gaps can be time-interpolated; edge gaps remain NaN.
    """
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    if end_time < start_time:
        start_time, end_time = end_time, start_time

    subset = original_df[
        (original_df["Time"] >= start_time)
        & (original_df["Time"] <= end_time)
    ][["Time", output_col]].copy()

    if subset.empty:
        return subset

    # Build a common hourly grid across the selected historical window.
    full_time = pd.date_range(
        start=start_time.floor("h"),
        end=end_time.ceil("h"),
        freq=expected_frequency,
    )
    event_df = pd.DataFrame({"Time": full_time}).merge(
        subset,
        on="Time",
        how="left",
    )

    event_df[output_col] = pd.to_numeric(
        event_df[output_col],
        errors="coerce",
    )

    if interpolate_inside:
        temp = event_df.set_index("Time")
        temp[output_col] = temp[output_col].interpolate(
            method="time",
            limit_area="inside",
        )
        event_df = temp.reset_index()

    return event_df


def _load_targets_response(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Design_Mode"] = "response"
    if "Target_ID" not in df.columns:
        df["Target_ID"] = np.arange(len(df), dtype=int)
    return df


def _load_targets_design(
    years: Sequence[Union[int, float, str]],
    directory: PathLike,
    filename_template: str,
    return_period_col: str = "Return_Period",
) -> pd.DataFrame:
    """
    Load design-based samples stored separately by return period/year.

    Example template:
        "Space_Representative_Samples_{year}yr.csv"

    `{year}` is replaced by each value in `years`.
    """
    directory = _as_path(directory)
    pieces = []

    for year in years:
        path = directory / filename_template.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"Design-sample file not found: {path}")

        part = pd.read_csv(path)
        part[return_period_col] = year
        part["Design_Mode"] = "design"
        part["Target_ID"] = np.arange(len(part), dtype=int)
        pieces.append(part)

    if not pieces:
        raise ValueError("No design-based sample files were loaded.")

    return pd.concat(pieces, ignore_index=True)


def load_target_samples(
    mode: str,
    *,
    response_samples_path: Optional[PathLike] = None,
    years: Optional[Sequence[Union[int, float, str]]] = None,
    design_samples_directory: Optional[PathLike] = None,
    design_filename_template: Optional[str] = None,
    return_period_col: str = "Return_Period",
) -> pd.DataFrame:
    """
    Load either response-based representative samples or design-based samples.

    Response-based:
        mode="response", response_samples_path="...csv"

    Design-based:
        mode="design",
        years=[1, 5, 10, 20, 50, 100],
        design_samples_directory="...",
        design_filename_template="..._{year}yr.csv"
    """
    mode = mode.lower()

    if mode == "response":
        if response_samples_path is None:
            raise ValueError("response_samples_path is required for mode='response'.")
        return _load_targets_response(response_samples_path)

    if mode == "design":
        if years is None or design_samples_directory is None or design_filename_template is None:
            raise ValueError(
                "years, design_samples_directory, and design_filename_template are "
                "required for mode='design'."
            )
        return _load_targets_design(
            years,
            design_samples_directory,
            design_filename_template,
            return_period_col=return_period_col,
        )

    raise ValueError("mode must be either 'response' or 'design'.")


def scale_and_align_events(
    *,
    center_variable: str,
    pot_path: PathLike,
    pot_time_col: str,
    pot_timeseries_path: PathLike,
    pot_timeseries_time_col: str,
    pot_timeseries_event_time_col: str,
    variables: Sequence[VariableSpec],
    output_directory: PathLike,
    mode: str = "response",
    response_samples_path: Optional[PathLike] = None,
    years: Optional[Sequence[Union[int, float, str]]] = None,
    design_samples_directory: Optional[PathLike] = None,
    design_filename_template: Optional[str] = None,
    return_period_col: str = "Return_Period",
    weight_col: str = "Weights",
    random_state: Optional[int] = 42,
    max_tries: int = 50,
    drop_incomplete_rows: bool = False,
    filename_prefix: str = "Event",
) -> tuple[pd.DataFrame, list[Path]]:
    """
    Run the complete POT analogue -> scale -> lag -> align workflow.

    Parameters
    ----------
    center_variable
        "NTR", "RF", or "RD". This variable is anchored at relative hour 0.
    pot_path
        Historical POT table for the chosen center-variable workflow.
        Example: POT_NTR_...csv for an NTR-centered workflow.
    pot_timeseries_path
        Historical POT-window table associated with that same POT table.
        It must contain `event_time` (or the supplied name), `time`, and all
        `pot_ts_value_col` / lag columns required by `variables`.
    variables
        Specs for NTR, RF, and RD. The center variable should have lag_col=None.
        Associated variables can have lag columns.
    mode
        "response" or "design".
    response_samples_path
        Representative-event CSV, typically containing Sim_NTR, Sim_RF, Sim_RD,
        and optionally Weights.
    years / design_samples_directory / design_filename_template
        Design-based target files by return period. `years` can be
        [1, 5, 10, 20, 50, 100], or any other set used in your analysis.

    Returns
    -------
    manifest, output_files
        `manifest` has one row per synthetic target event.
        `output_files` contains one merged CSV per target event.
    """
    center_variable = center_variable.upper()
    variable_map = {v.name.upper(): v for v in variables}

    if center_variable not in variable_map:
        raise ValueError(
            f"center_variable={center_variable!r} is not present in variables."
        )

    output_directory = _as_path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    targets = load_target_samples(
        mode,
        response_samples_path=response_samples_path,
        years=years,
        design_samples_directory=design_samples_directory,
        design_filename_template=design_filename_template,
        return_period_col=return_period_col,
    )

    required_target_cols = [v.target_col for v in variables]
    _require_columns(targets, required_target_cols, "Target samples")

    pot = pd.read_csv(pot_path)
    _require_columns(
        pot,
        [pot_time_col, *[v.pot_peak_col for v in variables]],
        "POT table",
    )
    pot = pot.copy()
    pot[pot_time_col] = _parse_datetime(pot[pot_time_col])
    pot = pot.dropna(subset=[pot_time_col]).reset_index(drop=True)

    ts_required = [
        pot_timeseries_time_col,
        pot_timeseries_event_time_col,
        *[v.pot_ts_value_col for v in variables],
        *[v.lag_col for v in variables if v.lag_col is not None],
    ]
    pot_ts = pd.read_csv(pot_timeseries_path)
    _require_columns(pot_ts, ts_required, "POT time-series")
    pot_ts = pot_ts.copy()
    pot_ts[pot_timeseries_time_col] = _parse_datetime(pot_ts[pot_timeseries_time_col])
    pot_ts[pot_timeseries_event_time_col] = _parse_datetime(
        pot_ts[pot_timeseries_event_time_col]
    )
    pot_ts = pot_ts.rename(columns={pot_timeseries_time_col: "time"})
    pot_ts = pot_ts.sort_values([pot_timeseries_event_time_col, "time"])

    # Load original series once, not inside the event loop.
    original_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for v in variables:
        for j, original in enumerate(v.original_series):
            original_cache[(v.name.upper(), j)] = _load_original_series(original)

    rng = np.random.default_rng(random_state)

    manifest_rows = []
    output_files: list[Path] = []

    for row_pos, target_row in targets.reset_index(drop=True).iterrows():
        event_frames = []
        event_meta = {
            "Target_Row": row_pos,
            "Center_Variable": center_variable,
            "Design_Mode": target_row.get("Design_Mode", mode),
        }

        if return_period_col in target_row.index:
            event_meta[return_period_col] = target_row[return_period_col]

        if weight_col in target_row.index:
            event_meta[weight_col] = target_row[weight_col]

        # Create each physical series independently, as done in the notebooks.
        for v in variables:
            vname = v.name.upper()
            target_value = float(target_row[v.target_col])

            # Availability restriction can differ among NTR/RF/RD.
            pot_v = pot.copy()
            if v.cutoff is not None:
                pot_v = pot_v[pot_v[pot_time_col] > pd.Timestamp(v.cutoff)].copy()

            if pot_v.empty:
                raise RuntimeError(
                    f"No POT events remain for {vname} after cutoff={v.cutoff!r}."
                )

            idx_local, selected_curve, sampling_probability = _choose_analogue_with_curve(
                pot=pot_v,
                pot_time_col=pot_time_col,
                pot_ts=pot_ts,
                event_time_col=pot_timeseries_event_time_col,
                historical_peak_col=v.pot_peak_col,
                target_value=target_value,
                rng=rng,
                power=v.sampling_power,
                min_difference=v.min_difference,
                max_tries=max_tries,
            )

            selected_pot_row = pot_v.iloc[idx_local]
            selected_event_time = selected_pot_row[pot_time_col]

            curve_reference = pd.to_numeric(
                selected_curve[v.pot_ts_value_col], errors="coerce"
            ).max()

            raw_scale, applied_scale = _bounded_scale_factor(
                target_value,
                curve_reference,
                v.scale_factor_min,
                v.scale_factor_max,
            )

            lag = 0.0 if vname == center_variable else _sample_lag(
                pot_ts,
                v.lag_col,
                rng,
                v.max_abs_lag_hours,
            )

            event_meta[f"{vname}_Target"] = target_value
            event_meta[f"{vname}_Historical_POT"] = float(
                selected_pot_row[v.pot_peak_col]
            )
            event_meta[f"{vname}_Historical_Event_Time"] = selected_event_time
            event_meta[f"{vname}_Sampling_Probability"] = sampling_probability
            event_meta[f"{vname}_Raw_Scale_Factor"] = raw_scale
            event_meta[f"{vname}_Applied_Scale_Factor"] = applied_scale
            event_meta[f"{vname}_Lag_Hours"] = lag

            # If an original hourly series is not supplied, scale the selected
            # POT-centered curve itself.
            if not v.original_series:
                out_col = v.output_col or vname
                event_df = pd.DataFrame(
                    {
                        "Time": selected_curve["time"].to_numpy(),
                        out_col: pd.to_numeric(
                            selected_curve[v.pot_ts_value_col], errors="coerce"
                        ).to_numpy() * applied_scale,
                    }
                )

                frame = _relative_frame(
                    event_df,
                    [out_col],
                    lag_hours=lag,
                    source_prefix=vname,
                    anchor_time=selected_event_time,
                )
                event_frames.append(frame)

            # Otherwise use the POT curve to define the historical event WINDOW,
            # then extract every high-resolution observation from the original series
            # within that window. This is essential when the POT curve is daily but
            # the boundary-condition series is hourly.
            else:
                curve_times = _parse_datetime(selected_curve["time"]).dropna()

                if curve_times.empty:
                    raise RuntimeError(
                        f"Selected {vname} historical analogue has no valid timestamps."
                    )

                window_start = curve_times.min()
                window_end = curve_times.max()

                for j, original in enumerate(v.original_series):
                    original_df = original_cache[(vname, j)]

                    event_df = _extract_original_window(
                        original_df,
                        start_time=window_start,
                        end_time=window_end,
                        output_col=original.output_col,
                        expected_frequency="1h",
                        interpolate_inside=True,
                    )

                    if event_df.empty:
                        warnings.warn(
                            f"No original {vname} data were available from "
                            f"{window_start} to {window_end} for "
                            f"{original.output_col!r}.",
                            RuntimeWarning,
                        )

                    event_df[original.output_col] = (
                        pd.to_numeric(
                            event_df[original.output_col],
                            errors="coerce",
                        )
                        * applied_scale
                    )

                    prefix = (
                        vname
                        if len(v.original_series) == 1
                        else f"{vname}_{j + 1}"
                    )

                    frame = _relative_frame(
                        event_df,
                        [original.output_col],
                        lag_hours=lag,
                        source_prefix=prefix,
                        anchor_time=selected_event_time,
                    )
                    event_frames.append(frame)

        if not event_frames:
            raise RuntimeError("No event frames were generated.")

        merged = event_frames[0]
        for frame in event_frames[1:]:
            merged = merged.merge(frame, on="rel_hour", how="outer")

        merged = merged.sort_values("rel_hour").reset_index(drop=True)

        # Build a synthetic aligned clock using the historical center-variable
        # selected-event time as the zero-hour reference.
        center_time = event_meta[f"{center_variable}_Historical_Event_Time"]
        merged["Time"] = pd.Timestamp(center_time) + pd.to_timedelta(
            merged["rel_hour"], unit="h"
        )

        merged[f"is_anchor_{center_variable}"] = merged["rel_hour"].eq(0)

        for v in variables:
            vname = v.name.upper()
            merged[f"Lag_Hour_{vname}"] = event_meta[f"{vname}_Lag_Hours"]
            merged[f"{vname}_Scale_Factor"] = event_meta[
                f"{vname}_Applied_Scale_Factor"
            ]

        if weight_col in target_row.index:
            merged["Weight_Flood"] = target_row[weight_col]

        if return_period_col in target_row.index:
            merged[return_period_col] = target_row[return_period_col]

        # Determine generated value columns, excluding bookkeeping/original-time cols.
        bookkeeping_prefixes = ("rel_hour", "Time", "is_anchor_", "Lag_Hour_")
        value_cols = [
            c for c in merged.columns
            if not c.endswith("_Time")
            and not c.endswith("_Scale_Factor")
            and c != "Weight_Flood"
            and c != return_period_col
            and not any(c.startswith(p) for p in bookkeeping_prefixes)
        ]

        # Never silently create an empty event because one station/variable has
        # missing observations. By default rows are retained. If strict completeness
        # is explicitly requested, only then require all physical value columns.
        if drop_incomplete_rows and value_cols:
            complete = merged.dropna(subset=value_cols).reset_index(drop=True)
            if complete.empty:
                missing_counts = merged[value_cols].isna().sum().to_dict()
                raise RuntimeError(
                    f"Event {row_pos} would become empty after requiring complete "
                    f"rows. Missing-value counts: {missing_counts}. "
                    "Use drop_incomplete_rows=False or inspect source coverage."
                )
            merged = complete

        # Regardless of strictness, discard rows on which every physical variable
        # is missing. These rows cannot be used as boundary conditions.
        if value_cols:
            merged = merged.dropna(subset=value_cols, how="all").reset_index(drop=True)

        if merged.empty:
            raise RuntimeError(
                f"Event {row_pos} produced zero usable rows before writing. "
                "Check historical POT timestamps and original-data coverage."
            )

        # Record coverage diagnostics in the manifest.
        event_meta["Output_Rows"] = len(merged)
        for col in value_cols:
            event_meta[f"Missing_{col}"] = int(merged[col].isna().sum())

        # Keep rel_hour because it is useful for debugging alignment.
        if mode.lower() == "design" and return_period_col in target_row.index:
            rp = str(target_row[return_period_col]).replace(".", "p")
            filename = f"{filename_prefix}_RP{rp}_{row_pos:05d}.csv"
        else:
            filename = f"{filename_prefix}_{row_pos:05d}.csv"

        output_path = output_directory / filename
        merged.to_csv(output_path, index=False)
        output_files.append(output_path)

        event_meta["Output_File"] = str(output_path)
        manifest_rows.append(event_meta)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_directory / f"{filename_prefix}_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return output_files


__all__ = [
    "OriginalSeries",
    "VariableSpec",
    "cumulative_rainfall_to_mm_per_hr",
    "inverse_difference_probabilities",
    "load_target_samples",
    "scale_and_align_events",
]
