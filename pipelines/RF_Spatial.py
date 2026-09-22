import os
os.environ["ZARR_V3_EXPERIMENTAL_API"] = "1"
import glob
import numpy as np
import pandas as pd
import os
import xarray as xr
import rioxarray
from rasterio.enums import Resampling 

def rainfall_spatial_distributed_gauges_wise(dir_path: str,
                                             nldas_path: str,
                                             timeseries_rf_time_col,
                                             timeseries_rf_value_col,
                                             nldas_rainfall_name,
                                             lat_nearest_point_to_gauge,
                                             lon_nearest_point_to_gauge,
                                             resample_resolution):
    pattern = os.path.join(dir_path, "*.csv")
    files = glob.glob(pattern)

    # --- open NLDAS once ---
    nldas_ds = xr.open_zarr(nldas_path, zarr_format=3, consolidated=False)
    # precompute a pandas datetime index for fast searchsorted
    nldas_time = pd.to_datetime(nldas_ds["time"].values)

    if not files:
        print(f"⚠️ No '*_RF.csv' files found in: {dir_path}")
        return

    for file in files:
        try:
            df = pd.read_csv(file)
            if timeseries_rf_time_col not in df.columns:
                print(f"   ❌ Skipping (no 'Time' col): {file}")
                continue
            # parse times
            df[timeseries_rf_time_col] = pd.to_datetime(df[timeseries_rf_time_col], errors="coerce")
            df = df.dropna(subset=[timeseries_rf_time_col])
            if df.empty:
                print(f"   ❌ Skipping (no valid times): {file}")
                continue

            # match to NLDAS time axis using searchsorted; clip to valid range
            event_times = xr.DataArray(
                df[timeseries_rf_time_col].values,
                dims="time"
            )

            nldas_matched = nldas_ds.sel(
                time=event_times,
                method="nearest",
                tolerance=pd.Timedelta("2h")
            )
            
            rain = nldas_matched[nldas_rainfall_name]  # check units: NLDAS Rainf is often kg m-2 s-1 (mm/s)
            nearest_gauge_ts = rain.sel(lon = lon_nearest_point_to_gauge, lat = lat_nearest_point_to_gauge, method='nearest')   # time series of basin mean
            
            nearest_gauge_peak = float(nearest_gauge_ts.max().compute().values)

            max_real_rf = pd.to_numeric(df[timeseries_rf_value_col], errors="coerce").max()
            max_real_rf = float(max_real_rf)  # scalar float
            
            if nearest_gauge_peak > 0:
                rf_scale = max_real_rf / nearest_gauge_peak
                print('max_real_rf: ', max_real_rf)
                print('nearest_gauge_peak: ', nearest_gauge_peak)
                print('rf_scale: ', rf_scale)
            else:
                rf_scale = np.nan
                print(f"⚠️ gauge peak is zero for {file}")

            # scale by mean RF scale factor (ignore NaNs)
            # rf_scale = float(np.nanmean(df["RF Scale Factor"].to_numpy()))
            if not np.isfinite(rf_scale):
                print(f"   ❌ Skipping (invalid RF Scale Factor): {file}")
                continue

            nldas_scaled = rain * rf_scale
            # Reproject to new resolution
            da = nldas_scaled.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.write_crs(4326)
            da_resampled = da.rio.reproject(
                dst_crs=da.rio.crs,
                resolution=resample_resolution,
                resampling=Resampling.bilinear,  # now correct
                align='center'
            )

            max_scaled = float(nldas_scaled.max().compute().values)
            print("max NLDAS scaled:", max_scaled)


            # write next to the CSV, rename suffix
            base_name = os.path.basename(file).replace(".csv", "_Spatial_RF.nc")
            out_path = os.path.join(dir_path, base_name)

            # ensure folder exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            da_resampled.to_netcdf(out_path)
            print(f"   ✅ Saved: {out_path}")

        except Exception as e:
            print(f"   💥 Error processing {file}: {e}")


def rainfall_spatial_distributed_basin_average_wise(
    dir_path: str,
    nldas_path: str,
    timeseries_rf_time_col,
    timeseries_rf_value_col,
    nldas_rainfall_name
):

    pattern = os.path.join(dir_path, "*.csv")
    files = glob.glob(pattern)

    # --------------------------------------------------
    # Open NLDAS dataset once
    # --------------------------------------------------
    nldas_ds = xr.open_zarr(nldas_path)

    if not files:
        print(f"⚠️ No CSV files found in: {dir_path}")
        return

    for file in files:

        try:

            print("\n--------------------------------------------")
            print(f"Processing: {os.path.basename(file)}")
            print("--------------------------------------------")

            # --------------------------------------------------
            # Read rainfall time series
            # --------------------------------------------------
            df = pd.read_csv(file)

            # Check required columns
            if timeseries_rf_time_col not in df.columns:
                print(
                    f"   ❌ Skipping "
                    f"(no '{timeseries_rf_time_col}' column): {file}"
                )
                continue

            if timeseries_rf_value_col not in df.columns:
                print(
                    f"   ❌ Skipping "
                    f"(no '{timeseries_rf_value_col}' column): {file}"
                )
                continue

            # --------------------------------------------------
            # Parse time and rainfall values
            # --------------------------------------------------
            df[timeseries_rf_time_col] = pd.to_datetime(
                df[timeseries_rf_time_col],
                errors="coerce"
            )

            df[timeseries_rf_value_col] = pd.to_numeric(
                df[timeseries_rf_value_col],
                errors="coerce"
            )

            # Remove invalid rows
            df = df.dropna(
                subset=[
                    timeseries_rf_time_col,
                    timeseries_rf_value_col
                ]
            )

            if df.empty:
                print(f"   ❌ Skipping (no valid rainfall data): {file}")
                continue

            # --------------------------------------------------
            # Match CSV event times to nearest NLDAS times
            # --------------------------------------------------
            event_times = xr.DataArray(
                df[timeseries_rf_time_col].values,
                dims="time"
            )

            nldas_matched = nldas_ds.sel(
                time=event_times,
                method="nearest"
            )

            # Extract rainfall variable
            nldas_rf = nldas_matched[nldas_rainfall_name]

            # --------------------------------------------------
            # Calculate NLDAS basin-average rainfall time series
            #
            # Change ("lat", "lon") to
            # ("latitude", "longitude") if those are your
            # actual NLDAS dimension names.
            # --------------------------------------------------
            basin_avg_ts = nldas_rf.mean(
                dim=("lat", "lon"),
                skipna=True
            )

            # --------------------------------------------------
            # Find peak of basin-average NLDAS rainfall
            # --------------------------------------------------
            basin_peak = float(
                basin_avg_ts.max(
                    dim="time",
                    skipna=True
                ).compute().values
            )

            # --------------------------------------------------
            # Peak rainfall from target/simulated time series
            # --------------------------------------------------
            max_real_rf = float(
                df[timeseries_rf_value_col].max()
            )

            print("max_real_rf:", max_real_rf)
            print("NLDAS basin-average peak:", basin_peak)

            # --------------------------------------------------
            # Calculate scale factor
            # --------------------------------------------------
            if basin_peak > 0:

                rf_scale = max_real_rf / basin_peak

            else:

                rf_scale = np.nan
                print(
                    f"   ⚠️ NLDAS basin-average peak is zero "
                    f"for {file}"
                )

            print("RF scale factor:", rf_scale)

            # --------------------------------------------------
            # Check scale factor
            # --------------------------------------------------
            if not np.isfinite(rf_scale):

                print(
                    f"   ❌ Skipping "
                    f"(invalid RF scale factor): {file}"
                )

                continue

            # --------------------------------------------------
            # Scale entire spatial NLDAS rainfall field
            # --------------------------------------------------
            nldas_scaled = nldas_rf * rf_scale

            # --------------------------------------------------
            # Diagnostic:
            # basin-average peak after scaling
            # --------------------------------------------------
            scaled_basin_avg_ts = nldas_scaled.mean(
                dim=("lat", "lon"),
                skipna=True
            )

            scaled_basin_peak = float(
                scaled_basin_avg_ts.max(
                    dim="time",
                    skipna=True
                ).compute().values
            )

            print(
                "Scaled NLDAS basin-average peak:",
                scaled_basin_peak
            )

            # --------------------------------------------------
            # Diagnostic:
            # absolute maximum pixel after scaling
            # --------------------------------------------------
            max_scaled_pixel = float(
                nldas_scaled.max(
                    skipna=True
                ).compute().values
            )

            print(
                "Maximum scaled NLDAS pixel:",
                max_scaled_pixel
            )

            # --------------------------------------------------
            # Output filename
            # --------------------------------------------------
            base_name = os.path.basename(file).replace(
                ".csv",
                "_Spatial_RF.nc"
            )

            out_path = os.path.join(
                dir_path,
                base_name
            )

            os.makedirs(
                os.path.dirname(out_path),
                exist_ok=True
            )

            # --------------------------------------------------
            # Save NetCDF
            # --------------------------------------------------
            nldas_scaled.to_netcdf(out_path)

            print(f"   ✅ Saved: {out_path}")

        except Exception as e:

            print(
                f"   💥 Error processing {file}: {e}"
            )