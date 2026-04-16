"""
identify_heatwaves.py
------------------------
Identifies and analyzes heatwaves using EAMxx simulation temperature data over CONUS.

The local 90th-percentile threshold is derived from ERA5 historical data (1981–2010).
EAMxx analysis data are interpolated to ERA5 coordinates before comparison.

Workflow
--------
1.  Read ERA5 historical temperature data and compute daily maximum temperatures.
2.  Estimate a day-of-year local 90th-percentile threshold for each grid cell.
3.  Read EAMxx simulation data, rename the temperature variable, and compute
    daily maximum temperatures.
4.  Interpolate EAMxx data to ERA5 lat/lon coordinates (bilinear).
5.  Build a boolean heatwave-day mask (Tmax > local 90th percentile), with NaNs
    preserved for ocean/invalid cells.
6.  Compute heatwave-day frequency (number of heatwave days per grid cell).
7.  Compute the temperature anomaly (Tmax − threshold) and keep only heatwave days.
8.  Compute and plot the maximum heatwave anomaly per grid cell.
9.  Identify the largest spatially contiguous heatwave region per day and plot
    its area, mean anomaly, and max anomaly as time series.
10. Compute the longest consecutive heatwave duration per grid cell and plot.
11. Perform 3-D (time × space) event labelling; compute duration, peak area, and
    mean anomaly for every event; plot a scatter of peak area vs. duration.

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as colors
import pandas as pd
import xarray as xr

from util_contour_levels import myDict_clevels
from util_heatwaves import (
    area_duration_anomaly,
    largest_contiguous_regions,
    longest_run,
    read_data,
    read_EAMxx_data,
)
from util_spatial_plots import cmap, create_metrics, xr_plot_US
from util_vector_plots import plot_xr_lineplot, plot_xr_scatterplot

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  ← edit here to change any run parameter
# ---------------------------------------------------------------------------

# ERA5 historical data — used only to build the percentile threshold
ERA5_HIST_CONFIG = dict(
    fpath="/pscratch/sd/e/esinha/ERA5_CONUS/",
    fname_abb="elmforc.ERA5.c2018.0.25d.t2m.",
    varnames="t2m",           # ERA5 variable name (also used as the common name)
)
HIST_CONFIG = dict(yr_start=1981, yr_end=2010, mon_start=1, mon_end=12)

# EAMxx simulation data — the analysis period
EAMXX_DATA_CONFIG = dict(
    fpath="/pscratch/sd/j/jli628/EAMxx_regridding/heatwave_201107_SSP245/",
    fname_abb="output.eamxx.INSTANT.nsteps_x9.",
    eamxx_varname="T_2m",     # native variable name inside the EAMxx files
    varnames="t2m",           # rename to this common name after reading
    data_source="EAMxx_SSP245",  # label used in output filenames and CSV columns
)
ANALYSIS_CONFIG = dict(
    yr_start=2011, yr_end=2011,
    mon_start=7,   mon_end=7,
    day_start=17,  day_end=24,   # day_start=17: first day is spin-up, skip it
)

PERCENTILE_VALUE = 0.9          # 90th percentile threshold
FIG_EXTENT = [235, 294, 24, 47] # [lon_min, lon_max, lat_min, lat_max]
FIG_WT, FIG_HT = 14, 9          # figure width / height

# Diagnostic grid cells for console checks (land vs. ocean)
LAND_CELL = dict(lat=37.75, lon=-95)
OCEAN_CELL = dict(lat=37.75, lon=-130)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _make_clevels(key: str) -> tuple[list | None, colors.BoundaryNorm | None]:
    """Return (levels, norm) from myDict_clevels, or (None, None) if the key is absent."""
    clevels = myDict_clevels.get(key)
    if clevels is None:
        log.warning("_make_clevels: key '%s' not found in myDict_clevels; using auto-levels.", key)
        return None, None
    if clevels:
        return clevels, colors.BoundaryNorm(boundaries=clevels, ncolors=len(clevels))
    return None, None


def _month_year_prefix(mon: int, yr: int) -> str:
    """Return a consistent plot-title prefix, e.g. 'July 2011'."""
    return f"{calendar.month_name[mon]} {yr}"


def _out_fname(
    tag: str,
    varnames: str,
    yr: int,
    data_source: str,
    ext: str = "png",
    output_dir: Path | None = None,
) -> Path:
    """Build a standardised output path.

    Parameters
    ----------
    output_dir : Path, optional
        Directory in which to write the file.  When *None* (default) the
        plain filename is returned, letting ``xr_plot_US`` / ``_resolve_out_path``
        fall back to the legacy ``figures/`` subfolder logic.
    """
    name = f"{data_source}_{varnames}_{tag}_{yr}.{ext}"
    return Path(output_dir) / name if output_dir is not None else Path(name)


def _save_timeseries(
    da: xr.DataArray,
    tag: str,
    varnames: str,
    yr: int,
    data_source: str,
    ylabel: str,
    title: str,
    output_dir: Path | None = None,
) -> None:
    """Save a DataArray time series as both a line-plot PNG and a CSV."""
    fpath = _out_fname(tag, varnames, yr, data_source, output_dir=output_dir)
    # work on a copy to avoid mutating the caller's DataArray
    da = da.copy()
    da["Data_Source"] = data_source
    df = da.to_dataframe().reset_index()
    # Add a dayofyear column so compare_heatwaves.py can align members on a
    # common x-axis regardless of each member's exact day range.
    if "time" in df.columns:
        df["dayofyear"] = pd.to_datetime(df["time"]).dt.dayofyear
    df.to_csv(fpath.with_suffix(".csv"), index=False)
    plot_xr_lineplot(da, fname=str(fpath), title=title, xlabel=None, ylabel=ylabel)
    log.info("Saved %s → %s", tag, fpath)


def _plot_spatial_map(
    da: xr.DataArray,
    clevels_key: str,
    title: str,
    out_fname: str | Path,
    fig_extent: list,
    stats: Any = ...,   # sentinel: compute from da; pass None to suppress
    fig_wt: int = FIG_WT,
    fig_ht: int = FIG_HT,
) -> None:
    """Compute contour levels, call xr_plot_US, and log the saved filename.

    Parameters
    ----------
    stats : tuple | None, optional
        Stats tuple passed directly to xr_plot_US.  When omitted (default),
        ``(max, mean, min)`` is computed from *da*.  Pass ``None`` explicitly
        to suppress the stats annotation in the figure.
    """
    if stats is ...:
        metrics = create_metrics(da)
        stats = (metrics["max"], metrics["mean"], metrics["min"])
    levels, norm = _make_clevels(clevels_key)
    xr_plot_US(
        da,
        stats,
        cmap_col=cmap,
        title=title,
        fig_wt=fig_wt,
        fig_ht=fig_ht,
        fig_extent=fig_extent,
        show_states=True,
        fname=out_fname,
        levels=levels,
        norm=norm,
    )
    log.info("Saved spatial map → %s", out_fname)


# ---------------------------------------------------------------------------
# Step 1-2: Compute ERA5 percentile threshold  (call ONCE; reuse across members)
# ---------------------------------------------------------------------------
def compute_threshold(
    era5_hist_cfg: dict,
    hist_cfg: dict,
    percentile_value: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Load ERA5 historical data, compute daily maxima, and return the
    day-of-year local percentile threshold for each grid cell.

    This is separated from load_and_prepare so the ensemble runner can
    call it once and pass ds_q90 (and ds_hist) to each member, avoiding
    repeated reading of the 30-year historical dataset.

    Parameters
    ----------
    era5_hist_cfg  : dict   Keys: fpath, fname_abb, varnames
    hist_cfg       : dict   Keys: yr_start, yr_end, mon_start, mon_end
    percentile_value : float  e.g. 0.9 for the 90th percentile

    Returns
    -------
    ds_hist : DataArray (time, lat, lon) – daily-max historical temperatures
    ds_q90  : DataArray (dayofyear, lat, lon) – percentile threshold
    """
    log.info(
        "Reading ERA5 historical data (%d–%d, months %d–%d) …",
        hist_cfg["yr_start"], hist_cfg["yr_end"],
        hist_cfg["mon_start"], hist_cfg["mon_end"],
    )
    ds_hist = read_data(
        hist_cfg["yr_start"], hist_cfg["yr_end"],
        hist_cfg["mon_start"], hist_cfg["mon_end"],
        era5_hist_cfg["fpath"], era5_hist_cfg["fname_abb"], era5_hist_cfg["varnames"],
    )
    ds_hist = ds_hist.resample(time="1D").max()

    # Normalise the time index to numpy datetime64 so that
    # groupby("time.dayofyear") works regardless of the NetCDF calendar.
    if hasattr(ds_hist.indexes["time"], "to_datetimeindex"):
        ds_hist = ds_hist.assign_coords(
            time=ds_hist.indexes["time"].to_datetimeindex()
        )

    log.info("Computing %.0f-th percentile threshold …", percentile_value * 100)
    ds_q90 = (
        ds_hist
        .groupby("time.dayofyear")
        .quantile(percentile_value, dim="time")
        .drop_vars("quantile")
    )
    return ds_hist, ds_q90


# ---------------------------------------------------------------------------
# Shared helper: build heatwave mask and anomaly from a daily-max DataArray
# ---------------------------------------------------------------------------
def _build_mask_and_anomaly(
    ds: xr.DataArray,
    ds_q90: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute the heatwave mask and temperature anomaly from daily-max data.

    Parameters
    ----------
    ds     : DataArray (time, lat, lon) — daily-max temperatures on ERA5 grid
    ds_q90 : DataArray (dayofyear, lat, lon) — percentile threshold

    Returns
    -------
    heatwave_mask : DataArray — True where Tmax > threshold; NaN for ocean
    anomaly       : DataArray — Tmax minus threshold (all times)
    """
    above_threshold = ds.groupby("time.dayofyear") > ds_q90
    heatwave_mask   = above_threshold.where(ds.notnull())
    anomaly         = ds.groupby("time.dayofyear") - ds_q90
    return heatwave_mask, anomaly


# ---------------------------------------------------------------------------
# Steps 3-5: Load EAMxx data and build heatwave mask
# ---------------------------------------------------------------------------
def load_and_prepare(
    eamxx_data_cfg: dict,
    analysis_cfg: dict,
    ds_hist: xr.DataArray,
    ds_q90: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Load EAMxx analysis data, interpolate to ERA5 coordinates, and build the
    heatwave mask and temperature anomaly.

    Parameters
    ----------
    eamxx_data_cfg : dict
        Keys: fpath, fname_abb, eamxx_varname, varnames, data_source.
    analysis_cfg : dict
        Keys: yr_start, yr_end, mon_start, mon_end, day_start, day_end.
    ds_hist : DataArray
        ERA5 daily-max historical temperatures — used for the target grid
        coordinates during interpolation.  Pre-compute via
        :func:`compute_threshold`.
    ds_q90  : DataArray
        Day-of-year local percentile threshold on the ERA5 grid.  Pre-compute
        via :func:`compute_threshold`.

    Returns
    -------
    heatwave_mask : DataArray (time, lat, lon) – bool, NaN for ocean cells
    anomaly       : DataArray (time, lat, lon) – Tmax minus threshold (all times)
    ds            : DataArray (time, lat, lon) – EAMxx daily-max on ERA5 grid
    ds_q90        : DataArray (dayofyear, lat, lon) – ERA5-derived threshold
    """

    # --- EAMxx analysis data ------------------------------------------------
    log.info(
        "Reading EAMxx data (%d–%d, month %d, days %d–%d) …",
        analysis_cfg["yr_start"], analysis_cfg["yr_end"],
        analysis_cfg["mon_start"],
        analysis_cfg["day_start"], analysis_cfg["day_end"],
    )
    ds = read_EAMxx_data(
        analysis_cfg["yr_start"], analysis_cfg["yr_end"],
        analysis_cfg["mon_start"], analysis_cfg["mon_end"],
        analysis_cfg["day_start"], analysis_cfg["day_end"],
        eamxx_data_cfg["fpath"], eamxx_data_cfg["fname_abb"],
        eamxx_data_cfg["eamxx_varname"], eamxx_data_cfg["varnames"],
    )
    log.info("EAMxx data loaded: %s", ds)

    # --- interpolate EAMxx to ERA5 lat/lon coordinates ----------------------
    log.info("Interpolating EAMxx data to ERA5 coordinates (bilinear) …")
    ds = ds.interp(lat=ds_hist.lat, lon=ds_hist.lon, method="linear")

    # --- daily maximum temperature ------------------------------------------
    ds = ds.resample(time="1D").max()

    # Normalise CFTimeIndex → numpy datetime64 so groupby("time.dayofyear")
    # and downstream pd.Timestamp usage work regardless of the NetCDF calendar.
    if hasattr(ds.indexes["time"], "to_datetimeindex"):
        ds = ds.assign_coords(time=ds.indexes["time"].to_datetimeindex())

    heatwave_mask, anomaly = _build_mask_and_anomaly(ds, ds_q90)
    return heatwave_mask, anomaly, ds, ds_q90


# ---------------------------------------------------------------------------
# Steps 3-5 (ERA5 observational path): Load ERA5 analysis data and build mask
# ---------------------------------------------------------------------------
def load_and_prepare_era5(
    era5_hist_cfg: dict,
    analysis_cfg: dict,
    ds_q90: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Read ERA5 data for the analysis period (e.g. July 2011), compute daily
    maxima, and build the heatwave mask and temperature anomaly.

    The analysis period is typically *outside* the historical baseline
    (1981–2010) used to build ``ds_q90``, so this function reads the ERA5
    files for the target year rather than slicing from ``ds_hist``.

    Parameters
    ----------
    era5_hist_cfg : dict
        Keys: fpath, fname_abb, varnames — same as passed to
        :func:`compute_threshold`.  Used to locate ERA5 files for the
        analysis year.
    analysis_cfg : dict
        Keys: yr_start, yr_end, mon_start, mon_end, day_start, day_end.
    ds_q90  : DataArray
        Day-of-year local percentile threshold on the ERA5 grid, returned
        by :func:`compute_threshold`.

    Returns
    -------
    heatwave_mask : DataArray (time, lat, lon) – bool, NaN for ocean cells
    anomaly       : DataArray (time, lat, lon) – Tmax minus threshold
    ds            : DataArray (time, lat, lon) – ERA5 daily-max for the period
    ds_q90        : DataArray (dayofyear, lat, lon) – ERA5-derived threshold
    """
    yr_start  = analysis_cfg["yr_start"]
    yr_end    = analysis_cfg["yr_end"]
    mon_start = analysis_cfg["mon_start"]
    mon_end   = analysis_cfg["mon_end"]
    day_start = analysis_cfg["day_start"]
    day_end   = analysis_cfg["day_end"]

    log.info(
        "Reading ERA5 analysis data (%d–%d, month %d, days %d–%d) …",
        yr_start, yr_end, mon_start, day_start, day_end,
    )
    ds = read_data(
        yr_start, yr_end,
        mon_start, mon_end,
        era5_hist_cfg["fpath"], era5_hist_cfg["fname_abb"], era5_hist_cfg["varnames"],
    )
    ds = ds.resample(time="1D").max()

    # Normalise CFTimeIndex → numpy datetime64 so groupby("time.dayofyear") works
    if hasattr(ds.indexes["time"], "to_datetimeindex"):
        ds = ds.assign_coords(time=ds.indexes["time"].to_datetimeindex())

    # Trim to the requested day range within the month
    ds = ds.sel(
        time=slice(
            f"{yr_start}-{mon_start:02d}-{day_start:02d}",
            f"{yr_end}-{mon_end:02d}-{day_end:02d}",
        )
    )
    log.info("ERA5 analysis period loaded: %s", ds)

    heatwave_mask, anomaly = _build_mask_and_anomaly(ds, ds_q90)
    return heatwave_mask, anomaly, ds, ds_q90


# ---------------------------------------------------------------------------
# Step 6: Heatwave-day frequency
# ---------------------------------------------------------------------------
def plot_frequency(
    heatwave_mask: xr.DataArray,
    varnames: str,
    analysis_cfg: dict,
    fig_extent: list,
    data_source: str,
    output_dir: Path | None = None,
) -> None:
    """Plot and save heatwave-day frequency map."""
    freq_days = heatwave_mask.fillna(False).sum(dim="time")
    prefix = _month_year_prefix(analysis_cfg["mon_start"], analysis_cfg["yr_start"])
    _plot_spatial_map(
        freq_days, "frequency",
        title=f"{prefix} - Heatwave-day frequency (days)",
        out_fname=_out_fname("heatwave_frequency", varnames, analysis_cfg["yr_start"], data_source, output_dir=output_dir),
        fig_extent=fig_extent,
    )


# ---------------------------------------------------------------------------
# Step 8: Maximum heatwave anomaly
# ---------------------------------------------------------------------------
def plot_max_anomaly(
    heatwave_anomaly: xr.DataArray,
    varnames: str,
    analysis_cfg: dict,
    fig_extent: list,
    data_source: str,
    output_dir: Path | None = None,
) -> None:
    """Plot and save maximum heatwave anomaly map."""
    max_anom = heatwave_anomaly.max(dim="time")
    prefix = _month_year_prefix(analysis_cfg["mon_start"], analysis_cfg["yr_start"])
    _plot_spatial_map(
        max_anom, "anomaly",
        title=f"{prefix} - Maximum heatwave anomaly (K)",
        out_fname=_out_fname("max_heatwave_anomaly", varnames, analysis_cfg["yr_start"], data_source, output_dir=output_dir),
        fig_extent=fig_extent,
    )


# ---------------------------------------------------------------------------
# Step 9: Largest contiguous heatwave region per day
# ---------------------------------------------------------------------------
def plot_largest_contiguous_region(
    heatwave_mask: xr.DataArray,
    anomaly: xr.DataArray,
    varnames: str,
    analysis_cfg: dict,
    fig_extent: list,
    data_source: str,
    connectivity: int = 2,
    output_dir: Path | None = None,
) -> None:
    """Compute and plot per-day largest contiguous heatwave region statistics."""
    largest_sizes, mean_anoms, max_anoms, largest_masks = largest_contiguous_regions(
        heatwave_mask, anomaly, connectivity=connectivity
    )

    if not largest_sizes.sizes:
        log.warning("No contiguous heatwave regions found; skipping plots.")
        return

    yr = analysis_cfg["yr_start"]
    prefix = _month_year_prefix(analysis_cfg["mon_start"], yr)

    _save_timeseries(
        largest_sizes,
        "largest_contiguous_heatwave_area_timeseries",
        varnames, yr, data_source,
        ylabel="Largest contiguous heatwave area (km²)",
        title=f"{prefix} - Largest contiguous heatwave area per day",
        output_dir=output_dir,
    )
    _save_timeseries(
        mean_anoms,
        "mean_anomaly_largest_contiguous_heatwave_area_timeseries",
        varnames, yr, data_source,
        ylabel="Mean anomaly (K)",
        title=f"{prefix} - Mean anomaly for the largest contiguous heatwave area per day",
        output_dir=output_dir,
    )
    _save_timeseries(
        max_anoms,
        "max_anomaly_largest_contiguous_heatwave_area_timeseries",
        varnames, yr, data_source,
        ylabel="Max anomaly (K)",
        title=f"{prefix} - Max anomaly for the largest contiguous heatwave area per day",
        output_dir=output_dir,
    )

    # Spatial map on the day with the largest area
    max_day_idx = int(largest_sizes.argmax().values)
    largest_region = largest_masks.isel(time=max_day_idx)
    _plot_spatial_map(
        largest_region,
        "largest_region",
        title=f"{prefix} - Largest heatwave region",
        out_fname=_out_fname("largest_heatwave_region", varnames, yr, data_source, output_dir=output_dir),
        fig_extent=fig_extent,
        stats=None,
    )


# ---------------------------------------------------------------------------
# Step 10: Longest consecutive heatwave duration
# ---------------------------------------------------------------------------
def plot_longest_duration(
    heatwave_mask: xr.DataArray,
    varnames: str,
    analysis_cfg: dict,
    fig_extent: list,
    data_source: str,
    output_dir: Path | None = None,
) -> None:
    """Compute per-grid-cell longest consecutive heatwave duration and plot."""
    longest_duration = xr.apply_ufunc(
        longest_run,
        heatwave_mask.chunk({"time": -1}),
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    prefix = _month_year_prefix(analysis_cfg["mon_start"], analysis_cfg["yr_start"])
    _plot_spatial_map(
        longest_duration, "duration",
        title=f"{prefix} - Longest heatwave duration (days)",
        out_fname=_out_fname("longest_heatwave_duration", varnames, analysis_cfg["yr_start"], data_source, output_dir=output_dir),
        fig_extent=fig_extent,
    )


# ---------------------------------------------------------------------------
# Step 11: Event-level 3-D analysis
# ---------------------------------------------------------------------------
def plot_event_scatter(
    heatwave_mask: xr.DataArray,
    anomaly: xr.DataArray,
    varnames: str,
    analysis_cfg: dict,
    data_source: str,
    connectivity: int = 2,
    output_dir: Path | None = None,
) -> None:
    """3-D event labelling and scatter plot of peak area vs. duration."""
    durations, max_areas, mean_anoms = area_duration_anomaly(
        heatwave_mask, anomaly, connectivity=connectivity
    )

    if not durations:
        log.warning("plot_event_scatter: no heatwave events found; skipping scatter plot.")
        return

    yr = analysis_cfg["yr_start"]
    prefix = _month_year_prefix(analysis_cfg["mon_start"], yr)
    fpath = _out_fname("event_area_duration_scatter", varnames, yr, data_source, output_dir=output_dir)

    plot_xr_scatterplot(
        x_val=durations,
        y_val=max_areas,
        c_col=mean_anoms,
        fname=str(fpath),
        title=f"{prefix} - Event peak area vs duration",
        xlabel="Event duration (days)",
        ylabel="Event peak area (km²)",
        cbar_label="Mean anomaly (K)",
    )
    log.info("Saved event scatter plot → %s", fpath)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    varnames    = EAMXX_DATA_CONFIG["varnames"]
    data_source = EAMXX_DATA_CONFIG["data_source"]

    # ---- 1-2: compute ERA5 threshold once ----------------------------------
    ds_hist, ds_q90 = compute_threshold(ERA5_HIST_CONFIG, HIST_CONFIG, PERCENTILE_VALUE)

    # ---- 3-5: load EAMxx data, interpolate, build mask ---------------------
    heatwave_mask, anomaly, _, _ = load_and_prepare(
        EAMXX_DATA_CONFIG, ANALYSIS_CONFIG,
        ds_hist=ds_hist, ds_q90=ds_q90,
    )

    # Quick sanity checks
    log.info("heatwave_mask (land cell):  %s", heatwave_mask.sel(**LAND_CELL).values)
    log.info("heatwave_mask (ocean cell): %s", heatwave_mask.sel(**OCEAN_CELL).values)

    # ---- 6: heatwave-day frequency -----------------------------------------
    plot_frequency(heatwave_mask, varnames, ANALYSIS_CONFIG, FIG_EXTENT, data_source)

    # ---- 7-8: anomaly and max-anomaly map ----------------------------------
    # Step 7: mask anomaly to heatwave days only
    heatwave_anomaly = anomaly.where(heatwave_mask)
    # Step 8: plot maximum heatwave anomaly per grid cell
    plot_max_anomaly(heatwave_anomaly, varnames, ANALYSIS_CONFIG, FIG_EXTENT, data_source)

    # ---- 9: largest contiguous region per day ------------------------------
    plot_largest_contiguous_region(
        heatwave_mask, anomaly, varnames, ANALYSIS_CONFIG, FIG_EXTENT, data_source
    )

    # ---- 10: longest duration map ------------------------------------------
    plot_longest_duration(heatwave_mask, varnames, ANALYSIS_CONFIG, FIG_EXTENT, data_source)

    # ---- 11: 3-D event scatter ---------------------------------------------
    plot_event_scatter(heatwave_mask, anomaly, varnames, ANALYSIS_CONFIG, data_source)

    log.info("Done.")


if __name__ == "__main__":
    main()

