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

The spatial maps from steps 6, 8 and 10 are also written as NetCDF
(``<data_source>_<var>_<tag>_<yr>.nc``) for ensemble statistics in
``plot_ensemble_stats.py``.

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
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

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

# Select which heatwave event to analyse.
# Choices:
#   "2011" — July 2011 heatwave (July 17–24, EAMxx SSP245 simulation)
#   "2021" — June–July 2021 heatwave (June 24 – July 7, update paths below)
HEATWAVE_SELECTOR = "2021"

# ERA5 historical data — used only to build the percentile threshold
ERA5_HIST_CONFIG = dict(
    fpath="/pscratch/sd/e/esinha/ERA5_CONUS/",
    fname_abb="elmforc.ERA5.c2018.0.25d.t2m.",
    varnames="t2m",           # ERA5 variable name (also used as the common name)
)
HIST_CONFIG = dict(yr_start=1981, yr_end=2010, mon_start=1, mon_end=12)

# ── July 2011 heatwave ──────────────────────────────────────────────────────
# EAMxx simulation data — the analysis period
EAMXX_DATA_CONFIG_2011 = dict(
    fpath="/pscratch/sd/j/jli628/EAMxx_regridding/heatwave_201107_SSP245/",
    fname_abb="output.eamxx.INSTANT.nsteps_x9.",
    eamxx_varname="T_2m",     # native variable name inside the EAMxx files
    varnames="t2m",           # rename to this common name after reading
    data_source="2011_SSP245_p0",  # label used in output filenames and CSV columns
)
ANALYSIS_CONFIG_2011 = dict(
    yr_start=2011, yr_end=2011,
    mon_start=7,   mon_end=7,
    day_start=17,  day_end=29,   # day_start=17: first day is spin-up, skip it
)

# ── June–July 2021 heatwave (24 June – 2 July 2021) ────────────────────────
# Update fpath to point to the actual EAMxx data for this event.
EAMXX_DATA_CONFIG_2021 = dict(
    fpath="/pscratch/sd/j/jli628/EAMxx_regridding/heatwave_202106/",  # ← update path
    fname_abb="output.eamxx.INSTANT.nsteps_x9.",
    eamxx_varname="T_2m",
    varnames="t2m",
    data_source="EAMxx_2021hw",
)
# day_start / day_end refer to the boundary days within their respective months.
# Because the heatwave spans two months the loader splits the read into
# two segments (June 24–30, July 1–2) and concatenates them automatically.
ANALYSIS_CONFIG_2021 = dict(
    yr_start=2021, yr_end=2021,
    mon_start=6,   mon_end=7,
    day_start=25,  day_end=5,
)

# ── Active configs (resolved from HEATWAVE_SELECTOR) ───────────────────────
_HW_CONFIGS = {
    "2011": (EAMXX_DATA_CONFIG_2011, ANALYSIS_CONFIG_2011),
    "2021": (EAMXX_DATA_CONFIG_2021, ANALYSIS_CONFIG_2021),
}
if HEATWAVE_SELECTOR not in _HW_CONFIGS:
    raise ValueError(
        f"Unknown HEATWAVE_SELECTOR '{HEATWAVE_SELECTOR}'. "
        f"Valid choices: {list(_HW_CONFIGS)}"
    )
EAMXX_DATA_CONFIG, ANALYSIS_CONFIG = _HW_CONFIGS[HEATWAVE_SELECTOR]

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

def _date_month_year_prefix(stday: int, stmon: int, endday: int, endmon: int, yr: int) -> str:
    """Return a consistent plot-title prefix, e.g. '24 June 2021'."""
    return f"{stday} {calendar.month_name[stmon]} - {endday} {calendar.month_name[endmon]} {yr}"

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


def _save_netcdf(
    da: xr.DataArray,
    tag: str,
    varnames: str,
    yr: int,
    data_source: str,
    long_name: str,
    units: str,
    period: str,
    output_dir: Path | None = None,
) -> None:
    """Save a 2-D (lat, lon) map as NetCDF for ensemble statistics.

    The file sits next to the PNG of the same *tag* and is read by
    ``plot_ensemble_stats.py``.
    """
    fpath = _out_fname(tag, varnames, yr, data_source, ext="nc", output_dir=output_dir)
    da = da.copy()
    da.name = tag
    da.attrs = dict(
        long_name=long_name,
        units=units,
        data_source=data_source,
        period=period,
    )
    da.to_netcdf(fpath)
    log.info("Saved %s NetCDF → %s", tag, fpath)


def _plot_spatial_on_axis(
    ax: Any,
    da: xr.DataArray,
    fig_extent: list,
    levels: list | None = None,
    norm: colors.BoundaryNorm | None = None,
    title: str | None = None,
    title_fontsize: int = 10,
    add_colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """Plot spatial data on a cartopy axis (for use in subplots).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object (with cartopy projection) to plot on.
    da : xr.DataArray
        Data to plot.
    fig_extent : list
        [lon_min, lon_max, lat_min, lat_max] for map extent.
    levels : list | None
        Contour levels. If None, matplotlib will auto-determine.
    norm : colors.BoundaryNorm | None
        Color normalization. If None, linear normalization is used.
    title : str | None
        Title for the subplot.
    title_fontsize : int
        Font size for the title (default 10).
    add_colorbar : bool
        Whether to add a colorbar to this specific subplot (default False).
    vmin, vmax : float | None
        Explicit min/max values for color scaling (ignored if levels/norm provided).

    Returns
    -------
    im : mappable
        The plot object (for creating colorbars).
    """
    
    # Set extent
    ax.set_extent([fig_extent[0], fig_extent[1], fig_extent[2], fig_extent[3]], 
                  crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    
    # Plot the data
    plot_kwargs = {
        'ax': ax,
        'transform': ccrs.PlateCarree(),
        'cmap': cmap,
        'add_colorbar': add_colorbar,
        'add_labels': False,
    }

    # Add levels/norm if provided, otherwise use vmin/vmax
    if levels is not None:
        plot_kwargs['levels'] = levels
    if norm is not None:
        plot_kwargs['norm'] = norm
    if levels is None and norm is None:
        if vmin is not None:
            plot_kwargs['vmin'] = vmin
        if vmax is not None:
            plot_kwargs['vmax'] = vmax
    
    im = da.plot(**plot_kwargs)
    
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    
    return im


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
    # Load data into memory to avoid dask chunking issues with quantile
    ds_hist_loaded = ds_hist.load()
    ds_q90 = (
        ds_hist_loaded
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
    heatwave_mask : DataArray — True where Tmax > threshold; NaN where either
                    Tmax or the ERA5 threshold is missing (ERA5 ocean cells)
    anomaly       : DataArray — Tmax minus threshold (all times)
    """
    anomaly         = ds.groupby("time.dayofyear") - ds_q90
    above_threshold = ds.groupby("time.dayofyear") > ds_q90
    # Mask on the anomaly, not on ds alone: EAMxx has values over the ocean
    # where the ERA5 threshold is NaN, and "Tmax > NaN" would give False
    # (counted as a non-heatwave day) instead of NaN.
    heatwave_mask   = above_threshold.where(anomaly.notnull())
    return heatwave_mask, anomaly


def load_era5_land_mask(
    era5_hist_cfg: dict = ERA5_HIST_CONFIG,
    hist_cfg: dict = HIST_CONFIG,
) -> xr.DataArray:
    """Boolean (lat, lon) mask, True where ERA5 has data (land cells).

    Read from the first ERA5 historical file, on the same grid as the
    threshold and the per-member outputs.
    """
    fname = (
        f"{era5_hist_cfg['fpath']}{era5_hist_cfg['fname_abb']}"
        f"{hist_cfg['yr_start']:04d}-{hist_cfg['mon_start']:02d}_CONUS.nc"
    )
    with xr.open_dataset(fname) as ds:
        land = ds[era5_hist_cfg["varnames"]].isel(time=0).notnull().load()
    return land.reset_coords(drop=True).rename("era5_land")


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
    ds = load_eamxx_daily_max(eamxx_data_cfg, analysis_cfg, target_grid=ds_hist)
    heatwave_mask, anomaly = _build_mask_and_anomaly(ds, ds_q90)
    return heatwave_mask, anomaly, ds, ds_q90


def load_eamxx_daily_max(
    eamxx_data_cfg: dict,
    analysis_cfg: dict,
    target_grid: xr.DataArray,
) -> xr.DataArray:
    """
    Read EAMxx analysis data, interpolate to the lat/lon of *target_grid*
    (bilinear), and return daily maximum temperatures.

    Parameters
    ----------
    eamxx_data_cfg : dict
        Keys: fpath, fname_abb, eamxx_varname, varnames.
    analysis_cfg : dict
        Keys: yr_start, yr_end, mon_start, mon_end, day_start, day_end.
    target_grid : DataArray
        Any array on the ERA5 grid (only its lat / lon coordinates are used).

    Returns
    -------
    ds : DataArray (time, lat, lon) – EAMxx daily-max on the target grid
    """

    # --- EAMxx analysis data ------------------------------------------------
    mon_start = analysis_cfg["mon_start"]
    mon_end   = analysis_cfg["mon_end"]
    day_start = analysis_cfg["day_start"]
    day_end   = analysis_cfg["day_end"]
    yr_start  = analysis_cfg["yr_start"]
    yr_end    = analysis_cfg["yr_end"]

    if mon_start != mon_end:
        # Cross-month heatwave: split read into two segments and concatenate.
        # Segment 1 — from day_start to the last day of mon_start.
        last_day_seg1 = calendar.monthrange(yr_start, mon_start)[1]
        log.info(
            "Reading EAMxx data — segment 1: %d-%02d, days %d–%d …",
            yr_start, mon_start, day_start, last_day_seg1,
        )
        ds_a = read_EAMxx_data(
            yr_start, yr_start,
            mon_start, mon_start,
            day_start, last_day_seg1,
            eamxx_data_cfg["fpath"], eamxx_data_cfg["fname_abb"],
            eamxx_data_cfg["eamxx_varname"], eamxx_data_cfg["varnames"],
        )
        # Segment 2 — from day 1 to day_end in mon_end.
        log.info(
            "Reading EAMxx data — segment 2: %d-%02d, days 1–%d …",
            yr_end, mon_end, day_end,
        )
        ds_b = read_EAMxx_data(
            yr_end, yr_end,
            mon_end, mon_end,
            1, day_end,
            eamxx_data_cfg["fpath"], eamxx_data_cfg["fname_abb"],
            eamxx_data_cfg["eamxx_varname"], eamxx_data_cfg["varnames"],
        )
        ds = xr.concat([ds_a, ds_b], dim="time")
    else:
        log.info(
            "Reading EAMxx data (%d–%d, month %d, days %d–%d) …",
            yr_start, yr_end, mon_start, day_start, day_end,
        )
        ds = read_EAMxx_data(
            yr_start, yr_end,
            mon_start, mon_end,
            day_start, day_end,
            eamxx_data_cfg["fpath"], eamxx_data_cfg["fname_abb"],
            eamxx_data_cfg["eamxx_varname"], eamxx_data_cfg["varnames"],
        )
    log.info("EAMxx data loaded: %s", ds)

    # --- interpolate EAMxx to ERA5 lat/lon coordinates ----------------------
    log.info("Interpolating EAMxx data to ERA5 coordinates (bilinear) …")
    ds = ds.interp(lat=target_grid.lat, lon=target_grid.lon, method="linear")

    # --- daily maximum temperature ------------------------------------------
    ds = ds.resample(time="1D").max()

    # Normalise CFTimeIndex → numpy datetime64 so groupby("time.dayofyear")
    # and downstream pd.Timestamp usage work regardless of the NetCDF calendar.
    if hasattr(ds.indexes["time"], "to_datetimeindex"):
        ds = ds.assign_coords(time=ds.indexes["time"].to_datetimeindex())
    return ds


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
    prefix = _date_month_year_prefix(
        analysis_cfg["day_start"], analysis_cfg["mon_start"],
        analysis_cfg["day_end"], analysis_cfg["mon_end"],
        analysis_cfg["yr_start"]
    )
    _plot_spatial_map(
        freq_days, "frequency",
        title=f"{prefix} - Heatwave-day frequency (days)",
        out_fname=_out_fname("heatwave_frequency", varnames, analysis_cfg["yr_start"], data_source, output_dir=output_dir),
        fig_extent=fig_extent,
    )
    # fillna(False) above turns ocean cells into 0; restore NaN there in the
    # NetCDF so ocean does not count as "0 heatwave days" in ensemble stats.
    valid = heatwave_mask.notnull().any(dim="time")
    _save_netcdf(
        freq_days.where(valid), "heatwave_frequency", varnames,
        analysis_cfg["yr_start"], data_source,
        long_name="Heatwave-day frequency", units="days", period=prefix,
        output_dir=output_dir,
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
    prefix = _date_month_year_prefix(
        analysis_cfg["day_start"], analysis_cfg["mon_start"],
        analysis_cfg["day_end"], analysis_cfg["mon_end"],
        analysis_cfg["yr_start"]
    )
    _plot_spatial_map(
        max_anom, "anomaly",
        title=f"{prefix} - Maximum heatwave anomaly (K)",
        out_fname=_out_fname("max_heatwave_anomaly", varnames, analysis_cfg["yr_start"], data_source, output_dir=output_dir),
        fig_extent=fig_extent,
    )
    _save_netcdf(
        max_anom, "max_heatwave_anomaly", varnames,
        analysis_cfg["yr_start"], data_source,
        long_name="Maximum heatwave anomaly", units="K", period=prefix,
        output_dir=output_dir,
    )


def plot_daily_anomaly_panels(
    anomaly: xr.DataArray,
    varnames: str,
    analysis_cfg: dict,
    fig_extent: list,
    data_source: str,
    output_dir: Path | None = None,
) -> None:
    """Plot and save multi-panel figure with anomaly map for each day.
    
    Creates a grid of subplots with one panel per day in the analysis period,
    showing the full temperature anomaly (not masked to heatwave areas).
    Grid dimensions are automatically determined based on the number of days.
    
    Parameters
    ----------
    anomaly : xr.DataArray
        Temperature anomaly (Tmax - threshold) for all grid points and times.
    varnames : str
        Variable name for output filename.
    analysis_cfg : dict
        Analysis configuration with time period info.
    fig_extent : list
        Map extent [lon_min, lon_max, lat_min, lat_max].
    data_source : str
        Data source label for output filename.
    output_dir : Path | None
        Output directory for the figure.
    """

    
    # Get the number of days
    n_days = len(anomaly.time)
    if n_days == 0:
        log.warning("No time steps in anomaly; skipping daily anomaly panels.")
        return
    
    # Check for valid data
    if int(anomaly.notnull().sum()) == 0:
        log.warning("Anomaly array has no valid data; skipping daily anomaly panels.")
        return
    
    # Get data range for auto-level computation if needed
    data_min = float(anomaly.min())
    data_max = float(anomaly.max())
    
    # Size each panel to match the map's actual aspect ratio (lon span / lat span).
    # Cartopy locks the GeoAxes aspect to the extent, so a panel cell taller than
    # this ratio just leaves dead space above/below the map — which is what was
    # showing up as "too much space between rows".
    lon_span = fig_extent[1] - fig_extent[0]
    lat_span = fig_extent[3] - fig_extent[2]
    panel_w = 5
    panel_h = panel_w * lat_span / lon_span + 0.5  # + headroom for the per-panel title

    # Determine grid layout: search candidate column counts and pick the one that
    # gives the best overall figure aspect ratio (close to 16:9) while penalizing
    # empty trailing panels. This accounts for the panel's real (wide) aspect ratio
    # instead of just balancing subplot counts into a square-ish grid of panels.
    target_aspect = 16 / 9
    best_ncols, best_score = 1, None
    for ncols_candidate in range(1, n_days + 1):
        nrows_candidate = int(np.ceil(n_days / ncols_candidate))
        fig_aspect = (ncols_candidate * panel_w) / (nrows_candidate * panel_h)
        empty = nrows_candidate * ncols_candidate - n_days
        score = abs(np.log(fig_aspect / target_aspect)) + 0.1 * (empty / n_days)
        if best_score is None or score < best_score:
            best_score = score
            best_ncols = ncols_candidate
    ncols = best_ncols
    nrows = int(np.ceil(n_days / ncols))

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(panel_w * ncols, panel_h * nrows), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)

    # Get contour levels once (reuse for all panels)
    levels, norm = _make_clevels("anomaly")
    
    # If levels don't cover the data range, compute symmetric range and use that
    vmin, vmax = None, None
    if levels is not None:
        if data_max > max(levels) or data_min < min(levels):
            levels, norm = None, None
    
    # If not using predefined levels, compute symmetric vmin/vmax for anomaly data
    if levels is None:
        abs_max = max(abs(data_min), abs(data_max))
        vmin, vmax = -abs_max, abs_max
    
    prefix = _date_month_year_prefix(
        analysis_cfg["day_start"], analysis_cfg["mon_start"],
        analysis_cfg["day_end"], analysis_cfg["mon_end"],
        analysis_cfg["yr_start"]
    )
    
    # Plot each day
    for idx, time_val in enumerate(anomaly.time.values):
        row = idx // ncols
        col = idx % ncols
        
        ax = axes[row, col]

        # Select data for this day
        day_anom = anomaly.sel(time=time_val)
        
        # Convert time to pandas timestamp for formatting
        time_pd = pd.Timestamp(time_val)
        day_title = time_pd.strftime("%Y-%m-%d")
        
        # Use the helper function to plot on this axis
        im = _plot_spatial_on_axis(
            ax=ax,
            da=day_anom,
            fig_extent=fig_extent,
            levels=levels,
            norm=norm,
            title=day_title,
            title_fontsize=10,
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax,
        )
    
    # Hide any unused subplots
    for idx in range(n_days, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        ax.axis('off')

    # Tighten spacing between panels now that each cell matches the map's aspect ratio
    fig.subplots_adjust(hspace=0.15, wspace=0.05)

    # Add a single colorbar for all panels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Anomaly (K)', fontsize=12)
    
    # Add overall title
    fig.suptitle(f"{prefix} - Daily heatwave anomaly (K)", fontsize=14, y=0.98)
    
    # Save figure
    out_fname = _out_fname("daily_heatwave_anomaly_panels", varnames, 
                           analysis_cfg["yr_start"], data_source, output_dir=output_dir)
    fig.savefig(out_fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved daily anomaly panels → %s", out_fname)


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
    prefix = _date_month_year_prefix(
        analysis_cfg["day_start"], analysis_cfg["mon_start"],
        analysis_cfg["day_end"], analysis_cfg["mon_end"],
        yr
    )
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
    ).compute()  # computed once; reused for CSV, NetCDF and the plot
    # Save per-grid-cell longest duration to CSV for ensemble comparison
    csv_path = _out_fname("longest_heatwave_duration", varnames, analysis_cfg["yr_start"], data_source, ext="csv", output_dir=output_dir)
    df_dur = longest_duration.to_dataframe(name="longest_duration").reset_index()
    df_dur["Data_Source"] = data_source
    df_dur.to_csv(csv_path, index=False)
    log.info("Saved longest duration CSV → %s", csv_path)

    prefix = _date_month_year_prefix(
        analysis_cfg["day_start"], analysis_cfg["mon_start"],
        analysis_cfg["day_end"], analysis_cfg["mon_end"],
        analysis_cfg["yr_start"]
    )
    _plot_spatial_map(
        longest_duration, "duration",
        title=f"{prefix} - Longest heatwave duration (days)",
        out_fname=_out_fname("longest_heatwave_duration", varnames, analysis_cfg["yr_start"], data_source, output_dir=output_dir),
        fig_extent=fig_extent,
    )
    _save_netcdf(
        longest_duration, "longest_heatwave_duration", varnames,
        analysis_cfg["yr_start"], data_source,
        long_name="Longest heatwave duration", units="days", period=prefix,
        output_dir=output_dir,
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
    prefix = _date_month_year_prefix(
        analysis_cfg["day_start"], analysis_cfg["mon_start"],
        analysis_cfg["day_end"], analysis_cfg["mon_end"],
        yr
    )
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

