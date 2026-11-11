"""
plot_pgw_amplification.py
-------------------------
Event-specific amplification of the heat dome under pseudo-global warming (PGW).

The PGW simulations add a monthly climatological warming signal (Δ, SSP2-4.5
2045 minus present) to the initial / boundary state.  If the heat dome simply
shifted with the background warming, PGW daily-max temperature would equal
Control daily-max temperature + Δ.  The residual

    amplification = PGW − (Control + Δ)

isolates the event-specific response: > 0 where the heat dome warms more than
the imposed background warming, < 0 where it warms less.

The model field and the delta are always the same quantity (``--variable``):

  t2m (default)  EAMxx T_2m — the same files and field as the heatwave
                 analysis (identify_heatwaves.py) — with the 2 m air-
                 temperature delta ``tas``
  skt            EAMxx skin temperature, from the reprocessed outputs in
                 EAMXX_SKT_DIR/heatwave_<201107|2021>_<exp>_<pert>/ (util_constants.py),
                 with the skin-temperature delta ``skt``

Inputs
------
  • Daily maximum of the selected EAMxx field for each CTRL and SSP245
    member, read with ``identify_heatwaves.load_eamxx_daily_max`` and
    interpolated to the ERA5 CONUS grid.  Cached per member as
    ``outputs/<member>/<member>_<t2m|skt>_daily_max_<year>.nc``.
  • Monthly PGW delta, 12 months × 721 × 1440 (0.25°), files in VARIABLES:
      tas  WRFPGWSSP245final_bilinear_tas_forcing_ERA5_2045.nc — lat (S→N)
           and lon (0–360) are stored in the file as 2-D arrays
      skt  ERA5_PGWSSP245_WRF_2045.nc — no coordinate variables; lat / lon
           come from an ERA5 pressure-level file, with latitude flipped
           (delta is ordered S→N, ERA5 N→S)
    Each day gets the delta of its calendar month (June delta for June days,
    July delta for July days).

Statistics
----------
  Ensemble means are taken over the members defined in
  run_heatwave_ensemble.py for each year:
    2011 (17–29 July)       CTRL / SSP245 × p0, p12, p24, n12, n24
    2021 (25 June – 5 July) CTRL / SSP245 × p0, p24, n24
  Members are paired by perturbation label (SSP245_p0 − CTRL_p0, …).  Only
  ERA5 land cells are kept.  Time series are area-weighted land means over
  REGIONS[year]: the Pacific Northwest (41–50°N, 125–110°W) for 2021 and
  the Southern Plains / Midwest / East core (30–45°N, 105–75°W) for 2011,
  outlined on the 2011 (CONUS) maps.  The Δ map panel titles instead give
  the area-weighted land mean over CONUS (48 contiguous states + DC, from
  the cached Natural Earth 1:50m state polygons).

Output (outputs/pgw_amplification/)
------
  <var> = t2m | skt;  <dvar> = tas | skt
  pgw_amplification_<var>_map_<year>.png   Control + Δ | PGW | PGW − (Control + Δ),
                                           time-mean over the map window, stippled
                                           where all member pairs agree on the sign
  pgw_amplification_<var>_map_<year>.nc    the three maps + Δ + sign agreement
  pgw_amplification_<var>_timeseries_<year>.png / .csv
                                           area-weighted regional mean daily max
                                           (ensemble mean and min–max range) and
                                           the amplification
  pgw_delta_<dvar>_map_<year>.png / pgw_delta_<dvar>_<year>.nc
                                           the monthly Δ added to Control (one
                                           panel per month), land cells
  pgw_amplification_<var>_focus_<year>.png (years in FOCUS_REGIONS, i.e. 2011)
                                           (a) ensemble-mean PGW − (Control + Δ)
                                           over the map window, stippled where
                                           all pairs agree, focus boxes outlined;
                                           (b) daily land mean over the Amplified
                                           (42–49°N, 101–92°W) and Damped
                                           (30–40°N, 109–97°W; CO, NM, west TX)
                                           boxes, mean and min–max over pairs
  pgw_amplification_<var>_focus_timeseries_<year>.csv
                                           the daily box means in (b), per pair

Usage
-----
  python plot_pgw_amplification.py                                   # T2m, 2011 and 2021, full period
  python plot_pgw_amplification.py --variable skt                    # skin temperature
  python plot_pgw_amplification.py --years 2011
  python plot_pgw_amplification.py --years 2011 --start 2011-07-20 --end 2011-07-23
  python plot_pgw_amplification.py --years 2021 --start 2021-06-26 --end 2021-06-29
  python plot_pgw_amplification.py --overwrite-cache                 # re-read EAMxx

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import argparse
import calendar
import logging
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from identify_heatwaves import FIG_EXTENT, load_eamxx_daily_max, load_era5_land_mask
from plot_ensemble_stats import EXPERIMENT_LABELS, _horizontal_colorbar, _plot_map
from run_heatwave_ensemble import OUTPUT_BASE_DIR, PNW_FIG_EXTENT, _MEMBER_LOOKUP
from util_constants import EAMXX_SKT_DIR, PGW_DELTA_DIR
from util_vector_plots import plot_mean_range

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
# Configuration
# ---------------------------------------------------------------------------
CTRL, PGW = "CTRL", "SSP245"

# One font size for all text in this script's figures (titles, axis labels,
# tick labels, legends, footnotes).  Overrides the 15-pt defaults that
# util_spatial_plots / util_vector_plots set on import.  Map titles and
# colorbar ticks from plot_ensemble_stats are fixed at 20 pt.
FONTSIZE = 20
plt.rcParams.update({key: FONTSIZE for key in (
    "font.size", "figure.titlesize", "axes.titlesize", "axes.labelsize",
    "xtick.labelsize", "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
)})


# Model field and its matching PGW delta, keyed by --variable (also the name
# used in cache / output filenames).
#   eamxx_fpath    None → the member's own files and T_2m from
#                  run_heatwave_ensemble.py (same as the heatwave analysis);
#                  otherwise per-year directory template of the reprocessed
#                  outputs holding eamxx_varname
#   delta_coord_file  None → lat (S→N) / lon (0–360) are stored in the delta
#                  file as 2-D "latitude" / "longitude"; otherwise read from
#                  this ERA5 file and flipped (ERA5 is N→S, delta S→N)
#   levels         colour levels (K) of the Control + Δ and PGW maps
VARIABLES = {
    "t2m": dict(
        label="Daily max 2 m temperature",
        eamxx_fpath=None,
        eamxx_varname=None,
        delta_file=PGW_DELTA_DIR / "WRFPGWSSP245final_bilinear_tas_forcing_ERA5_2045.nc",
        delta_var="tas",
        delta_coord_file=None,
        levels=list(range(288, 322, 2)),
    ),
    "skt": dict(
        label="Daily max skin temperature",
        eamxx_fpath={
            2011: f"{EAMXX_SKT_DIR}/heatwave_201107_{{exp}}_{{pert}}/",
            2021: f"{EAMXX_SKT_DIR}/heatwave_2021_{{exp}}_{{pert}}/",
        },
        eamxx_varname="skt",
        delta_file=PGW_DELTA_DIR / "ERA5_PGWSSP245_WRF_2045.nc",
        delta_var="skt",
        delta_coord_file=(PGW_DELTA_DIR /
                          "e5.oper.an.pl.128_130_t.ll025sc.2021062300_2021062323.nc"),
        levels=list(range(291, 342, 3)),
    ),
}
DEFAULT_VARIABLE = "t2m"

# Map extent and time-series averaging box per heatwave year,
# [lon_min, lon_max, lat_min, lat_max] in 0–360 lon.  The 2011 box covers
# the Southern Plains / Midwest / East core of the July 2011 heatwave.
YEARS = [2011, 2021]
FIG_EXTENTS = {2011: FIG_EXTENT, 2021: PNW_FIG_EXTENT}
REGIONS     = {2011: [255, 285, 30, 45], 2021: PNW_FIG_EXTENT}

# Amplification-focus figure, made only for years listed here: boxes (same
# convention) over which the daily amplification is averaged.  Each box is
# outlined on the map in the colour of its time series.
FOCUS_REGIONS = {
    2011: {
        # Minnesota / eastern Dakotas: PGW warms more than Δ
        "Amplified":   dict(box=[259, 268, 42, 49], color="#c8453b"),
        # Colorado, New Mexico and west Texas: PGW warms less than Δ
        "Damped": dict(box=[251, 263, 30, 40], color="#2a6fb0"),
    },
}

# Map colour levels (K); Control + Δ / PGW levels are in VARIABLES
AMP_LEVELS  = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
AMP_CMAP    = "RdBu_r"
DELTA_LEVELS = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
DELTA_CMAP   = "Reds"

# Time-series colours (one per series, used for line and min–max band)
SERIES_COLORS = {"ctrl_plus_delta": "#2a6fb0", "pgw": "#c8453b", "amp": "#4d4d4d"}

# Member directory names look like 2021_CTRL_p0, 2021_SSP245_n24, …
MEMBER_RE = re.compile(r"^(?P<year>\d{4})_(?P<exp>CTRL|SSP245)_(?P<pert>[pn]\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_pgw_delta(var: str, target_grid: xr.DataArray, months: list[int]) -> xr.DataArray:
    """Monthly PGW delta for *var* (VARIABLES key) on the lat/lon of *target_grid*.

    Returns
    -------
    DataArray (month, lat, lon) in K
    """
    cfg = VARIABLES[var]
    with xr.open_dataset(cfg["delta_file"]) as ds:
        if cfg["delta_coord_file"] is not None:
            with xr.open_dataset(cfg["delta_coord_file"]) as ds_coord:
                lat = ds_coord["latitude"].values[::-1]   # ERA5 is N→S; delta is S→N
                lon = ds_coord["longitude"].values         # 0 … 359.75
        else:
            # Regular grid stored as 2-D arrays: S→N, 0 … 359.75
            lat = ds["latitude"].values[:, 0]
            lon = ds["longitude"].values[0, :]

        da = ds[cfg["delta_var"]]
        # Dims are (time, lat, lon) under file-specific names, e.g. Time /
        # south_north / west_east; the 12 steps are January … December.
        da = da.rename({old: new for old, new in zip(da.dims, ("month", "lat", "lon"))
                        if old != new})
        delta = da.assign_coords(month=np.arange(1, 13), lat=lat, lon=lon).sel(month=months)

        # Read only the target domain (0–360 lon) before converting to −180–180
        lon360 = target_grid.lon % 360
        delta = delta.sel(
            lat=slice(float(target_grid.lat.min()) - 1, float(target_grid.lat.max()) + 1),
            lon=slice(float(lon360.min()) - 1, float(lon360.max()) + 1),
        ).load()
    delta = delta.assign_coords(lon=((delta.lon + 180) % 360) - 180).sortby("lon")

    # Same 0.25° grid as ERA5 CONUS, so this is an exact match
    delta = delta.interp(lat=target_grid.lat, lon=target_grid.lon, method="linear")
    delta.name = f"delta_{cfg['delta_var']}"
    delta.attrs = dict(units="K", source=str(cfg["delta_file"]), variable=cfg["delta_var"])
    return delta


def load_conus_mask(target_grid: xr.DataArray) -> xr.DataArray:
    """Boolean (lat, lon) mask on *target_grid*, True for cell centres inside
    the 48 contiguous US states + DC (Natural Earth 1:50m state polygons)."""
    reader = shpreader.Reader(
        shpreader.natural_earth("50m", "cultural", "admin_1_states_provinces_lakes"))
    conus = shapely.union_all([
        rec.geometry for rec in reader.records()
        if rec.attributes["adm0_a3"] == "USA"
        and rec.attributes["name"] not in ("Alaska", "Hawaii")
    ])
    lon2d, lat2d = np.meshgrid(target_grid.lon.values, target_grid.lat.values)
    return xr.DataArray(
        shapely.contains_xy(conus, lon2d, lat2d),
        coords=dict(lat=target_grid.lat, lon=target_grid.lon), dims=("lat", "lon"),
        name="conus",
    )


def load_member_daily_max(
    var: str,
    data_source: str,
    target_grid: xr.DataArray,
    output_base: Path,
    overwrite_cache: bool = False,
) -> xr.DataArray:
    """Daily max of *var* (VARIABLES key) for one member on the ERA5 grid,
    cached as NetCDF.

    The member's file prefix and analysis window come from
    run_heatwave_ensemble.py.  For variables with an ``eamxx_fpath`` the
    directory and EAMxx variable are swapped for the reprocessed outputs.
    """
    cfg = VARIABLES[var]
    member = _MEMBER_LOOKUP[data_source]
    cache = output_base / data_source / f"{data_source}_{var}_daily_max_{member.yr_start}.nc"
    if cache.exists() and not overwrite_cache:
        log.info("Reading cached daily-max %s → %s", var, cache)
        with xr.open_dataarray(cache) as da:
            return da.load()

    eamxx_cfg = member.to_eamxx_cfg()
    if cfg["eamxx_fpath"] is not None:
        m = MEMBER_RE.match(data_source)
        eamxx_cfg |= dict(
            fpath=cfg["eamxx_fpath"][member.yr_start].format(exp=m["exp"], pert=m["pert"]),
            eamxx_varname=cfg["eamxx_varname"],
            varnames=var,
        )
    log.info("Reading EAMxx daily-max %s for %s from %s …",
             eamxx_cfg["eamxx_varname"], data_source, eamxx_cfg["fpath"])
    da = load_eamxx_daily_max(
        eamxx_cfg, member.to_analysis_cfg(), target_grid=target_grid,
    ).load()
    da = da.reset_coords(drop=True).rename(f"{var}_max")
    da.attrs = dict(long_name=cfg["label"], units="K", data_source=data_source,
                    source_dir=eamxx_cfg["fpath"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(cache)
    log.info("Saved daily-max %s cache → %s", var, cache)
    return da


def load_experiment(
    var: str,
    year: int,
    exp: str,
    target_grid: xr.DataArray,
    output_base: Path,
    overwrite_cache: bool,
) -> xr.DataArray:
    """Stack all members of *exp* for *year* along a ``member`` dim (perturbation label)."""
    das = []
    for label in _MEMBER_LOOKUP:
        m = MEMBER_RE.match(label)
        if m is None or int(m["year"]) != year or m["exp"] != exp:
            continue
        da = load_member_daily_max(var, label, target_grid, output_base, overwrite_cache)
        das.append(da.expand_dims(member=[m["pert"]]))
    if not das:
        raise ValueError(f"No {exp} members defined for {year} in run_heatwave_ensemble.py")
    # Keep only the days shared by every member
    return xr.concat(das, dim="member", join="inner")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_amplification(
    ctrl: xr.DataArray,
    pgw: xr.DataArray,
    delta_monthly: xr.DataArray,
) -> xr.Dataset:
    """Control + Δ, PGW and their difference, per member and per day.

    Returns
    -------
    Dataset (member, time, lat, lon) with ctrl_plus_delta, pgw, amp and the
    daily delta (time, lat, lon).  ``amp`` is the per-pair difference, so only
    perturbations present in both experiments are kept.
    """
    ctrl, pgw = xr.align(ctrl, pgw, join="inner")
    delta = delta_monthly.sel(month=ctrl.time.dt.month).drop_vars("month")
    ctrl_plus_delta = ctrl + delta
    return xr.Dataset(dict(
        ctrl_plus_delta=ctrl_plus_delta,
        pgw=pgw,
        amp=pgw - ctrl_plus_delta,
        delta=delta,
    ))


def time_mean_maps(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Ensemble- and time-mean maps over [start, end], plus sign agreement of
    the per-pair amplification."""
    win = ds.sel(time=slice(start, end))
    pair_amp = win["amp"].mean("time")
    n_pairs = pair_amp.sizes["member"]
    agree = ((pair_amp > 0).sum("member") == n_pairs) | ((pair_amp < 0).sum("member") == n_pairs)
    return xr.Dataset(
        dict(
            ctrl_plus_delta=win["ctrl_plus_delta"].mean(["member", "time"]),
            pgw=win["pgw"].mean(["member", "time"]),
            amp=pair_amp.mean("member"),
            delta=win["delta"].mean("time"),
            sign_agreement=agree.where(pair_amp.notnull().all("member"), False),
        ),
        attrs=dict(start=start, end=end, n_pairs=n_pairs,
                   pairs=", ".join(map(str, pair_amp.member.values))),
    )


def regional_mean(da: xr.DataArray, extent: list) -> xr.DataArray:
    """Area-weighted (cos lat) mean over the valid cells inside *extent*
    ([lon_min, lon_max, lat_min, lat_max], lon in 0–360 or −180–180).
    NaN cells (ocean, once land-masked) are skipped."""
    lon_min, lon_max = (((np.array(extent[:2]) + 180) % 360) - 180)
    box = da.sel(lat=slice(extent[2], extent[3]), lon=slice(lon_min, lon_max))
    weights = np.cos(np.deg2rad(box.lat))
    return box.weighted(weights).mean(["lat", "lon"])


def masked_mean(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Area-weighted (cos lat) mean over the valid cells where *mask* is True."""
    weights = np.cos(np.deg2rad(da.lat))
    return da.where(mask).weighted(weights).mean(["lat", "lon"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """e.g. '25 June – 5 July 2021'."""
    if start == end:
        return f"{start.day} {start:%B %Y}"
    return f"{start.day} {start:%B} – {end.day} {end:%B %Y}"


def _region_label(region: list) -> str:
    """e.g. '41–50°N, 125–110°W' for [235, 250, 41, 50]."""
    lon_w, lon_e, lat_s, lat_n = region
    return f"{lat_s}–{lat_n}°N, {360 - lon_w}–{360 - lon_e}°W"


def _draw_box(ax, box: list, **kwargs) -> None:
    """Outline *box* [lon_min, lon_max, lat_min, lat_max] on a map axis."""
    lon_w, lon_e, lat_s, lat_n = box
    ax.plot([lon_w, lon_e, lon_e, lon_w, lon_w],
            [lat_s, lat_s, lat_n, lat_n, lat_s],
            transform=ccrs.PlateCarree(), **kwargs)


def _outline_region(axes: list, region: list, fig_extent: list) -> bool:
    """Dashed outline of the time-series box on each of *axes* when it is
    smaller than the map extent.  Returns True if drawn."""
    if list(region) == list(fig_extent):
        return False
    for ax in axes:
        _draw_box(ax, region, color="black", linewidth=2, linestyle="--")
    return True


# Legend handler drawing a (band, line) tuple as one overlaid entry
TUPLE_HANDLER = {tuple: HandlerTuple(ndivide=1)}


def _mean_range_handles(ax) -> list[tuple]:
    """One legend handle per plot_mean_range series on *ax*: its min–max band
    and mean line paired (use with handler_map=TUPLE_HANDLER)."""
    handles, _ = ax.get_legend_handles_labels()
    return [tuple(handles[i:i + 2]) for i in range(0, len(handles), 2)]


def _stipple(ax, agree: xr.DataArray) -> None:
    """Stipple the cells where *agree* is True (call inside a hatch rc_context)."""
    if bool(agree.any()):
        ax.contourf(
            agree.lon, agree.lat, agree.astype(float), levels=[0.5, 1.5],
            colors="none", hatches=["...."], transform=ccrs.PlateCarree(),
        )


def plot_delta_map(
    delta_monthly: xr.DataArray, var: str, year: int, fig_extent: list,
    conus: xr.DataArray, out_fname: Path,
) -> None:
    """One panel per month: the PGW Δ added to Control, on ERA5 land cells.

    Panel titles give the area-weighted land mean over CONUS (*conus* mask).
    """
    months = list(delta_monthly.month.values)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(months), figsize=(10 * len(months), 7), squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = list(axes[0])
    fig.subplots_adjust(wspace=0.06, bottom=0.2)
    for ax, mon in zip(axes, months):
        da = delta_monthly.sel(month=mon)
        im = _plot_map(
            ax, da, DELTA_LEVELS,
            title=(f"{calendar.month_name[mon]} — CONUS land mean "
                   f"{float(masked_mean(da, conus)):.2f} K"),
            fig_extent=fig_extent, cmap_col=DELTA_CMAP,
        )

    _horizontal_colorbar(fig, im, axes, DELTA_LEVELS, "K", ticks=DELTA_LEVELS[::2])
    fig.text(0.5, 0.0, "Land mean over CONUS (48 contiguous states + DC)",
             ha="center", va="top")
    fig.suptitle(f"{year} - PGW Δ{VARIABLES[var]['delta_var']} (SSP2-4.5 2045 − present), "
                 f"added to Control (K)", y=1.0)
    fig.savefig(out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved delta map → %s", out_fname)


def plot_amplification_map(
    maps: xr.Dataset, var: str, fig_extent: list, region: list,
    title: str, out_fname: Path,
) -> None:
    """1 × 3 panel: Control + Δ | PGW | PGW − (Control + Δ), stippled where
    every member pair agrees on the sign of the amplification.  The
    time-series averaging box (*region*) is outlined when it is smaller than
    the map extent."""
    levels = VARIABLES[var]["levels"]
    delta_var = VARIABLES[var]["delta_var"]
    n_pairs = maps.attrs["n_pairs"]
    with plt.rc_context({"hatch.linewidth": 0.6, "hatch.color": "black"}):
        fig, axes = plt.subplots(
            nrows=1, ncols=3, figsize=(27, 7),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        fig.subplots_adjust(wspace=0.06, bottom=0.2)

        im_abs = _plot_map(
            axes[0], maps["ctrl_plus_delta"], levels,
            title=f"{EXPERIMENT_LABELS[CTRL]} + Δ{delta_var} — Mean (n={n_pairs})",
            fig_extent=fig_extent,
        )
        _plot_map(
            axes[1], maps["pgw"], levels,
            title=f"{EXPERIMENT_LABELS[PGW]} — Mean (n={n_pairs})",
            fig_extent=fig_extent,
        )
        im_amp = _plot_map(
            axes[2], maps["amp"], AMP_LEVELS,
            title=f"{EXPERIMENT_LABELS[PGW]} − ({EXPERIMENT_LABELS[CTRL]} + Δ)",
            fig_extent=fig_extent, cmap_col=AMP_CMAP,
        )
        _stipple(axes[2], maps["sign_agreement"])
        _horizontal_colorbar(fig, im_abs, list(axes[:2]), levels, "K",
                             ticks=levels[1:-1:2])
        _horizontal_colorbar(fig, im_amp, [axes[2]], AMP_LEVELS, "K",
                             ticks=AMP_LEVELS[::2])
        note = f"Stippling: all {n_pairs} member pairs agree on the sign of the amplification"
        if _outline_region(axes, region, fig_extent):
            note += "; dashed box: time-series region"
        fig.text(0.5, 0.0, note, ha="center", va="top")
        fig.suptitle(title, y=1.0)
        fig.savefig(out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved amplification map → %s", out_fname)


def plot_amplification_focus(
    maps: xr.Dataset,
    series: dict[str, pd.DataFrame],
    year: int,
    fig_extent: list,
    window: tuple[pd.Timestamp, pd.Timestamp],
    title: str,
    out_fname: Path,
) -> None:
    """Two panels:
    (a) ensemble-mean amplification over the map window, stippled where every
        member pair agrees on the sign, with the FOCUS_REGIONS boxes outlined;
    (b) daily amplification averaged over each box (*series*: name →
        DataFrame time × member), mean over pairs and min–max range.  The map
        window is shaded when it is shorter than the simulated period."""
    focus = FOCUS_REGIONS[year]
    n_pairs = maps.attrs["n_pairs"]
    with plt.rc_context({"hatch.linewidth": 0.6, "hatch.color": "black"}):
        # Map ≈ 1.65 : 1 (like the other CONUS maps), time series ≈ 1.3 : 1
        fig = plt.figure(figsize=(19, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.22,
                              left=0.04, right=0.97, bottom=0.2, top=0.88)
        ax_m = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_t = fig.add_subplot(gs[1])

        # ---- (a) map --------------------------------------------------------
        im = _plot_map(
            ax_m, maps["amp"], AMP_LEVELS,
            title=f"(a) Ensemble mean, {_period_label(*window)}",
            fig_extent=fig_extent, cmap_col=AMP_CMAP,
        )
        _stipple(ax_m, maps["sign_agreement"])
        for reg in focus.values():
            _draw_box(ax_m, reg["box"], color=reg["color"], linewidth=3)
        _horizontal_colorbar(fig, im, [ax_m], AMP_LEVELS, "K", ticks=AMP_LEVELS[::2])

        # ---- (b) daily series per box -------------------------------------------
        for name, reg in focus.items():
            table = series[name]
            plot_mean_range(ax_t, table.index, table, list(table.columns),
                            color=reg["color"], label=name, marker="o")
        handles = _mean_range_handles(ax_t)
        #labels = [f"{name} (mean; min–max of {n_pairs} pairs)" for name in focus]
        labels = [name for name in focus]
        times = series[next(iter(focus))].index
        if tuple(window) != (times[0], times[-1]):
            handles.append(ax_t.axvspan(window[0], window[1],
                                        color="0.9", zorder=0))
            labels.append("Map window")
        ax_t.axhline(0, color="black", linewidth=0.8)
        y_lo, y_hi = ax_t.get_ylim()
        ax_t.set_ylim(y_lo, y_hi + 0.2 * (y_hi - y_lo))   # headroom for the legend
        ax_t.set_ylabel(f"{EXPERIMENT_LABELS[PGW]} − ({EXPERIMENT_LABELS[CTRL]} + Δ) (K)")
        ax_t.set_title("(b) Daily land mean over each box")
        ax_t.legend(handles, labels, loc="upper right", handler_map=TUPLE_HANDLER)
        ax_t.grid(alpha=0.3)
        ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        plt.setp(ax_t.get_xticklabels(), rotation=30, ha="right")

        fig.text(0.5, 0.0,
                 f"(a) Stippling: all {n_pairs} member pairs agree on the sign; "
                 f"boxes: regions averaged in (b)",
                 ha="center", va="top")
        fig.suptitle(title, y=1.0)
        fig.savefig(out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved amplification focus figure → %s", out_fname)


def plot_amplification_timeseries(
    table: pd.DataFrame, var: str, members: list[str], title: str, out_fname: Path,
) -> None:
    """Two stacked panels sharing the time axis:
    top    – regional-mean daily max, Control + Δ and PGW (mean and min–max)
    bottom – PGW − (Control + Δ) (mean over pairs and min–max)."""
    x = table.index
    fig, (ax_t, ax_a) = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 12), sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08),
    )
    plot_mean_range(ax_t, x, table, [f"ctrl_plus_delta_{m}" for m in members],
                    color=SERIES_COLORS["ctrl_plus_delta"], label="ctrl_plus_delta", marker="o")
    plot_mean_range(ax_t, x, table, [f"pgw_{m}" for m in members],
                    color=SERIES_COLORS["pgw"], label="pgw", marker="o")
    y_lo, y_hi = ax_t.get_ylim()
    ax_t.set_ylim(y_lo, y_hi + 0.25 * (y_hi - y_lo))   # headroom for the legend
    ax_t.set_ylabel(f"{VARIABLES[var]['label']} (K)")
    ax_t.legend(_mean_range_handles(ax_t),
                [f"{EXPERIMENT_LABELS[CTRL]} + Δ", EXPERIMENT_LABELS[PGW]],
                loc="upper center", ncol=2, handler_map=TUPLE_HANDLER)
    ax_t.grid(alpha=0.3)

    plot_mean_range(ax_a, x, table, [f"amp_{m}" for m in members],
                    color=SERIES_COLORS["amp"], label="amp", marker="o")
    ax_a.axhline(0, color="black", linewidth=0.8)
    ax_a.set_ylabel("Amplification (K)")
    ax_a.grid(alpha=0.3)
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.align_ylabels([ax_t, ax_a])

    fig.text(0.5, 0.07,
             f"Lines: mean of {len(members)} members (bottom: member pairs); "
             f"shading: min–max\nAmplification = "
             f"{EXPERIMENT_LABELS[PGW]} − ({EXPERIMENT_LABELS[CTRL]} + Δ)",
             ha="center", va="top")
    fig.suptitle(title, y=0.95)
    fig.savefig(out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved amplification time series → %s", out_fname)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    var: str,
    year: int,
    start: str | None,
    end: str | None,
    output_base: Path,
    out_dir: Path,
    land: xr.DataArray,
    overwrite_cache: bool,
) -> None:
    """Compute, save and plot the amplification maps and time series of *var*
    for one year."""
    cfg = VARIABLES[var]
    delta_var = cfg["delta_var"]
    fig_extent = FIG_EXTENTS[year]
    region = REGIONS[year]

    ctrl = load_experiment(var, year, CTRL, land, output_base, overwrite_cache)
    pgw  = load_experiment(var, year, PGW,  land, output_base, overwrite_cache)
    months = sorted(set(ctrl.time.dt.month.values.tolist()))
    delta_monthly = load_pgw_delta(var, land, months)

    # ---- delta maps ---------------------------------------------------------
    delta_land = delta_monthly.where(land)
    conus = load_conus_mask(land)
    for mon in months:
        log.info("%d: Δ%s month %d, CONUS land mean: %.2f K", year, delta_var, mon,
                 float(masked_mean(delta_land.sel(month=mon), conus)))
    delta_land.to_netcdf(out_dir / f"pgw_delta_{delta_var}_{year}.nc")
    plot_delta_map(
        delta_land, var, year, fig_extent, conus,
        out_fname=out_dir / f"pgw_delta_{delta_var}_map_{year}.png",
    )

    ds = compute_amplification(ctrl, pgw, delta_monthly).where(land)
    members = list(map(str, ds.member.values))
    times = pd.DatetimeIndex(ds.time.values)
    log.info("%d: member pairs %s; days %s – %s", year, members,
             f"{times[0]:%Y-%m-%d}", f"{times[-1]:%Y-%m-%d}")

    # ---- maps: time mean over the requested window --------------------------
    start = pd.Timestamp(start) if start else times[0]
    end   = pd.Timestamp(end)   if end   else times[-1]
    maps = time_mean_maps(ds, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
    period = _period_label(start, end)
    tag = f"{year}" if (start, end) == (times[0], times[-1]) else \
          f"{start:%Y%m%d}-{end:%Y%m%d}"

    maps.assign_attrs(units="K", model_variable=var, delta_variable=delta_var,
                      delta_file=str(cfg["delta_file"])).to_netcdf(
        out_dir / f"pgw_amplification_{var}_map_{tag}.nc")
    plot_amplification_map(
        maps, var, fig_extent=fig_extent, region=region,
        title=f"{period} - {cfg['label']} (K): heat-dome amplification beyond the PGW Δ",
        out_fname=out_dir / f"pgw_amplification_{var}_map_{tag}.png",
    )

    # ---- amplification focus: map + daily series over the focus boxes -------
    if year in FOCUS_REGIONS:
        series = {
            name: regional_mean(ds["amp"], reg["box"]).transpose("time", "member").to_pandas()
            for name, reg in FOCUS_REGIONS[year].items()
        }
        for name, table in series.items():
            pair_means = table.loc[start:end].mean()
            log.info("%d: %s amplification over %s: %+.2f K (pairs %s)", year, name, period,
                     float(pair_means.mean()),
                     ", ".join(f"{m} {v:+.2f}" for m, v in pair_means.items()))
        focus_table = pd.concat(
            [table.add_prefix(f"{name.lower()}_") for name, table in series.items()], axis=1)
        focus_table.columns.name = None
        focus_table.to_csv(out_dir / f"pgw_amplification_{var}_focus_timeseries_{year}.csv")
        plot_amplification_focus(
            maps, series, year, fig_extent, window=(start, end),
            title=(f"{cfg['label']}: amplification beyond the PGW Δ{delta_var}, "
                   f"{EXPERIMENT_LABELS[PGW]} − ({EXPERIMENT_LABELS[CTRL]} + Δ)"),
            out_fname=out_dir / f"pgw_amplification_{var}_focus_{tag}.png",
        )

    # ---- time series: regional land mean per member -------------------------
    # Columns <field>_<member> for each field, then the daily regional Δ
    reg = regional_mean(ds, region)
    table = pd.concat(
        [reg[field].transpose("time", "member").to_pandas().add_prefix(f"{field}_")
         for field in ["ctrl_plus_delta", "pgw", "amp"]]
        + [reg["delta"].to_pandas().rename("delta")],
        axis=1,
    )
    table.columns.name = None
    table.to_csv(out_dir / f"pgw_amplification_{var}_timeseries_{year}.csv")

    amp_mean = table[[f"amp_{m}" for m in members]].mean(axis=1)
    log.info("%d: regional-mean amplification (ensemble mean): %s", year,
             ", ".join(f"{t:%m-%d} {v:+.2f} K" for t, v in amp_mean.items()))
    log.info("%d: regional-mean amplification over map window %s: %+.2f K",
             year, period, float(amp_mean.loc[start:end].mean()))

    plot_amplification_timeseries(
        table, var, members,
        title=f"{_period_label(times[0], times[-1])} - land mean {_region_label(region)}",
        out_fname=out_dir / f"pgw_amplification_{var}_timeseries_{year}.png",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Heat-dome amplification: PGW − (Control + Δ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variable", choices=list(VARIABLES), default=DEFAULT_VARIABLE,
                   help="Model field and matching delta: t2m = T_2m + Δtas, "
                        "skt = skin temperature + Δskt.")
    p.add_argument("--years", nargs="+", type=int, choices=YEARS, default=YEARS)
    p.add_argument("--start", help="First day of the map average (YYYY-MM-DD); "
                                   "default: first simulated day.  Single year only.")
    p.add_argument("--end", help="Last day of the map average (YYYY-MM-DD); "
                                 "default: last simulated day.  Single year only.")
    p.add_argument("--output-base", type=Path, default=OUTPUT_BASE_DIR,
                   help="Directory holding the per-member output folders.")
    p.add_argument("--overwrite-cache", action="store_true",
                   help="Re-read EAMxx data even if the daily-max cache exists.")
    args = p.parse_args()
    if (args.start or args.end) and len(args.years) > 1:
        p.error("--start / --end need a single year, e.g. --years 2021")

    out_dir = args.output_base / "pgw_amplification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ERA5 land mask doubles as the target grid
    land = load_era5_land_mask()

    for year in args.years:
        run(args.variable, year, args.start, args.end, args.output_base, out_dir, land,
            args.overwrite_cache)


if __name__ == "__main__":
    main()
