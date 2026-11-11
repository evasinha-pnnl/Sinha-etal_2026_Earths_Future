"""
plot_ensemble_stats.py
----------------------
Ensemble mean and standard deviation of per-member heatwave maps.

Reads the per-member NetCDF maps written by ``identify_heatwaves.py`` (via
``run_heatwave_ensemble.py``) from ``outputs/<data_source>/`` and, for each
heatwave year, groups the EAMxx members by experiment (CTRL, SSP245).
ERA5 observational members are not included.

Statistics (per grid cell, across members)
------------------------------------------
  Mean  : NaN-skipping mean over members
  Std   : NaN-skipping sample standard deviation (ddof=1); NaN where fewer
          than two members have a valid value
  Count : number of members with a valid value (saved to NetCDF only)

Members are masked to ERA5 land cells before the statistics are computed.

Figure layout (one figure per metric and year, same map style and colormap
as the per-member plots from util_spatial_plots.xr_plot_US)
----------------------------------------------------------------------
              Mean            Std
  Control     mean_CTRL       std_CTRL
  PGW         mean_SSP245     std_SSP245

A second figure shows the same result as a difference (one row):

  Control mean  |  PGW mean  |  PGW − Control (mean)

The difference panel is stippled where every Control/PGW member pair
(matched by perturbation, e.g. SSP245_p12 − CTRL_p12) has the same sign of
the difference; cells where any pair is missing are not stippled.

Output
------
  outputs/ensemble_stats/
    ensemble_stats_<metric>_<year>.png
    ensemble_stats_<metric>_<year>.nc   ← mean / std / count with an experiment dim
    ensemble_diff_<metric>_<year>.png
    ensemble_diff_<metric>_<year>.nc    ← mean difference and sign agreement

Usage
-----
  python plot_ensemble_stats.py                                   # max anomaly, all years
  python plot_ensemble_stats.py --metrics heatwave_frequency      # another metric
  python plot_ensemble_stats.py --years 2021

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr

from identify_heatwaves import FIG_EXTENT, load_era5_land_mask
from run_heatwave_ensemble import OUTPUT_BASE_DIR, PNW_FIG_EXTENT
from util_contour_levels import myDict_clevels
from util_spatial_plots import cmap

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
VARNAMES    = "t2m"
YEARS       = [2011, 2021]
EXPERIMENTS = ["CTRL", "SSP245"]

# Metric tag (as used in the per-member NetCDF filenames) → plot settings.
# Mean and std panels use the contour levels of the per-member plots,
# myDict_clevels[clevels_key], so the spread is read on the same scale as
# the signal.  diff_levels are for the PGW − Control panel: a one-sided
# scale from 0, since the difference is almost everywhere positive;
# negative cells fall in the lower extend colour (DIFF_UNDER_COLOR).
METRICS = {
    "max_heatwave_anomaly": dict(
        label="Maximum heatwave anomaly", units="K",
        clevels_key="anomaly", diff_levels=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    ),
    "heatwave_frequency": dict(
        label="Heatwave-day frequency", units="days",
        clevels_key="frequency", diff_levels=[0, 2, 4, 6, 8, 10, 12],
    ),
    "longest_heatwave_duration": dict(
        label="Longest heatwave duration", units="days",
        clevels_key="duration", diff_levels=[0, 2, 4, 6, 8, 10, 12],
    ),
}
DIFF_CMAP        = "YlOrRd"
DIFF_UNDER_COLOR = "#2166ac"   # blue: cells where PGW − Control < 0
DEFAULT_METRICS = ["max_heatwave_anomaly"]

# Row labels for the figure
EXPERIMENT_LABELS = {"CTRL": "Control", "SSP245": "PGW"}

# Member directory names look like 2011_CTRL_p0, 2021_SSP245_n24, …
MEMBER_RE = re.compile(r"^(?P<year>\d{4})_(?P<exp>CTRL|SSP245)_(?P<pert>[pn]\d+)$")


# ---------------------------------------------------------------------------
# Loading and statistics
# ---------------------------------------------------------------------------
def load_members(
    output_base: Path,
    metric: str,
    year: int,
    land: xr.DataArray,
) -> tuple[dict[str, xr.DataArray], str | None]:
    """Stack per-member NetCDF maps along a ``member`` dim, one array per experiment.

    Each member is masked to ERA5 land cells (*land*), since older member
    files count coastal ocean as 0 heatwave days rather than NaN.

    Returns
    -------
    members : dict  experiment → DataArray (member, lat, lon)
    period  : str | None  analysis-period label taken from the file attributes
    """
    by_exp: dict[str, list[xr.DataArray]] = {exp: [] for exp in EXPERIMENTS}
    period = None
    for member_dir in sorted(p for p in output_base.iterdir() if p.is_dir()):
        m = MEMBER_RE.match(member_dir.name)
        if m is None or int(m["year"]) != year:
            continue
        fpath = member_dir / f"{member_dir.name}_{VARNAMES}_{metric}_{year}.nc"
        if not fpath.exists():
            log.warning("Missing %s — re-run run_heatwave_ensemble.py %s", fpath, member_dir.name)
            continue
        # units="days" would otherwise be decoded as timedelta64
        with xr.open_dataarray(fpath, decode_timedelta=False) as da:
            da = da.load()
        period = period or da.attrs.get("period")
        da = da.where(land.reindex_like(da, method="nearest", tolerance=1e-6))
        by_exp[m["exp"]].append(da.expand_dims(member=[m["pert"]]))

    members = {
        exp: xr.concat(das, dim="member")
        for exp, das in by_exp.items() if das
    }
    for exp, da in members.items():
        log.info("%s %d %s: members %s", metric, year, exp, list(da.member.values))
    return members, period


def compute_stats(da: xr.DataArray) -> xr.Dataset:
    """NaN-skipping mean, sample std (ddof=1) and valid-member count over ``member``."""
    with warnings.catch_warnings():
        # std over < 2 valid members → NaN with a "degrees of freedom" warning
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = da.mean(dim="member", skipna=True)
        std  = da.std(dim="member", ddof=1, skipna=True)
    count = da.notnull().sum(dim="member")
    return xr.Dataset(
        dict(mean=mean, std=std, count=count),
        attrs=dict(members=", ".join(map(str, da.member.values))),
    )


def compute_diff(members: dict[str, xr.DataArray], stats: dict[str, xr.Dataset]) -> xr.Dataset:
    """PGW − Control difference of the ensemble means, plus sign agreement.

    Sign agreement uses the member pairs matched by perturbation label
    (SSP245_p12 − CTRL_p12, …).  ``agree`` is True only where every pair has
    a valid value and all pair differences are > 0, or all are < 0.
    """
    diff_mean = stats["SSP245"]["mean"] - stats["CTRL"]["mean"]
    # Subtraction aligns on the shared "member" labels → one value per pair
    pair_diff = members["SSP245"] - members["CTRL"]
    n_pos = (pair_diff > 0).sum(dim="member")
    n_neg = (pair_diff < 0).sum(dim="member")
    n_pairs = pair_diff.sizes["member"]
    agree = (n_pos == n_pairs) | (n_neg == n_pairs)
    return xr.Dataset(
        dict(diff_mean=diff_mean, n_pairs_positive=n_pos,
             n_pairs_negative=n_neg, sign_agreement=agree),
        attrs=dict(n_pairs=n_pairs, pairs=", ".join(map(str, pair_diff.member.values))),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_map(
    ax, da: xr.DataArray, levels: list, title: str, fig_extent: list, cmap_col=cmap,
):
    """Draw one map in the style of util_spatial_plots.xr_plot_US."""
    im = da.plot(
        ax=ax, transform=ccrs.PlateCarree(), levels=levels, cmap=cmap_col,
        extend="both", add_colorbar=False, add_labels=False,
    )
    ax.set_title(title, fontsize=20)
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor="gainsboro")
    ax.add_feature(cfeature.LAKES, edgecolor="black")
    ax.add_feature(cfeature.STATES, edgecolor="black")
    ax.set_extent(fig_extent)
    ax.set_aspect("auto")
    return im


def plot_ensemble_panel(
    stats: dict[str, xr.Dataset],
    metric: str,
    year: int,
    period: str | None,
    fig_extent: list,
    out_fname: Path,
) -> None:
    """2 × 2 panel: rows = Control, PGW; columns = Mean, Std.

    All four panels share one colorbar (the per-member plot levels).
    """
    cfg = METRICS[metric]
    levels = myDict_clevels[cfg["clevels_key"]]
    columns = [("Mean", "mean"), ("Std", "std")]

    fig, axes = plt.subplots(
        nrows=len(EXPERIMENTS), ncols=len(columns), figsize=(18, 10.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(hspace=0.15, wspace=0.08)

    for c, (stat_name, var) in enumerate(columns):
        for r, exp in enumerate(EXPERIMENTS):
            ds = stats[exp]
            n = len(ds.attrs["members"].split(", "))
            im = _plot_map(
                axes[r, c], ds[var], levels,
                title=f"{EXPERIMENT_LABELS[exp]} — {stat_name} (n={n})",
                fig_extent=fig_extent,
            )

    # One vertical colorbar for all panels, labelled like the per-member
    # plots (ticks on the interior levels)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical",
                        fraction=0.02, pad=0.02, aspect=40)
    cbar.set_ticks(levels[1:-1])
    cbar.set_ticklabels([f"{lev:g}" for lev in levels[1:-1]])
    cbar.ax.tick_params(labelsize=20, length=0)
    cbar.ax.set_title(cfg["units"], fontsize=20, pad=15)  # e.g. "K", above the extend arrow

    title = f"{cfg['label']} ({cfg['units']}) — ensemble statistics"
    fig.suptitle(f"{period} - {title}" if period else f"{year} {title}",
                 fontsize=20, y=0.95)
    fig.savefig(out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ensemble panel → %s", out_fname)


def _horizontal_colorbar(fig, im, axes: list, levels: list, units: str, ticks: list):
    """Horizontal colorbar spanning *axes*, placed just below them."""
    x0 = axes[0].get_position().x0
    x1 = axes[-1].get_position().x1
    y0 = min(ax.get_position().y0 for ax in axes)
    width = x1 - x0
    cax = fig.add_axes([x0 + 0.1 * width, y0 - 0.09, 0.8 * width, 0.035])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{lev:g}" for lev in ticks])
    cbar.ax.tick_params(labelsize=20, length=0)
    cbar.ax.text(1.06, 0.5, units, transform=cbar.ax.transAxes,
                 fontsize=20, ha="left", va="center")
    return cbar


def plot_diff_panel(
    stats: dict[str, xr.Dataset],
    diff: xr.Dataset,
    metric: str,
    year: int,
    period: str | None,
    fig_extent: list,
    out_fname: Path,
) -> None:
    """1 × 3 panel: Control mean | PGW mean | PGW − Control, stippled where
    all member pairs agree on the sign of the difference."""
    cfg = METRICS[metric]
    levels = myDict_clevels[cfg["clevels_key"]]
    diff_levels = cfg["diff_levels"]
    n_pairs = diff.attrs["n_pairs"]

    # Thin, dense stippling that does not hide the colours underneath
    with plt.rc_context({"hatch.linewidth": 0.6, "hatch.color": "black"}):
        fig, axes = plt.subplots(
            nrows=1, ncols=3, figsize=(27, 7),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        fig.subplots_adjust(wspace=0.06, bottom=0.2)

        for ax, exp in zip(axes[:2], EXPERIMENTS):
            ds = stats[exp]
            n = len(ds.attrs["members"].split(", "))
            im_mean = _plot_map(
                ax, ds["mean"], levels,
                title=f"{EXPERIMENT_LABELS[exp]} — Mean (n={n})",
                fig_extent=fig_extent,
            )

        im_diff = _plot_map(
            axes[2], diff["diff_mean"], diff_levels,
            title=f"{EXPERIMENT_LABELS['SSP245']} − {EXPERIMENT_LABELS['CTRL']} (mean)",
            fig_extent=fig_extent, cmap_col=DIFF_CMAP,
        )
        # Make negative differences stand out from the one-sided scale
        diff_cmap = im_diff.get_cmap().copy()
        diff_cmap.set_under(DIFF_UNDER_COLOR)
        im_diff.set_cmap(diff_cmap)
        agree = diff["sign_agreement"]
        if bool(agree.any()):
            axes[2].contourf(
                agree.lon, agree.lat, agree.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["...."], transform=ccrs.PlateCarree(),
            )

        _horizontal_colorbar(fig, im_mean, list(axes[:2]), levels, cfg["units"],
                             ticks=levels[1:-1])
        _horizontal_colorbar(fig, im_diff, [axes[2]], diff_levels, cfg["units"],
                             ticks=diff_levels)
        fig.text(
            0.5, 0.0,
            f"Stippling: all {n_pairs} {EXPERIMENT_LABELS['CTRL']}/"
            f"{EXPERIMENT_LABELS['SSP245']} member pairs agree on the sign of the difference",
            ha="center", va="top", fontsize=18,
        )

        title = f"{cfg['label']} ({cfg['units']}) — ensemble mean and difference"
        fig.suptitle(f"{period} - {title}" if period else f"{year} {title}",
                     fontsize=20, y=1.0)
        fig.savefig(out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ensemble difference panel → %s", out_fname)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    metric: str, year: int, output_base: Path, stats_dir: Path, land: xr.DataArray,
) -> bool:
    """Compute, save and plot ensemble statistics for one metric and year."""
    members, period = load_members(output_base, metric, year, land)
    missing = [exp for exp in EXPERIMENTS if exp not in members]
    if missing:
        log.warning("%s %d: no members found for %s; skipping.", metric, year, missing)
        return False

    stats = {exp: compute_stats(da) for exp, da in members.items()}

    # Save statistics with an experiment dimension
    ds_out = xr.concat(
        [stats[exp] for exp in EXPERIMENTS],
        dim=pd.Index(EXPERIMENTS, name="experiment"),
        combine_attrs="drop",
    )
    for exp in EXPERIMENTS:
        ds_out.attrs[f"members_{exp}"] = stats[exp].attrs["members"]
    ds_out.attrs.update(
        metric=metric, units=METRICS[metric]["units"], period=period or str(year),
        std_ddof=1, nan_handling="skipna over members; ERA5 land cells only",
    )
    nc_path = stats_dir / f"ensemble_stats_{metric}_{year}.nc"
    ds_out.to_netcdf(nc_path)
    log.info("Saved ensemble stats NetCDF → %s", nc_path)

    fig_extent = PNW_FIG_EXTENT if year == 2021 else FIG_EXTENT
    plot_ensemble_panel(
        stats, metric, year, period, fig_extent,
        out_fname=stats_dir / f"ensemble_stats_{metric}_{year}.png",
    )

    # ---- difference view: Control mean | PGW mean | PGW − Control --------
    diff = compute_diff(members, stats)
    log.info("%s %d: sign agreement over %d pairs (%s); %.1f%% of valid cells stippled",
             metric, year, diff.attrs["n_pairs"], diff.attrs["pairs"],
             100 * float(diff["sign_agreement"].where(diff["diff_mean"].notnull()).mean()))
    diff_nc = stats_dir / f"ensemble_diff_{metric}_{year}.nc"
    diff.assign_attrs(metric=metric, units=METRICS[metric]["units"],
                      period=period or str(year)).to_netcdf(diff_nc)
    log.info("Saved ensemble difference NetCDF → %s", diff_nc)
    plot_diff_panel(
        stats, diff, metric, year, period, fig_extent,
        out_fname=stats_dir / f"ensemble_diff_{metric}_{year}.png",
    )
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot ensemble mean / std of per-member heatwave maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metrics", nargs="+", choices=list(METRICS), default=DEFAULT_METRICS)
    p.add_argument("--years", nargs="+", type=int, default=YEARS)
    p.add_argument("--output-base", type=Path, default=OUTPUT_BASE_DIR,
                   help="Directory holding the per-member output folders.")
    args = p.parse_args()

    stats_dir = args.output_base / "ensemble_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    land = load_era5_land_mask()

    for metric in args.metrics:
        for year in args.years:
            run(metric, year, args.output_base, stats_dir, land)


if __name__ == "__main__":
    main()
