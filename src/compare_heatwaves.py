"""
compare_heatwaves.py
--------------------
Reads per-member CSVs produced by run_heatwave_ensemble.py and creates
Control vs PGW (SSP245) comparison plots for each heatwave year:

  * Time series of the largest contiguous heatwave region (area, mean and
    max anomaly).  Metrics with ``ensemble_summary=True`` show the ensemble
    mean with a min–max band; the others show one line per member.
  * Longest heatwave duration per land grid cell, as a KDE (ensemble mean
    with min–max band) and as a dot-and-whisker plot of the fraction of land
    cells at each whole-day duration.

The ensemble member definitions and output directory are imported directly
from run_heatwave_ensemble so there is a single source of truth.

CSV layout (one file per member per metric):
  outputs/<data_source>/<data_source>_t2m_<tag>_<yr>.csv

Output plots (and a CSV of the plotted values) are written to:
  outputs/comparison/

Usage
-----
  python compare_heatwaves.py

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import datetime
import logging
from typing import NamedTuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Reuse member definitions and output path from the ensemble runner
from identify_heatwaves import load_era5_land_mask
from run_heatwave_ensemble import ENSEMBLE_MEMBERS, OUTPUT_BASE_DIR
from util_constants import (
    ANOMALY_PAIRS, EXPERIMENT_STYLES, FONTSIZE, MEMBER_COLORS,
    display_name, members_by_experiment, sorted_member_order,
)
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
# Output directory for comparison plots
# ---------------------------------------------------------------------------
COMPARISON_DIR = OUTPUT_BASE_DIR / "comparison"

# Label prefix for observational baseline members
OBS_LABEL_PREFIX = "ERA5_obs"

# KDE settings — match seaborn.kdeplot defaults (Scott bandwidth, cut=3, gridsize=200)
KDE_CUT      = 3
KDE_GRIDSIZE = 200

# ---------------------------------------------------------------------------
# Timeseries metrics to compare
# ---------------------------------------------------------------------------
class TimeseriesMetric(NamedTuple):
    csv_tag:      str    # tag in the CSV filename
    value_col:    str    # column to plot on the y-axis
    ylabel:       str    # y-axis label
    out_fname_tag: str   # stem of the output PNG filename
    scale:        float = 1.0   # multiply values by this before plotting
    ensemble_summary: bool = False  # True → Control/PGW mean + range instead of one line per member


TIMESERIES_METRICS: list[TimeseriesMetric] = [
    TimeseriesMetric(
        csv_tag       = "largest_contiguous_heatwave_area_timeseries",
        value_col     = "largest_region_area_km2",
        ylabel        = "Largest contiguous heatwave area\n(million km²)",
        out_fname_tag = "largest_contiguous_heatwave_area_comparison",
        scale         = 1e-6,   # km² → million km²
        ensemble_summary = True,
    ),
    TimeseriesMetric(
        csv_tag       = "mean_anomaly_largest_contiguous_heatwave_area_timeseries",
        value_col     = "largest_region_mean_anomaly_K",
        ylabel        = "Mean anomaly (K)",
        out_fname_tag = "mean_anomaly_largest_contiguous_heatwave_area_comparison",
    ),
    TimeseriesMetric(
        csv_tag       = "max_anomaly_largest_contiguous_heatwave_area_timeseries",
        value_col     = "largest_region_max_anomaly_K",
        ylabel        = "Maximum anomaly (K)",
        out_fname_tag = "max_anomaly_largest_contiguous_heatwave_area_comparison",
    ),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _get_xaxis_limits(year: int) -> tuple[datetime.date, datetime.date]:
    """
    Compute x-axis limits for the given year based on EAMxx member date ranges.
    
    Parameters
    ----------
    year : int — analysis year (e.g., 2011 or 2021)
    
    Returns
    -------
    tuple[datetime.date, datetime.date] — (x_min, x_max)
    """
    year_members = [m for m in ENSEMBLE_MEMBERS 
                    if m.yr_start == year and not m.data_source.startswith(OBS_LABEL_PREFIX)]
    if not year_members:
        raise ValueError(f"No EAMxx members found for year {year}")
    
    yr = year_members[0].yr_start
    mon_start = year_members[0].mon_start
    mon_end = year_members[0].mon_end
    x_min = datetime.date(yr, mon_start, min(m.day_start for m in year_members))
    x_max = datetime.date(yr, mon_end, max(m.day_end for m in year_members))
    return x_min, x_max


def _color(src: str) -> str:
    """Return color for the given data source."""
    if src in MEMBER_COLORS:
        return MEMBER_COLORS[src]
    return "#333333"

# ---------------------------------------------------------------------------
def load_ensemble_csvs(csv_tag: str, varnames: str = "t2m", year: int | None = None) -> pd.DataFrame:
    """
    Read one CSV per ensemble member for the given tag and concatenate them.

    Parameters
    ----------
    csv_tag   : str  — the tag portion of the filename
    varnames  : str  — variable name used in the filename (default 't2m')
    year      : int | None — filter members by year (e.g., 2011 or 2021); 
                             if None, load all members

    Returns
    -------
    pd.DataFrame — concatenated rows from all members, with Data_Source and
                   dayofyear columns.
    """
    members = ENSEMBLE_MEMBERS if year is None else [m for m in ENSEMBLE_MEMBERS if m.yr_start == year]
    dfs = []
    for member in members:
        yr = member.yr_start
        fname = (
            OUTPUT_BASE_DIR
            / member.data_source
            / f"{member.data_source}_{varnames}_{csv_tag}_{yr}.csv"
        )
        if not fname.exists():
            log.warning("CSV not found, skipping: %s", fname)
            continue
        df = pd.read_csv(fname)
        # Ensure dayofyear column exists (written by _save_timeseries; derive
        # from time as a fallback for CSVs written by older versions)
        if "dayofyear" not in df.columns and "time" in df.columns:
            df["dayofyear"] = pd.to_datetime(df["time"]).dt.dayofyear
        # Derive a proper date column for plotting with date-formatted x-axis
        df["date"] = pd.to_datetime(df["dayofyear"].astype(int) - 1,
                                    unit="D",
                                    origin=f"{yr}-01-01")
        dfs.append(df)
        log.info("Loaded %s", fname)

    if not dfs:
        year_str = f" for year {year}" if year else ""
        raise FileNotFoundError(
            f"No CSV files found for tag '{csv_tag}'{year_str}. "
            f"Run run_heatwave_ensemble.py first."
        )
    return pd.concat(dfs, ignore_index=True)


def _load_metric(metric: TimeseriesMetric, year: int, varnames: str) -> pd.DataFrame:
    """Load the per-member CSVs for *metric* and apply its unit scaling."""
    df = load_ensemble_csvs(metric.csv_tag, varnames, year=year)
    if metric.scale != 1.0:
        df[metric.value_col] = df[metric.value_col] * metric.scale
    return df


def _format_date_axis(fig, ax, year: int) -> None:
    """Daily ticks over the EAMxx analysis period for *year*."""
    ax.set_xlim(*_get_xaxis_limits(year))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)


# ---------------------------------------------------------------------------
def plot_comparison(
    metric: TimeseriesMetric,
    year: int,
    varnames: str = "t2m",
    show_obs: bool = True,
) -> None:
    """
    Load CSVs for all members of a given year and save a multi-member comparison line plot.

    Styling rules
    -------------
    * **Observation** (ERA5_obs_YYYY): thick solid black line (when show_obs=True).
    * **PGW members** (YYYY_SSP245_*): solid line.
    * **Control members** (YYYY_CTRL_*): dashed line.
    * Each ctrl/PGW anomaly pair shares the same colour (from MEMBER_COLORS).

    Parameters
    ----------
    metric    : TimeseriesMetric — metric to plot.
    year      : int              — analysis year (e.g., 2011 or 2021).
    varnames  : str              — variable name used in filenames (default 't2m').
    show_obs  : bool             — whether to include ERA5_obs in the plot
                                   (default True).
    """
    df = _load_metric(metric, year, varnames)

    _ctrl_members = set(ANOMALY_PAIRS.values())
    obs_label = f"{OBS_LABEL_PREFIX}_{year}"

    fig, ax = plt.subplots(figsize=(11, 6))

    # ── EAMxx members ────────────────────────────────────────────────────────
    # Build order: ctrl_0, PGW_0, ctrl_1, PGW_1, … so that the 4-column
    # legend reads "ctrl row" on top and "PGW row" on the bottom.
    _all_model = {
        m.data_source for m in ENSEMBLE_MEMBERS 
        if m.yr_start == year and not m.data_source.startswith(OBS_LABEL_PREFIX)
    }
    model_sources = sorted_member_order(_all_model)
    for src in model_sources:
        sub = df[df["Data_Source"] == src]
        if sub.empty:
            continue
        is_ctrl = src in _ctrl_members
        ax.plot(
            sub["date"], sub[metric.value_col],
            label=display_name(src),
            color=_color(src),
            linewidth=2,
            linestyle="--" if is_ctrl else "-",
        )

    # ── ERA5_obs ─────────────────────────────────────────────────────────────
    if show_obs:
        df_obs = df[df["Data_Source"] == obs_label]
        if not df_obs.empty:
            ax.plot(
                df_obs["date"], df_obs[metric.value_col],
                label=display_name(obs_label),
                color="black",
                linewidth=3,
                linestyle="-",
            )

    ax.set_ylabel(metric.ylabel, fontsize=FONTSIZE)
    _format_date_axis(fig, ax, year)
    ax.tick_params(labelsize=FONTSIZE)

    # One column per ctrl/PGW pair present this year, so the legend lays out
    # as a "ctrl row" on top and a "PGW row" below (matches model_sources
    # ordering from sorted_member_order above).
    _year_pairs = [
        (pgw, ctrl) for pgw, ctrl in ANOMALY_PAIRS.items()
        if pgw in _all_model or ctrl in _all_model
    ]
    ax.legend(
        loc="upper center",
        ncol=max(len(_year_pairs), 1),
        bbox_to_anchor=(0.5, -0.2),
        frameon=True,
        fontsize=FONTSIZE,
    )

    out_path = COMPARISON_DIR / f"{metric.out_fname_tag}_{year}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved comparison plot → %s", out_path)
    out_csv = COMPARISON_DIR / f"{metric.out_fname_tag}_{year}.csv"
    df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
def plot_comparison_ensemble(
    metric: TimeseriesMetric,
    year: int,
    varnames: str = "t2m",
    show_obs: bool = True,
) -> None:
    """
    Ensemble-summary version of :func:`plot_comparison`: for the Control and
    PGW (SSP245) ensembles of the given year, the line is the ensemble mean
    on each day and the shaded band spans the ensemble min–max.

    Uses EXPERIMENT_STYLES from util_constants.  ERA5_obs is drawn as
    a thick black line when *show_obs* is True.  Per-member values plus the
    mean / min / max are written to ``<out_fname_tag>_<year>.csv``.

    Parameters
    ----------
    metric    : TimeseriesMetric — metric to plot.
    year      : int              — analysis year (e.g., 2011 or 2021).
    varnames  : str              — variable name used in filenames (default 't2m').
    show_obs  : bool             — whether to include ERA5_obs in the plot
                                   (default True).
    """
    df = _load_metric(metric, year, varnames)

    # rows = date, columns = member
    wide = df.pivot_table(index="date", columns="Data_Source",
                          values=metric.value_col, aggfunc="first")
    exp_members = members_by_experiment(wide.columns)

    fig, ax = plt.subplots(figsize=(11, 6))
    for exp, style in EXPERIMENT_STYLES.items():
        cols = exp_members[exp]
        if not cols:
            continue
        stats = plot_mean_range(ax, wide.index, wide, cols, style["color"], style["label"])
        wide[[f"{exp}_mean", f"{exp}_min", f"{exp}_max"]] = stats.to_numpy()

    obs_label = f"{OBS_LABEL_PREFIX}_{year}"
    if show_obs and obs_label in wide.columns:
        ax.plot(wide.index, wide[obs_label], label=display_name(obs_label),
                color="black", linewidth=3, linestyle="-")

    ax.set_ylabel(metric.ylabel, fontsize=FONTSIZE)
    ax.set_ylim(bottom=0)
    _format_date_axis(fig, ax, year)
    ax.tick_params(labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, frameon=True)

    out_path = COMPARISON_DIR / f"{metric.out_fname_tag}_{year}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    wide.to_csv(out_path.with_suffix(".csv"))
    log.info("Saved ensemble comparison plot → %s", out_path)


# ---------------------------------------------------------------------------
def _load_land_durations(
    year: int, varnames: str = "t2m",
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """
    Read per-member longest-heatwave-duration CSVs for *year*, keeping ERA5
    land cells only (land cells with no heatwave day stay as 0).

    Returns
    -------
    durations   : dict  member → 1-D array of per-cell longest duration (days)
    exp_members : dict  "CTRL" / "SSP245" → member labels for this year
    """
    csv_tag = "longest_heatwave_duration"
    value_col = "longest_duration"

    exp_members = members_by_experiment(
        m.data_source for m in ENSEMBLE_MEMBERS if m.yr_start == year
    )

    # ERA5 land mask: older member CSVs count coastal ocean cells as 0-day
    # durations instead of NaN, so filter to land explicitly.
    land = (
        load_era5_land_mask()
        .to_dataframe()
        .reset_index()[["lat", "lon", "era5_land"]]
    )

    durations: dict[str, np.ndarray] = {}
    for src in exp_members["CTRL"] + exp_members["SSP245"]:
        fname = OUTPUT_BASE_DIR / src / f"{src}_{varnames}_{csv_tag}_{year}.csv"
        if not fname.exists():
            log.warning("CSV not found, skipping: %s", fname)
            continue
        df = pd.read_csv(fname)
        if value_col not in df.columns:
            log.warning("Column '%s' not found in %s; skipping.", value_col, fname)
            continue
        df = df.merge(land, on=["lat", "lon"], how="left")
        durations[src] = (
            df.loc[df["era5_land"].fillna(False).astype(bool), value_col].dropna().to_numpy()
        )
        log.info("Loaded %s", fname)
    return durations, exp_members


def plot_longest_duration_distribution(year: int, varnames: str = "t2m") -> None:
    """
    Read per-member longest-heatwave-duration CSVs and plot, in a single panel,
    the ensemble mean and range of the duration KDE for the Control and SSP245
    ensembles of the given year.

    A Gaussian KDE (same settings as seaborn.kdeplot) is evaluated for every
    member on a common duration grid, using ERA5 land cells only.  For each
    experiment the line is the ensemble-mean density and the shaded band
    spans the ensemble min–max.

    The per-member densities plus the ensemble mean / min / max are also
    written to ``longest_heatwave_duration_distribution_<year>.csv``.

    Parameters
    ----------
    year     : int — analysis year (e.g., 2011 or 2021)
    varnames : str — variable name used in filenames (default 't2m')
    """
    durations, exp_members = _load_land_durations(year, varnames)

    # Per-member KDE (land cells only; cells with no heatwave day stay as 0)
    kdes: dict[str, gaussian_kde] = {}
    support_lo, support_hi = np.inf, -np.inf
    for src, data in durations.items():
        kde = gaussian_kde(data)
        bw = kde.factor * data.std(ddof=1)
        support_lo = min(support_lo, data.min() - KDE_CUT * bw)
        support_hi = max(support_hi, data.max() + KDE_CUT * bw)
        kdes[src] = kde

    if not kdes:
        log.warning("No longest-duration CSVs found for %d; skipping distribution plot.", year)
        return

    # rows = duration grid (days), columns = member density
    grid = np.linspace(support_lo, support_hi, KDE_GRIDSIZE)
    df_kde = pd.DataFrame({src: kde(grid) for src, kde in kdes.items()},
                          index=pd.Index(grid, name="duration_days"))

    fig, ax = plt.subplots(figsize=(11, 6))
    for exp, style in EXPERIMENT_STYLES.items():
        cols = [src for src in exp_members[exp] if src in df_kde.columns]
        if not cols:
            continue
        stats = plot_mean_range(ax, grid, df_kde, cols, style["color"], style["label"])
        df_kde[[f"{exp}_mean", f"{exp}_min", f"{exp}_max"]] = stats.to_numpy()

    ax.set_xlabel("Longest consecutive heatwave duration (days)", fontsize=FONTSIZE)
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, frameon=True)
    ax.set_title(f"Distribution of longest heatwave duration per grid cell ({year})",
                 fontsize=FONTSIZE)

    out_path = COMPARISON_DIR / f"longest_heatwave_duration_distribution_{year}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    df_kde.to_csv(out_path.with_suffix(".csv"))
    log.info("Saved duration distribution plot → %s", out_path)


# ---------------------------------------------------------------------------
def plot_longest_duration_fraction(year: int, varnames: str = "t2m") -> None:
    """
    Dot-and-whisker plot of the fraction of land grid cells (%) at each whole-day
    longest heatwave duration (0 = no heatwave day … full analysis period),
    for the Control and SSP245 ensembles of the given year.

    Unlike the KDE in :func:`plot_longest_duration_distribution`, this keeps
    durations discrete, so nothing is shown below 0 days or beyond the
    length of the analysis period.

    For each experiment the dot is the ensemble mean and the whiskers span
    the ensemble min–max; Control and PGW (SSP245) sit side by side in each
    bin.  Per-member fractions plus mean / min / max are written to
    ``longest_heatwave_duration_fraction_<year>.csv``.

    Parameters
    ----------
    year     : int — analysis year (e.g., 2011 or 2021)
    varnames : str — variable name used in filenames (default 't2m')
    """
    durations, exp_members = _load_land_durations(year, varnames)
    if not durations:
        log.warning("No longest-duration CSVs found for %d; skipping fraction plot.", year)
        return

    # rows = duration (days), columns = member; missing durations → 0 %
    max_dur = int(max(d.max() for d in durations.values()))
    days = np.arange(max_dur + 1)
    df_frac = pd.DataFrame(
        {src: pd.Series(np.round(d).astype(int)).value_counts(normalize=True)
                .reindex(days, fill_value=0.0) * 100
         for src, d in durations.items()},
    )
    df_frac.index.name = "duration_days"

    offsets = {"CTRL": -0.15, "SSP245": 0.15}   # side by side in each bin
    markers = {"CTRL": "o", "SSP245": "s"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for exp, style in EXPERIMENT_STYLES.items():
        cols = [src for src in exp_members[exp] if src in df_frac.columns]
        if not cols:
            continue
        mean = df_frac[cols].mean(axis=1)
        lo   = df_frac[cols].min(axis=1)
        hi   = df_frac[cols].max(axis=1)
        df_frac[f"{exp}_mean"] = mean
        df_frac[f"{exp}_min"]  = lo
        df_frac[f"{exp}_max"]  = hi

        ax.errorbar(
            days + offsets[exp], mean, yerr=[mean - lo, hi - mean],
            fmt=markers[exp], markersize=8, color=style["color"], ecolor=style["color"],
            elinewidth=2, capsize=4,
            label=f"{style['label']} mean and range (n={len(cols)})",
        )

    ax.set_xlabel("Longest consecutive heatwave duration (days)", fontsize=FONTSIZE)
    ax.set_ylabel("Fraction of land grid cells (%)", fontsize=FONTSIZE)
    ax.set_xticks(days)
    ax.set_xlim(-0.5, max_dur + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    # Vertical lines between day bins (at 0.5, 1.5, …) via unlabelled minor ticks
    ax.set_xticks(np.arange(0.5, max_dur), minor=True)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", which="minor", alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, frameon=True)
    ax.set_title(f"Longest heatwave duration per land grid cell ({year})",
                 fontsize=FONTSIZE)

    out_path = COMPARISON_DIR / f"longest_heatwave_duration_fraction_{year}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    df_frac.to_csv(out_path.with_suffix(".csv"))
    log.info("Saved duration fraction plot → %s", out_path)


# ---------------------------------------------------------------------------
def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which years have data
    years = sorted(set(m.yr_start for m in ENSEMBLE_MEMBERS if not m.data_source.startswith(OBS_LABEL_PREFIX)))
    log.info("Creating comparison plots for years: %s", years)

    # Create separate plots for each year
    for year in years:
        log.info("Processing year %d...", year)
        
        for metric in TIMESERIES_METRICS:
            if metric.ensemble_summary:
                plot_comparison_ensemble(metric, year=year, show_obs=False)
            else:
                plot_comparison(metric, year=year, show_obs=False)
        
        plot_longest_duration_distribution(year=year)
        plot_longest_duration_fraction(year=year)

    log.info("All comparison plots written to %s", COMPARISON_DIR)


if __name__ == "__main__":
    main()
