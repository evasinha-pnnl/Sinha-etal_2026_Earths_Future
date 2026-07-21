"""
compare_heatwaves.py
--------------------
Reads per-member time-series CSVs produced by run_heatwave_ensemble.py and
creates multi-member comparison line plots.

The ensemble member definitions and output directory are imported directly
from run_heatwave_ensemble so there is a single source of truth.

CSV layout (one file per member per metric):
  outputs/<data_source>/<data_source>_t2m_<tag>_<yr>.csv
  Columns: time, dayofyear, Data_Source, <value_col>

Output plots are written to:
  outputs/comparison/

Notes
-----
ERA5_obs covers the full month (days 1-31); EAMxx members cover days 17-24.
Both are plotted on the same day-of-year x-axis.  ERA5_obs is drawn as a
dashed line so it is visually distinct from the EAMxx ensemble members.

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
import pandas as pd
import seaborn as sns

# Reuse member definitions and output path from the ensemble runner
from run_heatwave_ensemble import ENSEMBLE_MEMBERS, OUTPUT_BASE_DIR
from util_constants import MEMBER_COLORS, ANOMALY_PAIRS, FONTSIZE, sorted_member_order, display_name

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

# X-axis limits: computed per year in _get_xaxis_limits()

# ---------------------------------------------------------------------------
# Timeseries metrics to compare
# ---------------------------------------------------------------------------
class TimeseriesMetric(NamedTuple):
    csv_tag:      str    # tag in the CSV filename
    value_col:    str    # column to plot on the y-axis
    ylabel:       str    # y-axis label
    out_fname_tag: str   # stem of the output PNG filename
    scale:        float = 1.0   # multiply values by this before plotting


TIMESERIES_METRICS: list[TimeseriesMetric] = [
    TimeseriesMetric(
        csv_tag       = "largest_contiguous_heatwave_area_timeseries",
        value_col     = "largest_region_area_km2",
        ylabel        = "Largest contiguous heatwave area\n(million km²)",
        out_fname_tag = "largest_contiguous_heatwave_area_comparison",
        scale         = 1e-6,   # km² → million km²
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
    df = load_ensemble_csvs(metric.csv_tag, varnames, year=year)

    # Apply optional unit scaling (e.g. km² → million km²)
    if metric.scale != 1.0:
        df = df.copy()
        df[metric.value_col] = df[metric.value_col] * metric.scale

    _ctrl_members = set(ANOMALY_PAIRS.values())
    obs_label = f"{OBS_LABEL_PREFIX}_{year}"
    x_min, x_max = _get_xaxis_limits(year)

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
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    ax.tick_params(labelsize=FONTSIZE - 1)

    n_sources = len(df["Data_Source"].unique())
    ax.legend(
        loc="upper center",
        ncol=min(5, n_sources),
        bbox_to_anchor=(0.5, -0.2),
        frameon=True,
        fontsize=FONTSIZE - 1,
    )

    out_path = COMPARISON_DIR / f"{metric.out_fname_tag}_{year}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved comparison plot → %s", out_path)
    out_csv = COMPARISON_DIR / f"{metric.out_fname_tag}_{year}.csv"
    df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
def plot_longest_duration_distribution(year: int, varnames: str = "t2m") -> None:
    """
    Read per-member longest-heatwave-duration CSVs and create a facet histogram
    plot comparing each ctrl/SSP245 pair for the given year.

    Layout: variable rows × 2 columns (one row per ANOMALY_PAIRS entry present in the year).
    Each facet shows two overlapping density plots — ctrl (dashed) and its
    paired SSP245 member (solid) — using the shared MEMBER_COLORS.
    
    Parameters
    ----------
    year     : int — analysis year (e.g., 2011 or 2021)
    varnames : str — variable name used in filenames (default 't2m')
    """
    csv_tag = "longest_heatwave_duration"
    value_col = "longest_duration"

    # Load one CSV per member (ctrl + SSP245), skip missing
    # Filter to only include members from the specified year
    member_data: dict[str, pd.Series] = {}
    year_members = {m.data_source for m in ENSEMBLE_MEMBERS if m.yr_start == year}
    
    for pgw, ctrl in ANOMALY_PAIRS.items():
        # Skip pairs where neither member exists in this year
        if pgw not in year_members and ctrl not in year_members:
            continue
            
        for src in (ctrl, pgw):
            if src not in year_members:
                continue
            yr = next(
                (m.yr_start for m in ENSEMBLE_MEMBERS if m.data_source == src), None
            )
            if yr is None:
                continue
            fname = (
                OUTPUT_BASE_DIR
                / src
                / f"{src}_{varnames}_{csv_tag}_{yr}.csv"
            )
            if not fname.exists():
                log.warning("CSV not found, skipping: %s", fname)
                continue
            df = pd.read_csv(fname)
            if value_col not in df.columns:
                log.warning("Column '%s' not found in %s; skipping.", value_col, fname)
                continue
            # Drop NaNs (ocean cells) and keep only land cells with duration > 0
            member_data[src] = df[value_col].dropna()
            log.info("Loaded %s", fname)

    # Filter to only pairs that have data loaded
    pairs = [(pgw, ctrl) for pgw, ctrl in ANOMALY_PAIRS.items() 
             if pgw in member_data or ctrl in member_data]
    ncols = 2
    nrows = -(-len(pairs) // ncols)  # ceiling division

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4 * nrows),
                             sharex=True)
    axes_flat = axes.flatten()

    # Hide any unused panels
    for idx in range(len(pairs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax, (pgw, ctrl) in zip(axes_flat, pairs):
        for src, alpha in [(ctrl, 0.4), (pgw, 0.6)]:
            data = member_data.get(src)
            if data is None or data.empty:
                continue
            sns.kdeplot(
                data,
                ax=ax,
                color=MEMBER_COLORS.get(src, "#333333"),
                fill=False,
                alpha=1.0,
                linewidth=2.0,
                linestyle="--" if src == ctrl else "-",
                label=display_name(src),
            )
        ax.set_ylabel("Density", fontsize=FONTSIZE + 2)
        ax.tick_params(labelsize=FONTSIZE + 2)
        ax.legend(fontsize=FONTSIZE + 2, frameon=True)

    axes_flat[-1].set_xlabel("Longest consecutive heatwave duration (days)", fontsize=FONTSIZE + 2)
    # Also label the second-to-last visible axis if nrows*ncols > len(pairs)
    if len(pairs) % ncols != 0:
        axes_flat[len(pairs) - 1].set_xlabel("Longest consecutive heatwave duration (days)", fontsize=FONTSIZE + 2)
    fig.suptitle(f"Distribution of longest heatwave duration per grid cell ({year})",
                 fontsize=FONTSIZE+2, y=1.01)
    fig.tight_layout()

    out_path = COMPARISON_DIR / f"longest_heatwave_duration_distribution_{year}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved duration distribution plot → %s", out_path)


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
            plot_comparison(metric, year=year, show_obs=False)
        
        plot_longest_duration_distribution(year=year)

    log.info("All comparison plots written to %s", COMPARISON_DIR)


if __name__ == "__main__":
    main()
