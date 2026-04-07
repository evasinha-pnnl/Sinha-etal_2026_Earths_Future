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
ERA5_obs covers the full month (days 1–31); EAMxx members cover days 17–24.
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

# Reuse member definitions and output path from the ensemble runner
from run_heatwave_ensemble import ENSEMBLE_MEMBERS, OUTPUT_BASE_DIR

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

# Label of the observational baseline member (drawn as dashed)
OBS_LABEL = "ERA5_obs"

# X-axis limits: derived from the day range of the EAMxx members so the
# comparison plot always zooms in to the shared analysis window.
# EAMxx members are all non-ERA5 members; we take the union of their ranges.
_eamxx_members = [m for m in ENSEMBLE_MEMBERS if m.data_source != OBS_LABEL]
_eamxx_yr   = _eamxx_members[0].yr_start
_eamxx_mon  = _eamxx_members[0].mon_start
_XAXIS_MIN = datetime.date(_eamxx_yr, _eamxx_mon, min(m.day_start for m in _eamxx_members))
_XAXIS_MAX = datetime.date(_eamxx_yr, _eamxx_mon, max(m.day_end   for m in _eamxx_members))

# ---------------------------------------------------------------------------
# Timeseries metrics to compare
# ---------------------------------------------------------------------------
class TimeseriesMetric(NamedTuple):
    csv_tag:      str   # tag in the CSV filename
    value_col:    str   # column to plot on the y-axis
    ylabel:       str   # y-axis label
    title:        str   # plot title
    out_fname_tag: str  # stem of the output PNG filename


TIMESERIES_METRICS: list[TimeseriesMetric] = [
    TimeseriesMetric(
        csv_tag       = "largest_contiguous_heatwave_area_timeseries",
        value_col     = "largest_region_area_km2",
        ylabel        = "Heatwave area (km²)",
        title         = "Largest contiguous heatwave area",
        out_fname_tag = "largest_contiguous_heatwave_area_comparison",
    ),
    TimeseriesMetric(
        csv_tag       = "mean_anomaly_largest_contiguous_heatwave_area_timeseries",
        value_col     = "largest_region_mean_anomaly_K",
        ylabel        = "Mean anomaly (K)",
        title         = "Mean anomaly for the largest contiguous heatwave area",
        out_fname_tag = "mean_anomaly_largest_contiguous_heatwave_area_comparison",
    ),
    TimeseriesMetric(
        csv_tag       = "max_anomaly_largest_contiguous_heatwave_area_timeseries",
        value_col     = "largest_region_max_anomaly_K",
        ylabel        = "Maximum anomaly (K)",
        title         = "Maximum anomaly for the largest contiguous heatwave area",
        out_fname_tag = "max_anomaly_largest_contiguous_heatwave_area_comparison",
    ),
]


# ---------------------------------------------------------------------------
def load_ensemble_csvs(csv_tag: str, varnames: str = "t2m") -> pd.DataFrame:
    """
    Read one CSV per ensemble member for the given tag and concatenate them.

    Parameters
    ----------
    csv_tag   : str  — the tag portion of the filename
    varnames  : str  — variable name used in the filename (default 't2m')

    Returns
    -------
    pd.DataFrame — concatenated rows from all members, with Data_Source and
                   dayofyear columns.
    """
    dfs = []
    for member in ENSEMBLE_MEMBERS:
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
        raise FileNotFoundError(
            f"No CSV files found for tag '{csv_tag}'. "
            f"Run run_heatwave_ensemble.py first."
        )
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
def plot_comparison(metric: TimeseriesMetric, varnames: str = "t2m") -> None:
    """
    Load CSVs for all members and save a multi-member comparison line plot.

    ERA5_obs is drawn as a dashed line to distinguish the observational
    baseline from the EAMxx PGW ensemble members (solid lines).
    """
    df = load_ensemble_csvs(metric.csv_tag, varnames)

    # Split into obs and model members
    df_obs   = df[df["Data_Source"] == OBS_LABEL]
    df_model = df[df["Data_Source"] != OBS_LABEL]

    fig, ax = plt.subplots(figsize=(11, 6))

    # --- plot EAMxx members (solid lines) -----------------------------------
    model_sources = df_model["Data_Source"].unique()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, src in enumerate(model_sources):
        sub = df_model[df_model["Data_Source"] == src]
        ax.plot(
            sub["date"], sub[metric.value_col],
            label=src,
            color=color_cycle[i % len(color_cycle)],
            linewidth=2,
        )

    # --- plot ERA5_obs (dashed black line) ----------------------------------
    if not df_obs.empty:
        ax.plot(
            df_obs["date"], df_obs[metric.value_col],
            label=OBS_LABEL,
            color="black",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel(metric.ylabel, fontsize=14)
    ax.set_title(metric.title, fontsize=15)
    ax.set_xlim(_XAXIS_MIN, _XAXIS_MAX)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=30)
    ax.tick_params(labelsize=13)
    n_sources = len(df["Data_Source"].unique())
    ax.legend(
        loc="upper center",
        ncol=min(4, n_sources),
        bbox_to_anchor=(0.5, -0.15),
        frameon=True,
        fontsize=13,
    )

    out_path = COMPARISON_DIR / f"{metric.out_fname_tag}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved comparison plot → %s", out_path)


# ---------------------------------------------------------------------------
def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    for metric in TIMESERIES_METRICS:
        plot_comparison(metric)

    log.info("All comparison plots written to %s", COMPARISON_DIR)


if __name__ == "__main__":
    main()
