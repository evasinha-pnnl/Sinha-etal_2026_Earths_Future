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
# Colour: PGW member uses MEMBER_COLORS directly;
#         ctrl member uses the colour of its paired PGW member
def _color(src: str) -> str:
    if src in MEMBER_COLORS:
        return MEMBER_COLORS[src]
    return "#333333"

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
def plot_comparison(
    metric: TimeseriesMetric,
    varnames: str = "t2m",
    show_obs: bool = True,
) -> None:
    """
    Load CSVs for all members and save a multi-member comparison line plot.

    Styling rules
    -------------
    * **Observation** (ERA5_obs): thick solid black line (when show_obs=True).
    * **PGW members** (EAMxx_SSP245*): solid line.
    * **Control members** (EAMxx_ctrl*): dashed line.
    * Each ctrl/PGW anomaly pair shares the same colour (from MEMBER_COLORS).

    Parameters
    ----------
    metric    : TimeseriesMetric — metric to plot.
    varnames  : str              — variable name used in filenames (default 't2m').
    show_obs  : bool             — whether to include ERA5_obs in the plot
                                   (default True).
    """
    df = load_ensemble_csvs(metric.csv_tag, varnames)

    # Apply optional unit scaling (e.g. km² → million km²)
    if metric.scale != 1.0:
        df = df.copy()
        df[metric.value_col] = df[metric.value_col] * metric.scale

    _ctrl_members = set(ANOMALY_PAIRS.values())

    fig, ax = plt.subplots(figsize=(11, 6))

    # ── EAMxx members ────────────────────────────────────────────────────────
    # Build order: ctrl_0, PGW_0, ctrl_1, PGW_1, … so that the 4-column
    # legend reads "ctrl row" on top and "PGW row" on the bottom.
    _all_model = {
        m.data_source for m in ENSEMBLE_MEMBERS if m.data_source != OBS_LABEL
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
        df_obs = df[df["Data_Source"] == OBS_LABEL]
        if not df_obs.empty:
            ax.plot(
                df_obs["date"], df_obs[metric.value_col],
                label=display_name(OBS_LABEL),
                color="black",
                linewidth=3,
                linestyle="-",
            )

    ax.set_ylabel(metric.ylabel, fontsize=FONTSIZE)
    ax.set_xlim(_XAXIS_MIN, _XAXIS_MAX)
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

    out_path = COMPARISON_DIR / f"{metric.out_fname_tag}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved comparison plot → %s", out_path)
    out_csv = COMPARISON_DIR / f"{metric.out_fname_tag}.csv"
    df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
def plot_longest_duration_distribution(varnames: str = "t2m") -> None:
    """
    Read per-member longest-heatwave-duration CSVs and create a facet histogram
    plot comparing each ctrl/SSP245 pair.

    Layout: 5 rows × 1 column (one row per ANOMALY_PAIRS entry).
    Each facet shows two overlapping histograms — ctrl (hatched) and its
    paired SSP245 member (solid fill) — using the shared MEMBER_COLORS.
    """
    csv_tag = "longest_heatwave_duration"
    value_col = "longest_duration"

    # Load one CSV per member (ctrl + SSP245), skip missing
    member_data: dict[str, pd.Series] = {}
    for pgw, ctrl in ANOMALY_PAIRS.items():
        for src in (ctrl, pgw):
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

    pairs = list(ANOMALY_PAIRS.items())  # [(pgw, ctrl), ...]
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
    fig.suptitle("Distribution of longest heatwave duration per grid cell",
                 fontsize=FONTSIZE+2, y=1.01)
    fig.tight_layout()

    out_path = COMPARISON_DIR / "longest_heatwave_duration_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved duration distribution plot → %s", out_path)


# ---------------------------------------------------------------------------
def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    for metric in TIMESERIES_METRICS:
        plot_comparison(metric, show_obs=False)

    plot_longest_duration_distribution()

    log.info("All comparison plots written to %s", COMPARISON_DIR)


if __name__ == "__main__":
    main()
