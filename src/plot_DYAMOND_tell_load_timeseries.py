"""
plot_DYAMOND_tell_load_timeseries.py
-------------------------------------
Compare TELL-estimated hourly electricity load to EIA-930 observed demand
for one or more Balancing Authorities (BAs).

Two multi-panel figures are produced (one panel per BA, shared x-axis):
  1. Load overlay  — TELL estimated load vs EIA-930 demand (MW).
  2. Percent-difference — (EIA demand − TELL load) / EIA demand × 100 %.

Both figures are saved to ``image_output_dir`` when ``save_images=True``.

Usage
-----
Edit the constants in the ``__main__`` block at the bottom of this file,
then run:

    conda activate py313
    python plot_DYAMOND_tell_load_timeseries.py

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from util_constants import EIA930_DIR, OUTPUT_ROOT, TELL_OUTPUT_DIR

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
# Constants
# ---------------------------------------------------------------------------
FONTSIZE = 14


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def read_eia_data(eia_data_dir: Path, year: str) -> pd.DataFrame:
    """Read and concatenate the two EIA-930 half-year balance CSV files.

    Parameters
    ----------
    eia_data_dir : Path
        Directory containing ``EIA930_BALANCE_<year>_Jan_Jun.csv`` and
        ``EIA930_BALANCE_<year>_Jul_Dec.csv``.
    year : str
        Four-digit year string, e.g. ``'2016'``.

    Returns
    -------
    DataFrame with columns: BA, Time_UTC (datetime), Demand (MW), Region.
    """
    jan_jun = pd.read_csv(eia_data_dir / f"EIA930_BALANCE_{year}_Jan_Jun.csv")
    jul_dec = pd.read_csv(eia_data_dir / f"EIA930_BALANCE_{year}_Jul_Dec.csv")
    df = pd.concat([jan_jun, jul_dec], ignore_index=True)

    df = df.rename(columns={
        "Balancing Authority":       "BA",
        "Demand (MW) (Adjusted)":    "Demand",
        "UTC Time at End of Hour":   "Time_UTC",
    })
    df["Time_UTC"] = pd.to_datetime(df["Time_UTC"])
    return df[["BA", "Time_UTC", "Demand", "Region"]]


def read_tell_data(
    tell_data_dir: Path,
    ba: str,
    year: str,
    scenario: str,
) -> pd.DataFrame:
    """Read the TELL MLP output CSV for one BA.

    Parameters
    ----------
    tell_data_dir : Path
        Root TELL output directory.  The CSV is expected at
        ``<tell_data_dir>/<scenario>/<year>/<ba>_<year>_mlp_output.csv``.
    ba : str
        BA abbreviation, e.g. ``'PJM'``.
    year : str
        Four-digit year string.
    scenario : str
        Scenario label used as a subdirectory name, e.g. ``'DYAMOND1'``.

    Returns
    -------
    DataFrame with at least columns: Time_UTC (datetime), Load (MW).
    The last row (which wraps into the next month) is dropped.
    """
    csv_path = tell_data_dir / scenario / year / f"{ba}_{year}_mlp_output.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"TELL CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Time_UTC"])
    # Drop the sentinel last row that represents the start of the next month
    return df.iloc[:-1].copy()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _finalise_fig(
    fig: plt.Figure,
    path: Path,
    resolution: int,
    save: bool,
) -> None:
    """Save *fig* to *path* (when *save* is True) then close it."""
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=resolution, bbox_inches="tight")
        log.info("Saved → %s", path)
    plt.close(fig)


def plot_ba_panel(
    ba_list: list[str],
    year: str,
    scenario: str,
    tell_data_dir: Path,
    eia_df: pd.DataFrame,
    image_output_dir: Path,
    image_resolution: int = 150,
    save_images: bool = False,
) -> None:
    """Produce two multi-panel comparison figures — one panel per BA,
    stacked vertically with a shared x-axis.

    **Figure 1** (load overlay): TELL estimated load vs EIA-930 demand (MW).
    **Figure 2** (percent difference): (EIA − TELL) / EIA × 100 %.

    Layout mirrors ``_plot_ba_comparison_panel`` in analyze_tell_heatwave.py:
      • One narrow panel per BA, stacked vertically in a single column.
      • BA name + units written on the y-axis; no subplot titles.
      • All panels share the x-axis; tick labels only on the bottom panel.
      • Zero reference line on every percent-difference panel.
      • Single legend at the bottom of the figure.
      • Tight vertical spacing (hspace = 0).

    Parameters
    ----------
    ba_list : list[str]
        Balancing Authority abbreviations to include, e.g. ``['NYIS', 'PJM', 'MISO']``.
    year : str
        Four-digit year string.
    scenario : str
        Scenario label, e.g. ``'DYAMOND1'``.
    tell_data_dir : Path
        Root directory of TELL output CSVs.
    eia_df : pd.DataFrame
        Full-year EIA-930 DataFrame (all BAs) as returned by
        :func:`read_eia_data`.  Passed in so the caller reads it once.
    image_output_dir : Path
        Directory for output PNG files.
    image_resolution : int
        Output resolution in DPI (default 150).
    save_images : bool
        Write PNG files when True (default False).
    """
    # ── Load data for all BAs ────────────────────────────────────────────────
    # Store (tell_df, ba_eia_df) per BA; skip BAs with missing data.
    ba_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    t_global_min: pd.Timestamp | None = None
    t_global_max: pd.Timestamp | None = None

    for ba in ba_list:
        log.info("Loading data for BA=%s  year=%s  scenario=%s", ba, year, scenario)
        try:
            tell_df = read_tell_data(tell_data_dir, ba, year, scenario)
        except FileNotFoundError as exc:
            log.warning("%s — skipping.", exc)
            continue

        t_min, t_max = tell_df["Time_UTC"].min(), tell_df["Time_UTC"].max()
        ba_eia_df = eia_df[
            (eia_df["BA"] == ba) &
            (eia_df["Time_UTC"] >= t_min) &
            (eia_df["Time_UTC"] <= t_max)
        ].copy()

        if ba_eia_df.empty:
            log.warning("No EIA-930 data for BA=%s in %s — skipping.", ba, year)
            continue

        ba_data[ba] = (tell_df, ba_eia_df)

        # Track shared x-axis range across all BAs
        t_global_min = t_min if t_global_min is None else min(t_global_min, t_min)
        t_global_max = t_max if t_global_max is None else max(t_global_max, t_max)

    if not ba_data:
        log.warning("No data available for any BA — no figures produced.")
        return

    valid_bas = list(ba_data)
    n_bas = len(valid_bas)

    # ── Figure 1: load overlay ───────────────────────────────────────────────
    fig1, axes1 = plt.subplots(
        n_bas, 1,
        figsize=(18, 2.8 * n_bas + 1.0),
        sharex=True,
        squeeze=False,
    )
    fig1.subplots_adjust(hspace=0)

    legend_handles: list = []
    legend_labels:  list = []

    for row, ba in enumerate(valid_bas):
        ax = axes1[row, 0]
        tell_df, ba_eia_df = ba_data[ba]

        line_tell, = ax.plot(
            tell_df["Time_UTC"], tell_df["Load"],
            color="#FF0000", linewidth=1.0,
            label=f"TELL estimated based on {scenario}",
        )
        line_eia, = ax.plot(
            ba_eia_df["Time_UTC"], ba_eia_df["Demand"],
            color="#0000FF", linewidth=1.0,
            label="Observed (Source: EIA-930)",
        )
        if row == 0:
            legend_handles += [line_tell, line_eia]
            legend_labels  += [f"TELL estimated based on {scenario}", "Observed (Source: EIA-930)"]

        ax.set_ylabel(f"{ba} [MW]", fontsize=FONTSIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.tick_params(axis="x", labelsize=FONTSIZE,
                       labelbottom=(row == n_bas - 1))
        ax.set_xlim(t_global_min, t_global_max)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        ax.spines["bottom"].set_linewidth(0.4)

    axes1[-1, 0].xaxis.set_tick_params(rotation=30)
    fig1.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=2,
        fontsize=FONTSIZE, frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )
    #fig1.suptitle(
    #    f"Hourly load — TELL ({scenario}) vs EIA-930  |  {year}",
    #    fontsize=FONTSIZE, y=0.98,
    #)
    _finalise_fig(
        fig1,
        image_output_dir / f"{scenario}_TELL_BA_Hourly_Loads_{year}.png",
        image_resolution, save_images,
    )

    # ── Figure 2: percent-difference ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(
        n_bas, 1,
        figsize=(18, 2.8 * n_bas + 1.0),
        sharex=True,
        squeeze=False,
    )
    fig2.subplots_adjust(hspace=0)

    for row, ba in enumerate(valid_bas):
        ax = axes2[row, 0]
        tell_df, ba_eia_df = ba_data[ba]

        merged = pd.merge(
            ba_eia_df[["Time_UTC", "Demand"]],
            tell_df[["Time_UTC", "Load"]],
            on="Time_UTC",
            how="inner",
        )
        merged["pct_diff"] = (
            (merged["Demand"] - merged["Load"]) /
            merged["Demand"].replace(0, pd.NA)
            * 100.0
        )

        ax.plot(merged["Time_UTC"], merged["pct_diff"],
                color="#2ca02c", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(f"{ba} [% diff]", fontsize=FONTSIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.tick_params(axis="x", labelsize=FONTSIZE,
                       labelbottom=(row == n_bas - 1))
        ax.set_xlim(t_global_min, t_global_max)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        ax.spines["bottom"].set_linewidth(0.4)

    axes2[-1, 0].xaxis.set_tick_params(rotation=30)
    fig2.suptitle(
        f"% difference: (EIA demand − TELL load) / EIA demand  |  {scenario}  |  {year}",
        fontsize=FONTSIZE, y=0.98,
    )
    _finalise_fig(
        fig2,
        image_output_dir / f"{scenario}_TELL_BA_Hourly_Loads_pct_diff_{year}.png",
        image_resolution, save_images,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    BA_LIST = ["NYIS", "PJM", "MISO"]
    YEAR     = "2016"
    SCENARIO = "DYAMOND1"

    TELL_DATA_DIR    = TELL_OUTPUT_DIR
    EIA_DATA_DIR     = EIA930_DIR
    IMAGE_OUTPUT_DIR = OUTPUT_ROOT / SCENARIO

    # Read the full-year EIA data once and reuse it across all BAs
    eia_df = read_eia_data(EIA_DATA_DIR, YEAR)

    plot_ba_panel(
        ba_list=BA_LIST,
        year=YEAR,
        scenario=SCENARIO,
        tell_data_dir=TELL_DATA_DIR,
        eia_df=eia_df,
        image_output_dir=IMAGE_OUTPUT_DIR,
        image_resolution=150,
        save_images=True,
    )

    log.info("Done.")
