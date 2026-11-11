"""
compare_EIA930_EAMxx_tell_load_timeseries.py
-------------------------------------
Compare TELL-estimated hourly electricity load to EIA-930 observed demand
for one or more Balancing Authorities (BAs), across multiple TELL scenarios.

Three multi-panel figures are produced (one panel per BA, shared x-axis):
  1. Load overlay  — TELL estimated load (one line per scenario) vs
     EIA-930 demand (MW).
  2. Percent-difference — (EIA demand − TELL load) / EIA demand × 100 %,
     one line per scenario.
  3. Daily maximum load boxplot — distribution of daily peak load (MW) per
     BA, EIA-930 observed vs each TELL scenario.

All figures are saved to ``image_output_dir`` when ``save_images=True``.

Usage
-----
Edit the constants in the ``__main__`` block at the bottom of this file,
then run:

    conda activate py313
    python compare_EIA930_EAMxx_tell_load_timeseries.py

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
from matplotlib.patches import Patch
import pandas as pd

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

# Colors cycled across TELL scenarios; EIA-930 observations are always blue.
_EIA_COLOR = "#0000FF"
_SCENARIO_PALETTE = ["#FF0000", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]


def _build_color_map(scenario_list: list[str]) -> dict[str, str]:
    """Map ``'EIA-930'`` + each scenario in *scenario_list* to a plot color.

    Colors are assigned in order from ``_SCENARIO_PALETTE`` (cycling if there
    are more scenarios than palette entries) so a given scenario keeps the
    same color across the line-overlay and boxplot figures.
    """
    colors = {"EIA-930": _EIA_COLOR}
    for i, scenario in enumerate(scenario_list):
        colors[scenario] = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
    return colors


def _source_label(source: str) -> str:
    """Human-readable legend label for a data source key."""
    return "Observed (EIA-930)" if source == "EIA-930" else f"TELL ({source})"


# ---------------------------------------------------------------------------
# EIA data loading
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

# ---------------------------------------------------------------------------
# TELL data loading
# ---------------------------------------------------------------------------
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
    scenario_list: list[str],
    tell_data_dir: Path,
    eia_df: pd.DataFrame,
    image_output_dir: Path,
    image_resolution: int = 150,
    save_images: bool = False,
) -> None:
    """Produce two multi-panel comparison figures — one panel per BA,
    stacked vertically with a shared x-axis.

    **Figure 1** (load overlay): TELL estimated load for every scenario in
    *scenario_list* vs EIA-930 demand (MW) — i.e. one EIA line plus one line
    per scenario.
    **Figure 2** (percent difference): (EIA − TELL) / EIA × 100 %, one line
    per scenario.

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
    scenario_list : list[str]
        TELL scenario labels to overlay, e.g.
        ``['2021_ne128_25km', '2021_CTRL_p0', '2021_CTRL_n24', '2021_CTRL_p24']``.
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
    colors = _build_color_map(scenario_list)

    # ── Load data for all BAs / scenarios ────────────────────────────────────
    # Store {scenario: tell_df} and ba_eia_df per BA; skip BAs with no data.
    ba_data: dict[str, dict[str, pd.DataFrame]] = {}
    ba_eia: dict[str, pd.DataFrame] = {}
    t_global_min: pd.Timestamp | None = None
    t_global_max: pd.Timestamp | None = None

    for ba in ba_list:
        scenario_dfs: dict[str, pd.DataFrame] = {}
        for scenario in scenario_list:
            log.info("Loading data for BA=%s  year=%s  scenario=%s", ba, year, scenario)
            try:
                scenario_dfs[scenario] = read_tell_data(tell_data_dir, ba, year, scenario)
            except FileNotFoundError as exc:
                log.warning("%s — skipping.", exc)

        if not scenario_dfs:
            log.warning("No TELL data for BA=%s in any scenario — skipping.", ba)
            continue

        t_min = min(df["Time_UTC"].min() for df in scenario_dfs.values())
        t_max = max(df["Time_UTC"].max() for df in scenario_dfs.values())
        ba_eia_df = eia_df[
            (eia_df["BA"] == ba) &
            (eia_df["Time_UTC"] >= t_min) &
            (eia_df["Time_UTC"] <= t_max)
        ].copy()

        if ba_eia_df.empty:
            log.warning("No EIA-930 data for BA=%s in %s — skipping.", ba, year)
            continue

        ba_data[ba] = scenario_dfs
        ba_eia[ba] = ba_eia_df

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
        scenario_dfs = ba_data[ba]
        ba_eia_df = ba_eia[ba]

        line_eia, = ax.plot(
            ba_eia_df["Time_UTC"], ba_eia_df["Demand"],
            color=colors["EIA-930"], linewidth=1.0,
            label=_source_label("EIA-930"),
        )
        if row == 0:
            legend_handles.append(line_eia)
            legend_labels.append(_source_label("EIA-930"))

        for scenario in scenario_list:
            tell_df = scenario_dfs.get(scenario)
            if tell_df is None:
                continue
            line_tell, = ax.plot(
                tell_df["Time_UTC"], tell_df["Load"],
                color=colors[scenario], linewidth=1.0,
                label=_source_label(scenario),
            )
            if row == 0:
                legend_handles.append(line_tell)
                legend_labels.append(_source_label(scenario))

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
        loc="lower center", ncol=3,
        fontsize=FONTSIZE, frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )
    #fig1.suptitle(
    #    f"Hourly load — TELL scenarios vs EIA-930  |  {year}",
    #    fontsize=FONTSIZE, y=0.98,
    #)
    _finalise_fig(
        fig1,
        image_output_dir / f"AllScenarios_TELL_BA_Hourly_Loads_{year}.png",
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

    legend_handles2: list = []
    legend_labels2:  list = []

    for row, ba in enumerate(valid_bas):
        ax = axes2[row, 0]
        scenario_dfs = ba_data[ba]
        ba_eia_df = ba_eia[ba]

        for scenario in scenario_list:
            tell_df = scenario_dfs.get(scenario)
            if tell_df is None:
                continue

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

            line, = ax.plot(merged["Time_UTC"], merged["pct_diff"],
                             color=colors[scenario], linewidth=1.0,
                             label=_source_label(scenario))
            if row == 0:
                legend_handles2.append(line)
                legend_labels2.append(_source_label(scenario))

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(f"{ba} [% diff]", fontsize=FONTSIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.tick_params(axis="x", labelsize=FONTSIZE,
                       labelbottom=(row == n_bas - 1))
        ax.set_xlim(t_global_min, t_global_max)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        ax.spines["bottom"].set_linewidth(0.4)

    axes2[-1, 0].xaxis.set_tick_params(rotation=30)
    fig2.legend(
        legend_handles2, legend_labels2,
        loc="lower center", ncol=2,
        fontsize=FONTSIZE, frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig2.suptitle(
        f"% difference: (EIA demand − TELL load) / EIA demand  |  {year}",
        fontsize=FONTSIZE, y=0.98,
    )
    _finalise_fig(
        fig2,
        image_output_dir / f"AllScenarios_TELL_BA_Hourly_Loads_pct_diff_{year}.png",
        image_resolution, save_images,
    )


def plot_daily_max_load_boxplot(
    ba_list: list[str],
    year: str,
    scenario_list: list[str],
    tell_data_dir: Path,
    eia_df: pd.DataFrame,
    image_output_dir: Path,
    image_resolution: int = 150,
    save_images: bool = False,
) -> None:
    """Boxplot of daily maximum load (MW): EIA-930 observed vs each TELL
    scenario in *scenario_list*, grouped by BA.

    For each BA, the daily maximum load is computed by resampling the hourly
    series to daily resolution and taking the max. One box is drawn per data
    source (EIA-930 + each scenario); boxes for the same BA are grouped
    together along the x-axis, with a gap between BA groups. Color encodes
    data source and matches the colors used in :func:`plot_ba_panel`.

    Parameters
    ----------
    ba_list : list[str]
        Balancing Authority abbreviations to include.
    year : str
        Four-digit year string.
    scenario_list : list[str]
        TELL scenario labels to compare against EIA-930.
    tell_data_dir : Path
        Root directory of TELL output CSVs.
    eia_df : pd.DataFrame
        Full-year EIA-930 DataFrame (all BAs) as returned by
        :func:`read_eia_data`.
    image_output_dir : Path
        Directory for output PNG files.
    image_resolution : int
        Output resolution in DPI (default 150).
    save_images : bool
        Write PNG files when True (default False).
    """
    colors = _build_color_map(scenario_list)
    sources = ["EIA-930"] + scenario_list

    # box_data[ba][source] = list of daily maximum values
    box_data: dict[str, dict[str, list[float]]] = {}

    for ba in ba_list:
        per_source: dict[str, list[float]] = {}

        ba_eia_df = eia_df[eia_df["BA"] == ba]
        if not ba_eia_df.empty:
            daily_max = (
                ba_eia_df.set_index("Time_UTC")["Demand"]
                .resample("D").max().dropna()
            )
            if not daily_max.empty:
                per_source["EIA-930"] = daily_max.tolist()

        for scenario in scenario_list:
            log.info("Loading data for BA=%s  year=%s  scenario=%s", ba, year, scenario)
            try:
                tell_df = read_tell_data(tell_data_dir, ba, year, scenario)
            except FileNotFoundError as exc:
                log.warning("%s — skipping.", exc)
                continue
            daily_max = (
                tell_df.set_index("Time_UTC")["Load"]
                .resample("D").max().dropna()
            )
            if not daily_max.empty:
                per_source[scenario] = daily_max.tolist()

        if per_source:
            box_data[ba] = per_source
        else:
            log.warning("No data for BA=%s in any source — skipping.", ba)

    if not box_data:
        log.warning("No data available for any BA — no boxplot produced.")
        return

    valid_bas = list(box_data)
    n_sources = len(sources)
    group_width = n_sources + 1  # one empty slot between BA groups
    fig_width = max(10, 1.1 * group_width * len(valid_bas))

    fig, ax = plt.subplots(figsize=(fig_width, 7))

    group_centers: list[float] = []
    for gi, ba in enumerate(valid_bas):
        base = gi * group_width
        group_centers.append(base + (n_sources - 1) / 2)

        for si, source in enumerate(sources):
            values = box_data[ba].get(source)
            if not values:
                continue
            bp = ax.boxplot(
                values, positions=[base + si], widths=0.8,
                patch_artist=True, showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[source])
                patch.set_alpha(0.7)
            for median in bp["medians"]:
                median.set_color("black")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(valid_bas, fontsize=FONTSIZE)
    ax.set_xlim(-1, (len(valid_bas) - 1) * group_width + n_sources)
    ax.set_ylabel("Daily maximum load [MW]", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)

    legend_handles = [
        Patch(facecolor=colors[s], alpha=0.7, label=_source_label(s))
        for s in sources
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", ncol=len(sources),
        fontsize=FONTSIZE - 2, frameon=True, bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(
        f"Daily maximum load — EIA-930 vs TELL scenarios  |  {year}",
        fontsize=FONTSIZE, y=0.98,
    )
    _finalise_fig(
        fig,
        image_output_dir / f"AllScenarios_TELL_BA_DailyMaxLoad_Boxplot_{year}.png",
        image_resolution, save_images,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    BA_LIST = ["AVA", "BPAT", "CHPD", "DOPD", "GCPD", "IPCO", "PACW", "PGE", "PSEI", "SCL", "TPWR"]
    YEAR     = "2021"
    SCENARIO_LIST = [
        "2021_ne128_25km",
        "2021_CTRL_p0",
       # "2021_CTRL_n24",
       # "2021_CTRL_p24",
    ]

    TELL_DATA_DIR    = Path("/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL/TELL_outputs")
    EIA_DATA_DIR     = Path("/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL/EIA_930")
    IMAGE_OUTPUT_DIR = Path("/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL/src/outputs") / "EIA930_comparison"

    # Read the full-year EIA data once and reuse it across all BAs
    eia_df = read_eia_data(EIA_DATA_DIR, YEAR)

    plot_ba_panel(
        ba_list=BA_LIST,
        year=YEAR,
        scenario_list=SCENARIO_LIST,
        tell_data_dir=TELL_DATA_DIR,
        eia_df=eia_df,
        image_output_dir=IMAGE_OUTPUT_DIR,
        image_resolution=150,
        save_images=True,
    )

    plot_daily_max_load_boxplot(
        ba_list=BA_LIST,
        year=YEAR,
        scenario_list=SCENARIO_LIST,
        tell_data_dir=TELL_DATA_DIR,
        eia_df=eia_df,
        image_output_dir=IMAGE_OUTPUT_DIR,
        image_resolution=150,
        save_images=True,
    )

    log.info("Done.")
