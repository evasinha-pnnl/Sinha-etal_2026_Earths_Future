"""
interconnection_stress.py
-------------------------
Assess system-wide heat stress across the three U.S. electricity
interconnections (EIC, WIC, ERCOT) under future warming.

Scientific question
-------------------
Under future warming, does the likelihood of "system-wide" heat events that
simultaneously stress multiple U.S. interconnections increase, and by how
much in terms of load metrics?

Analyses implemented
--------------------
Analysis A — Interconnection-level load stress time-series
    Aggregate daily peak-load anomaly (sum over all BAs in each
    interconnection) for each member × interconnection × day.  Compare
    perturbation vs control to quantify warming-driven load increase at the
    interconnection scale.  Outputs:
      - Time-series line plot per interconnection, pert vs ctrl overlaid
      - Same expressed as % of control load
      - Summary table: mean and peak anomaly per interconnection × member

Analysis B — Daily co-occurrence index
    For each day, count how many interconnections simultaneously exceed a
    stress threshold (default: load anomaly > 0 GW, i.e. any positive
    anomaly).  Compare the co-occurrence count and its daily profile between
    perturbation and control members.  Outputs:
      - Bar chart of daily co-occurrence count (pert vs ctrl, side by side)
      - Summary table: days on which N interconnections were stressed

Inputs
------
``outputs/tell_heatwave_linkage/tell_load_daily_all_<year>.csv``
    Per-BA daily peak load for every BA and day (not filtered to
    heatwave-affected BAs), perturbation members, for the selected --year
    (written by every ``analyze_tell_heatwave.py --year <year>`` run).
    Columns: date, BA, data_source, peak_load, peak_load_anomaly, peak_load_ctrl
    Units: MW

    Interconnection totals sum *all* BAs in the interconnection, so every
    PGW − Control pair sums the same BA set and the totals are comparable
    across pairs and days.

``BA_to_Interconnection_Mapping.csv``  (tracked in the repo root)
    Static lookup: BA_Code → Interconnection (EIC | WIC | ERCOT).

Outputs (all written to OUTPUT_DIR, named with --year)
---------------------------------------------------------
tell_interconnection_load_anomaly_GW_<year>.csv
    Per (date, interconnection, data_source): sum_anomaly_GW, sum_ctrl_GW,
    anomaly_pct.

Each perturbation member's anomaly is one PGW − Control pair; the figures
summarise the pairs as an ensemble (mean with min–max range) rather than one
line or panel per member.

tell_interconnection_load_anomaly_timeseries_<year>.png
    Three-panel time-series (one per interconnection) — PGW − Control
    anomaly (GW): mean over pairs with a min–max band.

tell_interconnection_load_anomaly_timeseries_pct_<year>.png
    Same as above, expressed as % of control load.

tell_interconnection_load_absolute_timeseries_<year>.png
    Three-panel time-series (one per interconnection) — absolute peak load
    (GW) of the Control and PGW ensembles: mean with a min–max band each.

tell_interconnection_stress_heatmap_<year>.png
    Heatmap of the ensemble-mean interconnection-level load anomaly (GW);
    red border on cells where every pair exceeds the stress threshold.

tell_interconnection_cooccurrence_<year>.png
    Bar chart — daily number of simultaneously stressed interconnections:
    mean over pairs (bar) with min–max whiskers.

tell_interconnection_cooccurrence_summary_<year>.csv
    Per (date, data_source): list of stressed interconnections and co-
    occurrence count.

Usage
-----
    conda activate py39_tell
    python interconnection_stress.py [--year YEAR] [--load-csv PATH] [--ba-map CSV]
                                     [--output-dir DIR]
                                     [--stress-threshold-gw FLOAT]

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from util_constants import EXPERIMENT_STYLES, FONTSIZE, OUTPUT_ROOT, PERTURBATION_MEMBERS
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
# Configuration defaults
# ---------------------------------------------------------------------------
_SRC_DIR         = Path(__file__).parent
_ENSEMBLE_OUTPUT = OUTPUT_ROOT
_LINKAGE_DIR     = _ENSEMBLE_OUTPUT / "tell_heatwave_linkage"

# Matches Config.year's default in analyze_tell_heatwave.py
DEFAULT_YEAR = 2011

DEFAULT_BA_MAP_CSV = _SRC_DIR / "BA_to_Interconnection_Mapping.csv"
DEFAULT_OUTPUT_DIR = _LINKAGE_DIR


def _default_load_csv(year: int) -> Path:
    """Path to the tell_load_daily_all_<year>.csv written by analyze_tell_heatwave.py."""
    return _LINKAGE_DIR / f"tell_load_daily_all_{year}.csv"

# Interconnection display order and colours
INTERCONNECTIONS: list[str] = ["EIC", "WIC", "ERCOT"]
IC_COLORS: dict[str, str] = {
    "EIC":   "#1f77b4",   # blue
    "WIC":  "#ff7f0e",   # orange
    "ERCOT": "#9467bd",   # purple
}

# Style of the PGW − Control difference (mean line + range band), as in
# analyze_tell_heatwave.py / plot_affected_ba.py
DIFF_STYLE = dict(label="PGW − Control", color="black", band_color="gray")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    load_csv: Path,
    ba_map_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the daily all-BA load CSV and BA→interconnection map.

    Returns
    -------
    merged : daily load DataFrame with an added ``interconnection`` column
    ba_map : BA_Code → Interconnection lookup
    """
    for path, label in [(load_csv, "Daily load CSV"), (ba_map_csv, "BA map CSV")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    merged = pd.read_csv(load_csv, parse_dates=["date"])
    ba_map = pd.read_csv(ba_map_csv, dtype=str).rename(
        columns={"BA_Code": "BA", "Interconnection": "interconnection"}
    )

    # Validate required columns
    required = {"date", "BA", "data_source", "peak_load",
                "peak_load_anomaly", "peak_load_ctrl"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Daily load CSV missing columns: {missing}")
    log.info("Loaded daily load CSV: %d rows, %d BAs", len(merged), merged["BA"].nunique())

    # Attach interconnection
    merged = merged.merge(ba_map[["BA", "interconnection"]], on="BA", how="left")
    n_unmapped = merged["interconnection"].isna().sum()
    if n_unmapped:
        unmapped_bas = merged.loc[merged["interconnection"].isna(), "BA"].unique()
        log.warning(
            "%d rows have no interconnection mapping (BAs: %s) — dropping",
            n_unmapped, sorted(unmapped_bas),
        )
        merged = merged.dropna(subset=["interconnection"])

    log.info(
        "After BA mapping: %d rows | interconnections: %s",
        len(merged), merged["interconnection"].unique().tolist(),
    )
    return merged, ba_map


# ---------------------------------------------------------------------------
# Analysis A – Interconnection-level aggregation
# ---------------------------------------------------------------------------
def aggregate_by_interconnection(merged: pd.DataFrame) -> pd.DataFrame:
    """Sum peak-load anomaly, control load, and absolute perturbation load
    over all BAs per interconnection.

    Returns a DataFrame with columns:
        date, interconnection, data_source,
        sum_anomaly_MW, sum_ctrl_MW, sum_load_MW,
        sum_anomaly_GW, sum_ctrl_GW, sum_load_GW, anomaly_pct
    """
    # peak_load is the absolute perturbation load (MW); only present for
    # perturbation members — control rows carry it as NaN which is fine.
    agg_cols: dict[str, tuple[str, str]] = {
        "sum_anomaly_MW": ("peak_load_anomaly", "sum"),
        "sum_ctrl_MW":    ("peak_load_ctrl",    "sum"),
    }
    if "peak_load" in merged.columns:
        agg_cols["sum_load_MW"] = ("peak_load", "sum")

    agg = (
        merged
        .groupby(["date", "interconnection", "data_source"], as_index=False)
        .agg(**agg_cols)
    )
    agg["sum_anomaly_GW"] = agg["sum_anomaly_MW"] / 1e3
    agg["sum_ctrl_GW"]    = agg["sum_ctrl_MW"]    / 1e3
    if "sum_load_MW" in agg.columns:
        agg["sum_load_GW"] = agg["sum_load_MW"] / 1e3
    agg["anomaly_pct"]    = (
        agg["sum_anomaly_MW"] / agg["sum_ctrl_MW"].replace(0, np.nan) * 100
    )
    log.info("Interconnection aggregation: %d rows", len(agg))
    return agg


def _pair_members(ic_df: pd.DataFrame) -> list[str]:
    """Perturbation members (one per PGW − Control pair) present in *ic_df*."""
    present = set(ic_df["data_source"].unique())
    return [m for m in PERTURBATION_MEMBERS if m in present]


def _pair_table(ic_df: pd.DataFrame, ic: str, col: str, dates) -> pd.DataFrame:
    """date × pair table of *col* for one interconnection; days missing for a
    pair are NaN (gaps)."""
    return (
        ic_df[ic_df["interconnection"] == ic]
        .pivot_table(index="date", columns="data_source", values=col, aggfunc="first")
        .reindex(dates)
    )


def _stacked_ic_axes(n_ics: int, sharey: bool = False):
    """One narrow panel per interconnection, shared x-axis, no vertical gap."""
    fig, axes = plt.subplots(
        n_ics, 1,
        figsize=(10, 2.5 * n_ics + 1.2),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0)
    return fig, axes


def _style_ic_axis(ax, ylabel: str, is_bottom: bool) -> None:
    """IC name + units on the y-axis, 3 y-ticks, x labels on the bottom only."""
    ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=4)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.tick_params(axis="x", labelsize=FONTSIZE, labelbottom=is_bottom)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
    ax.spines["bottom"].set_linewidth(0.4)


def _bottom_legend(fig, axes, y_anchor: float) -> None:
    """Single legend below the panels, from the first panel with data."""
    handles, labels = next(
        (ax.get_legend_handles_labels() for ax in axes[:, 0]
         if ax.get_legend_handles_labels()[0]),
        ([], []),
    )
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=FONTSIZE, frameon=True, bbox_to_anchor=(0.5, y_anchor))
    else:
        log.warning("No data to plot; skipping legend.")


def plot_interconnection_timeseries(
    ic_df: pd.DataFrame,
    output_dir: Path,
    as_pct: bool = False,
    year: int = DEFAULT_YEAR,
) -> None:
    """Stacked time-series — one narrow panel per interconnection, shared x-axis.

    Each panel shows the PGW − Control peak-load anomaly as the mean over
    pairs (line with markers) and the pair min–max (shaded band).

    Layout:
      • No subplot titles — the interconnection name is written on the y-axis.
      • All panels share the x-axis; tick labels only on the bottom panel.
      • Zero-line drawn on every panel.
      • Single legend at the bottom of the figure.
      • Tight vertical spacing (hspace = 0).

    Outputs
    -------
    tell_interconnection_load_anomaly_timeseries_<year>.png      (as_pct=False)
    tell_interconnection_load_anomaly_timeseries_pct_<year>.png  (as_pct=True)
    """
    y_col   = "anomaly_pct"    if as_pct else "sum_anomaly_GW"
    y_unit  = "% ctrl"         if as_pct else "GW"
    fname   = (f"tell_interconnection_load_anomaly_timeseries_pct_{year}.png"
               if as_pct else f"tell_interconnection_load_anomaly_timeseries_{year}.png")

    ics   = [ic for ic in INTERCONNECTIONS if ic in ic_df["interconnection"].unique()]
    dates = sorted(ic_df["date"].unique())
    pairs = _pair_members(ic_df)
    fig, axes = _stacked_ic_axes(len(ics), sharey=as_pct)   # shared y only for %

    for row, ic in enumerate(ics):
        ax   = axes[row, 0]
        wide = _pair_table(ic_df, ic, y_col, dates)
        cols = [m for m in pairs if m in wide.columns]
        if cols:
            plot_mean_range(
                ax, wide.index, wide, cols, DIFF_STYLE["color"], DIFF_STYLE["label"],
                n_label=f"n={len(cols)} pairs", band_color=DIFF_STYLE["band_color"],
                alpha=0.3, marker="o", markersize=3,
            )
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        _style_ic_axis(ax, f"{ic} peak-load\nanomaly ({y_unit})", row == len(ics) - 1)
        if as_pct:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    axes[-1, 0].xaxis.set_tick_params(rotation=30)
    _bottom_legend(fig, axes, y_anchor=-0.12)
    fig.savefig(output_dir / fname, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", output_dir / fname)


def plot_interconnection_absolute_timeseries(
    ic_df: pd.DataFrame,
    output_dir: Path,
    year: int = DEFAULT_YEAR,
) -> None:
    """Stacked time-series of absolute peak load (GW) — one narrow panel per
    interconnection, shared x-axis.

    Each perturbation-member row carries its own summed load (``sum_load_GW``,
    PGW) and its matched control's (``sum_ctrl_GW``).  For the Control and
    PGW ensembles each panel shows the mean over pairs (line with markers)
    and the min–max (shaded band).

    Output: tell_interconnection_load_absolute_timeseries_<year>.png
    """
    if "sum_load_GW" not in ic_df.columns:
        log.warning(
            "sum_load_GW not present in aggregated table — "
            "skipping absolute load time-series plot."
        )
        return

    ics   = [ic for ic in INTERCONNECTIONS if ic in ic_df["interconnection"].unique()]
    dates = sorted(ic_df["date"].unique())
    pairs = _pair_members(ic_df)
    fig, axes = _stacked_ic_axes(len(ics))

    for row, ic in enumerate(ics):
        ax = axes[row, 0]
        for exp, col in (("CTRL", "sum_ctrl_GW"), ("SSP245", "sum_load_GW")):
            wide = _pair_table(ic_df, ic, col, dates)
            cols = [m for m in pairs if m in wide.columns]
            if cols:
                style = EXPERIMENT_STYLES[exp]
                plot_mean_range(ax, wide.index, wide, cols, style["color"], style["label"],
                                marker="o", markersize=3)
        _style_ic_axis(ax, f"{ic} (GW)", row == len(ics) - 1)

    axes[-1, 0].xaxis.set_tick_params(rotation=30)
    _bottom_legend(fig, axes, y_anchor=-0.16)
    fig.suptitle(
        f"Daily absolute peak load by interconnection  |  {year}",
        fontsize=FONTSIZE,
        y=0.98,
    )
    out = output_dir / f"tell_interconnection_load_absolute_timeseries_{year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Analysis B – Daily co-occurrence index
# ---------------------------------------------------------------------------
def compute_cooccurrence(
    ic_df: pd.DataFrame,
    stress_threshold_gw: float,
) -> pd.DataFrame:
    """For each day × member, count how many interconnections exceed the
    stress threshold (sum_anomaly_GW > stress_threshold_gw).

    Returns a DataFrame with columns:
        date, data_source, n_stressed, stressed_interconnections
    """
    rows = []
    for (date, member), grp in ic_df.groupby(["date", "data_source"]):
        stressed = grp.loc[
            grp["sum_anomaly_GW"] > stress_threshold_gw, "interconnection"
        ].tolist()
        rows.append({
            "date":                      date,
            "data_source":               member,
            "n_stressed":                len(stressed),
            "stressed_interconnections": "|".join(sorted(stressed)),
        })
    cooc = pd.DataFrame(rows).sort_values(["data_source", "date"])
    log.info(
        "Co-occurrence table: %d rows (threshold=%.1f GW)",
        len(cooc), stress_threshold_gw,
    )
    return cooc


def plot_cooccurrence(
    cooc_df: pd.DataFrame,
    stress_threshold_gw: float,
    output_dir: Path,
    year: int = DEFAULT_YEAR,
) -> None:
    """Bar chart of the daily number of simultaneously stressed
    interconnections: mean over PGW − Control pairs (bar) with min–max
    whiskers.  A pair with no row for a day counts as 0 stressed.

    Output: tell_interconnection_cooccurrence_<year>.png
    """
    counts = (
        cooc_df.pivot_table(index="date", columns="data_source",
                            values="n_stressed", aggfunc="first")
        .fillna(0)
    )
    pairs = [m for m in PERTURBATION_MEMBERS if m in counts.columns]
    counts = counts[pairs]
    mean, lo, hi = counts.mean(axis=1), counts.min(axis=1), counts.max(axis=1)

    x = np.arange(len(counts))
    date_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in counts.index]

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.8), 6))
    ax.bar(x, mean, width=0.7, color=DIFF_STYLE["band_color"], edgecolor="black",
           linewidth=0.8, label=f"Mean (n={len(pairs)} pairs)")
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none",
                ecolor="black", elinewidth=1.5, capsize=4, label="Range (min–max)")

    ax.set_xticks(x)
    ax.set_xticklabels(date_labels, rotation=30, ha="right", fontsize=FONTSIZE)
    ax.set_yticks(range(len(INTERCONNECTIONS) + 1))
    ax.set_yticklabels([str(n) for n in range(len(INTERCONNECTIONS) + 1)],
                       fontsize=FONTSIZE)
    ax.set_ylim(0, len(INTERCONNECTIONS) + 1.0)   # headroom for the legend
    ax.set_ylabel("# interconnections stressed simultaneously", fontsize=FONTSIZE)
    ax.set_title(
        f"Daily co-occurrence of stressed interconnections\n"
        f"Threshold: load anomaly > {stress_threshold_gw:.1f} GW  |  {year}\n",
        fontsize=FONTSIZE,
    )
    ax.legend(fontsize=FONTSIZE, loc="upper right", frameon=True)
    fig.tight_layout()

    out = output_dir / f"tell_interconnection_cooccurrence_{year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_cooccurrence_heatmap(
    ic_df: pd.DataFrame,
    stress_threshold_gw: float,
    output_dir: Path,
    year: int = DEFAULT_YEAR,
) -> None:
    """Heatmap: interconnection × date, cell = ensemble-mean anomaly (GW)
    over the PGW − Control pairs.

    A red border marks cells where every pair exceeds the stress threshold
    (a pair with no data for that cell does not count as exceeding).

    Output: tell_interconnection_stress_heatmap_<year>.png
    """
    pairs = _pair_members(ic_df)
    ics   = [ic for ic in INTERCONNECTIONS if ic in ic_df["interconnection"].unique()]
    dates = sorted(ic_df["date"].unique())
    date_labels = [pd.Timestamp(d).strftime("%b %d") for d in dates]

    # pair × interconnection × date
    tables = {ic: _pair_table(ic_df, ic, "sum_anomaly_GW", dates).reindex(columns=pairs)
              for ic in ics}
    stack = np.stack([np.stack([tables[ic][m].to_numpy() for ic in ics]) for m in pairs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells → NaN
        mean = np.nanmean(stack, axis=0)
    all_exceed = np.all(stack > stress_threshold_gw, axis=0)   # NaN → False
    vmax = float(np.nanmax(np.abs(mean))) if np.isfinite(mean).any() else 1.0

    fig, ax = plt.subplots(figsize=(10, 3.0 + len(ics) * 0.9))
    im = ax.imshow(mean, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Anomaly (GW)", shrink=0.8)

    for ci, di in zip(*np.nonzero(all_exceed)):
        ax.add_patch(plt.Rectangle(
            (di - 0.5, ci - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2,
        ))

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(date_labels, rotation=30, ha="right", fontsize=FONTSIZE)
    ax.set_yticks(range(len(ics)))
    ax.set_yticklabels(ics, fontsize=FONTSIZE)
    ax.set_title(f"Ensemble mean (n={len(pairs)} PGW − Control pairs)",
                 fontsize=FONTSIZE, fontweight="bold")

    fig.suptitle(
        "Interconnection-level load anomaly (GW) per day\n"
        f"Red border = all {len(pairs)} pairs exceed {stress_threshold_gw:.1f} GW  |  {year}",
        fontsize=FONTSIZE,
    )
    fig.tight_layout()
    out = output_dir / f"tell_interconnection_stress_heatmap_{year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Assess system-wide heat stress across U.S. interconnections "
            "under future warming (Analyses A + B)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=(
            "Heatwave event/year to analyse — must match the --year used when "
            "running analyze_tell_heatwave.py, since it determines the default "
            "--load-csv filename."
        ),
    )
    p.add_argument(
        "--load-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Path to tell_load_daily_all_<year>.csv (written by "
            "analyze_tell_heatwave.py). Defaults to the file for --year."
        ),
    )
    p.add_argument(
        "--ba-map",
        type=Path,
        default=DEFAULT_BA_MAP_CSV,
        metavar="CSV",
        help="BA → Interconnection mapping CSV (BA_Code, Interconnection).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Directory for output figures and CSVs.",
    )
    p.add_argument(
        "--stress-threshold-gw",
        type=float,
        default=0.0,
        metavar="GW",
        help=(
            "Minimum load anomaly (GW) for an interconnection to be counted as "
            "'stressed' in the co-occurrence analysis (Analysis B). "
            "Default 0.0 counts any positive anomaly."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    load_csv = args.load_csv if args.load_csv is not None else _default_load_csv(args.year)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
    merged, _ = load_data(load_csv, args.ba_map)

    # ── Analysis A: aggregate by interconnection ────────────────────────────
    log.info("Analysis A — aggregating load anomaly by interconnection …")
    ic_df = aggregate_by_interconnection(merged)

    # Save summary table
    ic_csv = args.output_dir / f"tell_interconnection_load_anomaly_GW_{args.year}.csv"
    ic_df.to_csv(ic_csv, index=False)
    log.info("Interconnection load table saved → %s", ic_csv)

    # Log summary
    for ic in INTERCONNECTIONS:
        sub = ic_df[ic_df["interconnection"] == ic]
        for member in sub["data_source"].unique():
            m = sub[sub["data_source"] == member]
            log.info(
                "  %-8s  %-20s  mean=%.2f GW  peak=%.2f GW",
                ic, member,
                m["sum_anomaly_GW"].mean(),
                m["sum_anomaly_GW"].max(),
            )

    log.info("Plotting interconnection time-series (GW) …")
    plot_interconnection_timeseries(ic_df, args.output_dir, as_pct=False, year=args.year)

    log.info("Plotting interconnection time-series (%% of ctrl) …")
    plot_interconnection_timeseries(ic_df, args.output_dir, as_pct=True, year=args.year)

    log.info("Plotting interconnection absolute load time-series …")
    plot_interconnection_absolute_timeseries(ic_df, args.output_dir, year=args.year)

    log.info("Plotting interconnection stress heatmap …")
    plot_cooccurrence_heatmap(ic_df, args.stress_threshold_gw, args.output_dir, year=args.year)

    # ── Analysis B: co-occurrence index ─────────────────────────────────────
    log.info(
        "Analysis B — computing co-occurrence index "
        "(threshold=%.1f GW) …", args.stress_threshold_gw,
    )
    cooc_df = compute_cooccurrence(ic_df, args.stress_threshold_gw)

    cooc_csv = args.output_dir / f"tell_interconnection_cooccurrence_summary_{args.year}.csv"
    cooc_df.to_csv(cooc_csv, index=False)
    log.info("Co-occurrence summary saved → %s", cooc_csv)
    log.info("\n%s", cooc_df.to_string(index=False))

    log.info("Plotting co-occurrence bar chart …")
    plot_cooccurrence(cooc_df, args.stress_threshold_gw, args.output_dir, year=args.year)

    log.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
