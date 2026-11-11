"""
plot_affected_ba.py
-------------------
Plot the number of heatwave-affected balancing authorities (BAs) per day for
the Control and PGW (SSP245) ensembles, one panel per BA-fraction threshold.

Reads ``tell_affected_bas_per_day_<year>_<threshold>.csv`` written by
``analyze_tell_heatwave.py --year <year> --threshold <threshold>`` and writes
two figures (plus a CSV of the plotted values) to
``outputs/tell_heatwave_linkage/``:

  * ``tell_affected_bas_per_day_multi_thresh_<year>.png`` — Control and PGW
    ensemble mean with a min–max band.
  * ``tell_affected_bas_per_day_pgw_minus_ctrl_multi_thresh_<year>.png`` —
    PGW − Control for each matched pair; mean over pairs with a min–max band.

Usage
-----
  python plot_affected_ba.py --year 2011

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from util_constants import ANOMALY_PAIRS, EXPERIMENT_STYLES, FONTSIZE, members_by_experiment
from util_vector_plots import plot_mean_range

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Heatwave ensemble output root (produced by run_heatwave_ensemble.py)
ENSEMBLE_OUTPUT = Path(__file__).parent / "outputs"

# Output directory for csv files containing the number of heatwave-affected BAs per day
OUTPUT_DIR = ENSEMBLE_OUTPUT / "tell_heatwave_linkage"

# BA thresholds to plot
THRESHOLDS = ['0.2', '0.4', '0.6', '0.8']

BA_AFFECTED_TAG      = "tell_affected_bas_per_day"
BA_ENSEMBLE_NAME_COL = "data_source"
BA_VALUE_COL         = "n_affected_bas"

# Style of the PGW − Control difference (mean line + range band)
DIFF_STYLE = dict(label="PGW − Control", color="black", band_color="gray")


# ---------------------------------------------------------------------------
# Load number of heatwave-affected BAs per day
# ---------------------------------------------------------------------------
def _load_ba_csv(year: int, thresh: str) -> pd.DataFrame:
    """Load one threshold's CSV (columns: date, data_source, n_affected_bas)."""
    fname = OUTPUT_DIR / f"{BA_AFFECTED_TAG}_{year}_{thresh}.csv"
    if not fname.exists():
        log.warning("BA CSV not found: %s", fname)
        return pd.DataFrame(columns=["date", BA_ENSEMBLE_NAME_COL, BA_VALUE_COL])
    return pd.read_csv(fname, parse_dates=["date"])


def _build_member_tables(year: int) -> dict[str, pd.DataFrame]:
    """Threshold → date × member table of n_affected_bas."""
    return {
        thresh: _load_ba_csv(year, thresh).pivot_table(
            index="date", columns=BA_ENSEMBLE_NAME_COL,
            values=BA_VALUE_COL, aggfunc="first",
        )
        for thresh in THRESHOLDS
    }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _facet_axes(n_panels: int):
    """2-column grid of shared-axis panels, unused panels hidden."""
    ncols = 2
    nrows = -(-n_panels // ncols)  # ceiling division
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    for idx in range(n_panels, nrows * ncols):
        axes.flat[idx].set_visible(False)
    return fig, axes


def _finish_facet_figure(fig, axes, out: Path, summary: list[pd.DataFrame],
                         year: int, ncol: int) -> None:
    """Add the shared legend, save the figure and the CSV of plotted values."""
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol,
                   bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=FONTSIZE)
    else:
        log.warning("No data for any member/threshold; skipping legend (year=%s)", year)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    if summary:
        pd.concat(summary).to_csv(out.with_suffix(".csv"))
    log.info("Saved plot → %s", out)


# ---------------------------------------------------------------------------
# Number of affected BAs: Control and PGW
# ---------------------------------------------------------------------------
def _plot_facet_grid(tables: dict[str, pd.DataFrame], year: int) -> None:
    """
    Facet plot (one panel per threshold) of the number of heatwave-affected
    BAs per day.  For the Control and PGW ensembles the line is the ensemble
    mean and the shaded band spans the ensemble min–max.
    """
    fig, axes = _facet_axes(len(tables))
    summary: list[pd.DataFrame] = []

    for ax, (thresh, wide) in zip(axes.flat, tables.items()):
        exp_members = members_by_experiment(wide.columns)
        for exp, style in EXPERIMENT_STYLES.items():
            cols = exp_members[exp]
            if not cols:
                continue
            stats = plot_mean_range(ax, wide.index, wide, cols, style["color"], style["label"])
            summary.append(stats.assign(Threshold=thresh, experiment=exp, n_members=len(cols)))

        ax.set_title(f"Threshold = {thresh}", fontsize=FONTSIZE)
        ax.set_ylabel("Number of BAs under heatwave", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE)
        ax.set_ylim(bottom=0)
        ax.grid(True)

    out = OUTPUT_DIR / f"tell_affected_bas_per_day_multi_thresh_{year}.png"
    _finish_facet_figure(fig, axes, out, summary, year, ncol=4)


# ---------------------------------------------------------------------------
# Difference plot: PGW − Control
# ---------------------------------------------------------------------------
def _plot_diff_facet_grid(tables: dict[str, pd.DataFrame], year: int) -> None:
    """
    Facet plot (one panel per threshold) of the PGW − Control difference in
    the number of heatwave-affected BAs per day.  Differences are taken for
    each matched pair (SSP245_p12 − CTRL_p12, …); the line is the mean over
    pairs and the shaded band spans the pair min–max.  A dashed zero-line is
    drawn for reference.
    """
    fig, axes = _facet_axes(len(tables))
    summary: list[pd.DataFrame] = []

    for ax, (thresh, wide) in zip(axes.flat, tables.items()):
        pairs = [(pgw, ctrl) for pgw, ctrl in ANOMALY_PAIRS.items()
                 if pgw in wide.columns and ctrl in wide.columns]
        ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=1)
        if pairs:
            diff = pd.DataFrame({pgw: wide[pgw] - wide[ctrl] for pgw, ctrl in pairs})
            stats = plot_mean_range(
                ax, diff.index, diff, list(diff.columns),
                DIFF_STYLE["color"], DIFF_STYLE["label"],
                n_label=f"n={len(pairs)} pairs",
                band_color=DIFF_STYLE["band_color"], alpha=0.3,
            )
            summary.append(stats.assign(Threshold=thresh, n_pairs=len(pairs)))

        ax.set_title(f"Threshold = {thresh}", fontsize=FONTSIZE)
        ax.set_ylabel("ΔBAs (PGW − Control)", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE)
        ax.grid(True)

    out = OUTPUT_DIR / f"tell_affected_bas_per_day_pgw_minus_ctrl_multi_thresh_{year}.png"
    _finish_facet_figure(fig, axes, out, summary, year, ncol=2)


# ---------------------------------------------------------------------------
def make_plots(year: int) -> None:
    """Create both facet figures for *year*."""
    tables = _build_member_tables(year)
    _plot_facet_grid(tables, year)
    _plot_diff_facet_grid(tables, year)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot number of heatwave-affected BAs per day, across thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--year",
        type=int,
        default=2011,
        choices=[2011, 2021],
        help=(
            "Heatwave event/year to plot — selects which "
            "tell_affected_bas_per_day_<year>_<threshold>.csv files to read "
            "(produced by analyze_tell_heatwave.py --year <year>)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    make_plots(parse_args().year)
