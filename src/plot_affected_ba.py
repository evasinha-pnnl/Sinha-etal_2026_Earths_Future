from __future__ import annotations

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from util_constants import FONTSIZE, MEMBER_COLORS, ANOMALY_PAIRS, sorted_member_order, display_name

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# Heatwave ensemble output root (produced by run_heatwave_ensemble.py)
ENSEMBLE_OUTPUT = Path(__file__).parent / "outputs"

# Output directory for csv files containing the number of heatwave-affected BAs per day
OUTPUT_DIR = ENSEMBLE_OUTPUT / "tell_heatwave_linkage"

# BA thresholds to plot
THRESHOLDS = ['0.2', '0.4', '0.6', '0.8']

BA_AFFECTED_TAG   = "tell_affected_bas_per_day"
BA_ENSEMBLE_NAME_COL   = "data_source"
BA_VALUE_COL = "n_affected_bas"

# ---------------------------------------------------------------------------
# Load number of heatwave-affected BAs per day
# ---------------------------------------------------------------------------
def _load_ba_csv(tag: str, year: int, thresh: str, value_col: str) -> pd.DataFrame:
    """Load one BA timeseries CSV and return a DataFrame with date and value columns."""
    fname = OUTPUT_DIR / f"{tag}_{year}_{thresh}.csv"
    if not fname.exists():
        log.warning("BA CSV not found: %s", fname)
        return pd.DataFrame(columns=["date", BA_ENSEMBLE_NAME_COL, value_col])
    return pd.read_csv(fname, parse_dates=["date"])
# ---------------------------------------------------------------------------

def _build_combined_df(year: int) -> pd.DataFrame:
    all_data = []
    for thresh in THRESHOLDS:
        df = _load_ba_csv(BA_AFFECTED_TAG, year, thresh, BA_VALUE_COL)
        df["Threshold"] = thresh
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)
# ---------------------------------------------------------------------------

def _plot_facet_grid(combined_df: pd.DataFrame, year: int) -> None:
    """
    Styling rules
    -------------
    * **PGW members** (2011_SSP245*): solid line.
    * **Control members** (2011_CTRL*): dashed line.
    * Each ctrl/PGW anomaly pair shares the same colour (from MEMBER_COLORS).
    """
    _ctrl_members = set(ANOMALY_PAIRS.values())

    def _color(src: str) -> str:
        if src in MEMBER_COLORS:
            return MEMBER_COLORS[src]
        return "#333333"

    thresholds = sorted(combined_df["Threshold"].unique())
    ncols = 2
    nrows = -(-len(thresholds) // ncols)  # ceiling division

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharex=True, sharey=True,
        squeeze=False,
    )

    all_sources = set(combined_df[BA_ENSEMBLE_NAME_COL].unique())
    model_sources = sorted_member_order(all_sources)

    legend_handles: list = []
    legend_labels: list = []

    for idx, thresh in enumerate(thresholds):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub_thresh = combined_df[combined_df["Threshold"] == thresh]

        for src in model_sources:
            sub = sub_thresh[sub_thresh[BA_ENSEMBLE_NAME_COL] == src]
            if sub.empty:
                continue
            is_ctrl = src in _ctrl_members
            line, = ax.plot(
                sub["date"], sub[BA_VALUE_COL],
                label=display_name(src),
                color=_color(src),
                linewidth=2,
                linestyle="--" if is_ctrl else "-",
            )
            if display_name(src) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(display_name(src))

        ax.set_title(f"Threshold = {thresh}", fontsize=FONTSIZE)
        ax.set_ylabel("Number of BAs under heatwave", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE)
        ax.grid(True)

    # Hide any unused axes
    for idx in range(len(thresholds), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(5, len(legend_labels)),
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
            fontsize=FONTSIZE,
        )
    else:
        log.warning("No data for any member/threshold; skipping legend for facet grid (year=%s)", year)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    out = OUTPUT_DIR / f"tell_affected_bas_per_day_multi_thresh_{year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", out)


# ---------------------------------------------------------------------------
# Difference plot: PGW − control
# ---------------------------------------------------------------------------

def _build_diff_df(year: int) -> pd.DataFrame:
    """
    For each (pgw, ctrl) pair in ANOMALY_PAIRS, compute the daily difference
    (pgw n_affected_bas − ctrl n_affected_bas) for every threshold.
    Returns a DataFrame with columns: date, pair_label, Threshold, diff.
    """
    all_rows = []
    for thresh in THRESHOLDS:
        df = _load_ba_csv(BA_AFFECTED_TAG, year, thresh, BA_VALUE_COL)
        if df.empty:
            continue
        for pgw, ctrl in ANOMALY_PAIRS.items():
            pgw_df = df[df[BA_ENSEMBLE_NAME_COL] == pgw][["date", BA_VALUE_COL]].rename(
                columns={BA_VALUE_COL: "pgw"}
            )
            ctrl_df = df[df[BA_ENSEMBLE_NAME_COL] == ctrl][["date", BA_VALUE_COL]].rename(
                columns={BA_VALUE_COL: "ctrl"}
            )
            merged = pgw_df.merge(ctrl_df, on="date", how="inner")
            merged["diff"] = merged["pgw"] - merged["ctrl"]
            merged["pair_label"] = f"{display_name(pgw)} − {display_name(ctrl)}"
            merged["pgw_member"] = pgw
            merged["Threshold"] = thresh
            all_rows.append(merged[["date", "pair_label", "pgw_member", "Threshold", "diff"]])
    if not all_rows:
        return pd.DataFrame(columns=["date", "pair_label", "pgw_member", "Threshold", "diff"])
    return pd.concat(all_rows, ignore_index=True)

# ---------------------------------------------------------------------------
def _plot_diff_facet_grid(diff_df: pd.DataFrame, year: int) -> None:
    """
    Facet plot (2 × 2) showing PGW − ctrl difference in number of heatwave-affected
    BAs per day, one subplot per threshold.  Each PGW/ctrl pair is drawn as a solid
    line whose colour matches MEMBER_COLORS for the PGW member.
    A dashed horizontal zero-line is drawn for reference.
    """
    def _color(pgw: str) -> str:
        return MEMBER_COLORS.get(pgw, "#333333")

    thresholds = sorted(diff_df["Threshold"].unique())
    ncols = 2
    nrows = -(-len(thresholds) // ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharex=True, sharey=True,
        squeeze=False,
    )

    pairs = diff_df[["pair_label", "pgw_member"]].drop_duplicates()
    pair_order = pairs.sort_values("pgw_member")

    legend_handles: list = []
    legend_labels: list = []

    for idx, thresh in enumerate(thresholds):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub_thresh = diff_df[diff_df["Threshold"] == thresh]

        ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=1)

        for _, pr in pair_order.iterrows():
            sub = sub_thresh[sub_thresh["pair_label"] == pr["pair_label"]]
            if sub.empty:
                continue
            line, = ax.plot(
                sub["date"], sub["diff"],
                label=pr["pair_label"],
                color=_color(pr["pgw_member"]),
                linewidth=2,
            )
            if pr["pair_label"] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(pr["pair_label"])

        ax.set_title(f"Threshold = {thresh}", fontsize=FONTSIZE)
        ax.set_ylabel("ΔBAs (PGW − ctrl)", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE)
        ax.grid(True)

    for idx in range(len(thresholds), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(3, len(legend_labels)),
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
            fontsize=FONTSIZE,
        )
    else:
        log.warning("No data for any pair/threshold; skipping legend for diff facet grid (year=%s)", year)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    out = OUTPUT_DIR / f"tell_affected_bas_per_day_pgw_minus_ctrl_multi_thresh_{year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", out)

# ---------------------------------------------------------------------------

def create_facet_plot(year: int) -> None:
    combined_df = _build_combined_df(year)
    _plot_facet_grid(combined_df, year)

# ---------------------------------------------------------------------------
def create_diff_facet_plot(year: int) -> None:
    """Create the PGW − ctrl difference facet plot for all thresholds."""
    diff_df = _build_diff_df(year)
    _plot_diff_facet_grid(diff_df, year)


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
    args = parse_args()
    create_facet_plot(args.year)
    create_diff_facet_plot(args.year)
