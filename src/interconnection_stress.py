"""
interconnection_stress.py
-------------------------
Assess system-wide heat stress across the three U.S. electricity
interconnections (EIC, WECC, ERCOT) under future warming.

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
``outputs/tell_heatwave_linkage/tell_heatwave_merged_<year>.csv``
    Per-BA daily peak-load anomaly on heatwave days, all members, for the
    selected --year (produced by ``analyze_tell_heatwave.py --year <year>``).
    Columns: date, BA, data_source, peak_load, peak_load_anomaly,
             peak_load_ctrl, hw_area_frac, hw_area_anomaly_km2,
             hw_mean_K, hw_max_K
    Units: MW (peak_load, peak_load_anomaly, peak_load_ctrl)

``BA_to_Interconnection_Mapping.csv``  (tracked in the repo root)
    Static lookup: BA_Code → Interconnection (EIC | WECC | ERCOT).

Outputs (all written to OUTPUT_DIR, named with --year)
---------------------------------------------------------
tell_interconnection_load_anomaly_GW_<year>.csv
    Per (date, interconnection, data_source): sum_anomaly_GW, sum_ctrl_GW,
    anomaly_pct.

tell_interconnection_load_anomaly_timeseries_<year>.png
    Three-panel time-series (one per interconnection) — perturbation and
    control lines, GW anomaly.

tell_interconnection_load_anomaly_timeseries_pct_<year>.png
    Same as above, expressed as % of control load.

tell_interconnection_load_absolute_timeseries_<year>.png
    Three-panel time-series (one per interconnection) — one pert/ctrl line
    pair per perturbation↔control anomaly pair defined for the selected
    year (e.g. for 2011: SSP245_p0/p12/p24/n12/n24 vs their matched CTRL).
    Y-axis = absolute peak load in GW.

tell_interconnection_stress_heatmap_<year>.png
    Heatmap of interconnection-level load anomaly (GW), one panel per
    perturbation member, red border on cells exceeding the stress threshold.

tell_interconnection_cooccurrence_<year>.png
    Stacked bar chart — daily count of simultaneously stressed
    interconnections, pert vs ctrl side by side.

tell_interconnection_cooccurrence_summary_<year>.csv
    Per (date, data_source): list of stressed interconnections and co-
    occurrence count.

Usage
-----
    conda activate py39_tell
    python interconnection_stress.py [--year YEAR] [--merged-csv PATH] [--ba-map CSV]
                                     [--output-dir DIR]
                                     [--stress-threshold-gw FLOAT]
                                     [--min-hw-frac FLOAT]

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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from util_constants import ANOMALY_PAIRS, FONTSIZE, MEMBER_COLORS, PERTURBATION_MEMBERS, sorted_member_order, display_name

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
_ENSEMBLE_OUTPUT = _SRC_DIR / "outputs"
_LINKAGE_DIR     = _ENSEMBLE_OUTPUT / "tell_heatwave_linkage"

# Matches Config.year's default in analyze_tell_heatwave.py
DEFAULT_YEAR = 2011

DEFAULT_BA_MAP_CSV = _SRC_DIR / "BA_to_Interconnection_Mapping.csv"
DEFAULT_OUTPUT_DIR = _LINKAGE_DIR


def _default_merged_csv(year: int) -> Path:
    """Path to the tell_heatwave_merged_<year>.csv written by analyze_tell_heatwave.py."""
    return _LINKAGE_DIR / f"tell_heatwave_merged_{year}.csv"

# Interconnection display order and colours
INTERCONNECTIONS: list[str] = ["EIC", "WECC", "ERCOT"]
IC_COLORS: dict[str, str] = {
    "EIC":   "#1f77b4",   # blue
    "WECC":  "#ff7f0e",   # orange
    "ERCOT": "#9467bd",   # purple
}

# Line styles for pert vs ctrl
PERT_LS  = "-"
CTRL_LS  = "--"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    merged_csv: Path,
    ba_map_csv: Path,
    min_hw_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the merged load CSV and BA→interconnection map.

    Returns
    -------
    merged : filtered merged DataFrame with an added ``interconnection`` column
    ba_map : BA_Code → Interconnection lookup
    """
    for path, label in [(merged_csv, "Merged CSV"), (ba_map_csv, "BA map CSV")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    merged = pd.read_csv(merged_csv, parse_dates=["date"])
    ba_map = pd.read_csv(ba_map_csv, dtype=str).rename(
        columns={"BA_Code": "BA", "Interconnection": "interconnection"}
    )

    # Validate required columns
    required = {"date", "BA", "data_source", "peak_load_anomaly",
                "peak_load_ctrl", "hw_area_frac"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Merged CSV missing columns: {missing}")

    # Apply hw_area_frac filter
    n_before = len(merged)
    merged = merged[merged["hw_area_frac"] > min_hw_frac].copy()
    log.info(
        "Loaded merged CSV: %d rows kept (of %d) after hw_area_frac > %.2f",
        len(merged), n_before, min_hw_frac,
    )

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


def plot_interconnection_timeseries(
    ic_df: pd.DataFrame,
    output_dir: Path,
    as_pct: bool = False,
    year: int = DEFAULT_YEAR,
) -> None:
    """Stacked time-series — one narrow panel per interconnection, shared x-axis.

    Layout mirrors the reference image:
      • No subplot titles — the interconnection name is written on the y-axis.
      • All panels share the x-axis; tick labels only on the bottom panel.
      • Zero-line drawn on every panel.
      • Single legend at the bottom of the figure.
      • Tight vertical spacing (hspace = 0).

    Outputs
    -------
    tell_interconnection_load_anomaly_timeseries.png      (as_pct=False)
    tell_interconnection_load_anomaly_timeseries_pct.png  (as_pct=True)
    """
    y_col   = "anomaly_pct"    if as_pct else "sum_anomaly_GW"
    y_unit  = "% ctrl"         if as_pct else "GW"
    fname   = (f"tell_interconnection_load_anomaly_timeseries_pct_{year}.png"
               if as_pct else f"tell_interconnection_load_anomaly_timeseries_{year}.png")

    ics = [ic for ic in INTERCONNECTIONS if ic in ic_df["interconnection"].unique()]
    n_ics = len(ics)

    fig, axes = plt.subplots(
        n_ics, 1,
        figsize=(10, 2.5 * n_ics + 1.2),
        sharex=True,
        sharey=as_pct,   # shared y-axis only for the % plot
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0)

    legend_handles: list = []
    legend_labels:  list = []

    for row, ic in enumerate(ics):
        ax  = axes[row, 0]
        sub = ic_df[ic_df["interconnection"] == ic].sort_values("date")

        for pert, ctrl in ANOMALY_PAIRS.items():
            color = MEMBER_COLORS.get(pert, "grey")
            for member, ls, lw, alpha in [
                (pert, PERT_LS,  2.0, 1.0),
                (ctrl, CTRL_LS,  1.4, 0.7),
            ]:
                msub = sub[sub["data_source"] == member]
                if msub.empty:
                    continue
                line, = ax.plot(
                    msub["date"], msub[y_col],
                    color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                    marker="o", markersize=3,
                    label=display_name(member),
                )
                # Collect legend entries only from the first panel
                if row == 0:
                    legend_handles.append(line)
                    legend_labels.append(display_name(member))

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        # IC name + units on the y-axis instead of a subplot title
        ax.set_ylabel(f"{ic} peak-load\nanomaly ({y_unit})", fontsize=FONTSIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.tick_params(axis="x", labelsize=FONTSIZE,
                       labelbottom=(row == n_ics - 1))
        if as_pct:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        # Keep only 3 y-ticks per panel to avoid crowding
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        # Draw a thin spine between panels
        ax.spines["bottom"].set_linewidth(0.4)

    # x-axis date formatting on the bottom panel only
    axes[-1, 0].xaxis.set_tick_params(rotation=30)

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=min(3, len(legend_handles)),
        fontsize=FONTSIZE,
        frameon=True,
        bbox_to_anchor=(0.5, -0.12),
    )
    #fig.suptitle(
    #    "Daily peak-load anomaly by interconnection",
    #    fontsize=FONTSIZE,
    #    y=0.98,
    #)
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

    Each panel has four lines:
      - SSP245     perturbation (solid,  green)
      - SSP245     control      (dashed, green)
      - SSP245_p12 perturbation (solid,  red)
      - SSP245_p12 control      (dashed, red)
      - SSP245_p24 perturbation (solid,  blue)
      - SSP245_p24 control      (dashed, blue)
      - SSP245_n12 perturbation (solid,  gray)
      - SSP245_n12 control      (dashed, gray)
      - SSP245_n24 perturbation (solid,  brown)
      - SSP245_n24 control      (dashed, brown)

    Layout mirrors the reference image:
      • No subplot titles — interconnection name written on the y-axis.
      • Shared x-axis; tick labels only on the bottom panel.
      • Tight vertical spacing (hspace = 0).
      • Single legend at the bottom of the figure.

    ``sum_load_GW``  is used for perturbation members (absolute pert load).
    ``sum_ctrl_GW``  is used for the matched control (read from pert rows).

    Output: tell_interconnection_load_absolute_timeseries.png
    """
    if "sum_load_GW" not in ic_df.columns:
        log.warning(
            "sum_load_GW not present in aggregated table — "
            "skipping absolute load time-series plot."
        )
        return

    ics   = [ic for ic in INTERCONNECTIONS if ic in ic_df["interconnection"].unique()]
    n_ics = len(ics)

    fig, axes = plt.subplots(
        n_ics, 1,
        figsize=(10, 2.5 * n_ics + 1.2),
        sharex=True,
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0)

    legend_handles: list = []
    legend_labels:  list = []

    for row, ic in enumerate(ics):
        ax  = axes[row, 0]
        sub = ic_df[ic_df["interconnection"] == ic].sort_values("date")

        for pert, ctrl in ANOMALY_PAIRS.items():
            color  = MEMBER_COLORS.get(pert, "grey")
            m_pert = sub[sub["data_source"] == pert]
            if m_pert.empty:
                continue

            line_ctrl, = ax.plot(
                m_pert["date"], m_pert["sum_ctrl_GW"],
                color=color, linestyle=CTRL_LS, linewidth=1.4, alpha=0.7,
                marker="o", markersize=3,
                label=display_name(ctrl),
            )
            line_pert, = ax.plot(
                m_pert["date"], m_pert["sum_load_GW"],
                color=color, linestyle=PERT_LS, linewidth=2.0, alpha=1.0,
                marker="o", markersize=3,
                label=display_name(pert),
            )
            if row == 0:
                legend_handles += [line_ctrl, line_pert]
                legend_labels  += [display_name(ctrl), display_name(pert)]

        # IC name + units as y-axis label; no subplot title
        ax.set_ylabel(f"{ic} (GW)", fontsize=FONTSIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.tick_params(axis="x", labelsize=FONTSIZE,
                       labelbottom=(row == n_ics - 1))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.spines["bottom"].set_linewidth(0.4)

    axes[-1, 0].xaxis.set_tick_params(rotation=30)

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=min(3, len(legend_handles)),
        fontsize=FONTSIZE,
        frameon=True,
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.suptitle(
        "Daily absolute peak load by interconnection\n"
        f"pert (solid) vs ctrl (dashed)  |  {year}",
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
    """Grouped bar chart: daily co-occurrence count, pert vs ctrl pairs.

    One group per day; within each group one bar per member, coloured by
    whether it is a perturbation or control run.

    Output: tell_interconnection_cooccurrence.png
    """
    all_members = sorted_member_order(set(cooc_df["data_source"].unique()))
    dates = sorted(cooc_df["date"].unique())
    n_dates   = len(dates)
    n_members = len(all_members)
    bar_width = 0.8 / n_members
    x         = np.arange(n_dates)
    date_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]

    _ctrl_members = set(ANOMALY_PAIRS.values())

    fig, ax = plt.subplots(figsize=(max(10, n_dates * 1.2), 7))

    for i, member in enumerate(all_members):
        m_sub  = cooc_df[cooc_df["data_source"] == member].set_index("date")
        counts = [int(m_sub.loc[d, "n_stressed"]) if d in m_sub.index else 0
                  for d in dates]
        x_pos  = x + (i - (n_members - 1) / 2.0) * bar_width
        is_ctrl   = member in _ctrl_members
        base_color = MEMBER_COLORS.get(member, "grey")
        rgba = (*mcolors.to_rgb(base_color), 0.35 if is_ctrl else 0.85)
        ax.bar(
            x_pos, counts,
            width=bar_width * 0.9,
            label=display_name(member),
            facecolor=rgba,
            edgecolor=base_color if is_ctrl else "white",
            linewidth=0.8,
            hatch="//" if is_ctrl else "",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(date_labels, rotation=30, ha="right", fontsize=FONTSIZE)
    ax.set_yticks(range(len(INTERCONNECTIONS) + 1))
    ax.set_yticklabels([str(n) for n in range(len(INTERCONNECTIONS) + 1)],
                       fontsize=FONTSIZE)
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
    """Heatmap: date × interconnection, cell = anomaly_GW.

    A red border around a cell indicates it exceeds the stress threshold.
    One subplot per perturbation member only (anomaly = pert − ctrl, so
    showing ctrl values here is not meaningful).

    Output: tell_interconnection_stress_heatmap.png
    """
    pert_members = [m for m in PERTURBATION_MEMBERS
                    if m in ic_df["data_source"].unique()]
    ics   = [ic for ic in INTERCONNECTIONS if ic in ic_df["interconnection"].unique()]
    dates = sorted(ic_df["date"].unique())
    date_labels = [pd.Timestamp(d).strftime("%b %d") for d in dates]

    n_members = len(pert_members)
    fig, axes = plt.subplots(
        1, n_members,
        figsize=(6 * n_members, 3.0 + len(ics) * 0.9),
        squeeze=False,
    )

    # Shared colour scale across all perturbation members
    all_vals = [
        ic_df.loc[ic_df["data_source"] == m, "sum_anomaly_GW"].values
        for m in pert_members
    ]
    vmax = max((np.nanmax(np.abs(v)) for v in all_vals if len(v)), default=1.0)

    for col, member in enumerate(pert_members):
        ax  = axes[0, col]
        sub = ic_df[ic_df["data_source"] == member]

        # Build matrix: rows = interconnections, cols = dates
        mat = np.full((len(ics), len(dates)), np.nan)
        for ci, ic in enumerate(ics):
            for di, d in enumerate(dates):
                cell = sub.loc[
                    (sub["interconnection"] == ic) & (sub["date"] == d),
                    "sum_anomaly_GW",
                ]
                if not cell.empty:
                    mat[ci, di] = float(cell.iloc[0])

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Anomaly (GW)", shrink=0.8)

        # Red borders on cells above threshold
        for ci in range(len(ics)):
            for di in range(len(dates)):
                if not np.isnan(mat[ci, di]) and mat[ci, di] > stress_threshold_gw:
                    ax.add_patch(plt.Rectangle(
                        (di - 0.5, ci - 0.5), 1, 1,
                        fill=False, edgecolor="red", linewidth=2,
                    ))

        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(date_labels, rotation=30, ha="right",
                           fontsize=FONTSIZE)
        ax.set_yticks(range(len(ics)))
        ax.set_yticklabels(ics, fontsize=FONTSIZE)
        ax.set_title(display_name(member), fontsize=FONTSIZE, fontweight="bold")

    fig.suptitle(
        "Interconnection-level load anomaly (GW) per day\n"
        f"Red border = exceeds {stress_threshold_gw:.1f} GW stress threshold  |  {year}",
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
            "--merged-csv filename."
        ),
    )
    p.add_argument(
        "--merged-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Path to tell_heatwave_merged_<year>.csv (produced by "
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
    p.add_argument(
        "--min-hw-frac",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Minimum BA heatwave-area fraction; rows at or below this are dropped.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    merged_csv = args.merged_csv if args.merged_csv is not None else _default_merged_csv(args.year)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
    merged, _ = load_data(merged_csv, args.ba_map, args.min_hw_frac)

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
