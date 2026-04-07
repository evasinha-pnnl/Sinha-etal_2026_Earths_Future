"""
mean_shift_decompose.py
-----------------------
⚠️  NOT APPLICABLE TO CURRENT DATA — see explanation below.

Why this analysis cannot be applied to the current EAMxx simulations
---------------------------------------------------------------------
The decomposition requires **non-heatwave days** within the same simulation
window to isolate the background mean-temperature-shift effect (the always-on
warming that is present regardless of whether a heatwave is occurring).

The EAMxx simulations (both perturbation and control) were run only for the
duration of the 2011 July heatwave event:

    ANALYSIS_WINDOW = ("2011-07-17", "2011-07-24")   ← 8 days, all heatwave

Because every day in ``tell_load_daily_all.csv`` falls within the heatwave
period, there are no non-heatwave days from which to estimate the
mean-shift baseline.  Running the script against the current data would
compute the baseline from zero rows and produce trivially wrong results
(baseline ≈ 0, heatwave excess ≈ total anomaly for every BA).

What would be needed to enable this analysis
--------------------------------------------
One of the following data extensions would make the decomposition valid:

  Option A — Extend the simulation window.
    Re-run the EAMxx perturbation and control simulations over a longer period
    that includes both heatwave and non-heatwave days (e.g. the full July 2011
    month, or the full JJA season).  The non-heatwave days within the extended
    window provide a clean baseline.

  Option B — Use climatological TELL output.
    If TELL has been driven by a multi-year climatological EAMxx run (not
    heatwave-targeted), the long-term mean daily load can serve as the
    mean-shift reference.  The baseline would be:

        B(BA, member) = mean peak_load_anomaly over all non-heatwave days
                        in the climatological run for the same calendar month.

  Option C — ERA5-driven TELL as reference.
    Use the ERA5-driven TELL output (``ERA5_obs`` member) over a multi-year
    period to establish the climatological load–temperature relationship.
    Project the EAMxx mean-temperature change onto that relationship to
    estimate how much load increase is attributable to the mean shift alone.

To activate the script once suitable data is available
------------------------------------------------------
  1. Extend ANALYSIS_WINDOW (or supply a separate non-heatwave CSV).
  2. Remove or update this warning block in the docstring.
  3. Remove the ``raise NotImplementedError`` at the top of ``main()``.
  4. Run:  python mean_shift_decompose.py [--output-dir DIR] [--threshold FRAC]

-------------------------------------------------------------------------------

Scientific question
-------------------
How much of the increase in peak electricity demand under warming arises from
**mean temperature shifts** (a background always-on warming effect present
even on non-heatwave days) versus the **extra push from the heatwave itself**
(the additional load that occurs on days when the heatwave is active, above
and beyond the background shift)?

Decomposition
-------------
For each perturbation member and each BA:

  1. **Mean-shift baseline** (B):
     Mean daily peak-load anomaly (perturbation − its matched control) on
     **non-heatwave days** within the analysis window.  These days have the
     same background warming but no active heatwave, so they isolate the
     temperature-mean-shift effect.

  2. **Heatwave excess** (E):
     Mean daily peak-load anomaly on **heatwave days** minus B.
     This isolates the additional load attributable to the heatwave itself.

  Total anomaly on a heatwave day ≈ B + E.

Inputs (all produced by analyze_tell_heatwave.py)
-------------------------------------------------
``outputs/tell_heatwave_linkage/tell_load_daily_all.csv``
    All perturbation-member daily peak-load anomalies (heatwave + non-heatwave
    days) within the analysis window.
    Columns: date, BA, data_source, peak_load, peak_load_anomaly, peak_load_ctrl

``outputs/tell_heatwave_linkage/tell_heatwave_affected_bas.csv``
    Spatial-filter output: heatwave area fraction per (member, date, BA).
    Columns: data_source, date, BA, hw_area_frac

Outputs (all written to OUTPUT_DIR)
------------------------------------
tell_mean_shift_decomp_summary.csv
    Per (member, BA): baseline B, mean heatwave excess E, mean total anomaly,
    fraction of total anomaly attributable to mean shift vs heatwave excess.

tell_mean_shift_decomp_stacked_bar.png
    Stacked bar chart — one group of bars per BA (top BAs), split into
    mean-shift baseline (grey) and heatwave excess (coloured by member).

tell_mean_shift_decomp_timeseries.png
    Daily time-series per BA × member showing total anomaly decomposed into
    the two components (stacked area plot).

tell_mean_shift_decomp_fraction_bar.png
    100 % stacked bar showing what fraction of the total anomaly comes from
    the mean-shift baseline vs heatwave excess, for each BA × member.

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
import matplotlib.ticker as mticker
import numpy as np
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
# Configuration defaults
# ---------------------------------------------------------------------------
_SRC_DIR          = Path(__file__).parent
_ENSEMBLE_OUTPUT  = _SRC_DIR / "outputs"
_LINKAGE_DIR      = _ENSEMBLE_OUTPUT / "tell_heatwave_linkage"

DEFAULT_LOAD_CSV   = _LINKAGE_DIR / "tell_load_daily_all.csv"
DEFAULT_FILTER_CSV = _LINKAGE_DIR / "tell_heatwave_affected_bas.csv"
DEFAULT_OUTPUT_DIR = _LINKAGE_DIR

# Minimum BA heatwave-area fraction to classify a BA×day as a heatwave day.
# Should match the value used when running analyze_tell_heatwave.py.
DEFAULT_HW_FRAC_THRESHOLD = 0.0

# Perturbation members to analyse
PERTURBATION_MEMBERS: list[str] = ["EAMxx_SSP245", "EAMxx_SSP245_p12", "EAMxx_SSP245_p24", "EAMxx_SSP245_n24"]

# Number of top BAs to show in summary plots (ranked by mean total anomaly)
N_TOP_BAS = 8

# Colour map for members — kept consistent with analyze_tell_heatwave.py
MEMBER_COLORS: dict[str, str] = {
    "EAMxx_SSP245":      "#2ca02c",   # green
    "EAMxx_SSP245_p12":  "#d62728",   # red
    "EAMxx_SSP245_p24":  "#9467bd",   # purple
    "EAMxx_SSP245_n24":  "#8c564b",   # brown
}

# Neutral colour for the mean-shift baseline bars
BASELINE_COLOR = "#aec7e8"   # light blue

FONTSIZE = 12

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            "Run analyze_tell_heatwave.py first."
        )


def load_inputs(
    load_csv: Path,
    filter_csv: Path,
    hw_frac_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required CSVs and return (load_daily, hw_flag, heatwave_days).

    Returns
    -------
    load_daily : date, BA, data_source, peak_load_anomaly, peak_load_ctrl
    hw_flag    : date, BA, data_source, hw_area_frac, is_heatwave_day
    """    _check_exists(load_csv,   "Full daily load CSV")
    _check_exists(filter_csv, "Spatial filter CSV")

    load_daily = pd.read_csv(load_csv,   parse_dates=["date"])
    hw_frac    = pd.read_csv(filter_csv, parse_dates=["date"], dtype={"BA": str})

    # Mark heatwave days: hw_area_frac > threshold
    hw_frac["is_heatwave_day"] = hw_frac["hw_area_frac"] > hw_frac_threshold

    log.info(
        "Loaded %d load rows across %d BAs and %d members",
        len(load_daily), load_daily["BA"].nunique(), load_daily["data_source"].nunique(),
    )
    log.info(
        "Spatial filter: %d BA×day entries, threshold=%.2f",
        len(hw_frac), hw_frac_threshold,
    )
    return load_daily, hw_frac


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
def compute_decomposition(
    load_daily: pd.DataFrame,
    hw_frac: pd.DataFrame,
    members: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose peak-load anomaly into mean-shift baseline + heatwave excess.

    Parameters
    ----------
    load_daily : DataFrame with columns date, BA, data_source,
                 peak_load_anomaly, peak_load_ctrl.
    hw_frac    : DataFrame with columns date, BA, data_source,
                 hw_area_frac, is_heatwave_day.
    members    : Perturbation members to analyse.

    Returns
    -------
    daily_decomp : Per (date, BA, member) — peak_load_anomaly_GW, baseline_GW,
                   hw_excess_GW, is_heatwave_day.
    summary      : Per (BA, member) — baseline_GW, hw_excess_GW,
                   mean_total_anomaly_GW, baseline_frac, hw_excess_frac.
    """
    # Merge heatwave flag onto load_daily
    daily = load_daily[load_daily["data_source"].isin(members)].copy()
    daily["peak_load_anomaly_GW"] = daily["peak_load_anomaly"] / 1e3
    daily["peak_load_ctrl_GW"]    = daily["peak_load_ctrl"]    / 1e3

    # Only keep rows for perturbation members in the spatial filter
    hw_flag = hw_frac[hw_frac["data_source"].isin(members)][
        ["date", "BA", "data_source", "hw_area_frac", "is_heatwave_day"]
    ]

    daily = daily.merge(hw_flag, on=["date", "BA", "data_source"], how="left")

    # Days not in the spatial filter have no heatwave information → treat as
    # non-heatwave (is_heatwave_day = False).
    daily["is_heatwave_day"] = daily["is_heatwave_day"].fillna(False)

    # ── Compute mean-shift baseline per BA × member ─────────────────────────
    # Non-heatwave days: is_heatwave_day == False
    non_hw = daily[~daily["is_heatwave_day"]]
    baseline = (
        non_hw
        .groupby(["BA", "data_source"])["peak_load_anomaly_GW"]
        .mean()
        .rename("baseline_GW")
        .reset_index()
    )
    log.info(
        "Baseline computed from non-heatwave days: %d BA×member pairs",
        len(baseline),
    )

    # ── Merge baseline back onto all days ───────────────────────────────────
    daily = daily.merge(baseline, on=["BA", "data_source"], how="left")
    # For BAs that have no non-heatwave day in the window, baseline = 0
    daily["baseline_GW"] = daily["baseline_GW"].fillna(0.0)

    # Heatwave excess on each heatwave day = anomaly - baseline
    daily["hw_excess_GW"] = np.where(
        daily["is_heatwave_day"],
        daily["peak_load_anomaly_GW"] - daily["baseline_GW"],
        0.0,
    )

    # ── Summary table: mean over heatwave days ───────────────────────────────
    hw_days = daily[daily["is_heatwave_day"]]
    if hw_days.empty:
        log.warning("No heatwave days found in daily data — check threshold.")
        summary = pd.DataFrame()
    else:
        summary = (
            hw_days
            .groupby(["BA", "data_source"])
            .agg(
                baseline_GW       = ("baseline_GW",            "first"),
                hw_excess_GW      = ("hw_excess_GW",           "mean"),
                mean_total_GW     = ("peak_load_anomaly_GW",   "mean"),
                n_heatwave_days   = ("is_heatwave_day",        "sum"),
            )
            .reset_index()
        )
        total_abs = summary["baseline_GW"].abs() + summary["hw_excess_GW"].abs()
        summary["baseline_frac"]  = np.where(
            total_abs > 0,
            summary["baseline_GW"].abs() / total_abs,
            np.nan,
        )
        summary["hw_excess_frac"] = np.where(
            total_abs > 0,
            summary["hw_excess_GW"].abs() / total_abs,
            np.nan,
        )
        log.info(
            "Decomposition summary: %d BA×member rows", len(summary)
        )

    return daily, summary


# ---------------------------------------------------------------------------
# Figure 1 — Stacked bar chart: baseline vs heatwave excess per BA
# ---------------------------------------------------------------------------
def plot_stacked_bar(
    summary: pd.DataFrame,
    members: list[str],
    output_dir: Path,
    n_top: int = N_TOP_BAS,
) -> None:
    """Grouped stacked bar chart — mean-shift baseline vs heatwave excess.

    One group per member; within each group one bar per top BA.
    The bar is split into: baseline (light blue) + heatwave excess (member colour).

    Output: tell_mean_shift_decomp_stacked_bar.png
    """
    if summary.empty:
        log.warning("Empty summary — skipping stacked bar chart.")
        return

    # Select top BAs by absolute mean total anomaly, averaged across members
    top_bas = (
        summary.groupby("BA")["mean_total_GW"]
        .mean()
        .abs()
        .nlargest(n_top)
        .index.tolist()
    )
    sub = summary[summary["BA"].isin(top_bas)].copy()

    n_bas      = len(top_bas)
    n_members  = len(members)
    bar_width  = 0.8 / n_members
    x          = np.arange(n_bas)

    fig, ax = plt.subplots(figsize=(max(10, n_bas * 1.4), 6))

    for i, member in enumerate(members):
        color  = MEMBER_COLORS.get(member, "grey")
        m_sub  = sub[sub["data_source"] == member].set_index("BA").reindex(top_bas)

        baseline  = m_sub["baseline_GW"].fillna(0.0).values
        hw_excess = m_sub["hw_excess_GW"].fillna(0.0).values
        x_pos     = x + (i - (n_members - 1) / 2.0) * bar_width

        # Baseline bar (always drawn from zero)
        ax.bar(
            x_pos, baseline,
            width=bar_width * 0.9,
            color=BASELINE_COLOR,
            label=f"Mean-shift baseline ({member})" if i == 0 else "_nolegend_",
            edgecolor="white",
            linewidth=0.4,
        )
        # Heatwave-excess bar stacked on top of baseline
        ax.bar(
            x_pos, hw_excess,
            width=bar_width * 0.9,
            bottom=baseline,
            color=color,
            alpha=0.85,
            label=f"Heatwave excess — {member}",
            edgecolor="white",
            linewidth=0.4,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(top_bas, rotation=30, ha="right", fontsize=FONTSIZE)
    ax.set_ylabel("Peak-load anomaly (GW)", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.set_title(
        "Mean-shift baseline vs heatwave excess — top BAs by mean total anomaly\n"
        "July 2011  |  light blue = background warming  |  colour = heatwave-driven",
        fontsize=FONTSIZE,
    )
    # Compact legend: deduplicate
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=FONTSIZE - 1,
        loc="upper right",
        ncol=1,
        frameon=True,
    )
    fig.tight_layout()

    out = output_dir / "tell_mean_shift_decomp_stacked_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Figure 2 — Daily time-series: stacked-area decomposition per BA × member
# ---------------------------------------------------------------------------
def plot_decomp_timeseries(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    members: list[str],
    output_dir: Path,
    n_top: int = N_TOP_BAS,
) -> None:
    """Stacked-area daily time-series per BA — one subplot per BA, per member.

    Layout: rows = top BAs, columns = members.
    Each panel shows:
      - Grey filled area: mean-shift baseline (constant across the window)
      - Coloured filled area: heatwave excess on heatwave days (zero on others)
      - Black line: observed total anomaly

    Output: tell_mean_shift_decomp_timeseries.png
    """
    if summary.empty:
        log.warning("Empty summary — skipping time-series plot.")
        return

    top_bas = (
        summary.groupby("BA")["mean_total_GW"]
        .mean()
        .abs()
        .nlargest(n_top)
        .index.tolist()
    )

    n_bas     = len(top_bas)
    n_members = len(members)

    fig, axes = plt.subplots(
        n_bas, n_members,
        figsize=(6 * n_members, 3.5 * n_bas),
        sharex=True,
        squeeze=False,
    )

    for row, ba in enumerate(top_bas):
        for col, member in enumerate(members):
            ax    = axes[row, col]
            color = MEMBER_COLORS.get(member, "grey")

            sub = daily[
                (daily["BA"] == ba) & (daily["data_source"] == member)
            ].sort_values("date")

            if sub.empty:
                ax.set_visible(False)
                continue

            dates    = sub["date"].values
            total    = sub["peak_load_anomaly_GW"].values
            baseline = sub["baseline_GW"].values          # constant per BA×member
            excess   = sub["hw_excess_GW"].values         # non-zero only on HW days

            # Baseline band (drawn first so it sits behind excess)
            ax.fill_between(
                dates,
                0, baseline,
                step="mid",
                color=BASELINE_COLOR,
                alpha=0.8,
                label="Mean-shift baseline",
            )
            # Heatwave-excess band stacked on top of baseline
            ax.fill_between(
                dates,
                baseline, baseline + excess,
                step="mid",
                color=color,
                alpha=0.7,
                label="Heatwave excess",
            )
            # Total anomaly line
            ax.step(
                dates, total,
                where="mid",
                color="black",
                linewidth=1.5,
                label="Total anomaly",
            )
            ax.axhline(0, color="black", linewidth=0.6, linestyle=":")

            if col == 0:
                ax.set_ylabel(f"BA: {ba}\n\nAnomaly (GW)", fontsize=FONTSIZE - 1)
            if row == 0:
                ax.set_title(member, fontsize=FONTSIZE, fontweight="bold")
            ax.tick_params(labelsize=FONTSIZE - 1)
            ax.xaxis.label.set_visible(False)

    fig.autofmt_xdate(rotation=30)

    # Single shared legend from the last visible axes
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=FONTSIZE,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Daily peak-load anomaly decomposed into mean-shift baseline + heatwave excess\n"
        "July 2011",
        fontsize=FONTSIZE,
    )
    fig.tight_layout()

    out = output_dir / "tell_mean_shift_decomp_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Figure 3 — Fraction bar chart: what share is baseline vs HW excess?
# ---------------------------------------------------------------------------
def plot_fraction_bar(
    summary: pd.DataFrame,
    members: list[str],
    output_dir: Path,
    n_top: int = N_TOP_BAS,
) -> None:
    """100 % stacked bar showing fraction of |anomaly| from each component.

    Output: tell_mean_shift_decomp_fraction_bar.png
    """
    if summary.empty:
        log.warning("Empty summary — skipping fraction bar chart.")
        return

    top_bas = (
        summary.groupby("BA")["mean_total_GW"]
        .mean()
        .abs()
        .nlargest(n_top)
        .index.tolist()
    )
    sub = summary[summary["BA"].isin(top_bas)].copy()

    n_bas     = len(top_bas)
    n_members = len(members)
    bar_width = 0.8 / n_members
    x         = np.arange(n_bas)

    fig, ax = plt.subplots(figsize=(max(10, n_bas * 1.4), 5))

    for i, member in enumerate(members):
        color = MEMBER_COLORS.get(member, "grey")
        m_sub = sub[sub["data_source"] == member].set_index("BA").reindex(top_bas)

        baseline_frac  = m_sub["baseline_frac"].fillna(0.5).values * 100
        hw_excess_frac = m_sub["hw_excess_frac"].fillna(0.5).values * 100
        x_pos          = x + (i - (n_members - 1) / 2.0) * bar_width

        ax.bar(
            x_pos, baseline_frac,
            width=bar_width * 0.9,
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=0.4,
            label="Mean-shift baseline" if i == 0 else "_nolegend_",
        )
        ax.bar(
            x_pos, hw_excess_frac,
            width=bar_width * 0.9,
            bottom=baseline_frac,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label=f"Heatwave excess — {member}",
        )

    ax.axhline(50, color="black", linewidth=0.7, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(top_bas, rotation=30, ha="right", fontsize=FONTSIZE)
    ax.set_ylabel("Fraction of |total anomaly|", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.set_title(
        "Fraction of peak-load anomaly from mean-shift vs heatwave excess\n"
        "July 2011  |  top BAs by mean absolute total anomaly",
        fontsize=FONTSIZE,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=FONTSIZE - 1,
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()

    out = output_dir / "tell_mean_shift_decomp_fraction_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Decompose warming-driven peak-load anomaly into "
            "mean-shift baseline and heatwave excess."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--load-csv",
        type=Path,
        default=DEFAULT_LOAD_CSV,
        metavar="CSV",
        help="Path to tell_load_daily_all.csv (produced by analyze_tell_heatwave.py).",
    )
    p.add_argument(
        "--filter-csv",
        type=Path,
        default=DEFAULT_FILTER_CSV,
        metavar="CSV",
        help="Path to tell_heatwave_affected_bas.csv (spatial filter cache).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Directory for output figures and summary CSV.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_HW_FRAC_THRESHOLD,
        metavar="FRAC",
        help=(
            "Minimum BA heatwave-area fraction to classify a day as a heatwave day "
            "(should match the value used in analyze_tell_heatwave.py)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()    # parsed early so the dead code below stays valid
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "\n\n"
        "mean_shift_decompose.py is NOT applicable to the current EAMxx data.\n\n"
        "The EAMxx simulations cover only the heatwave window (2011-07-17 – 2011-07-24).\n"
        "Every day in the dataset is a heatwave day, so there are no non-heatwave\n"
        "days from which to estimate the mean-shift baseline.\n\n"
        "See the module docstring for details and for what data would be needed\n"
        "to enable this analysis.\n"
    )

    # Load inputs
    load_daily, hw_frac = load_inputs(
        args.load_csv, args.filter_csv, args.threshold,
    )

    # Keep only perturbation members present in the data
    members_present = [m for m in PERTURBATION_MEMBERS
                       if m in load_daily["data_source"].unique()]
    if not members_present:
        raise ValueError(
            f"None of {PERTURBATION_MEMBERS} found in load CSV. "
            f"Available: {load_daily['data_source'].unique().tolist()}"
        )
    log.info("Analysing members: %s", members_present)

    # Decompose
    log.info("Computing mean-shift / heatwave-excess decomposition …")
    daily_decomp, summary = compute_decomposition(
        load_daily, hw_frac, members_present,
    )

    # Save summary CSV
    csv_out = args.output_dir / "tell_mean_shift_decomp_summary.csv"
    summary.to_csv(csv_out, index=False)
    log.info("Summary saved → %s", csv_out)
    log.info("\n%s", summary.to_string(index=False))

    # Figures
    log.info("Plotting stacked bar chart …")
    plot_stacked_bar(summary, members_present, args.output_dir)

    log.info("Plotting daily time-series decomposition …")
    plot_decomp_timeseries(daily_decomp, summary, members_present, args.output_dir)

    log.info("Plotting fraction bar chart …")
    plot_fraction_bar(summary, members_present, args.output_dir)

    log.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
