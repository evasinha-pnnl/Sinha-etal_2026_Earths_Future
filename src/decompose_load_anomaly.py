"""
decompose_load_anomaly.py
-------------------------
Partial-correlation / multiple-linear-regression decomposition of the
warming-driven peak-electricity-load anomaly.

Scientific question
-------------------
How much of the increase in peak electricity demand under warming arises from
mean temperature shifts versus changes in heatwave spatial extent anomaly, mean
intensity, or peak intensity?

Approach (Analysis 1)
---------------------
Fit an OLS regression per perturbation member:

    peak_load_anomaly_GW  ~  β₀
                           + β₁ · hw_area_anomaly_km2  (spatial extent anomaly)
                           + β₂ · hw_mean_K             (mean intensity anomaly)
                           + β₃ · hw_max_K              (peak intensity anomaly)

Predictors are standardised (zero mean, unit variance) before fitting so
that the resulting standardised coefficients β* are directly comparable —
each β* tells you how many standard deviations of load anomaly are associated
with a one-standard-deviation change in that predictor, holding all others
constant.

Input
-----
``outputs/tell_heatwave_linkage/tell_heatwave_merged_<year>.csv``
Written by ``analyze_tell_heatwave.py --year <year>``.  Required columns:
  date, BA, data_source, peak_load_anomaly, peak_load_ctrl,
  hw_area_frac, hw_area_anomaly_km2, hw_mean_K, hw_max_K

Only rows where ``hw_area_frac > --min-hw-frac`` (default 0.0) are used in
the regression, so the model is fitted exclusively on BAs that experience the
heatwave.  Increase ``--min-hw-frac`` (e.g. to 0.1 or 0.25) to restrict to
BAs substantially covered by the heatwave.

Outputs (all written to OUTPUT_DIR, named with --year)
----------------------------------------------------------
tell_load_decomp_std_coeffs_<year>.png
    Horizontal grouped bar chart of standardised OLS coefficients β* per
    member, with 95 % confidence intervals.

tell_load_decomp_partial_regression_<year>.png
    2 × 3 partial-regression (added-variable) plots — one panel per
    (member × predictor), showing the effect of each predictor after
    partialling out the other two.

tell_load_decomp_regression_table_<year>.csv
    Tidy CSV with columns: member, predictor, beta_std, ci_low_95,
    ci_high_95, p_value, r2_full.

Usage
-----
    conda activate py39_tell
    python decompose_load_anomaly.py [--year YEAR] [--merged-csv PATH] [--output-dir DIR] [--min-hw-frac FRAC]

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
import numpy as np
import pandas as pd
from scipy import stats

from util_constants import FONTSIZE, MEMBER_COLORS, PERTURBATION_MEMBERS

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
_SRC_DIR     = Path(__file__).parent
_ENSEMBLE_OUTPUT = _SRC_DIR / "outputs"
_LINKAGE_DIR = _ENSEMBLE_OUTPUT / "tell_heatwave_linkage"

# Matches Config.year's default in analyze_tell_heatwave.py
DEFAULT_YEAR = 2011

DEFAULT_OUTPUT_DIR = _LINKAGE_DIR


def _default_merged_csv(year: int) -> Path:
    """Path to the tell_heatwave_merged_<year>.csv written by analyze_tell_heatwave.py."""
    return _LINKAGE_DIR / f"tell_heatwave_merged_{year}.csv"

# Perturbation members to analyse (must match data_source values in merged CSV)
PERTURBATION_MEMBERS: list[str] = ["2011_SSP245_p0", "2011_SSP245_p12", "2011_SSP245_p24", "2011_SSP245_n12", "2011_SSP245_n24"]

# Heatwave predictors: (column_name, display_label, scale_for_display)
PREDICTORS: list[tuple[str, str, float]] = [
    ("hw_area_anomaly_km2", "Spatial extent anomaly\n(×10⁶ km²)", 1e6),
    ("hw_mean_K",           "Mean intensity anomaly\n(K)",          1.0),
    ("hw_max_K",            "Peak intensity anomaly\n(K)",           1.0),
]

# Colour map for members — keep consistent with analyze_tell_heatwave.py
MEMBER_COLORS: dict[str, str] = {
    "2011_SSP245_p0":   "#2ca02c",   # green
    "2011_SSP245_p12":  "#d62728",   # red
    "2011_SSP245_p24":  "#9467bd",   # purple
    "2011_SSP245_n12":  "#7f7f7f",   # gray
    "2011_SSP245_n24":  "#8c564b",   # brown
}

FONTSIZE = 12

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_merged(csv_path: Path, min_hw_frac: float = 0.0) -> pd.DataFrame:
    """Load and validate the merged heatwave × load CSV.

    Only rows where ``hw_area_frac > min_hw_frac`` are kept, ensuring the
    regression uses only BAs that genuinely experience the heatwave.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Merged CSV not found: {csv_path}\n"
            "Run analyze_tell_heatwave.py first to generate it."
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"date", "BA", "data_source", "peak_load_anomaly",
                "peak_load_ctrl", "hw_area_frac",
                "hw_area_anomaly_km2", "hw_mean_K", "hw_max_K"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Merged CSV is missing columns: {missing}")

    n_before = len(df)
    df = df[df["hw_area_frac"] > min_hw_frac].copy()
    log.info(
        "Loaded merged CSV: %d rows kept (of %d) after hw_area_frac > %.2f filter "
        "| %d BAs, %d members",
        len(df), n_before, min_hw_frac,
        df["BA"].nunique(), df["data_source"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------
def _standardise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of *df* with *cols* standardised (mean=0, std=1).

    Columns with zero variance are set to 0.0 and a warning is emitted.
    """
    out = df.copy()
    for col in cols:
        mu, sigma = df[col].mean(), df[col].std(ddof=1)
        if sigma > 0:
            out[col] = (df[col] - mu) / sigma
        else:
            log.warning("Column '%s' has zero variance — standardised values set to 0.", col)
            out[col] = 0.0
    return out


def fit_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
) -> dict:
    """Fit OLS via scipy.stats.linregress generalised to multiple predictors.

    Uses the normal-equations approach (pure numpy) so there is no
    dependency on statsmodels.

    Returns a dict with keys:
        beta        — array of all coefficients including intercept (β₀ first)
        beta_std    — slope coefficients only (intercept excluded); these are
                      standardised β* values when the inputs have been
                      standardised before calling this function
        se          — standard errors for all coefficients (intercept included)
        t_values    — t statistics for all coefficients
        p_values    — two-sided p values for all coefficients
        r2          — coefficient of determination R²
        r2_adj      — adjusted R²
        n           — number of observations
        df_resid    — degrees of freedom residual
    """
    sub = df[[y_col] + x_cols].dropna()
    n   = len(sub)
    k   = len(x_cols)            # number of predictors (excl. intercept)

    if n < k + 2:
        log.warning("Too few observations (%d) to fit %d-predictor model.", n, k)
        return {}

    y = sub[y_col].values
    X = np.column_stack([np.ones(n), sub[x_cols].values])  # design matrix

    # Normal equations: β = (XᵀX)⁻¹ Xᵀy
    XtX     = X.T @ X
    Xty     = X.T @ y
    beta    = np.linalg.solve(XtX, Xty)

    # Residuals and variance
    y_hat      = X @ beta
    residuals  = y - y_hat
    df_resid   = n - k - 1
    sigma2     = (residuals @ residuals) / df_resid

    # Standard errors
    cov_beta = sigma2 * np.linalg.inv(XtX)
    se       = np.sqrt(np.diag(cov_beta))

    # t statistics and p values (two-sided)
    t_values = beta / se
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df=df_resid)

    # R²
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = residuals @ residuals
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 else float("nan")

    return dict(
        beta=beta,
        beta_std=beta[1:],      # slope coefficients only (intercept excluded)
        se=se,
        t_values=t_values,
        p_values=p_values,
        r2=r2,
        r2_adj=r2_adj,
        n=n,
        df_resid=df_resid,
    )


def run_regression_all_members(
    df: pd.DataFrame,
    members: list[str],
    predictors: list[tuple[str, str, float]],
) -> pd.DataFrame:
    """Fit standardised OLS for every member and return a tidy results table.

    The response variable is ``peak_load_anomaly`` converted to GW.
    Predictors are standardised before fitting so coefficients are comparable.

    Returns
    -------
    DataFrame with columns:
        member, predictor, beta_std, se_std, ci_low_95, ci_high_95,
        t_value, p_value, r2_full, r2_adj_full, n
    """
    pred_cols   = [p[0] for p in predictors]
    pred_labels = [p[1].replace("\n", " ") for p in predictors]

    rows: list[dict] = []

    for member in members:
        sub = df[df["data_source"] == member].copy()
        sub["y_gw"] = sub["peak_load_anomaly"] / 1e3   # MW → GW

        # Standardise predictors AND response so β* are fully comparable
        std_df = _standardise(sub, pred_cols + ["y_gw"])

        result = fit_ols(std_df, y_col="y_gw", x_cols=pred_cols)
        if not result:
            continue

        df_resid = result["df_resid"]
        t_crit   = stats.t.ppf(0.975, df=df_resid)   # 95 % two-sided CI

        for j, (pred_col, pred_label) in enumerate(zip(pred_cols, pred_labels)):
            beta_j = result["beta_std"][j]
            se_j   = result["se"][j + 1]          # +1 to skip intercept SE
            rows.append({
                "member":       member,
                "predictor":    pred_label,
                "beta_std":     round(beta_j, 4),
                "se_std":       round(se_j, 4),
                "ci_low_95":    round(beta_j - t_crit * se_j, 4),
                "ci_high_95":   round(beta_j + t_crit * se_j, 4),
                "t_value":      round(result["t_values"][j + 1], 4),
                "p_value":      round(result["p_values"][j + 1], 4),
                "r2_full":      round(result["r2"], 4),
                "r2_adj_full":  round(result["r2_adj"], 4),
                "n":            result["n"],
            })
            log.info(
                "  %-20s  %-30s  β*=%+.3f  p=%.3f",
                member, pred_label, beta_j, result["p_values"][j + 1],
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1 — Standardised coefficient bar chart
# ---------------------------------------------------------------------------
def plot_std_coefficients(
    reg_table: pd.DataFrame,
    members: list[str],
    predictors: list[tuple[str, str, float]],
    output_dir: Path,
    year: int = DEFAULT_YEAR,
) -> None:
    """Horizontal grouped bar chart of standardised OLS coefficients.

    One group per predictor; within each group one bar per member.
    Error bars show 95 % confidence intervals.

    Output: tell_load_decomp_std_coeffs.png
    """
    pred_labels = [p[1] for p in predictors]
    n_preds     = len(pred_labels)
    n_members   = len(members)

    bar_height  = 0.35
    group_gap   = 0.1
    group_height = n_members * bar_height + group_gap
    y_centres    = np.arange(n_preds) * group_height

    fig, ax = plt.subplots(figsize=(9, 2.5 + n_preds * 0.9))

    for i, member in enumerate(members):
        color  = MEMBER_COLORS.get(member, "grey")
        sub    = reg_table[reg_table["member"] == member]

        betas  = []
        ci_low = []
        ci_hi  = []
        for pred_label in pred_labels:
            row = sub[sub["predictor"] == pred_label.replace("\n", " ")]
            if row.empty:
                betas.append(0.0); ci_low.append(0.0); ci_hi.append(0.0)
            else:
                betas.append(float(row["beta_std"]))
                ci_low.append(float(row["ci_low_95"]))
                ci_hi.append(float(row["ci_high_95"]))

        betas  = np.array(betas)
        ci_low = np.array(ci_low)
        ci_hi  = np.array(ci_hi)
        xerr   = np.vstack([betas - ci_low, ci_hi - betas])   # shape (2, n_preds)

        y_pos  = y_centres + (i - (n_members - 1) / 2.0) * bar_height

        ax.barh(
            y_pos, betas,
            height=bar_height * 0.85,
            color=color,
            alpha=0.85,
            label=member,
            xerr=xerr,
            error_kw=dict(elinewidth=1.2, capsize=3, ecolor="black"),
        )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(y_centres)
    ax.set_yticklabels(pred_labels, fontsize=FONTSIZE)
    ax.set_xlabel("Standardised coefficient β*", fontsize=FONTSIZE)
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.set_title(
        "OLS decomposition of peak-load anomaly\n"
        f"Standardised coefficients with 95 % CI  —  {year}",
        fontsize=FONTSIZE,
    )
    ax.legend(fontsize=FONTSIZE, loc="lower right")
    fig.tight_layout()

    out = output_dir / f"tell_load_decomp_std_coeffs_{year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Figure 2 — Partial-regression (added-variable) plots
# ---------------------------------------------------------------------------
def plot_partial_regression(
    df: pd.DataFrame,
    members: list[str],
    predictors: list[tuple[str, str, float]],
    output_dir: Path,
    year: int = DEFAULT_YEAR,
) -> None:
    """Partial-regression (added-variable) plots.

    Layout: rows = members, columns = predictors.
    Each panel shows the partial effect of one predictor on the response
    after removing the linear influence of all other predictors.

    Output: tell_load_decomp_partial_regression.png
    """
    pred_cols   = [p[0] for p in predictors]
    pred_labels = [p[1] for p in predictors]
    n_members   = len(members)
    n_preds     = len(pred_cols)

    fig, axes = plt.subplots(
        n_members, n_preds,
        figsize=(5 * n_preds, 4 * n_members),
        squeeze=False,
    )

    for row_idx, member in enumerate(members):
        color = MEMBER_COLORS.get(member, "grey")
        sub   = df[df["data_source"] == member].copy()
        sub["y_gw"] = sub["peak_load_anomaly"] / 1e3

        data  = sub[pred_cols + ["y_gw"]].dropna()
        y_arr = data["y_gw"].values

        for col_idx, (pred_col, pred_label, pred_scale) in enumerate(predictors):
            ax = axes[row_idx, col_idx]

            # Residualise y on all OTHER predictors
            other_cols = [c for c in pred_cols if c != pred_col]
            X_other    = np.column_stack([np.ones(len(data)),
                                          data[other_cols].values])
            b_y        = np.linalg.lstsq(X_other, y_arr, rcond=None)[0]
            resid_y    = y_arr - X_other @ b_y

            # Residualise focal predictor on all OTHER predictors
            x_arr   = data[pred_col].values
            b_x     = np.linalg.lstsq(X_other, x_arr, rcond=None)[0]
            resid_x = x_arr - X_other @ b_x

            # OLS on residuals → partial slope
            slope, intercept, r_val, p_val, _ = stats.linregress(resid_x, resid_y)

            # Scale x residuals for display
            resid_x_disp = resid_x / pred_scale

            ax.scatter(
                resid_x_disp, resid_y,
                color=color, alpha=0.55, s=20, edgecolors="none",
            )
            x_line = np.linspace(resid_x_disp.min(), resid_x_disp.max(), 100)
            ax.plot(
                x_line,
                intercept + slope * pred_scale * x_line,
                color=color, linewidth=1.8, linestyle="--",
            )
            ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
            ax.axvline(0, color="black", linewidth=0.6, linestyle=":")

            # Annotate with partial r and p-value
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                  ("*"   if p_val < 0.05  else "ns"))
            ax.text(
                0.97, 0.05,
                f"r = {r_val:.2f}  {sig}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=FONTSIZE - 1,
            )

            if row_idx == 0:
                ax.set_title(pred_label, fontsize=FONTSIZE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{member}\n\nPartial resid.\npeak-load anomaly (GW)",
                    fontsize=FONTSIZE - 1,
                )
            ax.tick_params(labelsize=FONTSIZE - 1)

            # x-axis label only on bottom row
            if row_idx == n_members - 1:
                ax.set_xlabel(
                    f"Partial resid. {pred_label}",
                    fontsize=FONTSIZE - 1,
                )

    fig.suptitle(
        "Partial-regression plots — load anomaly vs each heatwave characteristic\n"
        f"(each predictor residualised on the other two)  |  {year}",
        fontsize=FONTSIZE,
    )
    fig.tight_layout()

    out = output_dir / f"tell_load_decomp_partial_regression_{year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OLS decomposition of warming-driven peak-load anomaly into "
            "heatwave spatial extent, mean intensity, and peak intensity."
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
            "Path to tell_heatwave_merged_<year>.csv produced by "
            "analyze_tell_heatwave.py. Defaults to the file for --year."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Directory for output figures and CSV table.",
    )
    p.add_argument(
        "--min-hw-frac",
        type=float,
        default=0.0,
        metavar="FRAC",
        help=(
            "Minimum BA heatwave-area fraction (0–1) required for a row to be "
            "included in the regression. Only BAs where the heatwave covers more "
            "than this fraction of BA territory on that day are used. "
            "E.g. 0.1 keeps BAs with ≥10%% overlap."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    merged_csv = args.merged_csv if args.merged_csv is not None else _default_merged_csv(args.year)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data — keep only BAs that genuinely experience the heatwave
    df = load_merged(merged_csv, min_hw_frac=args.min_hw_frac)

    # Keep only perturbation members that are actually present in the data
    members_present = [m for m in PERTURBATION_MEMBERS
                       if m in df["data_source"].unique()]
    if not members_present:
        raise ValueError(
            f"None of {PERTURBATION_MEMBERS} found in merged CSV. "
            f"Available: {df['data_source'].unique().tolist()}"
        )
    log.info("Analysing members: %s", members_present)

    # Fit OLS and build results table
    log.info("Fitting standardised OLS models …")
    reg_table = run_regression_all_members(df, members_present, PREDICTORS)

    # Save regression table
    csv_out = args.output_dir / f"tell_load_decomp_regression_table_{args.year}.csv"
    reg_table.to_csv(csv_out, index=False)
    log.info("Regression table saved → %s", csv_out)
    log.info("\n%s", reg_table.to_string(index=False))

    # Figure 1 — standardised coefficient chart
    log.info("Plotting standardised coefficients …")
    plot_std_coefficients(reg_table, members_present, PREDICTORS, args.output_dir, year=args.year)

    # Figure 2 — partial-regression plots
    log.info("Plotting partial-regression panels …")
    plot_partial_regression(df, members_present, PREDICTORS, args.output_dir, year=args.year)

    log.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
