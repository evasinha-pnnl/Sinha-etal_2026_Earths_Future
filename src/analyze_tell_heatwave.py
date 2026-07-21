"""
analyze_tell_heatwave.py
------------------------
Links TELL estimated electricity load to heatwave events identified by
identify_heatwaves / run_heatwave_ensemble.

Everything below is selected by a single ``--year`` CLI argument (2011 or
2021): the TELL member directories, the perturbation↔control anomaly pairs,
the analysis window, and the plot label all resolve together from that one
value (see ``_resolve_hw_tell_config`` / ``_resolve_anomaly_pairs`` and the
``Config`` dataclass), so they can't drift out of sync with each other.

Workflow
--------
1.  Load TELL hourly load CSVs for every member of the selected year's TELL
    member set (10 members for 2011: 5 CTRL + 5 SSP245 offsets p0/p12/p24/
    n12/n24; currently 2 active members for 2021 — CTRL_p0 and SSP245_p0 —
    the rest are commented out in ``TELL_MEMBER_DIRS_2021`` until those
    simulations are complete).
2.  Compute load anomaly = perturbation member load − its *matched* control
    load, per BA, for each (perturbation, control) pair defined for the
    selected year, e.g. for 2011: SSP245_p0 − CTRL_p0, SSP245_p12 − CTRL_p12,
    SSP245_p24 − CTRL_p24, SSP245_n12 − CTRL_n12, SSP245_n24 − CTRL_n24.
3.  Load per-member heatwave event CSVs produced by run_heatwave_ensemble
    (largest contiguous area, mean/max anomaly time series), and compute the
    heatwave-area anomaly (perturbation − matched control).
4.  Aggregate hourly load to daily peak (max) per BA per member.
5.  **Spatial filter**: for each perturbation member, recompute the daily
    heatwave mask on the ERA5 grid, map it onto the EAMxx native grid via
    ``grid_cell_to_ba.EAMxx.parquet``, and for each day identify only those
    BAs whose territory has ≥ ``HW_BA_FRAC_THRESHOLD`` fractional overlap
    with active heatwave cells.  Peak-load anomaly rows for BAs that are
    *not* under the heatwave on a given day are dropped.
6.  Merge heatwave characteristics (area, duration, mean/peak intensity) with
    the spatially-filtered daily peak-load anomaly over the selected year's
    analysis window.
7.  Produce output figures and tables:
    a. Bar chart — number of heatwave-affected BAs per day per member
       (``plot_affected_bas_per_day``).
    b. Per-BA peak-load-anomaly time series — one figure per (unit, BA set)
       combination: GW / % of control, for a fixed BA set and for the BAs
       with the largest mean anomaly (``plot_load_anomaly_timeseries_per_ba_comparison``,
       4 figures total).
    c. Per-BA heatwave-fraction time series — BA fractional area under the
       heatwave per day / member (``_plot_ba_comparison_panel_affected_area``).
    d. Scatter matrix — peak-load anomaly vs heatwave spatial extent, mean
       intensity, and peak intensity, one point per (BA × day), coloured by
       member, with a per-member OLS trend line (``plot_scatter_matrix``).
    e. Correlation table — Pearson r between peak-load anomaly and each
       heatwave characteristic, per member, saved as
       ``tell_heatwave_correlations_<year>.csv`` (``compute_correlations``).

All output figures/CSVs are named with the selected ``--year`` (e.g.
``tell_heatwave_merged_2011.csv``, ``tell_load_vs_heatwave_scatter_2021.png``),
so runs for different years share ``cfg.output_dir`` without overwriting
each other.

Configuration
-------------
Use --year to select the heatwave event/year (2011 or 2021). Edit TELL_BASE
or the TELL_MEMBER_DIRS_* / ANALYSIS_WINDOW_* / FIXED_BAS_*_* constants at
the top of this file to change paths, add/re-enable members as new
simulations complete, change which BAs are highlighted in the per-BA
comparison plots, or add a new event/year — ANOMALY_PAIRS (in
util_constants.py) is filtered down to whichever pairs exist in the
selected year's TELL_MEMBER_DIRS automatically.

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Re-use heatwave mask computation from the ensemble pipeline
from identify_heatwaves import (
    ERA5_HIST_CONFIG,
    HIST_CONFIG,
    PERCENTILE_VALUE,
    compute_threshold,
    load_and_prepare,
)
from run_heatwave_ensemble import (
    EAMXX_BASE_PATH,
    _MEMBER_LOOKUP,
)
from util_constants import ANOMALY_PAIRS, FONTSIZE, MEMBER_COLORS, sorted_member_order, display_name

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
# Configuration  ← defaults; all overridable via CLI args or env vars
# ---------------------------------------------------------------------------

# Root directory of TELL output CSVs.
# Override with the TELL_BASE environment variable:
#   export TELL_BASE=/path/to/TELL_outputs
_TELL_BASE_DEFAULT = (
    "/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL/TELL_outputs"
)
TELL_BASE = Path(os.environ.get("TELL_BASE", _TELL_BASE_DEFAULT))

# ── July 2011 heatwave ──────────────────────────────────────────────────────
TELL_MEMBER_DIRS_2011: dict[str, str] = {
    "2011_CTRL_p0":      "2011_CTRL_p0",
    "2011_CTRL_p12":     "2011_CTRL_p12",
    "2011_CTRL_p24":     "2011_CTRL_p24",
    "2011_CTRL_n12":     "2011_CTRL_n12",
    "2011_CTRL_n24":     "2011_CTRL_n24",
    "2011_SSP245_p0":    "2011_SSP245_p0",
    "2011_SSP245_p12":   "2011_SSP245_p12",
    "2011_SSP245_p24":   "2011_SSP245_p24",
    "2011_SSP245_n12":   "2011_SSP245_n12",
    "2011_SSP245_n24":   "2011_SSP245_n24",
}
ANALYSIS_WINDOW_2011 = ("2011-07-17", "2011-07-28")
YEAR_2011            = 2011

# BAs highlighted in the per-BA comparison plots (see
# _plot_ba_comparison_panel_affected_area / _plot_ba_comparison_panel).
#FIXED_BAS_AFFECTED_AREA_2011: list[str] = ["PJM", "MISO", "SWPP", "ERCO"]
FIXED_BAS_AFFECTED_AREA_2011: list[str] = ["AZPS", "PACE", "PNM", "WACM"]
FIXED_BAS_LOAD_ANOMALY_2011:  list[str] = ["AECI", "LGEE", "SWPP", "PJM"]

# ── June–July 2021 heatwave (24 June – 2 July 2021) ────────────────────────
# Update TELL subdirectory names once the 2021 TELL outputs are available.
TELL_MEMBER_DIRS_2021: dict[str, str] = {
    "2021_CTRL_p0":      "2021_CTRL_p0",           # ← update subdirectory names
    # "2021_CTRL_p12":     "2021_CTRL_p12",
    # "2021_CTRL_p24":     "2021_CTRL_p24",
    # "2021_CTRL_n12":     "2021_CTRL_n12",
    # "2021_CTRL_n24":     "2021_CTRL_n24",
    "2021_SSP245_p0":    "2021_SSP245_p0",
    # "2021_SSP245_p12":   "2021_SSP245_p12",
    # "2021_SSP245_p24":   "2021_SSP245_p24",
    # "2021_SSP245_n12":   "2021_SSP245_n12",
    # "2021_SSP245_n24":   "2021_SSP245_n24",
}
ANALYSIS_WINDOW_2021 = ("2021-06-24", "2021-07-10")
YEAR_2021            = 2021

# BAs highlighted in the per-BA comparison plots for the 2021 heatwave.
# Update these once the 2021 spatial filter results show which BAs are
# under the heatwave for this event.
FIXED_BAS_AFFECTED_AREA_2021: list[str] = ["BPAT", "GCPD", "IPCO", "PACW"]
FIXED_BAS_LOAD_ANOMALY_2021:  list[str] = ["BPAT", "GCPD", "IPCO", "PACW"]

# ── Per-year configs, selected at runtime via --year (see Config / main()) ──
_HW_TELL_CONFIGS: dict[int, tuple[dict[str, str], tuple[str, str], str, list[str], list[str]]] = {
    YEAR_2011: (
        TELL_MEMBER_DIRS_2011, ANALYSIS_WINDOW_2011, "July 2011",
        FIXED_BAS_AFFECTED_AREA_2011, FIXED_BAS_LOAD_ANOMALY_2011,
    ),
    YEAR_2021: (
        TELL_MEMBER_DIRS_2021, ANALYSIS_WINDOW_2021, "June–July 2021",
        FIXED_BAS_AFFECTED_AREA_2021, FIXED_BAS_LOAD_ANOMALY_2021,
    ),
}


def _resolve_hw_tell_config(
    year: int,
) -> tuple[dict[str, str], tuple[str, str], str, list[str], list[str]]:
    """Return (tell_member_dirs, analysis_window, hw_label,
    fixed_bas_affected_area, fixed_bas_load_anomaly) for *year*."""
    if year not in _HW_TELL_CONFIGS:
        raise ValueError(
            f"Unknown year {year} for TELL/heatwave configuration. "
            f"Valid choices: {sorted(_HW_TELL_CONFIGS)}"
        )
    return _HW_TELL_CONFIGS[year]


def _resolve_anomaly_pairs(tell_member_dirs: dict[str, str]) -> dict[str, str]:
    """Restrict ANOMALY_PAIRS to the perturbation/control pairs present in
    *tell_member_dirs*, so functions that loop over anomaly pairs don't pull
    in members from a different heatwave event/year."""
    return {
        pert: ctrl for pert, ctrl in ANOMALY_PAIRS.items()
        if pert in tell_member_dirs and ctrl in tell_member_dirs
    }

# Heatwave ensemble output root (produced by run_heatwave_ensemble.py)
ENSEMBLE_OUTPUT = Path(__file__).parent / "outputs"

# Heatwave CSV tags & their value columns (must match _save_timeseries output)
HW_AREA_TAG   = "largest_contiguous_heatwave_area_timeseries"
HW_MEAN_TAG   = "mean_anomaly_largest_contiguous_heatwave_area_timeseries"
HW_MAX_TAG    = "max_anomaly_largest_contiguous_heatwave_area_timeseries"
HW_AREA_COL   = "largest_region_area_km2"
HW_MEAN_COL   = "largest_region_mean_anomaly_K"
HW_MAX_COL    = "largest_region_max_anomaly_K"

VARNAMES = "t2m"

# Output directory for TELL–heatwave linkage figures
OUTPUT_DIR = ENSEMBLE_OUTPUT / "tell_heatwave_linkage"

# MEMBER_COLORS, ANOMALY_PAIRS, and FONTSIZE are imported from util_constants.

# ---------------------------------------------------------------------------
# Spatial filtering configuration
# ---------------------------------------------------------------------------
# Parquet mapping EAMxx grid-cell index → BA Code (built by build_grid_to_ba.py)
GRID_TO_BA_PQ = Path(__file__).parent / "grid_cell_to_ba.EAMxx.parquet"

# EAMxx native grid dimension used for flat-index decoding (ilat * EAMXX_NLON + ilon)
EAMXX_NLON = 1841

# Minimum fraction of a BA's weighted grid-cell area that must lie under
# the heatwave for that BA to be considered "affected" on a given day.
# 0.0 ⇒ any overlap counts; 0.5 ⇒ majority overlap required.
HW_BA_FRAC_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Runtime configuration  (populated in main() from CLI args)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Holds all runtime-configurable settings for one analysis run.

    Functions that previously read module-level globals (TELL_MEMBER_DIRS,
    ANALYSIS_WINDOW, HW_LABEL, ANOMALY_PAIRS, …) now accept a ``Config``
    instance, which makes them safe to call from tests or other scripts
    without mutating global state — and keeps the member set, analysis
    window, and label consistent with ``year`` instead of drifting from a
    separate hardcoded selector.
    """
    year:                 int   = YEAR_2011
    hw_ba_frac_threshold: float = HW_BA_FRAC_THRESHOLD
    tell_base:            Path  = field(default_factory=lambda: TELL_BASE)
    output_dir:           Path  = field(default_factory=lambda: ENSEMBLE_OUTPUT / "tell_heatwave_linkage")
    tell_member_dirs:     dict[str, str]  = field(default_factory=lambda: TELL_MEMBER_DIRS_2011)
    analysis_window:      tuple[str, str] = ANALYSIS_WINDOW_2011
    hw_label:             str = "July 2011"
    anomaly_pairs:        dict[str, str]  = field(
        default_factory=lambda: _resolve_anomaly_pairs(TELL_MEMBER_DIRS_2011)
    )
    fixed_bas_affected_area: list[str] = field(
        default_factory=lambda: list(FIXED_BAS_AFFECTED_AREA_2011)
    )
    fixed_bas_load_anomaly:  list[str] = field(
        default_factory=lambda: list(FIXED_BAS_LOAD_ANOMALY_2011)
    )


# ---------------------------------------------------------------------------
# Step 1 – Load TELL hourly data
# ---------------------------------------------------------------------------
def load_tell_all_members(cfg: Config) -> pd.DataFrame:
    """
    Read all BA CSVs for every member and return a single DataFrame with
    columns: Time_UTC, Load, BA, data_source.

    Only hours within cfg.analysis_window are kept.
    """
    dfs: list[pd.DataFrame] = []
    t_start, t_end = cfg.analysis_window

    for data_source, subdir in cfg.tell_member_dirs.items():
        tell_dir = cfg.tell_base / subdir / str(cfg.year)
        csv_files = sorted(tell_dir.glob(f"*_{cfg.year}_mlp_output.csv"))
        if not csv_files:
            log.warning("No TELL CSVs found for %s in %s", data_source, tell_dir)
            continue

        member_dfs: list[pd.DataFrame] = []
        for fpath in csv_files:
            df = pd.read_csv(fpath, parse_dates=["Time_UTC"])
            df = df[
                (df["Time_UTC"] >= t_start) &
                (df["Time_UTC"] <= t_end + " 23:59:59")
            ]
            member_dfs.append(df)

        if member_dfs:
            combined = pd.concat(member_dfs, ignore_index=True)
            combined["data_source"] = data_source
            dfs.append(combined)
            log.info("Loaded TELL for %s: %d BA-hours", data_source, len(combined))

    if not dfs:
        raise FileNotFoundError(
            f"No TELL data loaded. Check TELL_BASE={cfg.tell_base} and cfg.analysis_window."
        )
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 2 – Compute load anomaly (perturbation − its matched control)
# ---------------------------------------------------------------------------
def compute_load_anomaly(tell_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    For each perturbation member, compute load_anomaly against its matched
    control as defined in cfg.anomaly_pairs (ANOMALY_PAIRS restricted to the
    members of the active year's TELL_MEMBER_DIRS), e.g. for 2011:

        2011_SSP245_p0   − 2011_CTRL_p0
        2011_SSP245_p12  − 2011_CTRL_p12
        2011_SSP245_p24  − 2011_CTRL_p24
        2011_SSP245_n12  − 2011_CTRL_n12
        2011_SSP245_n24  − 2011_CTRL_n24

    Returns only the perturbation rows with an added ``load_anomaly`` column.
    """
    result_dfs: list[pd.DataFrame] = []

    for pert_member, ctrl_member in cfg.anomaly_pairs.items():
        ctrl = (
            tell_df[tell_df["data_source"] == ctrl_member]
            [["Time_UTC", "BA", "Load"]]
            .rename(columns={"Load": "Load_ctrl"})
        )
        pert = tell_df[tell_df["data_source"] == pert_member].copy()
        merged = pert.merge(ctrl, on=["Time_UTC", "BA"], how="left")
        merged["load_anomaly"] = merged["Load"] - merged["Load_ctrl"]
        result_dfs.append(merged)
        log.info(
            "Anomaly: %s − %s  (%d BA-hour pairs)",
            pert_member, ctrl_member, merged["load_anomaly"].notna().sum(),
        )

    return pd.concat(result_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3 – Load heatwave characteristic CSVs
# ---------------------------------------------------------------------------
def _load_hw_csv(data_source: str, tag: str, value_col: str, cfg: Config) -> pd.DataFrame:
    """Load one heatwave timeseries CSV and return (date, value_col) columns."""
    fname = (
        ENSEMBLE_OUTPUT
        / data_source
        / f"{data_source}_{VARNAMES}_{tag}_{cfg.year}.csv"
    )
    if not fname.exists():
        log.warning("Heatwave CSV not found: %s", fname)
        return pd.DataFrame(columns=["date", value_col])
    df = pd.read_csv(fname, parse_dates=["time"])
    df["date"] = df["time"].dt.normalize()
    n_before = len(df)
    df = df[["date", value_col]].drop_duplicates("date")
    if len(df) < n_before:
        log.debug(
            "_load_hw_csv: dropped %d duplicate date rows from %s",
            n_before - len(df), fname.name,
        )
    return df


def load_heatwave_characteristics(cfg: Config) -> pd.DataFrame:
    """
    Load area, mean-anomaly, and max-anomaly heatwave CSVs for all perturbation
    members (keys of ANOMALY_PAIRS) and return a merged DataFrame with columns:
    data_source, date, hw_area_km2, hw_area_anomaly_km2, hw_mean_K, hw_max_K.

    ``hw_area_km2``         — absolute heatwave area in the perturbation run.
    ``hw_area_anomaly_km2`` — area anomaly = perturbation area − matched control
                              area (analogous to hw_mean_K / hw_max_K).
    """
    rows: list[pd.DataFrame] = []
    for data_source, ctrl_source in cfg.anomaly_pairs.items():
        area_df      = _load_hw_csv(data_source, HW_AREA_TAG, HW_AREA_COL, cfg)
        ctrl_area_df = _load_hw_csv(ctrl_source,  HW_AREA_TAG, HW_AREA_COL, cfg)
        mean_df      = _load_hw_csv(data_source, HW_MEAN_TAG, HW_MEAN_COL, cfg)
        max_df       = _load_hw_csv(data_source, HW_MAX_TAG,  HW_MAX_COL,  cfg)

        if area_df.empty:
            continue

        merged = (
            area_df
            .merge(mean_df, on="date", how="outer")
            .merge(max_df,  on="date", how="outer")
        )
        merged = merged.rename(columns={
            HW_AREA_COL: "hw_area_km2",
            HW_MEAN_COL: "hw_mean_K",
            HW_MAX_COL:  "hw_max_K",
        })

        # Compute area anomaly: perturbation − matched control
        if not ctrl_area_df.empty:
            ctrl_area_df = ctrl_area_df.rename(
                columns={HW_AREA_COL: "_ctrl_area_km2"}
            )
            merged = merged.merge(ctrl_area_df, on="date", how="left")
            merged["hw_area_anomaly_km2"] = (
                merged["hw_area_km2"] - merged["_ctrl_area_km2"]
            )
            merged = merged.drop(columns=["_ctrl_area_km2"])
        else:
            log.warning(
                "Control area CSV not found for %s → hw_area_anomaly_km2 will be NaN",
                data_source,
            )
            merged["hw_area_anomaly_km2"] = float("nan")

        merged["data_source"] = data_source
        rows.append(merged)

    if not rows:
        raise FileNotFoundError(
            "No heatwave CSVs found. Run run_heatwave_ensemble.py first."
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4 – Aggregate load to daily peak per BA
# ---------------------------------------------------------------------------
def daily_peak_load(tell_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly load (and anomaly) to daily peak (max) per BA × member."""
    tell_df = tell_df.copy()
    tell_df["date"] = tell_df["Time_UTC"].dt.normalize()

    agg = (
        tell_df
        .groupby(["date", "BA", "data_source"], as_index=False)
        .agg(
            peak_load=("Load", "max"),
            peak_load_anomaly=("load_anomaly", "max"),
            peak_load_ctrl=("Load_ctrl", "max"),
        )
    )
    return agg


# ---------------------------------------------------------------------------
# Step 5 – Spatial filtering helpers (heatwave mask → BA overlap fractions)
# ---------------------------------------------------------------------------
def _load_grid_to_ba() -> pd.DataFrame:
    """Load the grid_cell_to_ba parquet and cache it."""
    log.info("Loading grid-cell → BA mapping from %s", GRID_TO_BA_PQ)
    df = pd.read_parquet(GRID_TO_BA_PQ)
    log.info("  %d rows, %d unique cells, %d unique BAs",
             len(df), df["cell_index"].nunique(), df["BA_Code"].nunique())
    return df


def _cell_index_to_latlon(
    cell_indices: np.ndarray,
    eamxx_lat: np.ndarray,
    eamxx_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat EAMxx cell indices to (lat, lon) coordinate arrays.

    cell_index = ilat * EAMXX_NLON + ilon
    """
    ilat = cell_indices // EAMXX_NLON
    ilon = cell_indices % EAMXX_NLON
    return eamxx_lat[ilat], eamxx_lon[ilon]


def _load_eamxx_grid_coords() -> tuple[np.ndarray, np.ndarray]:
    """Read lat/lon coordinate arrays from the first EAMxx file on disk.

    Uses 2011_CTRL_p0  as the reference member — all members share the same
    native grid, so any member would work.
    """
    member = _MEMBER_LOOKUP["2011_CTRL_p0"]
    fpath = Path(EAMXX_BASE_PATH) / member.subdir
    nc_files = sorted(fpath.glob(f"{member.fname_abb}*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No EAMxx files found in {fpath}")
    with xr.open_dataset(nc_files[0]) as ds:
        lat = ds["lat"].values.copy()
        lon = ds["lon"].values.copy()
    log.info("EAMxx grid: %d lats × %d lons", len(lat), len(lon))
    return lat, lon


def compute_heatwave_mask_for_member(
    data_source: str,
    ds_hist: xr.DataArray,
    ds_q90: xr.DataArray,
) -> xr.DataArray:
    """Recompute the daily heatwave mask for a given perturbation member.

    Parameters
    ----------
    data_source : str — member label (key in _MEMBER_LOOKUP).
    ds_hist     : DataArray — ERA5 daily-max historical (for the target grid).
    ds_q90      : DataArray — ERA5-derived day-of-year threshold.

    Returns
    -------
    heatwave_mask : DataArray (time, lat, lon) on the ERA5 CONUS grid.
        True where daily Tmax exceeds the 90th-percentile threshold.
        Time coordinate is normalised to numpy datetime64.
    """
    member = _MEMBER_LOOKUP[data_source]
    eamxx_data_cfg = member.to_eamxx_cfg()
    analysis_cfg   = member.to_analysis_cfg()

    heatwave_mask, _, _, _ = load_and_prepare(
        eamxx_data_cfg, analysis_cfg,
        ds_hist=ds_hist, ds_q90=ds_q90,
    )

    return heatwave_mask


def _nearest_index(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """For each value, return the index of the nearest element in *grid*.

    Uses ``np.searchsorted`` on the sorted grid — O(n log m), constant
    memory — instead of building an (n × m) distance matrix.
    """
    sorted_grid = np.sort(grid)
    sort_order = np.argsort(grid)

    idx = np.searchsorted(sorted_grid, values, side="left")
    idx = np.clip(idx, 1, len(sorted_grid) - 1)

    # Pick whichever neighbour is closer
    left = sorted_grid[idx - 1]
    right = sorted_grid[idx]
    pick_left = (np.abs(values - left) <= np.abs(values - right))
    nearest_sorted_idx = np.where(pick_left, idx - 1, idx)

    # Map back to the original (potentially unsorted) grid order
    return sort_order[nearest_sorted_idx]


def compute_daily_hw_bas(
    heatwave_mask: xr.DataArray,
    grid_to_ba: pd.DataFrame,
    eamxx_lat: np.ndarray,
    eamxx_lon: np.ndarray,
) -> dict[str, dict[str, float]]:
    """For each day in the heatwave mask, compute the heatwave area fraction for all BAs.

    For every (date, BA) pair, returns the fraction of the BA's weighted
    grid-cell area that lies under an active heatwave cell.  Thresholding
    is deferred to the caller (``apply_spatial_filter``).

    Returns
    -------
    dict mapping ISO date string → {BA_Code: hw_area_frac (0–1)}.
    """
    era5_lat = heatwave_mask.lat.values
    era5_lon = heatwave_mask.lon.values
    unique_cells = grid_to_ba["cell_index"].unique()
    u_lats, u_lons = _cell_index_to_latlon(unique_cells, eamxx_lat, eamxx_lon)
    u_ilat = _nearest_index(era5_lat, u_lats)
    u_ilon = _nearest_index(era5_lon, u_lons)
    cell_to_ilat = pd.Series(u_ilat, index=unique_cells)
    cell_to_ilon = pd.Series(u_ilon, index=unique_cells)
    row_ilat = grid_to_ba["cell_index"].map(cell_to_ilat).values
    row_ilon = grid_to_ba["cell_index"].map(cell_to_ilon).values
    ba_total_weight = grid_to_ba.groupby("BA_Code")["ba_weight"].sum()
    daily_ba_fracs: dict[str, dict[str, float]] = {}
    for t in heatwave_mask.time.values:
        day_str = str(pd.Timestamp(t).date())
        mask_2d = heatwave_mask.sel(time=t).values
        hw_flags = mask_2d[row_ilat, row_ilon].astype(bool)
        hw_ba_weight = grid_to_ba["ba_weight"].values * hw_flags
        ba_hw_weight = pd.Series(hw_ba_weight, index=grid_to_ba.index).groupby(
            grid_to_ba["BA_Code"]
        ).sum()
        hw_frac = ba_hw_weight / ba_total_weight
        daily_ba_fracs[day_str] = hw_frac.to_dict()
        log.info("  %s: %d / %d BAs have any heatwave overlap",
                 day_str, (hw_frac > 0).sum(), len(ba_total_weight))
    return daily_ba_fracs


def load_spatial_filter_from_csv(
    csv_path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Reconstruct the spatial filter from a previously saved CSV.

    The CSV must have columns: data_source, date, BA, hw_area_frac
    (written by the slow-path save block in ``main()``).

    Returns
    -------
    Same structure as ``build_spatial_filter``:
    member → {date_str → {BA_Code → hw_area_frac}}
    """
    log.info("Loading spatial filter from cached CSV: %s", csv_path)
    df = pd.read_csv(csv_path, dtype={"BA": str})
    spatial_filter: dict[str, dict[str, dict[str, float]]] = {}
    for (member, date_str), grp in df.groupby(["data_source", "date"]):
        spatial_filter.setdefault(member, {})[date_str] = dict(
            zip(grp["BA"], grp["hw_area_frac"])
        )
    log.info("  Loaded %d members from cached filter", len(spatial_filter))
    return spatial_filter


def build_spatial_filter(
    members: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Build the full spatial filter: member → {date → {BA_Code: hw_frac}}."""
    grid_to_ba = _load_grid_to_ba()
    eamxx_lat, eamxx_lon = _load_eamxx_grid_coords()
    log.info("Computing ERA5 threshold for spatial filter …")
    ds_hist, ds_q90 = compute_threshold(
        ERA5_HIST_CONFIG, HIST_CONFIG, PERCENTILE_VALUE,
    )
    spatial_filter: dict[str, dict[str, dict[str, float]]] = {}
    for member_name in members:
        log.info("Computing heatwave mask for %s …", member_name)
        heatwave_mask = compute_heatwave_mask_for_member(
            member_name, ds_hist, ds_q90,
        )
        log.info("  Heatwave mask shape: %s, time range: %s → %s",
                 heatwave_mask.shape,
                 str(heatwave_mask.time.values[0])[:10],
                 str(heatwave_mask.time.values[-1])[:10])
        daily_ba_fracs = compute_daily_hw_bas(
            heatwave_mask, grid_to_ba, eamxx_lat, eamxx_lon,
        )
        spatial_filter[member_name] = daily_ba_fracs
    return spatial_filter


def apply_spatial_filter(
    load_daily: pd.DataFrame,
    spatial_filter: dict[str, dict[str, dict[str, float]]],
    cfg: Config,
) -> pd.DataFrame:
    """Filter daily peak-load anomaly to only heatwave-affected BAs per day.

    Parameters
    ----------
    load_daily     : DataFrame with columns date, BA, data_source, peak_load_anomaly.
    spatial_filter : member → {date_str → {BA_Code: hw_area_frac}}.
    cfg            : Config instance — ``cfg.hw_ba_frac_threshold`` sets the
                     minimum hw_area_frac for a BA to be kept.

    Returns
    -------
    Filtered copy of load_daily with an added ``hw_area_frac`` column so that
    downstream scripts (e.g. decompose_load_anomaly.py) can apply their own
    stricter overlap threshold without re-running the spatial filter.
    """
    keep_mask = pd.Series(False, index=load_daily.index)
    # Build a lookup: (member, date, BA) → hw_area_frac for all kept rows
    frac_records: list[dict] = []

    for member, daily_bas in spatial_filter.items():
        for date_str, ba_frac_dict in daily_bas.items():
            # Keep only BAs whose heatwave fraction exceeds the threshold
            affected_bas = {ba: frac for ba, frac in ba_frac_dict.items()
                            if frac > cfg.hw_ba_frac_threshold}
            date_val = pd.Timestamp(date_str)
            row_mask = (
                (load_daily["data_source"] == member) &
                (load_daily["date"] == date_val) &
                (load_daily["BA"].isin(affected_bas))
            )
            keep_mask |= row_mask
            for ba, frac in affected_bas.items():
                frac_records.append({
                    "data_source": member,
                    "date": date_val,
                    "BA": ba,
                    "hw_area_frac": frac,
                })

    filtered = load_daily[keep_mask].copy()

    # Attach hw_area_frac so the merged CSV carries it for downstream use
    if frac_records:
        frac_df = pd.DataFrame(frac_records)
        filtered = filtered.merge(frac_df, on=["data_source", "date", "BA"], how="left")

    log.info("Spatial filter: %d / %d rows kept (%.1f%%)",
             len(filtered), len(load_daily),
             100 * len(filtered) / max(len(load_daily), 1))
    return filtered

# ---------------------------------------------------------------------------
# Step 7a – Per-BA comparison: subplots per BA, lines per member
# ---------------------------------------------------------------------------
def _top_bas_by_hw_frac(
    spatial_filter: dict[str, dict[str, dict[str, float]]],
    perturbation_members: list[str],
    n: int = 5,
) -> list[str]:
    """Return the *n* BAs with the highest mean heatwave-area fraction,
    averaged across all perturbation members and days."""
    rows = []
    for member in perturbation_members:
        for date_str, ba_frac_dict in spatial_filter.get(member, {}).items():
            for ba, frac in ba_frac_dict.items():
                rows.append({"BA": ba, "hw_area_frac": frac})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    top = (
        df.groupby("BA")["hw_area_frac"]
        .mean()
        .nlargest(n)
        .index.tolist()
    )
    return top


def _top_bas_by_peak_anomaly(
    load_anom_daily: pd.DataFrame,
    perturbation_members: list[str],
    n: int = 4,
) -> list[str]:
    """Return the *n* BAs with the highest mean absolute peak-load anomaly,
    averaged across all perturbation members and days."""
    sub = load_anom_daily[load_anom_daily["data_source"].isin(perturbation_members)]
    if sub.empty:
        return []
    top = (
        sub.groupby("BA")["peak_load_anomaly"]
        .mean()
        .abs()
        .nlargest(n)
        .index.tolist()
    )
    return top

# ---------------------------------------------------------------------------
def _plot_ba_comparison_panel_affected_area(
    spatial_filter: dict[str, dict[str, dict[str, float]]],
    cfg: Config,
    *,
    bas: list[str] | None = None,
    fname_suffix: str = "",
    subtitle: str = "",
) -> None:
    """Create a per-BA comparison figure — one narrow panel per BA, shared x-axis.

    Layout: one panel per BA stacked vertically, no subplot titles, BA name on
    y-axis, shared x-axis with tick labels only on the bottom panel, zero
    reference line, single legend at the bottom, tight vertical spacing.

    Parameters
    ----------
    spatial_filter:  dict[str, dict[str, dict[str, float]]],
    bas             : Explicit list of BA codes to show (default:
                      cfg.fixed_bas_affected_area).
    fname_suffix    : String appended before ``.png`` to distinguish outputs.
    subtitle        : Optional subtitle line.
    """
    top_bas = bas if bas is not None else cfg.fixed_bas_affected_area
    n_bas   = len(top_bas)

    _ctrl_members = set(cfg.anomaly_pairs.values())
    _cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _color(member: str, idx: int) -> str:
        return MEMBER_COLORS.get(member, _cycle_colors[idx % len(_cycle_colors)])

    rows = []
    for member, daily_bas in spatial_filter.items():
        for date_str, ba_frac_dict in daily_bas.items():
            for ba, frac in ba_frac_dict.items():
                rows.append({
                "data_source": member,
                "BA": ba,
                "date": pd.Timestamp(date_str),
                "hw_area_frac": frac
            })

    df = pd.DataFrame(rows)

    members = list(spatial_filter)
    # Reorder: ctrl_0, PGW_0, ctrl_1, PGW_1, … so the legend reads
    # "ctrl row" on top and "PGW row" on the bottom with ncol=4.
    members = sorted_member_order(set(members))

    fig, axes = plt.subplots(
        n_bas, 1,
        figsize=(12, 3.2 * n_bas + 1.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0)

    legend_handles: list = []
    legend_labels:  list = []


    for row, ba in enumerate(top_bas):
        ax = axes[row, 0]

        for i, member in enumerate(members):
            color = _color(member, i)
            is_ctrl = member in _ctrl_members
            sub = df[
                (df["data_source"] == member) &
                (df["BA"] == ba)
            ].sort_values("date")

            if sub.empty:
                continue

            line, = ax.plot(
                sub["date"], sub["hw_area_frac"],
                color=color,
                linewidth=2,
                linestyle="--" if is_ctrl else "-",
                marker="o",
                markersize=3,
                label=display_name(member)
            )
            if display_name(member) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(display_name(member))

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(f"{ba} fraction\nin heatwave", fontsize=FONTSIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.tick_params(axis="x", labelsize=FONTSIZE,
                       labelbottom=(row == n_bas - 1))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        ax.spines["bottom"].set_linewidth(0.4)

    axes[-1, 0].xaxis.set_tick_params(rotation=30)

    fname = f"ba_comparison{fname_suffix}_{cfg.year}.png"

    if subtitle:
        fig.suptitle(subtitle, fontsize=FONTSIZE)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=min(5, len(legend_handles)),
            fontsize=FONTSIZE,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )
    else:
        log.warning("No data for any BA in %s; skipping legend for %s", top_bas, fname)

    out = cfg.output_dir / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)

# ---------------------------------------------------------------------------
def _plot_ba_comparison_panel(
    load_anom_daily: pd.DataFrame,
    hw_chars: pd.DataFrame,
    spatial_filter: dict[str, dict[str, dict[str, float]]],
    cfg: Config,
    *,
    as_pct: bool = False,
    bas: list[str] | None = None,
    fname_suffix: str = "",
    subtitle: str = "",
) -> None:
    """Create a per-BA comparison figure — one narrow panel per BA, shared x-axis.

    Layout: one panel per BA stacked vertically, no subplot titles, BA name on
    y-axis, shared x-axis with tick labels only on the bottom panel, zero
    reference line, single legend at the bottom, tight vertical spacing.

    Parameters
    ----------
    load_anom_daily : DataFrame with columns date, BA, data_source,
                      peak_load_anomaly, peak_load_ctrl.
    hw_chars        : Accepted for API compatibility; not plotted here.
    spatial_filter  : Accepted for API compatibility; not used here.
    as_pct          : If True plot anomaly as % of control; otherwise in GW.
    bas             : Explicit list of BA codes to show (default:
                      cfg.fixed_bas_load_anomaly).
    fname_suffix    : String appended before ``.png`` to distinguish outputs.
    subtitle        : Optional subtitle line.
    """
    top_bas = bas if bas is not None else cfg.fixed_bas_load_anomaly
    members = list(cfg.anomaly_pairs)
    n_bas   = len(top_bas)

    _cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _color(member: str, idx: int) -> str:
        return MEMBER_COLORS.get(member, _cycle_colors[idx % len(_cycle_colors)])

    y_unit = "% ctrl" if as_pct else "GW"

    fig, axes = plt.subplots(
        n_bas, 1,
        figsize=(10, 3.2 * n_bas + 1.5),
        sharex=True,
        sharey=as_pct,
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0)

    legend_handles: list = []
    legend_labels:  list = []

    for row, ba in enumerate(top_bas):
        ax = axes[row, 0]

        for i, member in enumerate(members):
            color = _color(member, i)
            sub = load_anom_daily[
                (load_anom_daily["data_source"] == member) &
                (load_anom_daily["BA"] == ba)
            ].sort_values("date")

            if sub.empty:
                continue

            if as_pct:
                y = (
                    sub["peak_load_anomaly"]
                    / sub["peak_load_ctrl"].replace(0, np.nan)
                    * 100.0
                )
            else:
                y = sub["peak_load_anomaly"] / 1e3   # MW → GW

            line, = ax.plot(
                sub["date"], y,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=display_name(member)
            )
            if display_name(member) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(display_name(member))

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(f"{ba} peak-load\nanomaly ({y_unit})", fontsize=FONTSIZE, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE)
        ax.tick_params(axis="x", labelsize=FONTSIZE,
                       labelbottom=(row == n_bas - 1))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
        ax.spines["bottom"].set_linewidth(0.4)

    axes[-1, 0].xaxis.set_tick_params(rotation=30)

    if as_pct:
        fname = f"tell_load_anomaly_pct_timeseries_per_ba_comparison{fname_suffix}_{cfg.year}.png"
    else:
        fname = f"tell_load_anomaly_timeseries_per_ba_comparison{fname_suffix}_{cfg.year}.png"

    if subtitle:
        fig.suptitle(subtitle, fontsize=FONTSIZE)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=min(3, len(legend_handles)),
            fontsize=FONTSIZE,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )
    else:
        log.warning("No data for any BA in %s; skipping legend for %s", top_bas, fname)

    out = cfg.output_dir / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)

# ---------------------------------------------------------------------------

def plot_load_anomaly_timeseries_per_ba_comparison(
    load_anom_daily: pd.DataFrame,
    hw_chars: pd.DataFrame,
    spatial_filter: dict[str, dict[str, dict[str, float]]],
    cfg: Config,
) -> None:
    """Plot per-BA comparison of peak-load anomaly across perturbation members.

    Produces four figures in two pairs (absolute GW and % of control):

    **Pair 1 — fixed BAs with strong heatwave signal** (AECI, LGEE, SWPP, PJM):
      tell_load_anomaly_timeseries_per_ba_comparison.png
      tell_load_anomaly_pct_timeseries_per_ba_comparison.png

    **Pair 2 — top BAs by largest mean peak-load anomaly**:
      tell_load_anomaly_timeseries_per_ba_comparison_top_anomaly.png
      tell_load_anomaly_pct_timeseries_per_ba_comparison_top_anomaly.png
    """
    # ── Pair 1: fixed BAs with strongest heatwave signal ────────────────────
    _plot_ba_comparison_panel(
        load_anom_daily, hw_chars, spatial_filter, cfg,
        as_pct=False,
    )
    _plot_ba_comparison_panel(
        load_anom_daily, hw_chars, spatial_filter, cfg,
        as_pct=True,
    )

    # ── Pair 2: BAs with largest mean absolute peak-load anomaly ────────────
    members = list(cfg.anomaly_pairs)
    top_anom_bas = _top_bas_by_peak_anomaly(load_anom_daily, members, n=4)
    if top_anom_bas:
        _plot_ba_comparison_panel(
            load_anom_daily, hw_chars, spatial_filter, cfg,
            as_pct=False,
            bas=top_anom_bas,
            fname_suffix="_top_anomaly",
        )
        _plot_ba_comparison_panel(
            load_anom_daily, hw_chars, spatial_filter, cfg,
            as_pct=True,
            bas=top_anom_bas,
            fname_suffix="_top_anomaly",
        )
    else:
        log.warning("No BAs found for top-anomaly comparison plot; skipping.")


# ---------------------------------------------------------------------------
# Step 7b – Scatter matrix: peak-load anomaly vs heatwave characteristics
# ---------------------------------------------------------------------------
def plot_scatter_matrix(merged: pd.DataFrame, cfg: Config) -> None:
    """
    Three scatter sub-plots, one per heatwave characteristic:
      - spatial extent anomaly  (hw_area_anomaly_km2)
      - mean intensity          (hw_mean_K)
      - peak intensity          (hw_max_K)

    Each point = one (BA × day × member).  Points are coloured by member.
    A per-member OLS trend line is added.
    """
    characteristics = [
        ("hw_area_anomaly_km2", "Heatwave spatial extent anomaly (×10⁶ km²)", 1e6),
        ("hw_mean_K",           "Heatwave mean intensity (K)",                  1.0),
        ("hw_max_K",            "Heatwave peak intensity (K)",                  1.0),
    ]

    members = list(cfg.anomaly_pairs)   # perturbation members only

    fig, axes = plt.subplots(1, len(characteristics), figsize=(6 * len(characteristics), 5))

    for ax, (hw_col, xlabel, scale) in zip(axes, characteristics):
        for member in members:
            sub = merged[merged["data_source"] == member].dropna(
                subset=[hw_col, "peak_load_anomaly"]
            )
            if sub.empty:
                continue
            x = sub[hw_col] / scale
            y = sub["peak_load_anomaly"] / 1e3   # → GW

            ax.scatter(
                x, y,
                color=MEMBER_COLORS.get(member, "grey"),
                alpha=0.55,
                s=30,
                label=display_name(member),
                edgecolors="none",
            )

            # OLS trend line
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(
                    x_line, np.polyval(coeffs, x_line),
                    color=MEMBER_COLORS.get(member, "grey"),
                    linewidth=1.8,
                    linestyle="--",
                )

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
        ax.set_ylabel("Peak load anomaly vs ctrl (GW)", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE - 1)

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(),
        loc="lower center",
        ncol=min(3, len(members)),
        bbox_to_anchor=(0.5, -0.08),
        fontsize=FONTSIZE,
        frameon=True,
    )

    fig.suptitle(
        f"Peak-load anomaly vs heatwave characteristics — {cfg.hw_label}\n"
        "(each point = one BA × day; dashed = OLS trend per member)",
        fontsize=FONTSIZE,
    )
    fig.tight_layout()

    out = cfg.output_dir / f"tell_load_vs_heatwave_scatter_{cfg.year}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ---------------------------------------------------------------------------
# Step 7c – Correlation summary table
# ---------------------------------------------------------------------------
def compute_correlations(merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute Pearson r between peak_load_anomaly and each heatwave
    characteristic, stratified by member.

    Returns a tidy DataFrame and saves it as a CSV.
    """
    hw_cols = {
        "hw_area_anomaly_km2": "Spatial extent anomaly",
        "hw_mean_K":           "Mean intensity",
        "hw_max_K":            "Peak intensity",
    }
    rows = []
    for member in cfg.anomaly_pairs:   # perturbation members only
        sub = merged[merged["data_source"] == member].dropna(
            subset=list(hw_cols.keys()) + ["peak_load_anomaly"]
        )
        for col, label in hw_cols.items():
            if len(sub) < 3:
                r = float("nan")
            else:
                r = sub["peak_load_anomaly"].corr(sub[col])
            rows.append({"Member": member, "HW characteristic": label, "Pearson r": round(r, 3)})

    corr_df = pd.DataFrame(rows)
    out_csv = cfg.output_dir / f"tell_heatwave_correlations_{cfg.year}.csv"
    corr_df.to_csv(out_csv, index=False)
    log.info("Correlation table saved → %s", out_csv)
    log.info("\n%s", corr_df.to_string(index=False))
    return corr_df


# ---------------------------------------------------------------------------
# Step 7d – Bar chart: number of heatwave-affected BAs per day
# ---------------------------------------------------------------------------
def plot_affected_bas_per_day(
    spatial_filter: dict[str, dict[str, dict[str, float]]],
    cfg: Config,
) -> None:
    """Bar chart showing how many BAs are under the heatwave each day,
    for all ensemble members (controls and perturbations)."""
    rows = []
    for member, daily_bas in spatial_filter.items():
        for date_str, ba_frac_dict in daily_bas.items():
            n_affected = sum(1 for frac in ba_frac_dict.values()
                             if frac > cfg.hw_ba_frac_threshold)
            rows.append({
                "date": pd.Timestamp(date_str),
                "data_source": member,
                "n_affected_bas": n_affected,
            })
    df = pd.DataFrame(rows).sort_values(["data_source", "date"])

    members = list(spatial_filter)
    # Reorder: ctrl_0, PGW_0, ctrl_1, PGW_1, … so the legend reads
    # "ctrl row" on top and "PGW row" on the bottom with ncol=4.
    members = sorted_member_order(set(members))
    n_members = len(members)
    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.8 / n_members          # total group width ≈ 0.8
    x = np.arange(n_dates)

    _ctrl_members = set(cfg.anomaly_pairs.values())

    for i, member in enumerate(members):
        sub = df[df["data_source"] == member].set_index("date").reindex(dates)
        is_ctrl = member in _ctrl_members
        base_color = MEMBER_COLORS.get(member, "grey")
        # Use an RGBA facecolor so alpha only dims the fill, not the hatch.
        # The hatch edgecolor is set separately at full opacity so it stays visible.
        rgba = (*mcolors.to_rgb(base_color), 0.25 if is_ctrl else 0.85)
        ax.bar(
            x + i * bar_width,
            sub["n_affected_bas"].fillna(0),
            bar_width,
            label=display_name(member),
            facecolor=rgba,
            edgecolor=base_color if is_ctrl else "white",
            linewidth=0.8,
            hatch="//" if is_ctrl else "",
        )

    ax.set_xticks(x + bar_width * (n_members - 1) / 2)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=30)
    ax.set_ylabel("Number of BAs under heatwave", fontsize=FONTSIZE)
    ax.set_title(
        f"Threshold: fraction in heatwave > {cfg.hw_ba_frac_threshold:.2f}  |  {cfg.hw_label}\n",
        fontsize=FONTSIZE,
    )


    ax.legend(
        loc="upper center",
        ncol=min(5, n_members),
        bbox_to_anchor=(0.5, -0.3),
        fontsize=FONTSIZE - 1,
        frameon=True,
    )
    fig.tight_layout()

    out = cfg.output_dir / f"tell_affected_bas_per_day_{cfg.year}_{cfg.hw_ba_frac_threshold}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)
    out_csv = cfg.output_dir / f"tell_affected_bas_per_day_{cfg.year}_{cfg.hw_ba_frac_threshold}.csv"
    df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    All arguments are optional; defaults fall back to the module-level
    constants so the script can also be imported and called programmatically.
    """
    p = argparse.ArgumentParser(
        description="Link TELL electricity load to EAMxx heatwave events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--year",
        type=int,
        default=YEAR_2011,
        choices=sorted(_HW_TELL_CONFIGS),
        help=(
            "Heatwave event/year to analyse — selects the TELL member "
            "directories, analysis window, and label (used to locate TELL "
            "CSVs and heatwave output)."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=HW_BA_FRAC_THRESHOLD,
        metavar="FRAC",
        help=(
            "Minimum fraction of a BA's area that must lie under the heatwave "
            "for that BA to be considered affected (0.0 = any overlap, "
            "0.5 = majority overlap)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for output figures and CSVs. "
            f"Defaults to outputs/tell_heatwave_linkage/ "
            f"relative to this script."
        ),
    )
    p.add_argument(
        "--tell-base",
        type=Path,
        default=TELL_BASE,
        metavar="DIR",
        help=(
            "Root directory of TELL output CSVs. "
            "Can also be set via the TELL_BASE environment variable; "
            "the CLI flag takes precedence."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Resolve the TELL member set / analysis window / label for the selected
    # year, and restrict ANOMALY_PAIRS to that year's members, so every
    # downstream function is consistent with args.year instead of a
    # separately hardcoded selector.
    (
        tell_member_dirs, analysis_window, hw_label,
        fixed_bas_affected_area, fixed_bas_load_anomaly,
    ) = _resolve_hw_tell_config(args.year)
    anomaly_pairs = _resolve_anomaly_pairs(tell_member_dirs)

    # Build a Config from CLI args; no module-level globals are mutated.
    cfg = Config(
        year                    = args.year,
        hw_ba_frac_threshold    = args.threshold,
        tell_base               = args.tell_base,
        output_dir              = args.output_dir
                                  if args.output_dir is not None
                                  else ENSEMBLE_OUTPUT / "tell_heatwave_linkage",
        tell_member_dirs        = tell_member_dirs,
        analysis_window         = analysis_window,
        hw_label                = hw_label,
        anomaly_pairs           = anomaly_pairs,
        fixed_bas_affected_area = fixed_bas_affected_area,
        fixed_bas_load_anomaly  = fixed_bas_load_anomaly,
    )

    log.info(
        "Configuration: year=%d (%s)  threshold=%.2f  tell_base=%s  output_dir=%s  members=%s",
        cfg.year, cfg.hw_label, cfg.hw_ba_frac_threshold, cfg.tell_base, cfg.output_dir,
        list(cfg.tell_member_dirs),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load TELL hourly data for all members
    log.info("Loading TELL hourly load data …")
    tell_df = load_tell_all_members(cfg)

    # 2. Compute load anomaly (each perturbation − its matched control)
    log.info(
        "Computing load anomalies: %s",
        ", ".join(f"{p} − {c}" for p, c in cfg.anomaly_pairs.items()),
    )
    tell_anom = compute_load_anomaly(tell_df, cfg)

    # 3. Aggregate to daily peak
    log.info("Aggregating to daily peak load …")
    load_daily = daily_peak_load(tell_anom)

    # 4. Load heatwave characteristics
    log.info("Loading heatwave characteristics …")
    hw_chars = load_heatwave_characteristics(cfg)

    # 5. Spatial filtering: keep only BAs overlapping the heatwave per day
    all_members = list(cfg.tell_member_dirs.keys())          # all members for this year
    perturbation_members = list(cfg.anomaly_pairs.keys())    # perturbation only
    filter_csv = cfg.output_dir / f"tell_heatwave_affected_bas_{cfg.year}.csv"

    if filter_csv.exists():
        # ── Validate cached spatial filter ──────────────────────────────────
        _probe = pd.read_csv(filter_csv, nrows=0)
        # Guard 1: old-format CSV that lacks the hw_area_frac column.
        if "hw_area_frac" not in _probe.columns:
            log.warning(
                "Cached spatial filter %s is missing column 'hw_area_frac' "
                "(old format) — deleting and recomputing.", filter_csv
            )
            filter_csv.unlink()
        else:
            # Guard 2: cached CSV was built with a different (smaller) member
            # set — e.g. new ensemble members have been added since the last
            # run.  Any member in cfg.tell_member_dirs that is absent from
            # the cache means we must rebuild.
            _cached_members = set(pd.read_csv(filter_csv, usecols=["data_source"])["data_source"])
            _expected_members = set(all_members)
            _missing = _expected_members - _cached_members
            if _missing:
                log.warning(
                    "Cached spatial filter is missing members %s — "
                    "deleting and recomputing.", sorted(_missing)
                )
                filter_csv.unlink()

#    if filter_csv.exists():
#        # Guard 3: cached CSV doesn't cover the full analysis window — e.g.
#        # more simulation days were added for an existing member since the
#        # cache was built (member set unchanged, so Guard 2 wouldn't catch
#        # this).
#        _cached_dates = pd.read_csv(filter_csv, usecols=["data_source", "date"])
#        _cached_dates["date"] = pd.to_datetime(_cached_dates["date"])
#        _window_start = pd.Timestamp(analysis_window[0])
#        _window_end = pd.Timestamp(analysis_window[1])
#        _stale_members = [
#            member for member in all_members
#            if (_mdates := _cached_dates.loc[_cached_dates["data_source"] == member, "date"]).empty
#            or _mdates.max() < _window_end
#            or _mdates.min() > _window_start
#        ]
#        if _stale_members:
#            log.warning(
#                "Cached spatial filter does not cover the full analysis "
#                "window %s for members %s — deleting and recomputing.",
#                analysis_window, _stale_members,
#            )
#            filter_csv.unlink()

    if filter_csv.exists():
        log.info(
            "Cached spatial filter found (%s) — skipping recomputation.", filter_csv
        )
        spatial_filter = load_spatial_filter_from_csv(filter_csv)
    else:
        # ── Slow path: recompute from heatwave masks ─────────────────────────
        log.info("Building spatial filter (heatwave mask → BA overlap) …")
        spatial_filter = build_spatial_filter(all_members)

        # Save BA heatwave area fractions CSV (columns: data_source, date, BA, hw_area_frac)
        filter_rows = []
        for member, daily_bas in spatial_filter.items():
            for date_str, ba_frac_dict in daily_bas.items():
                for ba, hw_frac in ba_frac_dict.items():
                    filter_rows.append({
                        "data_source": member,
                        "date":        date_str,
                        "BA":          ba,
                        "hw_area_frac": hw_frac,
                    })
        filter_df = pd.DataFrame(filter_rows)
        filter_df.to_csv(filter_csv, index=False)
        log.info(
            "Spatial filter saved → %s  (%d rows)", filter_csv, len(filter_df)
        )

    # For load-anomaly filtering, use only perturbation members
    perturbation_filter = {
        m: spatial_filter[m] for m in perturbation_members
    }
    log.info("Applying spatial filter to daily peak-load anomaly …")
    load_daily_filtered = apply_spatial_filter(load_daily, perturbation_filter, cfg)

    # 6. Merge filtered load anomaly with heatwave characteristics
    log.info("Merging filtered load anomaly with heatwave characteristics …")
    merged = load_daily_filtered.merge(hw_chars, on=["date", "data_source"], how="inner")
    log.info("Merged dataset: %d rows (BA × day × member)", len(merged))

    # Save full (unfiltered) daily peak-load anomaly for downstream analysis
    # (needed by mean-shift decomposition in mean_shift_decompose.py)
    load_daily_csv = cfg.output_dir / f"tell_load_daily_all_{cfg.year}.csv"
    load_daily[load_daily["data_source"].isin(perturbation_members)].to_csv(
        load_daily_csv, index=False
    )
    log.info("Full daily load saved → %s", load_daily_csv)

    # Save merged table for further analysis
    merged_csv = cfg.output_dir / f"tell_heatwave_merged_{cfg.year}.csv"
    merged.to_csv(merged_csv, index=False)
    log.info("Merged data saved → %s", merged_csv)

    # 7. Plots
    log.info("Plotting affected BAs per day …")
    plot_affected_bas_per_day(spatial_filter, cfg)

    log.info("Plotting per-BA load anomaly timeseries comparison …")
    plot_load_anomaly_timeseries_per_ba_comparison(
        load_daily_filtered, hw_chars, spatial_filter, cfg
    )

    _plot_ba_comparison_panel_affected_area(
    spatial_filter, cfg, fname_suffix="_affected_area", subtitle="BA fraction under heatwave")

    log.info("Plotting scatter matrix (spatially filtered) …")
    plot_scatter_matrix(merged, cfg)

    log.info("Computing correlations (spatially filtered) …")
    compute_correlations(merged, cfg)

    log.info("Done. Outputs in %s", cfg.output_dir)


if __name__ == "__main__":
    main()
