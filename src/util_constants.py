"""
util_constants.py
-----------------
Shared configuration constants used across the TELL–heatwave analysis
scripts (analyze_tell_heatwave.py, compare_heatwaves.py,
interconnection_stress.py, mean_shift_decompose.py, plot_affected_ba.py).

Centralising these here avoids copy-paste drift between files.  Input data
locations are also defined here (see "Input data locations" below).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Input data locations
# ---------------------------------------------------------------------------
# By default every path resolves to the NERSC location used for the paper.
# Set the DATA_ROOT environment variable to the unpacked data archive
# instead, and every path resolves to the matching subdirectory below it:
#   export DATA_ROOT=/path/to/EAMxx-TELL-data
DATA_ROOT = os.environ.get("DATA_ROOT")

_NERSC_PROJECT = "/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL"


def _data_path(archive_subdir: str, nersc_path: str) -> Path:
    """Return ``DATA_ROOT/archive_subdir`` if DATA_ROOT is set, else *nersc_path*."""
    if DATA_ROOT:
        return Path(DATA_ROOT) / archive_subdir
    return Path(nersc_path)


# Regridded SCREAM (EAMxx) output, one subdirectory per ensemble member
EAMXX_REGRID_DIR = _data_path("eamxx_regridded", "/pscratch/sd/j/jli628/EAMxx_regridding")
# Reprocessed SCREAM output holding skin temperature (plot_pgw_amplification.py --variable skt)
EAMXX_SKT_DIR    = _data_path("eamxx_skt", "/pscratch/sd/e/esinha/EAMxx")
# PGW temperature deltas (SSP2-4.5 2031-2060 minus 1981-2010)
PGW_DELTA_DIR    = _data_path("pgw_deltas", "/pscratch/sd/j/jli628/EAMxx_regridding/PGW_detals")
# ERA5 2-m temperature over CONUS, 1981-2010 (heatwave threshold and land mask)
ERA5_DIR         = _data_path("era5", "/pscratch/sd/e/esinha/ERA5_CONUS")
# TELL hourly load output, one subdirectory per ensemble member / scenario
TELL_OUTPUT_DIR  = _data_path("tell_outputs", f"{_NERSC_PROJECT}/TELL_outputs")
# EIA-930 balance files (EIA930_BALANCE_<year>_<Jan_Jun|Jul_Dec>.csv)
EIA930_DIR       = _data_path("eia930", f"{_NERSC_PROJECT}/EIA_930")
# im3components wrf_to_tell files: grid-cell -> county weights, plus
# wrf_to_tell_ancillary_data/ (BA service territory, county shapefile)
WRF_TO_TELL_DIR  = _data_path(
    "wrf_to_tell", f"{_NERSC_PROJECT}/im3components/im3components/wrf_to_tell"
)

# Intermediate products (per-member heatwave maps, CSVs, caches) and figures.
# Override with OUTPUT_ROOT to read archived intermediate products from
# elsewhere; defaults to src/outputs/ next to these scripts.
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", Path(__file__).parent / "outputs"))

# ---------------------------------------------------------------------------
# Ensemble member colours  (consistent across all figures)
# ---------------------------------------------------------------------------
MEMBER_COLORS: dict[str, str] = {
    "ERA5_obs_2011":     "#000000",   # black
    "ERA5_obs_2021":     "#000000",   # black
    "2011_CTRL_p0":      "#0000ff",   # blue
    "2011_CTRL_p12":     "#ffa500",   # orange
    "2011_CTRL_p24":     "#ff0000",   # red
    "2011_CTRL_n12":     "#7f7f7f",   # gray
    "2011_CTRL_n24":     "#008000",   # green
    "2011_SSP245_p0":    "#0000ff",   # blue
    "2011_SSP245_p12":   "#ffa500",   # orange
    "2011_SSP245_p24":   "#ff0000",   # red
    "2011_SSP245_n12":   "#7f7f7f",   # gray
    "2011_SSP245_n24":   "#008000",   # green
    "2021_CTRL_p0":      "#2a78d6",   # blue
    "2021_CTRL_p24":     "#eb6834",   # orange
    "2021_CTRL_n24":     "#1baf7a",   # aqua
    "2021_SSP245_p0":    "#2a78d6",   # blue
    "2021_SSP245_p24":   "#eb6834",   # orange
    "2021_SSP245_n24":   "#1baf7a",   # aqua
}

# ---------------------------------------------------------------------------
# Anomaly pairs: perturbation member → its matched control member
# ---------------------------------------------------------------------------
ANOMALY_PAIRS: dict[str, str] = {
    "2011_SSP245_p0":    "2011_CTRL_p0",
    "2011_SSP245_p12":   "2011_CTRL_p12",
    "2011_SSP245_p24":   "2011_CTRL_p24",
    "2011_SSP245_n12":   "2011_CTRL_n12",
    "2011_SSP245_n24":   "2011_CTRL_n24",
    "2021_SSP245_p0":    "2021_CTRL_p0",
    "2021_SSP245_p24":   "2021_CTRL_p24",
    "2021_SSP245_n24":   "2021_CTRL_n24",
}

PERTURBATION_MEMBERS: list[str] = list(ANOMALY_PAIRS.keys())

# ---------------------------------------------------------------------------
# Ensemble-summary styles: one entry per experiment (mean line + range band)
# ---------------------------------------------------------------------------
EXPERIMENT_STYLES: dict[str, dict[str, str]] = {
    "CTRL":   dict(label="Control", color="#2a78d6"),
    "SSP245": dict(label="PGW",     color="#eb6834"),
}


def members_by_experiment(available) -> dict[str, list[str]]:
    """Split the member labels in *available* into Control and PGW lists.

    Keys match EXPERIMENT_STYLES ("CTRL", "SSP245"); members keep the
    ANOMALY_PAIRS order, so Control and PGW lists line up pair by pair.
    """
    available = set(available)
    return {
        "CTRL":   [m for m in ANOMALY_PAIRS.values() if m in available],
        "SSP245": [m for m in ANOMALY_PAIRS.keys()   if m in available],
    }

# ---------------------------------------------------------------------------
# Default font size for figures
# ---------------------------------------------------------------------------
FONTSIZE: int = 14


# ---------------------------------------------------------------------------
# Shared helper: human-readable display name for a member label
# ---------------------------------------------------------------------------
_DISPLAY_NAMES: dict[str, str] = {
    "ERA5_obs_2011":     "ERA5 2011",
    "ERA5_obs_2021":     "ERA5 2021",
    "2011_CTRL_p0":      "Control",
    "2011_CTRL_p12":     "Control +12h",
    "2011_CTRL_p24":     "Control +24h",
    "2011_CTRL_n12":     "Control -12h",
    "2011_CTRL_n24":     "Control -24h",
    "2011_SSP245_p0":    "SSP245",
    "2011_SSP245_p12":   "SSP245 +12h",
    "2011_SSP245_p24":   "SSP245 +24h",
    "2011_SSP245_n12":   "SSP245 -12h",
    "2011_SSP245_n24":   "SSP245 -24h",
    "2021_CTRL_p0":      "2021 Control",
    "2021_CTRL_p24":     "2021 Control +24h",
    "2021_CTRL_n24":     "2021 Control -24h",
    "2021_SSP245_p0":    "2021 SSP245",
    "2021_SSP245_p24":   "2021 SSP245 +24h",
    "2021_SSP245_n24":   "2021 SSP245 -24h",
}


def display_name(member: str) -> str:
    """Return a short human-readable label for a member string.

    ``"2011_CTRL_p0"`` → ``"Control"``,
    ``"2011_SSP245_p12"`` → ``"SSP245 +12h"``, etc.
    Members not in the lookup table are returned unchanged.
    """
    return _DISPLAY_NAMES.get(member, member)


# ---------------------------------------------------------------------------
# Shared helper: ctrl-then-PGW member ordering for consistent legends
# ---------------------------------------------------------------------------
def sorted_member_order(available: set[str]) -> list[str]:
    """Return members in ctrl₀, PGW₀, ctrl₁, PGW₁, … order.

    With ``ncol=4`` this produces a two-row legend where the top row is all
    control members (dashed lines / hatched bars) and the bottom row is all
    PGW members (solid lines / filled bars).

    Parameters
    ----------
    available : set of member labels present in the current dataset/filter.

    Returns
    -------
    Ordered list; members not covered by ANOMALY_PAIRS are appended at the
    end (e.g. ERA5_obs).
    """
    ordered: list[str] = []
    for pgw, ctrl in ANOMALY_PAIRS.items():
        if ctrl in available:
            ordered.append(ctrl)
        if pgw in available:
            ordered.append(pgw)
    for m in available:
        if m not in ordered:
            ordered.append(m)
    return ordered
