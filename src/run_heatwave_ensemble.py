"""
run_heatwave_ensemble.py
------------------------
Loops over PGW ensemble members and runs the full heatwave analysis pipeline
(from identify_heatwaves) for each member.

Ensemble members
----------------
Five members are currently defined:

  ERA5 observational baseline:
    ERA5_obs  — ERA5 July 2011, days 1–31, from ERA5_HIST_CONFIG path

  EAMxx PGW simulations under /pscratch/sd/j/jli628/EAMxx_regridding/:

  Member name           Subdir
  ─────────────────     ───────────────────────────
  EAMxx_ctrl            heatwave_201107/
  EAMxx_ctrl_p12        heatwave_201107_p12/
  EAMxx_ctrl_p24        heatwave_201107_p24/
  EAMxx_ctrl_n12        heatwave_201107_n12/
  EAMxx_ctrl_n24        heatwave_201107_n24/
  EAMxx_SSP245          heatwave_201107_SSP245/
  EAMxx_SSP245_p12      heatwave_201107_SSP245_p12/
  EAMxx_SSP245_p24      heatwave_201107_SSP245_p24/
  EAMxx_SSP245_n12      heatwave_201107_SSP245_n12/
  EAMxx_SSP245_n24      heatwave_201107_SSP245_n24/

  EAMxx members share:
    • fname_abb  = output.eamxx.INSTANT.nsteps_x9.
    • eamxx_varname = T_2m  →  renamed to t2m
    • period: July 2011, days 17–24  (day 16 = spin-up, skipped)

Output
------
  outputs/
    <data_source>/   ← plots and CSVs written directly here (no figures/ subfolder)

Usage
-----
  python run_heatwave_ensemble.py                            # run all members (parallel)
  python run_heatwave_ensemble.py EAMxx_ctrl                 # run one specific member
  python run_heatwave_ensemble.py EAMxx_ctrl EAMxx_SSP245    # run a subset
  python run_heatwave_ensemble.py --workers 2                # limit parallel workers

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

import logging
import sys
import traceback
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Re-use all analysis functions and shared config from identify_heatwaves
from identify_heatwaves import (
    ERA5_HIST_CONFIG,
    HIST_CONFIG,
    PERCENTILE_VALUE,
    FIG_EXTENT,
    LAND_CELL,
    OCEAN_CELL,
    compute_threshold,
    load_and_prepare,
    load_and_prepare_era5,
    plot_frequency,
    plot_max_anomaly,
    plot_largest_contiguous_region,
    plot_longest_duration,
    plot_event_scatter,
)

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
# Paths
# ---------------------------------------------------------------------------
EAMXX_BASE_PATH = "/pscratch/sd/j/jli628/EAMxx_regridding/"
OUTPUT_BASE_DIR = Path(__file__).parent / "outputs"

# ---------------------------------------------------------------------------
# Ensemble member definitions
# ← Add / remove / modify members here; everything else is automatic.
# ---------------------------------------------------------------------------
@dataclass
class EnsembleMember:
    """Configuration for one PGW ensemble member.

    For the ERA5 observational member, set ``subdir = ""`` — this signals
    ``run_member`` to use ``load_and_prepare_era5`` instead of the EAMxx path.
    """
    data_source:   str        # label used in filenames and CSV columns
    subdir:        str        # subdirectory under EAMXX_BASE_PATH; "" = ERA5 obs
    fname_abb:     str  = "output.eamxx.INSTANT.nsteps_x9."
    eamxx_varname: str  = "T_2m"
    varnames:      str  = "t2m"
    yr_start:      int  = 2011
    yr_end:        int  = 2011
    mon_start:     int  = 7
    mon_end:       int  = 7
    day_start:     int  = 17   # day 16 is spin-up → skip
    day_end:       int  = 24

    @property
    def is_era5_obs(self) -> bool:
        """True when this member represents the ERA5 observational dataset."""
        return self.subdir == ""

    def to_eamxx_cfg(self) -> dict:
        """Return the eamxx_data_cfg dict expected by load_and_prepare."""
        return dict(
            fpath         = EAMXX_BASE_PATH + self.subdir,
            fname_abb     = self.fname_abb,
            eamxx_varname = self.eamxx_varname,
            varnames      = self.varnames,
            data_source   = self.data_source,
        )

    def to_analysis_cfg(self) -> dict:
        """Return the analysis_cfg dict expected by load_and_prepare."""
        return dict(
            yr_start  = self.yr_start,
            yr_end    = self.yr_end,
            mon_start = self.mon_start,
            mon_end   = self.mon_end,
            day_start = self.day_start,
            day_end   = self.day_end,
        )


ENSEMBLE_MEMBERS: list[EnsembleMember] = [
    # ERA5 observational baseline (subdir="" signals the ERA5 obs path)
    EnsembleMember(
        data_source = "ERA5_obs",
        subdir      = "",
        day_start   = 1,    # ERA5 has full month coverage; no spin-up to skip
        day_end     = 31,
    ),
    EnsembleMember(
        data_source = "EAMxx_ctrl",
        subdir      = "heatwave_201107/",
    ),
    EnsembleMember(
        data_source = "EAMxx_ctrl_p12",
        subdir      = "heatwave_201107_p12/",
    ),
    EnsembleMember(
        data_source = "EAMxx_ctrl_p24",
        subdir      = "heatwave_201107_p24/",
    ),
    EnsembleMember(
        data_source = "EAMxx_ctrl_n12",
        subdir      = "heatwave_201107_n12/",
    ),
    EnsembleMember(
        data_source = "EAMxx_ctrl_n24",
        subdir      = "heatwave_201107_n24/",
    ),
    EnsembleMember(
        data_source = "EAMxx_SSP245",
        subdir      = "heatwave_201107_SSP245/",
    ),
    EnsembleMember(
        data_source = "EAMxx_SSP245_p12",
        subdir      = "heatwave_201107_SSP245_p12/",
    ),
    EnsembleMember(
        data_source = "EAMxx_SSP245_p24",
        subdir      = "heatwave_201107_SSP245_p24/",
    ),
    EnsembleMember(
        data_source = "EAMxx_SSP245_n12",
        subdir      = "heatwave_201107_SSP245_n12/",
    ),
    EnsembleMember(
        data_source = "EAMxx_SSP245_n24",
        subdir      = "heatwave_201107_SSP245_n24/",
    ),
]

# Index for quick lookup by data_source label
_MEMBER_LOOKUP: dict[str, EnsembleMember] = {m.data_source: m for m in ENSEMBLE_MEMBERS}


# ---------------------------------------------------------------------------
# Per-member runner
# ---------------------------------------------------------------------------
def run_member(member: EnsembleMember, output_base: Path) -> bool:
    """
    Run the full heatwave analysis pipeline for a single ensemble member.

    Each call computes the ERA5 threshold independently so that this function
    is safe to execute in a separate process (avoids xarray/dask pickling
    issues when sharing pre-computed datasets across processes).

    Parameters
    ----------
    member      : EnsembleMember — configuration for this member.
    output_base : Path           — base directory for ensemble output.

    Returns
    -------
    bool : True on success, False if an exception was caught.
    """
    # For ERA5 obs the fpath logged is the ERA5 historical data path
    fpath = ERA5_HIST_CONFIG["fpath"] if member.is_era5_obs else (
        EAMXX_BASE_PATH + member.subdir
    )

    # --- validate EAMxx input directory (ERA5 path is pre-validated by compute_threshold) --
    if not member.is_era5_obs and not Path(fpath).is_dir():
        log.error("Member %s: directory not found → %s", member.data_source, fpath)
        return False

    log.info("=" * 70)
    log.info("Processing member : %s", member.data_source)
    log.info("  data directory  : %s", fpath)
    log.info("  period          : %d-%02d  days %d–%d",
             member.yr_start, member.mon_start, member.day_start, member.day_end)
    log.info("=" * 70)

    member_dir = output_base / member.data_source
    member_dir.mkdir(parents=True, exist_ok=True)

    try:
        analysis_cfg = member.to_analysis_cfg()

        # ---- steps 1-2: compute ERA5 threshold (per-process) ---------------
        ds_hist, ds_q90 = compute_threshold(ERA5_HIST_CONFIG, HIST_CONFIG, PERCENTILE_VALUE)

        # ---- steps 3-5: load data, build mask ------------------------------
        if member.is_era5_obs:
            heatwave_mask, anomaly, _, _ = load_and_prepare_era5(
                ERA5_HIST_CONFIG,
                analysis_cfg,
                ds_q90=ds_q90,
            )
        else:
            eamxx_data_cfg = member.to_eamxx_cfg()
            heatwave_mask, anomaly, _, _ = load_and_prepare(
                eamxx_data_cfg, analysis_cfg,
                ds_hist=ds_hist, ds_q90=ds_q90,
            )

        # sanity checks
        log.info("heatwave_mask (land cell):  %s",
                 heatwave_mask.sel(**LAND_CELL).values)
        log.info("heatwave_mask (ocean cell): %s",
                 heatwave_mask.sel(**OCEAN_CELL).values)

        # ---- step 6: heatwave-day frequency map ----------------------------
        plot_frequency(
            heatwave_mask, member.varnames,
            analysis_cfg, FIG_EXTENT, member.data_source,
            output_dir=member_dir,
        )

        # ---- steps 7-8: anomaly and max-anomaly map ------------------------
        heatwave_anomaly = anomaly.where(heatwave_mask)
        plot_max_anomaly(
            heatwave_anomaly, member.varnames,
            analysis_cfg, FIG_EXTENT, member.data_source,
            output_dir=member_dir,
        )

        # ---- step 9: largest contiguous region per day ---------------------
        plot_largest_contiguous_region(
            heatwave_mask, anomaly, member.varnames,
            analysis_cfg, FIG_EXTENT, member.data_source,
            output_dir=member_dir,
        )

        # ---- step 10: longest duration map ---------------------------------
        plot_longest_duration(
            heatwave_mask, member.varnames,
            analysis_cfg, FIG_EXTENT, member.data_source,
            output_dir=member_dir,
        )

        # ---- step 11: 3-D event scatter ------------------------------------
        plot_event_scatter(
            heatwave_mask, anomaly, member.varnames,
            analysis_cfg, member.data_source,
            output_dir=member_dir,
        )

        log.info("Member %s completed successfully → %s",
                 member.data_source, member_dir)
        return True

    except Exception:
        log.error("Member %s FAILED:\n%s",
                  member.data_source, traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Run the ensemble in parallel.  Optionally pass member data_source labels
    as command-line arguments to process only a subset, e.g.:

        python run_heatwave_ensemble.py EAMxx_ctrl EAMxx_SSP245
        python run_heatwave_ensemble.py --workers 2
    """
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    p = argparse.ArgumentParser(
        description="Run the heatwave analysis pipeline for all ensemble members.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "members",
        nargs="*",
        metavar="MEMBER",
        help=(
            "Data-source label(s) to run (e.g. EAMxx_ctrl EAMxx_SSP245). "
            "Omit to run all members."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of parallel worker processes (default: one per member).",
    )
    args = p.parse_args()

    if args.members:
        unknown = [r for r in args.members if r not in _MEMBER_LOOKUP]
        if unknown:
            p.error(f"Unknown member(s): {unknown}.  Available: {list(_MEMBER_LOOKUP)}")
        members_to_run = [_MEMBER_LOOKUP[r] for r in args.members]
    else:
        members_to_run = ENSEMBLE_MEMBERS

    max_workers = args.workers

    n_members   = len(members_to_run)
    n_workers   = min(max_workers or n_members, n_members)
    log.info("Running %d ensemble member(s) with up to %d worker(s): %s",
             n_members, n_workers, [m.data_source for m in members_to_run])

    # Dispatch members to a process pool; each worker computes its own
    # ERA5 threshold to avoid xarray/dask cross-process pickling issues.
    results: dict[str, bool] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(run_member, member, OUTPUT_BASE_DIR): member.data_source
            for member in members_to_run
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                log.error("Member %s raised an unhandled exception:\n%s",
                          name, traceback.format_exc())
                results[name] = False

    # Summary
    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    log.info("")
    log.info("=" * 70)
    log.info("ENSEMBLE RUN SUMMARY")
    log.info("=" * 70)
    for name, ok in results.items():
        status = "OK  ✓" if ok else "FAILED ✗"
        log.info("  %-30s  %s", name, status)
    log.info("-" * 70)
    log.info("  %d / %d member(s) completed successfully.",
             n_ok, len(results))
    if n_fail:
        log.warning("  %d member(s) failed — see log above for details.",
                    n_fail)
    log.info("=" * 70)

    # Non-zero exit code if any member failed (useful for batch scripts)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
