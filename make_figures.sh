#!/bin/bash
# Regenerate every manuscript figure from the archived intermediate products.
#
# Does NOT need the raw or regridded SCREAM output, provided OUTPUT_ROOT holds
# the archived per-member outputs (heatwave maps/CSVs, daily-max caches, and
# the tell_heatwave_affected_bas_<year>.csv spatial-filter caches).
#
# Usage:
#   export DATA_ROOT=/path/to/EAMxx-TELL-data
#   export OUTPUT_ROOT=$DATA_ROOT/outputs
#   bash make_figures.sh

set -euo pipefail
cd "$(dirname "$0")/src"

# ── Evaluation against EIA-930 (DYAMOND1, August 2016) ──────────────────────
python plot_ba_service_territory.py
python plot_DYAMOND_tell_load_timeseries.py

# ── Heatwave characteristics ────────────────────────────────────────────────
python plot_ensemble_stats.py
python compare_heatwaves.py

# ── Event-specific amplification ────────────────────────────────────────────
python plot_pgw_amplification.py --years 2011 --start 2011-07-20 --end 2011-07-23
python plot_pgw_amplification.py --years 2011    # full event (SI)
python plot_pgw_amplification.py --years 2021

# ── TELL load, affected BAs, interconnections ───────────────────────────────
for yr in 2011 2021; do
  for th in 0.2 0.4 0.6 0.8; do
    python analyze_tell_heatwave.py --year "$yr" --threshold "$th"
  done
  python plot_affected_ba.py --year "$yr"        # needs all four thresholds
  python interconnection_stress.py --year "$yr"  # needs tell_load_daily_all_<year>.csv
done

# ── Composite panels ────────────────────────────────────────────────────────
python merge_images.py
