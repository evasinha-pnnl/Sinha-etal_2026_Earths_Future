# EAMxx-TELL – Heatwave Analysis Scripts

Analysis pipeline linking **SCREAM pseudo-global-warming (PGW) ensemble**
heatwave events to **TELL** electricity-load estimates over CONUS, for the
July 2011 central/eastern U.S. heatwave and the June–July 2021 Pacific
Northwest heatwave.

> **Naming:** SCREAM is built on the EAMxx code base, so scripts, variables,
> and file names use `EAMxx`; the manuscript uses `SCREAM`. Members labelled
> `SSP245` are the PGW members (Control + SSP2-4.5 mid-century deltas).

See the [top-level README](../README.md) for the data archive, the
figure-to-script map, and how to reproduce the figures.

---

## Repository Layout

```
src/
├── identify_heatwaves.py          # Core heatwave detection & plotting library
├── run_heatwave_ensemble.py       # Run heatwave detection for every ensemble member
├── compare_heatwaves.py           # Compare heatwave metrics (area, duration, …) across members
├── plot_ensemble_stats.py         # Ensemble mean / std maps of per-member heatwave metrics
├── plot_pgw_amplification.py      # Event-specific amplification: PGW − (Control + Δ)
├── build_grid_to_ba.py            # Build grid-cell → BA mapping parquet (one-time)
├── analyze_tell_heatwave.py       # Link TELL load to heatwave events (main script)
├── plot_affected_ba.py            # Number of heatwave-affected BAs per day
├── interconnection_stress.py      # Load anomalies across U.S. interconnections
├── plot_DYAMOND_tell_load_timeseries.py  # TELL (DYAMOND1) vs EIA-930 evaluation
├── plot_ba_service_territory.py   # BA service-territory and interconnection maps
├── merge_images.py                # Composite panels from individual figures
├── run.sh                         # SLURM batch script used on Perlmutter
├── util_constants.py              # Shared constants and ALL input data paths
├── util_heatwaves.py              # Heatwave utility functions
├── util_spatial_plots.py          # Spatial plotting helpers
├── util_vector_plots.py           # Line / scatter plotting helpers
├── util_contour_levels.py         # Contour level definitions
├── BA_to_Interconnection_Mapping.csv  # BA → interconnection lookup
└── README.md                      # This file
```

Exploratory scripts not used in the manuscript are in `../exploratory/`.

Generated at runtime (not tracked by git; location set by `OUTPUT_ROOT`,
default `src/outputs/`):

```
outputs/
  <member>/                        # Per-member heatwave plots and CSVs
  tell_heatwave_linkage/           # TELL–heatwave linkage figures and CSVs
grid_cell_to_ba.EAMxx.parquet     # Built by build_grid_to_ba.py
```

---

## Prerequisites

### 1. System

- NERSC Perlmutter (or any Linux system with access to the data paths below)
- Conda / Mamba

### 2. Conda environment

The analysis scripts were run in the `py313` conda environment on Perlmutter:

```bash
conda activate py313
pip install -r ../requirements.txt
```

> **Note**: The [TELL model](https://github.com/IMMM-SFA/tell) is not on PyPI.
> Install it separately if needed:
> ```bash
> git clone https://github.com/IMMM-SFA/tell.git
> pip install -e tell/
> ```

### 3. Input data paths

All input locations are defined in one place, `util_constants.py`
(section *Input data locations*). By default they point to the NERSC paths
used for the paper. To run from the data archive instead, set:

```bash
export DATA_ROOT=/path/to/EAMxx-TELL-data   # unpacked data archive
export OUTPUT_ROOT=$DATA_ROOT/outputs            # archived intermediate products
```

| Constant | Archive subdirectory | NERSC default |
|----------|---------------------|---------------|
| `EAMXX_REGRID_DIR` | `eamxx_regridded/` | `/pscratch/sd/j/jli628/EAMxx_regridding/` |
| `EAMXX_SKT_DIR` | `eamxx_skt/` | `/pscratch/sd/e/esinha/EAMxx/` |
| `PGW_DELTA_DIR` | `pgw_deltas/` | `/pscratch/sd/j/jli628/EAMxx_regridding/PGW_detals/` |
| `ERA5_DIR` | `era5/` | `/pscratch/sd/e/esinha/ERA5_CONUS/` |
| `TELL_OUTPUT_DIR` | `tell_outputs/` | `<project>/TELL_outputs/` |
| `EIA930_DIR` | `eia930/` | `<project>/EIA_930/` |
| `WRF_TO_TELL_DIR` | `wrf_to_tell/` | `<project>/im3components/im3components/wrf_to_tell/` |

`<project>` = `/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL`.
`analyze_tell_heatwave.py` also honours `TELL_BASE` / `--tell-base`.

### 4. Ensemble members

Subdirectories are relative to `EAMXX_REGRID_DIR`.

| Label | Subdirectory | Description |
|-------|-------------|-------------|
| `ERA5_obs_2011` | (ERA5 data) | ERA5 July 2011 |
| `ERA5_obs_2021` | (ERA5 data) | ERA5 June–July 2021 |
| `2011_CTRL_p0` | `heatwave_201107/` | 2011 Control, no initialization offset |
| `2011_CTRL_p12` | `heatwave_201107_p12/` | 2011 Control, +12 h initialization offset |
| `2011_CTRL_p24` | `heatwave_201107_p24/` | 2011 Control, +24 h initialization offset |
| `2011_CTRL_n12` | `heatwave_201107_n12/` | 2011 Control, −12 h initialization offset |
| `2011_CTRL_n24` | `heatwave_201107_n24/` | 2011 Control, −24 h initialization offset |
| `2011_SSP245_p0` | `heatwave_201107_SSP245/` | 2011 PGW (SSP2-4.5 deltas), no initialization offset |
| `2011_SSP245_p12` | `heatwave_201107_SSP245_p12/` | 2011 PGW (SSP2-4.5 deltas), +12 h initialization offset |
| `2011_SSP245_p24` | `heatwave_201107_SSP245_p24/` | 2011 PGW (SSP2-4.5 deltas), +24 h initialization offset |
| `2011_SSP245_n12` | `heatwave_201107_SSP245_n12/` | 2011 PGW (SSP2-4.5 deltas), −12 h initialization offset |
| `2011_SSP245_n24` | `heatwave_201107_SSP245_n24/` | 2011 PGW (SSP2-4.5 deltas), −24 h initialization offset |
| `2021_CTRL_p0`  | `heatwave_2021_CTRL_p0/` | 2021 Control, no initialization offset |
| `2021_CTRL_p24` | `heatwave_2021_CTRL_p24/` | 2021 Control, +24 h initialization offset |
| `2021_CTRL_n24` | `heatwave_2021_CTRL_n24/` | 2021 Control, −24 h initialization offset |
| `2021_SSP245_p0`  | `heatwave_2021_SSP245_p0/`  | 2021 PGW (SSP2-4.5 deltas), no initialization offset |
| `2021_SSP245_p24` | `heatwave_2021_SSP245_p24/` | 2021 PGW (SSP2-4.5 deltas), +24 h initialization offset |
| `2021_SSP245_n24` | `heatwave_2021_SSP245_n24/` | 2021 PGW (SSP2-4.5 deltas), −24 h initialization offset |

---

## Usage

Scripts must be run **in order**: each step produces outputs consumed by the next.

### Step 1 – Build the grid-cell → BA mapping (one-time setup)

Only needs to be run once. Produces `grid_cell_to_ba.EAMxx.parquet` in this
directory.

```bash
python build_grid_to_ba.py
```

### Step 2 – Run the heatwave ensemble pipeline

Runs `identify_heatwaves.py` for every ensemble member **in parallel** and
writes per-member maps (NetCDF), plots and CSVs to `outputs/<member>/`.
This is the only step (besides `plot_pgw_amplification.py` without its cache
and the first run of `analyze_tell_heatwave.py`) that reads the regridded
SCREAM output.

```bash
# Run all members (default: parallel with all available CPUs)
python run_heatwave_ensemble.py

# Run a single member
python run_heatwave_ensemble.py 2011_CTRL_p0

# Run a subset with limited workers
python run_heatwave_ensemble.py 2011_CTRL_p0 2011_SSP245_p0 --workers 2
```

### Step 3 – Compare heatwave characteristics across members

```bash
python compare_heatwaves.py
```

Outputs figures to `outputs/` comparing frequency, intensity, and
duration across all members.

### Step 4 – Link TELL load to heatwave events

```bash
# Default run (year=2011, uses TELL_BASE env var or built-in fallback path)
python analyze_tell_heatwave.py

# Select the year (2011 or 2021) — determines the TELL member set, analysis
# window, and the _<year> suffix on every output file/figure
python analyze_tell_heatwave.py --year 2011

# Override the TELL data root via environment variable
export TELL_BASE=/path/to/TELL_outputs
python analyze_tell_heatwave.py --year 2011

# Override individual settings via CLI flags
python analyze_tell_heatwave.py --year 2011 --threshold 0.5 --output-dir /tmp/results

# See all options
python analyze_tell_heatwave.py --help
```

Available CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | `2011` | Heatwave event/year to analyse (`2011` or `2021`) — selects the TELL member directories, analysis window, and plot label; also determines the `_<year>` suffix on output filenames |
| `--threshold` | `0.0` | Min BA area fraction under heatwave (0 = any overlap) |
| `--output-dir` | `outputs/tell_heatwave_linkage/` | Output directory for figures and CSVs |
| `--tell-base` | `$TELL_BASE` or built-in path | Root directory of TELL output CSVs |

This is the main analysis script. On first run (per year) it performs an
expensive spatial-filter computation (heatwave mask → BA overlap fractions)
and caches the result to:

```
outputs/tell_heatwave_linkage/tell_heatwave_affected_bas_<year>.csv
```

Subsequent runs for the same `--year` load from cache automatically — the
cache is also invalidated and rebuilt automatically if the member set or the
underlying data's date range changes (e.g. new simulation days are added). To
force a recompute manually, delete that CSV file.

### Step 5 - Plot heatwave-affected BAs per day for various thresholds

python plot_affected_ba.py

Plot the number of heatwave-affected balancing authorities (BAs) per day for
the Control and PGW (SSP245) ensembles, one panel per BA-fraction threshold.


### Step 6 – Interconnection-level stress analysis (Analyses A + B)

Answers: *Does future warming increase the simultaneous stress on multiple
U.S. electricity interconnections, and by how much in load terms?*

Uses `BA_to_Interconnection_Mapping.csv` (repo root) to map each BA to one of
the three U.S. interconnections (EIC, WIC, ERCOT), then sums the daily
peak load of *all* BAs in each interconnection from
`tell_load_daily_all_<year>.csv` (written by every Step 4 run, not filtered
to heatwave-affected BAs), so every PGW − Control pair sums the same BA set.

**Analysis A — Load stress time-series**
Sums daily peak-load anomaly over all BAs per interconnection × member.
Compares perturbation vs control to quantify warming-driven load increase at
the interconnection scale.

**Analysis B — Daily co-occurrence index**
Counts how many interconnections simultaneously exceed a configurable stress
threshold on each day.  Compares co-occurrence counts between perturbation and
control members.

> **Note on probability framing:** with 5 (2011) or 3 (2021) PGW members
> over a single event,
> the results are descriptive (magnitude and co-occurrence during this specific
> event) rather than statistical (return-period or likelihood changes).  A
> larger ensemble or longer simulation window would be needed to compute robust
> probability changes.

```bash
# Default run (year=2011, threshold = any positive anomaly)
python interconnection_stress.py

# Select the year (must match the --year used in Step 4)
python interconnection_stress.py --year 2011

# Use a more stringent stress threshold (e.g. 5 GW)
python interconnection_stress.py --stress-threshold-gw 5.0
python interconnection_stress.py --stress-threshold-gw 5.0 --year 2021

# Override paths
python interconnection_stress.py \
    --year 2011 \
    --load-csv   /path/to/tell_load_daily_all_2011.csv \
    --ba-map     /path/to/BA_to_Interconnection_Mapping.csv \
    --output-dir /path/to/output/

# See all options
python interconnection_stress.py --help
```

Available CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | `2011` | Heatwave event/year to analyse — must match the `--year` used in Step 4; determines the default `--load-csv` filename |
| `--load-csv` | `outputs/tell_heatwave_linkage/tell_load_daily_all_<year>.csv` | Input daily all-BA peak load from Step 4 |
| `--ba-map` | `BA_to_Interconnection_Mapping.csv` | BA → Interconnection lookup CSV |
| `--output-dir` | `outputs/tell_heatwave_linkage/` | Output directory for figures and CSVs |
| `--stress-threshold-gw` | `0.0` | Min load anomaly (GW) for an interconnection to count as "stressed" |

---

### Step 7 – Event-specific amplification and ensemble maps

```bash
python plot_ensemble_stats.py                        # ensemble mean / std heatwave maps
python plot_pgw_amplification.py --years 2011 \
    --start 2011-07-20 --end 2011-07-23              # amplification, peak window
python plot_pgw_amplification.py --years 2011        # amplification, full event (SI)
```

`plot_pgw_amplification.py` caches each member's daily-maximum field as
`outputs/<member>/<member>_t2m_daily_max_<year>.nc`; with those caches in place
it does not need the SCREAM output.

The mean-shift / heatwave-excess decomposition formerly listed here is not
applicable to these simulations (every simulated day is a heatwave day) and has
moved to `../exploratory/mean_shift_decompose.py`.

---

## Outputs

All outputs are written to `outputs/tell_heatwave_linkage/`. Every filename
below is suffixed with `_<year>` (e.g. `tell_heatwave_threshold_0.8_2011.csv`), so
runs for different `--year` values coexist without overwriting each other:

| File | Description |
|------|-------------|
| `tell_heatwave_affected_bas_<year>.csv` | Cache: BA heatwave area fractions per member per day |
| `tell_heatwave_threshold_<threshold>_<year>.csv` | Merged daily peak-load anomaly + heatwave characteristics, for BA-days with heatwave-area fraction > `<threshold>` |
| `tell_affected_bas_per_day_<year>_<threshold>.png` | Bar chart: # of heatwave-affected BAs per day, all members |
| `tell_load_anomaly_timeseries_per_ba_<year>.png` | Daily peak-load anomaly (GW) for top 5 BAs — one subplot per member |
| `tell_load_anomaly_pct_timeseries_per_ba_<year>.png` | Same, as % of control load |
| `tell_load_anomaly_timeseries_per_ba_comparison_<year>.png` | Fixed BA set: PGW − Control peak-load anomaly (GW), mean over pairs with min–max band; BA-days with any heatwave overlap |
| `tell_load_anomaly_pct_timeseries_per_ba_comparison_<year>.png` | Same, as % of control load |
| `tell_load_anomaly_timeseries_per_ba_comparison_top_anomaly_<year>.png` | Same, for the top 4 BAs by largest mean load anomaly |
| `tell_load_anomaly_pct_timeseries_per_ba_comparison_top_anomaly_<year>.png` | Same, as % of control load |
| `tell_load_decomp_regression_table_<year>.csv` | OLS results: β*, SE, 95% CI, p-value, R² per member × predictor |
| `tell_load_decomp_std_coeffs_<year>.png` | Standardised OLS coefficients β* with 95% CI — grouped bar chart |
| `tell_load_decomp_partial_regression_<year>.png` | Partial-regression plots — load anomaly vs each HW characteristic |
| `tell_interconnection_load_anomaly_GW_<year>.csv` | Per (date, interconnection, member): summed load anomaly GW, ctrl GW, anomaly % |
| `tell_interconnection_load_timeseries_<year>.png` | Three-panel time-series — load anomaly (GW) per interconnection, pert vs ctrl |
| `tell_interconnection_load_timeseries_pct_<year>.png` | Same, expressed as % of control load |
| `tell_interconnection_stress_heatmap_<year>.png` | Date × interconnection heatmap of load anomaly (GW); red border = above threshold |
| `tell_interconnection_cooccurrence_summary_<year>.csv` | Per (date, member): count and list of simultaneously stressed interconnections |
| `tell_interconnection_cooccurrence_<year>.png` | Daily co-occurrence bar chart — # of simultaneously stressed interconnections |

---

## Configuration

All user-configurable settings can be changed via **CLI flags** (preferred),
**environment variables**, or by editing the constants at the top of each script.

Key settings in `analyze_tell_heatwave.py`:

| Constant | Env var | CLI flag | Default | Description |
|----------|---------|----------|---------|-------------|
| `TELL_BASE` | `TELL_BASE` | `--tell-base` | `/global/.../TELL_outputs` | Root directory of TELL output CSVs |
| `YEAR` | — | `--year` | `2011` | Simulation year |
| `HW_BA_FRAC_THRESHOLD` | — | `--threshold` | `0.0` | Min BA area fraction under heatwave |
| `OUTPUT_DIR` | — | `--output-dir` | `outputs/tell_heatwave_linkage/` | Output directory |

---

## Authors

Eva Sinha &lt;eva.sinha@pnnl.gov&gt;
