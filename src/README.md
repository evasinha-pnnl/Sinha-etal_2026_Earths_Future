# LDRD-EAMxx-TELL – Heatwave Analysis Scripts

Analysis pipeline linking **EAMxx PGW ensemble** heatwave events to
**TELL** electricity-load projections over CONUS (focus: July 2011).

---

## Repository Layout

```
src/
├── identify_heatwaves.py          # Core heatwave detection & plotting library
├── run_heatwave_ensemble.py       # Run the full pipeline for all 4 EAMxx members
├── compare_heatwaves.py           # Compare heatwave metrics across members
├── build_grid_to_ba.py            # Build grid-cell → BA mapping parquet (one-time)
├── analyze_tell_heatwave.py       # Link TELL load to heatwave events (main script)
├── util_heatwaves.py              # Heatwave utility functions
├── util_spatial_plots.py          # Spatial plotting helpers
├── util_contour_levels.py         # Contour level definitions
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

Generated at runtime (not tracked by git):

```
ensemble_output/
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

The scripts run inside the `py39_tell` conda environment:

```bash
conda activate py39_tell
pip install -r requirements.txt
```

> **Note**: The [TELL model](https://github.com/IMMM-SFA/tell) is not on PyPI.
> Install it separately if needed:
> ```bash
> git clone https://github.com/IMMM-SFA/tell.git
> pip install -e tell/
> ```

### 3. Input data paths

| Data | Path on NERSC |
|------|--------------|
| EAMxx regridded NetCDF files | `/pscratch/sd/j/jli628/EAMxx_regridding/` |
| ERA5 historical temperature | configured inside `identify_heatwaves.py` → `ERA5_HIST_CONFIG` |
| TELL output CSVs | `/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL/TELL_outputs/` |
| Grid-cell → county weight parquet | `/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL/im3components/im3components/wrf_to_tell/grid_cell_to_county_weight.EAMxx.parquet` |
| BA service territory CSV | `/global/homes/e/esinha/wrk/E3SM_projects/LDRD-EAMxx-TELL/im3components/im3components/wrf_to_tell/wrf_to_tell_ancillary_data/ba_service_territory_2019.csv` |

### 4. EAMxx ensemble members

| Label | Subdirectory | Description |
|-------|-------------|-------------|
| `EAMxx_ctrl` | `heatwave_201107/` | Present-day control |
| `EAMxx_ctrl_p12` | `heatwave_201107_p12/` | Present-day control +1.2 K |
| `EAMxx_SSP245` | `heatwave_201107_SSP245/` | SSP2-4.5 perturbation |
| `EAMxx_SSP245_p12` | `heatwave_201107_SSP245_p12/` | SSP2-4.5 +1.2 K perturbation |

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

Runs `identify_heatwaves.py` for all four EAMxx members **in parallel** and
writes per-member plots and CSVs to `ensemble_output/<member>/`.

```bash
# Run all members (default: parallel with all available CPUs)
python run_heatwave_ensemble.py

# Run a single member
python run_heatwave_ensemble.py EAMxx_ctrl

# Run a subset with limited workers
python run_heatwave_ensemble.py EAMxx_ctrl EAMxx_SSP245 --workers 2
```

### Step 3 – Compare heatwave characteristics across members

```bash
python compare_heatwaves.py
```

Outputs figures to `ensemble_output/` comparing frequency, intensity, and
duration across all members.

### Step 4 – Link TELL load to heatwave events

```bash
# Default run (uses TELL_BASE env var or built-in fallback path)
python analyze_tell_heatwave.py

# Override the TELL data root via environment variable
export TELL_BASE=/path/to/TELL_outputs
python analyze_tell_heatwave.py

# Override individual settings via CLI flags
python analyze_tell_heatwave.py --year 2011 --threshold 0.5 --output-dir /tmp/results

# See all options
python analyze_tell_heatwave.py --help
```

Available CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | `2011` | Simulation year (locates TELL CSVs and heatwave output) |
| `--threshold` | `0.0` | Min BA area fraction under heatwave (0 = any overlap) |
| `--output-dir` | `ensemble_output/tell_heatwave_linkage/` | Output directory for figures and CSVs |
| `--tell-base` | `$TELL_BASE` or built-in path | Root directory of TELL output CSVs |

This is the main analysis script. On first run it performs an expensive
spatial-filter computation (heatwave mask → BA overlap fractions) and caches
the result to:

```
ensemble_output/tell_heatwave_linkage/tell_heatwave_affected_bas.csv
```

Subsequent runs load from cache automatically. To force a recompute, delete
that CSV file.

---

## Outputs

All outputs are written to `ensemble_output/tell_heatwave_linkage/`:

| File | Description |
|------|-------------|
| `tell_heatwave_affected_bas.csv` | Cache: BA heatwave area fractions per member per day |
| `tell_heatwave_merged.csv` | Merged daily peak-load anomaly + heatwave characteristics |
| `tell_heatwave_correlations.csv` | Pearson r table (load anomaly vs heatwave metrics) |
| `tell_affected_bas_per_day.png` | Bar chart: # of heatwave-affected BAs per day, all members |
| `tell_load_anomaly_timeseries_per_ba.png` | Daily peak-load anomaly (GW) for top 5 BAs — one subplot per member |
| `tell_load_anomaly_pct_timeseries_per_ba.png` | Same, as % of control load |
| `tell_load_anomaly_timeseries_per_ba_comparison.png` | Daily peak-load anomaly (GW) for top 5 BAs — one subplot per BA, lines per member |
| `tell_load_anomaly_pct_timeseries_per_ba_comparison.png` | Same, as % of control load |
| `tell_load_vs_heatwave_scatter.png` | Scatter: load anomaly vs spatial extent / intensity |

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
| `OUTPUT_DIR` | — | `--output-dir` | `ensemble_output/tell_heatwave_linkage/` | Output directory |

---

## Authors

Eva Sinha &lt;eva.sinha@pnnl.gov&gt;
