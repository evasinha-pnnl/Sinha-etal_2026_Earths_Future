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
├── decompose_load_anomaly.py      # OLS decomposition: load anomaly vs HW drivers
├── interconnection_stress.py      # System-wide stress across U.S. interconnections (Analyses A + B)
├── mean_shift_decompose.py        # ⚠️ NOT APPLICABLE — see script docstring (requires non-heatwave days)
├── util_heatwaves.py              # Heatwave utility functions
├── util_spatial_plots.py          # Spatial plotting helpers
├── util_contour_levels.py         # Contour level definitions
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

Generated at runtime (not tracked by git):

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
| `ERA5_obs_2011` | (ERA5 data) | ERA5 observations July 2011 |
| `ERA5_obs_2021` | (ERA5 data) | ERA5 observations June-July 2021 |
| `2011_CTRL_p0` | `heatwave_201107/` | Present-day control |
| `2011_CTRL_p12` | `heatwave_201107_p12/` | Present-day control|
| `2011_CTRL_p24` | `heatwave_201107_p24/` | Present-day control|
| `2011_CTRL_n12` | `heatwave_201107_n12/` | Present-day control|
| `2011_CTRL_n24` | `heatwave_201107_n24/` | Present-day control|
| `2011_SSP245_p0` | `heatwave_201107_SSP245/` | SSP2-4.5 perturbation |
| `2011_SSP245_p12` | `heatwave_201107_SSP245_p12/` | SSP2-4.5 +1.2 K perturbation |
| `2011_SSP245_p24` | `heatwave_201107_SSP245_p24/` | SSP2-4.5 +1.2 K perturbation |
| `2011_SSP245_n12` | `heatwave_201107_SSP245_n12/` | SSP2-4.5 +1.2 K perturbation |
| `2011_SSP245_n24` | `heatwave_201107_SSP245_n24/` | SSP2-4.5 +1.2 K perturbation |
| `2021_CTRL_p0` | `heatwave_2021_CTRL_p0/` | 2021 control |
| `2021_SSP245_p0` | `heatwave_2021_SSP245_p0/` | 2021 SSP2-4.5 perturbation |

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
writes per-member plots and CSVs to `outputs/<member>/`.

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

### Step 5 – OLS decomposition of the load anomaly (Analysis 1)

Answers: *How much of the warming-driven peak-load increase is explained by
heatwave spatial extent, mean intensity, or peak intensity?*

Reads the `tell_heatwave_merged_<year>.csv` written by Step 4 and fits a
standardised OLS model per perturbation member:

```
peak_load_anomaly_GW  ~  β₀ + β₁·hw_area_km2 + β₂·hw_mean_K + β₃·hw_max_K
```

All predictors and the response are standardised before fitting, so the
resulting β* coefficients are directly comparable across predictors.

```bash
# Default run (year=2011, reads from outputs/tell_heatwave_linkage/)
python decompose_load_anomaly.py

# Select the year (must match the --year used in Step 4)
python decompose_load_anomaly.py --year 2011

# Override input / output paths
python decompose_load_anomaly.py \
    --year 2011 \
    --merged-csv /path/to/tell_heatwave_merged_2011.csv \
    --output-dir /path/to/output/

# See all options
python decompose_load_anomaly.py --help
```

Available CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | `2011` | Heatwave event/year to analyse — must match the `--year` used in Step 4; determines the default `--merged-csv` filename |
| `--merged-csv` | `outputs/tell_heatwave_linkage/tell_heatwave_merged_<year>.csv` | Input merged CSV from Step 4 |
| `--output-dir` | `outputs/tell_heatwave_linkage/` | Output directory for figures and CSV |
| `--min-hw-frac` | `0.0` | Minimum BA heatwave-area fraction (0–1) for inclusion in regression. Rows with `hw_area_frac ≤` this value are dropped. E.g. `0.1` keeps only BAs where ≥10 % of territory is under the heatwave. |

---

### Step 6 – Interconnection-level stress analysis (Analyses A + B)

Answers: *Does future warming increase the simultaneous stress on multiple
U.S. electricity interconnections, and by how much in load terms?*

Uses `BA_to_Interconnection_Mapping.csv` (repo root) to map each BA to one of
the three U.S. interconnections (EIC, WECC, ERCOT), then aggregates
`tell_heatwave_merged_<year>.csv` to the interconnection level.

**Analysis A — Load stress time-series**
Sums daily peak-load anomaly over all BAs per interconnection × member.
Compares perturbation vs control to quantify warming-driven load increase at
the interconnection scale.

**Analysis B — Daily co-occurrence index**
Counts how many interconnections simultaneously exceed a configurable stress
threshold on each day.  Compares co-occurrence counts between perturbation and
control members.

> **Note on probability framing:** with only 2 perturbation members × 8 days,
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

# Override paths
python interconnection_stress.py \
    --year 2011 \
    --merged-csv /path/to/tell_heatwave_merged_2011.csv \
    --ba-map     /path/to/BA_to_Interconnection_Mapping.csv \
    --output-dir /path/to/output/

# See all options
python interconnection_stress.py --help
```

Available CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | `2011` | Heatwave event/year to analyse — must match the `--year` used in Step 4; determines the default `--merged-csv` filename |
| `--merged-csv` | `outputs/tell_heatwave_linkage/tell_heatwave_merged_<year>.csv` | Input merged CSV from Step 4 |
| `--ba-map` | `BA_to_Interconnection_Mapping.csv` | BA → Interconnection lookup CSV |
| `--output-dir` | `outputs/tell_heatwave_linkage/` | Output directory for figures and CSVs |
| `--stress-threshold-gw` | `0.0` | Min load anomaly (GW) for an interconnection to count as "stressed" |
| `--min-hw-frac` | `0.0` | Min BA heatwave-area fraction; rows at or below this are excluded |

---

### Step 7 – Mean-shift baseline vs heatwave-excess decomposition (Analysis 2)

> ⚠️ **NOT APPLICABLE to the current EAMxx data.**
>
> The EAMxx simulations cover only the heatwave window (`2011-07-17` –
> `2011-07-24`).  Every day in the dataset is a heatwave day, so there are no
> non-heatwave days from which to estimate the mean-shift baseline.  Running
> the script raises a `NotImplementedError` with a detailed explanation.
>
> **What would be needed:** re-run EAMxx over a longer window (e.g. full July
> or JJA) so that non-heatwave days are available, or use a multi-year
> climatological TELL run as the reference.  See the `mean_shift_decompose.py`
> module docstring for full details and alternative approaches.

*Algorithm (for future use):* Reads `tell_load_daily_all_<year>.csv` (all days,
produced by Step 4) and `tell_heatwave_affected_bas_<year>.csv` (spatial filter)
and partitions the peak-load anomaly into two additive components per BA × member:

- **Mean-shift baseline** — mean anomaly on **non-heatwave days** (isolates
  the always-on background warming signal)
- **Heatwave excess** — anomaly on heatwave days minus the baseline (isolates
  the incremental load driven by the heatwave itself)

```bash
# Default run (year=2011, reads from outputs/tell_heatwave_linkage/)
python mean_shift_decompose.py

# Select the year (must match the --year used in Step 4)
python mean_shift_decompose.py --year 2011

# Override paths or BA-fraction threshold
python mean_shift_decompose.py \
    --year 2011 \
    --load-csv   /path/to/tell_load_daily_all_2011.csv \
    --filter-csv /path/to/tell_heatwave_affected_bas_2011.csv \
    --output-dir /path/to/output/ \
    --threshold  0.0

# See all options
python mean_shift_decompose.py --help
```

Available CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--year` | `2011` | Heatwave event/year to analyse — must match the `--year` used in Step 4; determines the default `--load-csv` / `--filter-csv` filenames |
| `--load-csv` | `outputs/tell_heatwave_linkage/tell_load_daily_all_<year>.csv` | All-days load CSV from Step 4 |
| `--filter-csv` | `outputs/tell_heatwave_linkage/tell_heatwave_affected_bas_<year>.csv` | Spatial filter CSV from Step 4 |
| `--output-dir` | `outputs/tell_heatwave_linkage/` | Output directory for figures and CSV |
| `--threshold` | `0.0` | Min BA heatwave-area fraction to classify a day as a heatwave day |

---

## Outputs

All outputs are written to `outputs/tell_heatwave_linkage/`. Every filename
below is suffixed with `_<year>` (e.g. `tell_heatwave_merged_2011.csv`), so
runs for different `--year` values coexist without overwriting each other:

| File | Description |
|------|-------------|
| `tell_heatwave_affected_bas_<year>.csv` | Cache: BA heatwave area fractions per member per day |
| `tell_heatwave_merged_<year>.csv` | Merged daily peak-load anomaly + heatwave characteristics |
| `tell_heatwave_correlations_<year>.csv` | Pearson r table (load anomaly vs heatwave metrics) |
| `tell_affected_bas_per_day_<year>_<threshold>.png` | Bar chart: # of heatwave-affected BAs per day, all members |
| `tell_load_anomaly_timeseries_per_ba_<year>.png` | Daily peak-load anomaly (GW) for top 5 BAs — one subplot per member |
| `tell_load_anomaly_pct_timeseries_per_ba_<year>.png` | Same, as % of control load |
| `tell_load_anomaly_timeseries_per_ba_comparison_<year>.png` | 2×2 grid: top 4 BAs by HW fraction — lines per member (GW) |
| `tell_load_anomaly_pct_timeseries_per_ba_comparison_<year>.png` | Same, as % of control load |
| `tell_load_anomaly_timeseries_per_ba_comparison_top_anomaly_<year>.png` | 2×2 grid: top 4 BAs by largest load anomaly — lines per member (GW) |
| `tell_load_anomaly_pct_timeseries_per_ba_comparison_top_anomaly_<year>.png` | Same, as % of control load |
| `tell_load_vs_heatwave_scatter_<year>.png` | Scatter: load anomaly vs spatial extent / intensity |
| `tell_load_decomp_regression_table_<year>.csv` | OLS results: β*, SE, 95% CI, p-value, R² per member × predictor |
| `tell_load_decomp_std_coeffs_<year>.png` | Standardised OLS coefficients β* with 95% CI — grouped bar chart |
| `tell_load_decomp_partial_regression_<year>.png` | Partial-regression plots — load anomaly vs each HW characteristic |
| `tell_interconnection_load_anomaly_GW_<year>.csv` | Per (date, interconnection, member): summed load anomaly GW, ctrl GW, anomaly % |
| `tell_interconnection_load_timeseries_<year>.png` | Three-panel time-series — load anomaly (GW) per interconnection, pert vs ctrl |
| `tell_interconnection_load_timeseries_pct_<year>.png` | Same, expressed as % of control load |
| `tell_interconnection_stress_heatmap_<year>.png` | Date × interconnection heatmap of load anomaly (GW); red border = above threshold |
| `tell_interconnection_cooccurrence_summary_<year>.csv` | Per (date, member): count and list of simultaneously stressed interconnections |
| `tell_interconnection_cooccurrence_<year>.png` | Daily co-occurrence bar chart — # of simultaneously stressed interconnections |
| `tell_mean_shift_decomp_summary_<year>.csv` | Per (BA, member): baseline, mean HW excess, total anomaly, component fractions |
| `tell_mean_shift_decomp_stacked_bar_<year>.png` | Grouped stacked bars — baseline (light blue) vs HW excess (colour) for top BAs |
| `tell_mean_shift_decomp_timeseries_<year>.png` | Daily stacked-area time-series: baseline + HW excess per top BA × member |
| `tell_mean_shift_decomp_fraction_bar_<year>.png` | 100 % stacked bar: fraction of total anomaly from each component per BA |

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
