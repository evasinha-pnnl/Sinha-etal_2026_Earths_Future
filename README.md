# Sinha-etal_2026_Earths_Future

**Mid-century warming broadens extreme heatwaves and synchronizes peak electricity demand across the U.S.**

Eva Sinha<sup>1\*</sup>, Jianfeng Li<sup>1</sup>, Casey D. Burleyson<sup>1</sup>, Katherine V. Calvin<sup>2</sup>, and L. Ruby Leung<sup>1</sup>

<sup>1</sup> Pacific Northwest National Laboratory, Richland, WA, USA  
<sup>2</sup> University of Maryland, College Park, MD, United States

\* corresponding author: eva.sinha@pnnl.gov

## Abstract

<!-- TODO: paste the final abstract -->

## Journal reference

<!-- TODO: add once published -->

## Code reference

<!-- TODO: Zenodo DOI for the tagged release of this repository -->

## Data reference

### Input data

| Dataset | Description | Reference |
|---------|-------------|-----------|
| SCREAM (E3SM) simulations | Regridded daily output of the Control and PGW ensembles (July 2011, June–July 2021) | <!-- TODO: Zenodo / DataHub DOI --> |
| PGW temperature deltas | CMIP6 11-model mean, SSP2-4.5, 2031–2060 minus 1981–2010 | <!-- TODO: DOI --> |
| ERA5 | Hourly 2-m temperature over CONUS, 1981–2011 and 2021 | Hersbach et al. (2020), doi:10.1002/qj.3803 |
| EIA-930 | Hourly balancing-authority demand, 2016 | U.S. Energy Information Administration, Form EIA-930 |
| BA service territories | County-to-BA mapping and grid-cell weights (`wrf_to_tell`) | <!-- TODO: DOI --> |

### Output data

| Dataset | Description | Reference |
|---------|-------------|-----------|
| Intermediate products | Per-member heatwave maps and metrics, TELL hourly load, cached daily-maximum fields | https://doi.org/10.5281/zenodo.24405570 |

## Contributing modeling software

| Model | Version | Repository link | DOI |
|-------|---------|-----------------|-----|
| E3SM / SCREAM (EAMxx) | SCREAMv1 | https://github.com/E3SM-Project/E3SM | https://www.osti.gov/doecode/biblio/144883 |
| TELL | v1.2.1 tell | https://github.com/IMMM-SFA/tell | https://zenodo.org/records/13354910 |
| `wrf_to_tell` | <!-- TODO: commit --> | https://github.com/IMMM-SFA/im3components/tree/main/im3components/wrf_to_tell | — |

> **Naming:** SCREAM is built on the EAMxx code base, so scripts and files in
> this repository use `EAMxx`; the manuscript uses `SCREAM`. Members labelled
> `SSP245` are the PGW members.

## Reproduce my experiment

The SCREAM simulations and TELL runs were performed on NERSC Perlmutter.
The steps below document the workflow; steps 1–3 require resources and data
that are not part of this repository.

1. **Run SCREAM.** Control and PGW ensembles with initialization offsets of
   0, ±12 h and ±24 h (2011) or 0 and ±24 h (2021).
   <!-- TODO: case-setup scripts, namelist changes, and how the PGW deltas were applied -->
2. **Regrid SCREAM output** to a regular latitude–longitude grid.
   <!-- TODO: regridding script / reference -->
3. **Run TELL** on the SCREAM meteorology using the
   `wrf_to_tell` conversion. <!-- TODO: TELL run script and trained-model version -->
4. **Run the analysis** in `src/` (details in [src/README.md](src/README.md)):

   ```bash
   cd src
   python build_grid_to_ba.py          # one-time grid-cell → BA mapping
   bash run.sh                         # heatwave detection, TELL linkage, affected BAs, interconnections
   ```

## Reproduce my figures

1. Download and unpack the data archive (see *Data reference*).
2. Create the Python environment:

   ```bash
   conda create -n eamxx-tell python=3.13
   conda activate eamxx-tell
   pip install -r requirements.txt
   ```

3. Point the scripts to the archive and run every figure step:

   ```bash
   export DATA_ROOT=/path/to/EAMxx-TELL-data
   export OUTPUT_ROOT=$DATA_ROOT/outputs
   bash make_figures.sh
   ```

All input paths are defined in [`src/util_constants.py`](src/util_constants.py).
Without `DATA_ROOT` they default to the NERSC locations used for the paper.

| Figure | Description | Script(s) | Output (under `OUTPUT_ROOT`) |
|--------|-------------|-----------|------------------------------|
| SI 1 | TELL (DYAMOND1) vs EIA-930 hourly load, NYIS / PJM / MISO | `plot_DYAMOND_tell_load_timeseries.py`, `plot_ba_service_territory.py`, `merge_images.py` | `DYAMOND1/` |
| 1 | Ensemble mean / std of maximum temperature exceedance | `plot_ensemble_stats.py` | `ensemble_stats/` |
| 2 and 3 | Heatwave area and duration, Control vs PGW | `compare_heatwaves.py` | `comparison/` |
| 4 | Event-specific amplification, PGW − (Control + Δ) | `plot_pgw_amplification.py` | `pgw_amplification/` |
| 5 | Number of heatwave-affected BAs per day | `plot_affected_ba.py` | `tell_heatwave_linkage/` |
| 6 and 7 | BA peak-load anomalies (2011, 2021) | `analyze_tell_heatwave.py` | `tell_heatwave_linkage/` |
| SI 6 | Interconnection load anomalies | `interconnection_stress.py` | `tell_heatwave_linkage/` |
| SI 3 | BA service territories and interconnections | `plot_ba_service_territory.py` | `BA_service_territory/` |

## Repository layout

```
├── README.md            # This file
├── LICENSE
├── requirements.txt     # Python dependencies
├── make_figures.sh      # Regenerate all figures from the data archive
├── src/                 # Analysis pipeline (see src/README.md)
└── exploratory/         # Scripts not used in the manuscript
```

## Acknowledgments

This research was supported by the Earth and Biological Sciences Directorate (EBSD)’s Laboratory
Directed Research and Development (LDRD) Program at Pacific Northwest National Laboratory (PNNL).
PNNL is operated by Battelle for the U.S. Department of Energy under Contract DE-AC05-76RLO1830.
This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of
Science User Facility supported by the Office of Science of the U.S. Department of Energy under
Contract No. DE-AC02-05CH11231 using NERSC award BER-ERCAP0037494.
