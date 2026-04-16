"""
plot_EAMxx.py
-------------
Make CONUS spatial plots of EAMxx / SCREAMv1 output variables.

Each entry in PLOT_CONFIGS defines one dataset to process.  For every
variable listed, the script slices the CONUS domain, computes basic
statistics, and calls xr_plot_US() to produce a PNG.

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

import matplotlib as mpl
mpl.use('Agg')
import xarray as xr

from util_spatial_plots import cmap, create_metrics, xr_plot_US
from util_contour_levels import *  # noqa: F401,F403  (supplies contour-level constants)

# ---------------------------------------------------------------------------
# Shared plot settings
# ---------------------------------------------------------------------------
FIG_EXTENT = [235, 293, 24, 47]   # [lon_min, lon_max, lat_min, lat_max]
FIG_WT     = 14
FIG_HT     = 8

# CONUS lat/lon slice (degrees)
LAT_SLICE = slice(24, 50)
LON_SLICE = slice(-125, -66)

# ---------------------------------------------------------------------------
# Dataset configurations
# Each dict describes one NetCDF file + the variables to plot.
# ---------------------------------------------------------------------------
PLOT_CONFIGS = [
    {
        "fpath":       "/pscratch/sd/e/esinha/DYAMOND1_SCREAMv1/",
        "fname":       "output.scream.SurfVars.INSTANT.nmins_x15.2016-08-01-00900_rgr_rot.nc",
        "varnames":    ["T2"],
        "title_label": "EAMxx",
        "date_label":  "2016-08-01",
        "time_index":  20,          # index along the time dimension to plot
        "show_states": True,
    },
    {
        "fpath":       "/pscratch/sd/j/jli628/EAMxx_regridding/heatwave_201107/",
        "fname":       "output.eamxx.INSTANT.nsteps_x9.2011-07-24-32400.nc",
        "varnames":    ["T_2m"],
        "title_label": "EAMxx",
        "date_label":  "2011-07-24",
        "time_index":  None,        # no time dimension — plot the 2-D field directly
        "show_states": False,
    },
    {
        "fpath":       "/pscratch/sd/e/esinha/E3SMv3/20260303_I20TREAMELMCNPRDCTCBCPHSBGC_ne30pg2_f09_oEC60to30v3/",
        "fname":       "20260303_I20TREAMELMCNPRDCTCBCPHSBGC_ne30pg2_f09_oEC60to30v3.eam.h0.2011-07_rgr.nc",
        "varnames":    ["TREFHT"],
        "title_label": "EAM",
        "date_label":  "2011-07",
        "time_index":  None,        # no time dimension — plot the 2-D field directly
        "show_states": False,
    },
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def plot_variable(ds, var, title_label, date_label, time_index, show_states):
    """Slice the CONUS domain, compute stats, and save a spatial PNG.

    Parameters
    ----------
    ds          : xr.Dataset — open dataset.
    var         : str        — variable name inside ds.
    title_label : str        — title string used in the title and filename.
    date_label  : str        — date string used in the title and filename.
    time_index  : int | None — if not None, select this index along the time
                               dimension before plotting.
    show_states : bool       — draw US state boundaries.
    """
    plot_data   = ds[var]
    conus_data  = plot_data.sel(lat=LAT_SLICE, lon=LON_SLICE)

    field = conus_data[time_index] if time_index is not None else conus_data

    metrics = create_metrics(conus_data)
    stats   = (metrics["max"], metrics["mean"], metrics["min"])

    title    = f"{title_label} output on {date_label} - {var}"
    out_fname = f"{var}_{title_label}_output_US_{date_label}.png"

    xr_plot_US(
        field,
        stats,
        cmap_col    = cmap,
        title       = title,
        fig_wt      = FIG_WT,
        fig_ht      = FIG_HT,
        fig_extent  = FIG_EXTENT,
        show_states = show_states,
        fname       = out_fname,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for cfg in PLOT_CONFIGS:
        ds = xr.open_dataset(cfg["fpath"] + cfg["fname"], decode_times=False)
        for var in cfg["varnames"]:
            plot_variable(
                ds          = ds,
                var         = var,
                title_label = cfg["title_label"],
                date_label  = cfg["date_label"],
                time_index  = cfg["time_index"],
                show_states = cfg["show_states"],
            )
        ds.close()
