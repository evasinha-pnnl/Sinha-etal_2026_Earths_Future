"""
Python modules for spatial plots
"""

from pathlib import Path
import matplotlib as mpl
mpl.use('agg')
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

__author__ = 'Eva Sinha'
__email__  = 'eva.sinha@pnnl.gov'

plt.rc('figure', titlesize=15)
plt.rc('legend', fontsize=15)
plt.rc('axes',   labelsize=15, titlesize=15)
plt.rc('xtick',  labelsize=15)
plt.rc('ytick',  labelsize=15)
plt.rc('figure', figsize=(11, 8.5))

# Colormap for sptial plots
colormap = 'WhiteBlueGreenYellowRed.rgb'
rgb_arr = np.loadtxt(colormap)
rgb_arr = rgb_arr / 255.0
cmap = LinearSegmentedColormap.from_list(name=colormap, colors=rgb_arr)

diff_cmap = 'BrBG'

# -----------------------------------------------------------
# Source: https://github.com/E3SM-Project/e3sm_diags/blob/master/e3sm_diags/driver/lat_lon_driver.py
def create_metrics(da):
    """Creates the mean, max, min, rmse, corr in a dictionary"""
    # For input None, metrics are instantiated to 999.999.
    # Apply float() to make sure the elements in metrics_dict are JSON serializable, i.e. np.float64 type is JSON serializable, but not np.float32.
    missing_value = 999.999
    metrics_dict = {}

    metrics_dict = { 
        'min': float(da.min()) if da is not None else missing_value,
        'max': float(da.max()) if da is not None else missing_value,
        #'mean': float(mean(da)) if da is not None else missing_value,# Was giving error - You have specified an invalid axis
        'mean': float(da.mean()) if da is not None else missing_value,
    }   

    return(metrics_dict)

# -----------------------------------------------------------
def xr_plot_US(da_plot, stats, cmap_col, title, fig_wt, fig_ht, fig_extent, show_states, fname, levels=None, norm=None):

    # If fname has no directory component, save into the local figures/ folder
    # (legacy behaviour).  If an absolute or relative path with directories is
    # provided, save directly to that path — no chdir needed.
    fname = Path(fname)
    if fname.parent == Path('.'):
        out_path = Path('figures') / fname
        out_path.parent.mkdir(exist_ok=True)
    else:
        out_path = fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(fig_wt, fig_ht))
    ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection = ccrs.LambertConformal(central_longitude=-95, central_latitude=37.5))

    # spatial plot using xarray
    fg = da_plot.plot(ax          = ax, 
                      transform   = ccrs.PlateCarree(), # coordinate system of data
                      levels      = levels,
                     # norm        = norm,
                      cmap        = cmap_col,
                      extend      = 'both',
                      add_colorbar= False)

    # Add title
    ax.set_title(title, fontsize=20)

    # Add colorbar
    cax = fig.add_axes([ax.get_position().x1+0.04, ax.get_position().y0, 0.015, ax.get_position().height])
    cbar = plt.colorbar(fg, cax=cax)

    if levels is None:
       cbar.ax.tick_params(labelsize=20, length=0)

    else:
       maxval = np.amax(np.absolute(levels[1:-1]))
       if maxval < 0.2:
          fmt = "%5.3f"
          pad = 28
       elif maxval < 10.0:
          fmt = "%5.2f"
          pad = 40
       elif maxval < 100.0:
          fmt = "%5.0f"
          pad = 25
       elif maxval < 10000.0:
          fmt = "%5.0f"
          pad = 30
       elif maxval > 9999.0:
          fmt = "%.0f"
          pad = 50
       else:
          fmt = "%6.1f"
          pad = 40

       cbar.set_ticks(levels[1:-1])
       labels = [fmt % level for level in levels[1:-1]]
       cbar.ax.set_yticklabels(labels, ha='right')
       cbar.ax.tick_params(labelsize=15, pad=pad, length=0)

    # Min, Mean, Max
    #ax.text(x=0.80, y=1.0, s='Max\nMean\nMin', ha='left', fontsize=20, transform=ax.transAxes)
    #ax.text(x=1.0, y=1.0, s='%.2f\n%.2f\n%.2f'%stats[0:3], ha='right', fontsize=20, transform=ax.transAxes)

    # Add additional features like coastline, oceans, and lakes
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='gainsboro') # To add a background grey color
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    if(show_states):
       ax.add_feature(cfeature.STATES, edgecolor='black')

    # Define map extent
    ax.set_extent(fig_extent)

    plt.savefig(out_path, bbox_inches='tight')

    plt.close(fig=None)
    
# -----------------------------------------------------------

def xr_plot_global(da_plot, stats, cmap_col, title, fig_wt, fig_ht, levels, norm, fname):

    # Same path-aware save logic as xr_plot_US
    fname = Path(fname)
    if fname.parent == Path('.'):
        out_path = Path('figures') / fname
        out_path.parent.mkdir(exist_ok=True)
    else:
        out_path = fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(fig_wt, fig_ht))
    ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection = ccrs.Robinson(central_longitude=0))

    # spatial plot using xarray
    fg = da_plot.plot.contourf(ax          = ax, 
                               transform   = ccrs.PlateCarree(), # coordinate system of data
                               levels      = levels, 
                               norm        = norm,
                               cmap        = cmap_col,
                               extend      = 'both',
                               add_colorbar= False)

    # Add title
    ax.set_title(title)

    # Add colorbar
    cax = fig.add_axes([ax.get_position().x1+0.04, ax.get_position().y0, 0.015, ax.get_position().height])
    cbar = plt.colorbar(fg, cax=cax)

    if levels is None:
       cbar.ax.tick_params(labelsize=15, length=0)

    else:
       maxval = np.amax(np.absolute(levels[1:-1]))
       if maxval < 0.2:
          fmt = "%5.3f"
          pad = 28
       elif maxval < 10.0:
          fmt = "%5.2f"
          pad = 40
       elif maxval < 100.0:
          fmt = "%5.0f"
          pad = 25
       elif maxval < 10000.0:
          fmt = "%5.0f"
          pad = 30
       elif maxval > 9999.0:
          fmt = "%.0f"
          pad = 50
       else:
          fmt = "%6.1f"
          pad = 40

       cbar.set_ticks(levels[1:-1])
       labels = [fmt % level for level in levels[1:-1]]
       cbar.ax.set_yticklabels(labels, ha='right')
       cbar.ax.tick_params(labelsize=15, pad=pad, length=0)

    # Min, Mean, Max
    ax.text(x=0.80, y=1.0, s='Max\nMean\nMin', ha='left', fontsize=20, transform=ax.transAxes)
    ax.text(x=1.0, y=1.0, s='%.2f\n%.2f\n%.2f'%stats[0:3], ha='right', fontsize=20, transform=ax.transAxes)

    # Add additional features like coastline and oceans
    ax.coastlines(lw = 0.6)

    plt.savefig(out_path, bbox_inches='tight')

    plt.close(fig=None)

# -----------------------------------------------------------

