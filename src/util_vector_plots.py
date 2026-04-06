from pathlib import Path
import numpy as np
import seaborn as sns
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__author__ = 'Eva Sinha'
__email__  = 'eva.sinha@pnnl.gov'

plt.rc('figure', titlesize=15)
plt.rc('legend', fontsize=15, title_fontsize=15)
plt.rc('axes',   labelsize=15, titlesize=15)
plt.rc('xtick',  labelsize=15)
plt.rc('ytick',  labelsize=15)
plt.rc('figure', figsize=(11, 8.5))

# ---------------------------------------------------------------------------
def _resolve_out_path(fname) -> Path:
    """Return the save path, mirroring the logic in util_spatial_plots:
    plain filenames go into figures/, absolute/relative paths save directly."""
    p = Path(fname)
    if p.parent == Path('.'):
        out = Path('figures') / p
        out.parent.mkdir(exist_ok=True)
    else:
        out = p
        out.parent.mkdir(parents=True, exist_ok=True)
    return out

# -----------------------------------------------------------
# Make line plot
def plot_sns_lineplot(df, x_col, y_col, fname, hue_col=None, title=None, xlabel=None, ylabel=None):

    out_path = _resolve_out_path(fname)

    g = sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, linewidth=2.0, estimator=None)

    # Modify axis labels
    g.set(xlabel=xlabel, ylabel=ylabel, title=title)

    # Move legend to bottom
    if(hue_col is not None):
      sns.move_legend(g, 'upper center', ncol=len(np.unique(df[hue_col])), bbox_to_anchor=(0.5, -0.1), title=None, frameon=True, fontsize=15)

    plt.savefig(out_path, bbox_inches='tight')

    plt.close(fig=None)

# -----------------------------------------------------------
# Make line plot
def plot_xr_lineplot(ds, fname, title=None, xlabel=None, ylabel=None):

    out_path = _resolve_out_path(fname)

    #plt.plot(ds['time'].values, ds.values, marker='o')
    ds.plot.line(marker='o')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(out_path, bbox_inches='tight')

    plt.close(fig=None)

# -----------------------------------------------------------
# Make scatter plot
def plot_xr_scatterplot(x_val, y_val, c_col, fname, title=None, xlabel=None, ylabel=None, cbar_label=None):

    out_path = _resolve_out_path(fname)

    c_col_arr = np.array(c_col)

    if c_col_arr.size == 0:
        logging.getLogger(__name__).warning(
            "plot_xr_scatterplot: no data to plot (empty inputs); skipping %s", out_path
        )
        return

    c_min, c_max = np.nanmin(c_col_arr), np.nanmax(c_col_arr)
    c_range = c_max - c_min if c_max != c_min else 1.0
    sc = plt.scatter(x_val, y_val, c=c_col_arr, cmap='YlOrRd',
          s=20 + 80 * (c_col_arr - c_min) / c_range,
          edgecolor=None, alpha=0.8)

    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)

    plt.savefig(out_path, bbox_inches='tight')

    plt.close(fig=None)

# -----------------------------------------------------------
