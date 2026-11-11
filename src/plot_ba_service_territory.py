"""
plot_ba_service_territory.py
-----------------------------
Plot the county-level service territory for one or more Balancing Authorities
(BAs) overlaid on a CONUS basemap, and produce a single map showing all three
NERC interconnections (EIC, WIC, ERCOT) coloured by interconnection.

Figures produced
----------------
* One PNG per BA  : ``BA_Service_Territory_<BA>.png``
* One combined PNG: ``Interconnections_Map.png``

Data inputs
-----------
* County shapefile       : ``tl_2020_us_county.shp``  (TIGER/Line 2020)
* BA–county mapping      : ``ba_service_territory_2019.csv``
  Columns used: ``County_FIPS`` (int), ``BA_Code`` (str)
* BA–interconnection map : ``BA_to_Interconnection_Mapping.csv``
  Columns used: ``BA_Code`` (str), ``Interconnection`` (str)

Usage
-----
    conda activate py313
    python plot_ba_service_territory.py

Authors
-------
Eva Sinha  <eva.sinha@pnnl.gov>
"""

from __future__ import annotations

# ── PROJ database path must be set BEFORE geopandas/pyproj are imported ─────
import os
# Defaults to the active conda environment's PROJ database
import sys
_PROJ_DB = os.environ.get("PROJ_DATA", os.path.join(sys.prefix, "share", "proj"))
os.environ["PROJ_DATA"]    = _PROJ_DB
os.environ["PROJ_LIB"]     = _PROJ_DB
os.environ["PROJ_NETWORK"] = "OFF"
# ────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path

import pyproj
pyproj.datadir.set_data_dir(_PROJ_DB)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd

from util_constants import OUTPUT_ROOT, WRF_TO_TELL_DIR
import pandas as pd

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
# Constants
# ---------------------------------------------------------------------------
FONTSIZE = 14

# Highlight colour for the selected BA
BA_COLOR = "#d62728"   # red
# Base colour for all other CONUS counties
BASE_COLOR = "#d9d9d9"  # light grey
# State boundary colour
STATE_COLOR = "#555555"

# Colours for the three NERC interconnections
INTERCONNECTION_COLORS: dict[str, str] = {
    "EIC":   "#1f77b4",   # blue
    "WIC":  "#2ca02c",   # green
    "ERCOT": "#ff7f0e",   # orange
}

# CONUS bounding box (longitude, latitude)
CONUS_XLIM = (-125.0, -66.0)
CONUS_YLIM = (  24.0,  50.0)

# FIPS codes for Alaska (02) and Hawaii (15) — excluded from CONUS view
_NON_CONUS_STATE_FP = {"02", "15"}

# Shared cartopy projection and CONUS extent — used by all plot functions
_PROJ = ccrs.LambertConformal(central_longitude=-96, central_latitude=37.5,
                              standard_parallels=(29.5, 45.5))
_PC   = ccrs.PlateCarree()
_CONUS_EXTENT = [-119.5, -74.0, 25.0, 50.0]   # [W, E, S, N] in degrees


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# LCC projection string — shared by loading and plotting code
_LCC_PROJ = "+proj=lcc +lon_0=-96 +lat_0=37.5 +lat_1=29.5 +lat_2=45.5 +datum=WGS84"


def load_county_gdf(shapefile: Path) -> gpd.GeoDataFrame:
    """Load the county shapefile, pre-project to LCC, and drop non-CONUS.

    Reprojecting here (once) avoids repeating the expensive ``to_crs()`` call
    for every BA inside the plotting loop.
    """
    log.info("Reading county shapefile: %s", shapefile)
    gdf = gpd.read_file(shapefile)

    # Zero-pad GEOID to 5 characters (state 2-digit + county 3-digit)
    gdf["GEOID"] = gdf["GEOID"].str.zfill(5)
    gdf["COUNTY_FIPS_INT"] = gdf["GEOID"].astype(int)

    # Drop Alaska, Hawaii, and territories
    gdf = gdf[~gdf["STATEFP"].isin(_NON_CONUS_STATE_FP)].copy()
    # Keep only 48 contiguous states + DC (FIPS 01–56, excluding 02 & 15)
    gdf = gdf[gdf["STATEFP"].astype(int) <= 56].copy()

    # Reproject directly to target LCC — no intermediate WGS-84 step
    gdf = gdf.to_crs(_LCC_PROJ)
    return gdf


def load_ba_territory_df(ba_csv: Path) -> pd.DataFrame:
    """Read and return the full BA–county mapping DataFrame.

    Columns: ``County_FIPS`` (int), ``BA_Code`` (str), and others.
    Load this once and pass it to :func:`get_ba_fips` to avoid re-reading
    the CSV for every BA.
    """
    log.info("Reading BA service territory CSV: %s", ba_csv)
    return pd.read_csv(ba_csv)


def get_ba_fips(ba_df: pd.DataFrame, ba_code: str) -> set[int]:
    """Return the set of County_FIPS integers for *ba_code*.

    Parameters
    ----------
    ba_df : DataFrame
        Full BA–county mapping as returned by :func:`load_ba_territory_df`.
    ba_code : str
        BA abbreviation, e.g. ``'PJM'``.
    """
    fips = ba_df.loc[ba_df["BA_Code"] == ba_code, "County_FIPS"]
    if fips.empty:
        raise ValueError(f"BA code '{ba_code}' not found in the BA territory data")
    return set(fips.astype(int))


def load_interconnection_fips(
    ba_df: pd.DataFrame,
    interconnection_csv: Path,
) -> dict[str, set[int]]:
    """Return a mapping of interconnection name → set of County_FIPS integers.

    Joins the BA–county DataFrame with ``BA_to_Interconnection_Mapping.csv``
    on ``BA_Code``.

    Parameters
    ----------
    ba_df : DataFrame
        Full BA–county mapping as returned by :func:`load_ba_territory_df`.
    interconnection_csv : Path
        BA–interconnection mapping CSV.

    Returns
    -------
    dict with keys ``'EIC'``, ``'WIC'``, ``'ERCOT'`` and values that are
    sets of integer County_FIPS codes.
    """
    log.info("Building interconnection → county FIPS mapping")
    ic_df  = pd.read_csv(interconnection_csv)
    merged = ba_df.merge(ic_df, on="BA_Code", how="left")

    return {
        str(ic): set(grp["County_FIPS"].dropna().astype(int))
        for ic, grp in merged.groupby("Interconnection")
    }


# ---------------------------------------------------------------------------
# State-boundary dissolve
# ---------------------------------------------------------------------------
def state_boundaries(county_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve county polygons to state polygons for boundary overlay.

    Accepts a GeoDataFrame in any CRS; returns boundaries in the same CRS.
    Pre-compute this once after ``load_county_gdf`` and pass it into the
    plotting function to avoid repeating the dissolve for every BA.
    """
    return county_gdf.dissolve(by="STATEFP").reset_index()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _make_ax(figsize: tuple[float, float] = (13, 8)) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure + LCC cartopy axes clipped to the CONUS extent."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": _PROJ})
    ax.set_extent(_CONUS_EXTENT, crs=_PC)
    ax.tick_params(labelsize=FONTSIZE - 1)
    return fig, ax


def _save_close(
    fig: plt.Figure,
    out_path: Path,
    resolution: int,
    save: bool,
) -> None:
    """Save *fig* to *out_path* (when *save* is True) then close it."""
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=resolution, bbox_inches="tight")
        log.info("Saved → %s", out_path)
    plt.close(fig)


def _assign_colors(
    county_gdf: gpd.GeoDataFrame,
    fips_color_map: dict[int, str],
) -> list[str]:
    """Return a list of colours, one per county row, using *fips_color_map*.

    Counties not present in the mapping receive ``BASE_COLOR``.
    """
    return county_gdf["COUNTY_FIPS_INT"].map(
        lambda f: fips_color_map.get(f, BASE_COLOR)
    ).tolist()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_ba_service_territory(
    ba_code: str,
    plot_str: str,
    county_gdf: gpd.GeoDataFrame,
    states_gdf: gpd.GeoDataFrame,
    ba_fips: set[int],
    image_output_dir: Path,
    image_resolution: int = 150,
    save_images: bool = False,
) -> None:
    """Individual service territory map for one BA.

    Parameters
    ----------
    ba_code : str
        BA abbreviation used in the title and output filename.
    county_gdf, states_gdf : GeoDataFrame
        CONUS county / state geometries in LCC CRS.
    ba_fips : set[int]
        County FIPS integers belonging to this BA.
    image_output_dir : Path
        Output directory for PNG files.
    image_resolution : int
        DPI (default 150).
    save_images : bool
        Write PNG when True.
    """
    log.info("BA=%s — %d counties in service territory",
             ba_code, sum(f in ba_fips for f in county_gdf["COUNTY_FIPS_INT"]))

    fips_color = {f: BA_COLOR for f in ba_fips}
    colors = _assign_colors(county_gdf, fips_color)

    fig, ax = _make_ax()
    county_gdf.plot(ax=ax, color=colors, edgecolor="white", linewidth=0.2)
    states_gdf.boundary.plot(ax=ax, edgecolor=STATE_COLOR, linewidth=0.6)

    ax.legend(
        handles=[
            mpatches.Patch(color=BA_COLOR,   label=f"{ba_code} {plot_str}"), #service territory"),
            mpatches.Patch(color=BASE_COLOR,  label="Other CONUS counties"),
        ],
        loc="lower left", fontsize=FONTSIZE - 1, frameon=True,
    )

    _save_close(fig, image_output_dir / f"{plot_str}_Service_Territory_{ba_code}.png",
                image_resolution, save_images)


def plot_all_bas_map(
    ba_list: list[str],
    county_gdf: gpd.GeoDataFrame,
    states_gdf: gpd.GeoDataFrame,
    ba_fips_map: dict[str, set[int]],
    image_output_dir: Path,
    image_resolution: int = 150,
    save_images: bool = False,
) -> None:
    """Single CONUS map with each selected BA in a distinct colour.

    Parameters
    ----------
    ba_list : list[str]
        Ordered list of BA codes to highlight.
    county_gdf, states_gdf : GeoDataFrame
        CONUS county / state geometries in LCC CRS.
    ba_fips_map : dict[str, set[int]]
        BA code → County_FIPS set (pre-loaded by the caller).
    image_output_dir : Path
        Output directory for the PNG.
    image_resolution : int
        DPI (default 150).
    save_images : bool
        Write PNG when True.
    """
    #_BA_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    _BA_PALETTE = [
    "#D55E00",  # vermillion
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#004D40",  # dark teal (in place of black)
]
    ba_color_map = {ba: _BA_PALETTE[i % len(_BA_PALETTE)]
                    for i, ba in enumerate(ba_list)}

    log.info("Plotting combined BA map: %s", ba_list)

    # Build fips → colour lookup (first BA wins for overlapping counties)
    fips_color: dict[int, str] = {}
    for ba in reversed(ba_list):          # reverse so first BA wins
        for f in ba_fips_map.get(ba, set()):
            fips_color[f] = ba_color_map[ba]

    colors = _assign_colors(county_gdf, fips_color)

    fig, ax = _make_ax()
    county_gdf.plot(ax=ax, color=colors, edgecolor="white", linewidth=0.2)
    states_gdf.boundary.plot(ax=ax, edgecolor=STATE_COLOR, linewidth=0.6)

    legend_handles = [
        mpatches.Patch(color=ba_color_map[ba], label=f"{ba}")# BA  service territory")
        for ba in ba_list
    ] + [mpatches.Patch(color=BASE_COLOR, label="Other CONUS counties")]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=FONTSIZE - 1, frameon=True, ncols=2)

    fname = "BA_Service_Territory_" + "_".join(ba_list) + ".png"
    _save_close(fig, image_output_dir / fname, image_resolution, save_images)


def plot_interconnection_map(
    county_gdf: gpd.GeoDataFrame,
    states_gdf: gpd.GeoDataFrame,
    ic_fips: dict[str, set[int]],
    image_output_dir: Path,
    image_resolution: int = 150,
    save_images: bool = False,
) -> None:
    """Single CONUS map with each NERC interconnection in a distinct colour.

    Parameters
    ----------
    county_gdf, states_gdf : GeoDataFrame
        CONUS county / state geometries in LCC CRS.
    ic_fips : dict[str, set[int]]
        Interconnection name → County_FIPS set (from
        :func:`load_interconnection_fips`).
    image_output_dir : Path
        Output directory for the PNG.
    image_resolution : int
        DPI (default 150).
    save_images : bool
        Write PNG when True.
    """
    log.info("Plotting interconnections map (%d interconnections)", len(ic_fips))

    # Build fips → colour lookup
    fips_color: dict[int, str] = {}
    for ic, fips_set in ic_fips.items():
        color = INTERCONNECTION_COLORS.get(ic, BASE_COLOR)
        for f in fips_set:
            fips_color[f] = color

    colors = _assign_colors(county_gdf, fips_color)

    fig, ax = _make_ax()
    county_gdf.plot(ax=ax, color=colors, edgecolor="white", linewidth=0.2)
    states_gdf.boundary.plot(ax=ax, edgecolor=STATE_COLOR, linewidth=0.6)

    legend_handles = [
        mpatches.Patch(color=INTERCONNECTION_COLORS[ic], label=ic)
        for ic in ["EIC", "WIC", "ERCOT"] if ic in ic_fips
    ] + [mpatches.Patch(color=BASE_COLOR, label="Unassigned")]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=FONTSIZE - 1, frameon=True,
              title="Interconnection", title_fontsize=FONTSIZE - 1)

    _save_close(fig, image_output_dir / "Interconnections_Map.png",
                image_resolution, save_images)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    #BA_LIST = ["NYIS", "PJM", "MISO"]
    #BA_LIST = ["PJM", "MISO", "SWPP", "ERCO"]
    #BA_LIST = ["AECI", "LGEE", "SWPP", "PJM"]
    BA_LIST = ["AZPS", "PACE", "PNM", "WACM"]
    BA_LIST = ["AZPS", "ERCO", "MISO", "PACE", "PNM", "PJM", "SWPP", "WACM"]
    #BA_LIST = ["AVA", "BPAT", "CHPD", "DOPD", "GCPD", "IPCO", "PACW", "PGE", "PSEI", "SCL", "TPWR"]
    #BA_LIST = ["AECI", "ERCO", "LGEE", "MISO", "NYIS", "PJM", "SWPP"]
    #BA_LIST = ["CISO", "NEVP", "NWMT", "PACE", "WACM", "WALC", "WAUW"]
    #BA_LIST = ["BPAT", "GCPD", "IPCO", "PACW"]

    COUNTY_SHP = WRF_TO_TELL_DIR / "wrf_to_tell_ancillary_data" / "tl_2020_us_county.shp"
    BA_CSV = WRF_TO_TELL_DIR / "wrf_to_tell_ancillary_data" / "ba_service_territory_2019.csv"
    INTERCONNECTION_CSV = Path(__file__).parent / "BA_to_Interconnection_Mapping.csv"
    IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "BA_service_territory"

    # Load county shapefile and state boundaries once — reused for all plots
    county_gdf = load_county_gdf(COUNTY_SHP)
    states_gdf = state_boundaries(county_gdf)

    # Read BA–county CSV once — reused by BA plots and interconnection mapping
    ba_df = load_ba_territory_df(BA_CSV)

    # ── Per-BA individual service territory maps ─────────────────────────────
    ba_fips_map: dict[str, set[int]] = {}
    for ba in BA_LIST:
        try:
            ba_fips_map[ba] = get_ba_fips(ba_df, ba)
        except ValueError as exc:
            log.warning("%s — skipping.", exc)
            continue

        plot_ba_service_territory(
            ba_code=ba,
            plot_str="BA",
            county_gdf=county_gdf,
            states_gdf=states_gdf,
            ba_fips=ba_fips_map[ba],
            image_output_dir=IMAGE_OUTPUT_DIR,
            image_resolution=150,
            save_images=True,
        )

    # ── Combined map: all three BAs in one figure ────────────────────────────
    if ba_fips_map:
        plot_all_bas_map(
            ba_list=[ba for ba in BA_LIST if ba in ba_fips_map],
            county_gdf=county_gdf,
            states_gdf=states_gdf,
            ba_fips_map=ba_fips_map,
            image_output_dir=IMAGE_OUTPUT_DIR,
            image_resolution=150,
            save_images=True,
        )

    # ── Individual interconnection maps + combined interconnections map ───────
    ic_fips = load_interconnection_fips(ba_df, INTERCONNECTION_CSV)

    for ic_name, fips_set in ic_fips.items():
        plot_ba_service_territory(
            ba_code=ic_name,
            plot_str="Interconnection",
            county_gdf=county_gdf,
            states_gdf=states_gdf,
            ba_fips=fips_set,
            image_output_dir=IMAGE_OUTPUT_DIR,
            image_resolution=150,
            save_images=True,
        )

    plot_interconnection_map(
        county_gdf=county_gdf,
        states_gdf=states_gdf,
        ic_fips=ic_fips,
        image_output_dir=IMAGE_OUTPUT_DIR,
        image_resolution=150,
        save_images=True,
    )

    log.info("Done.")
