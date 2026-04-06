'''
Python modules for making spatial plots of ELM outputs
'''
import glob
import logging
import numpy as np
import xarray as xr
from scipy import ndimage as ndi

__author__ = 'Eva Sinha'
__email__  = 'eva.sinha@pnnl.gov'

log = logging.getLogger(__name__)

# -----------------------------------------------------------
def read_data(yr_start, yr_end, mon_start, mon_end, fpath, fname_abb, varnames, decode_times=True):
  """Read ELM model output for select variables
     param: yr_start      start year for reading model output
     param: yr_end        end year for reading model output
     param: mon_start     start month for reading model output
     param: mon_end       end month for reading model output
     param: fpath         directory path
     param: fname_abb     model run case id
     param: varnames      variable name for subsetting
     :return:             data array with model output
  """

  # Read names of all NetCDF files within the given year range
  fnames = []
  for yr in range(int(yr_start), int(yr_end)+1, 1):
    for mon in range(int(mon_start), int(mon_end)+1):
      fnames.append(fpath + fname_abb + str(yr).zfill(4) + '-' + str(mon).zfill(2) + '_CONUS.nc')
      #fnames.append(glob.glob(fpath + fname_abb + str(yr).zfill(4) + str(mon).zfill(2) + '*.nc')[0])

  # Open a multiple netCDF data file and load the data into xarrays
  ds = xr.open_mfdataset(fnames, decode_times=decode_times, combine='nested', 
         concat_dim='time', data_vars='minimal', compat='no_conflicts')

  # Only keep select variables in the data array
  ds = ds[varnames]

  return(ds)

# -----------------------------------------------------------
def read_EAMxx_data(yr_start, yr_end, mon_start, mon_end, day_start, day_end, fpath, fname_abb, varname, rename_var, decode_times=True):
  """Read ELM model output for select variables
     param: yr_start      start year for reading model output
     param: yr_end        end year for reading model output
     param: mon_start     start month for reading model output
     param: mon_end       end month for reading model output
     param: fpath         directory path
     param: fname_abb     model run case id
     param: varname      variable name for subsetting
     :return:             data array with model output
  """

  # Read names of all NetCDF files within the given year range
  fnames = []
  for yr in range(int(yr_start), int(yr_end)+1, 1):
    for mon in range(int(mon_start), int(mon_end)+1):
      for day in range(int(day_start), int(day_end)+1):
        for sec in range(0, 86400, 3600):  # Read every fourth time-step
          fnames.extend(glob.glob(fpath + fname_abb + str(yr).zfill(4) + '-' + str(mon).zfill(2) + '-' + str(day).zfill(2) + '-' + str(sec).zfill(5) + '*.nc'))

  # Sort the combined list of file names
  fnames.sort()

  # Open a multiple netCDF data file and load the data into xarrays
  ds = xr.open_mfdataset(fnames, decode_times=decode_times, combine='nested', 
         concat_dim='time', data_vars='minimal', compat='no_conflicts')

  # Rename coordinates
  ds = ds.rename({varname:rename_var})

  # Only keep select variables in the data array
  ds = ds[rename_var]

  return(ds)

# -----------------------------------------------------------
def longest_run(arr):
    # arr: 1D boolean array (time) or numeric array containing NaNs
    # Returns the length of the longest run of consecutive True values
    # If all values in arr are NA/NaN, return np.nan
    a = np.asarray(arr)

    # If array is empty, return 0.0
    if a.size == 0:
        return 0.0

    # Determine NaN positions safely (works for numeric arrays; non-numeric -> no NaNs)
    try:
        isnan = np.isnan(a)
    except Exception:
        isnan = np.zeros(a.shape, dtype=bool)

    # If every element is NaN -> return NA (np.nan)
    if np.all(isnan):
        return np.nan

    # Build boolean mask: True where value is truthy AND not NaN.
    try:
        truth = a.astype(bool)
    except Exception:
        truth = np.array([bool(x) for x in a])

    mask = np.logical_and(~isnan, truth)

    # Compute longest consecutive True run, treating NaNs as breaks
    max_run = 0
    current_run = 0
    for val in mask:
        if val:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return float(max_run)
# -----------------------------------------------------------
def compute_gridcell_areas(lat, lon, radius_km=6371.0):
    """
    Compute approximate grid-cell areas (km^2) for a regular lat/lon grid.
    lat, lon: 1D arrays (degrees) for cell centers.
    Returns an xarray.DataArray with dims (lat, lon) in km^2.
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError('lat and lon must be 1D')

    # compute edges
    lat_edges = np.empty(lat.size + 1, dtype=float)
    lon_edges = np.empty(lon.size + 1, dtype=float)

    # lat edges: midpoints, extrapolate at ends
    lat_diff = np.diff(lat)
    lat_edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    lat_edges[0] = lat[0] - 0.5 * lat_diff[0]
    lat_edges[-1] = lat[-1] + 0.5 * lat_diff[-1]

    # lon edges
    lon_diff = np.diff(lon)
    lon_edges[1:-1] = 0.5 * (lon[:-1] + lon[1:])
    lon_edges[0] = lon[0] - 0.5 * lon_diff[0]
    lon_edges[-1] = lon[-1] + 0.5 * lon_diff[-1]

    # convert to radians
    deg2rad = np.pi / 180.0
    lat_edges_rad = lat_edges * deg2rad
    lon_edges_rad = lon_edges * deg2rad

    # compute dlon for each cell column
    dlon = np.abs(np.diff(lon_edges_rad))  # length = nlon

    # compute area per (lat, lon) cell: R^2 * dlon * (sin(lat_north)-sin(lat_south))
    area = np.empty((lat.size, lon.size), dtype=float)
    for i in range(lat.size):
        lat_s = lat_edges_rad[i]
        lat_n = lat_edges_rad[i+1]
        lat_factor = np.abs(np.sin(lat_n) - np.sin(lat_s))
        area[i, :] = (radius_km**2) * lat_factor * dlon

    # make DataArray with coords
    area_da = xr.DataArray(area, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    return area_da

# -----------------------------------------------------------
def _proc(arr2d, area2d, anom2d, connectivity):
    """
    Identifies the largest contiguous region of True values in a 2D array (excluding NaNs), 
    and computes its total area and mean and max anomaly using corresponding arrays.
    """
    try:
        arr_bool = arr2d.astype(bool)
    except Exception:
        arr_bool = np.array([bool(x) for x in arr2d.flat]).reshape(arr2d.shape)

    nan_mask = np.isnan(arr2d)
    valid_heatwave_mask = np.logical_and(~nan_mask, arr_bool)

    if not valid_heatwave_mask.any():
        # return area, mean_anomaly, max_anomaly, mask
        return 0.0, np.nan, np.nan, np.zeros_like(arr2d, dtype=bool)

    # Generate a structuring element that will consider features connected even if they touch diagonally
    struct = ndi.generate_binary_structure(2, 1 if connectivity == 1 else 2)
    labels, _ = ndi.label(valid_heatwave_mask, structure=struct)

    # compute area sum per label using bincount with weights
    labels_flat = labels.ravel()
    weights_flat = area2d.ravel()
    sums = np.bincount(labels_flat, weights=weights_flat)
    if sums.size <= 1:
        return 0.0, np.nan, np.zeros_like(arr2d, dtype=bool)
    sums[0] = 0.0
    largest_label = sums.argmax()
    largest_area = float(sums[largest_label])
    largest_mask = (labels == largest_label)

    # mean and max anomaly over largest_mask (ignore NaNs)
    try:
        anom_vals = anom2d[largest_mask]
        anom_vals = anom_vals[~np.isnan(anom_vals)]
        mean_anom = float(anom_vals.mean()) if anom_vals.size > 0 else np.nan
        max_anom = float(anom_vals.max()) if anom_vals.size > 0 else np.nan
    except Exception:
        mean_anom = np.nan
        max_anom = np.nan

    return largest_area, mean_anom, max_anom, largest_mask

# -----------------------------------------------------------
def largest_contiguous_regions(heatwave_mask, anomaly, connectivity=1):
    """
    Identify the largest spatially-contiguous True region in the 2D (lat, lon)
    field for each day in the provided DataArray and return the area (km^2).

    Parameters
    - heatwave_mask: xarray.DataArray with dims (time, lat, lon). Values should
      be boolean (True where heatwave condition holds) with NaNs preserved for
      masked/ocean cells.
    anomaly : xarray.DataArray
        Anomaly values with the same shape as `heatwave_mask`, used to compute mean anomaly for each event.
    - connectivity: int, 1 for 4-connectivity, 2 for 8-connectivity.

    Returns
    - sizes: xarray.DataArray with dim (time,) giving the area (km^2)
      of the largest contiguous region for each day (float).
    - masks: xarray.DataArray with dims (time, lat, lon) boolean mask marking
      the largest contiguous region for each day (False where none).
    """

    # prepare area weights array
    lat = heatwave_mask['lat'].values
    lon = heatwave_mask['lon'].values
    area_weights = compute_gridcell_areas(lat, lon)

    sizes, mean_anoms, max_anoms, masks = xr.apply_ufunc(
        _proc,
        heatwave_mask,
        area_weights,
        anomaly,
        xr.DataArray(np.array(connectivity)),
        input_core_dims=[["lat", "lon"], ["lat", "lon"], ["lat", "lon"], []],
        output_core_dims=[[], [], [], ["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, bool],
    )

    sizes.name = "largest_region_area_km2"
    mean_anoms.name = "largest_region_mean_anomaly_K"
    max_anoms.name = "largest_region_max_anomaly_K"
    masks.name = "largest_region_mask"

    return sizes, mean_anoms, max_anoms, masks
# -----------------------------------------------------------
def area_duration_anomaly(heatwave_mask, anomaly, connectivity=1):
    """
    Identifies and analyzes the largest contiguous 3D regions (events) in a heatwave mask over time and space.

    This function labels connected regions in a 3D boolean mask (time, latitude, longitude) representing heatwave occurrences,
    computes the duration, maximum spatial extent, and mean anomaly for each event.

    Parameters
    ----------
    heatwave_mask : xarray.DataArray
        Boolean mask (True/False) indicating heatwave occurrence, with dimensions (time, lat, lon).
    anomaly : xarray.DataArray
        Anomaly values with the same shape as `heatwave_mask`, used to compute mean anomaly for each event.
    connectivity : int, optional
        Connectivity for spatial neighbors. Use 1 for 4-connectivity (default), 2 for 8-connectivity.

    Returns
    -------
    durations : list of int
        Duration (number of time steps) for each contiguous event.
    max_areas : list of float
        Maximum spatial area (sum of grid cell areas) covered by each event at any time step.
    mean_anoms : list of float
        Mean anomaly value over all cells in each event.

    Notes
    -----
    - This function may be memory intensive for large datasets, as it converts DataArrays to numpy arrays.
    - The function prints the number of detected events and skips analysis if no heatwave cells are found.

    Suggested better name
    ---------------------
    detect_heatwave_events_3d
    """
    # Note: this step converts DataArrays to numpy arrays and may be memory intensive for large datasets.

    durations  = []
    max_areas  = []
    mean_anoms = []

    hw_mask_arr = heatwave_mask.fillna(False).values.astype(bool)  # shape (time, lat, lon)
    if hw_mask_arr.sum() == 0:
        log.warning("area_duration_anomaly: no heatwave cells found; returning empty event lists.")
        return durations, max_areas, mean_anoms

    # build 3D structuring element: connect time neighbors and spatial neighbors according to `connectivity`
    space_struct = ndi.generate_binary_structure(2, 1 if connectivity == 1 else 2)
    structure3d = np.zeros((3,) + space_struct.shape, dtype=int)
    structure3d[1, :, :] = space_struct
    # connect adjacent time steps only at the same spatial position (no diagonal in time)
    structure3d[0, 1, 1] = 1
    structure3d[2, 1, 1] = 1

    labels3d, n_events = ndi.label(hw_mask_arr, structure=structure3d)
    log.info("area_duration_anomaly: %d 3D connected event(s) found.", n_events)

    # prepare area weights array
    lat = heatwave_mask['lat'].values
    lon = heatwave_mask['lon'].values
    area_weights = compute_gridcell_areas(lat, lon)
    area_arr = area_weights.values  # shape (lat, lon)

    # anomaly values as numpy array (time, lat, lon)
    anom_arr = anomaly.values

    for eid in range(1, n_events + 1):
        idx = np.where(labels3d == eid)
        if len(idx[0]) == 0:
            continue
        times = np.unique(idx[0])
        duration = times.size
        # compute area per time for this event
        area_times = []
        for t in times:
            mask_t = labels3d[t] == eid
            area_t = (area_arr * mask_t).sum()
            area_times.append(area_t)
        max_area = float(np.max(area_times)) if area_times else 0.0
        # mean anomaly over all cells of the event (ignore NaNs)
        vals = anom_arr[labels3d == eid]
        vals = vals[~np.isnan(vals)]
        mean_anom_val = float(vals.mean()) if vals.size > 0 else np.nan

        durations.append(duration)
        max_areas.append(max_area)
        mean_anoms.append(mean_anom_val)

    return durations, max_areas, mean_anoms
# -----------------------------------------------------------
