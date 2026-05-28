import numpy as np
import os
import xarray as xr
import rasterio

from geopy.geocoders import Nominatim
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.windows import from_bounds, Window
from xgb_0_prep_params import *


def relative_humidity(T, Td):
    """
    Calculate relative humidity (%) from air temperature (T) and dew point temperature (Td).
    Temperatures should be in degrees Celsius.
    """
    A = 17.62
    B = 243.12  # °C

    def saturation_vapor_pressure(temp):
        return 6.112 * np.exp((A * temp) / (B + temp))

    e_Td = saturation_vapor_pressure(Td)
    e_T = saturation_vapor_pressure(T)

    RH = 100 * (e_Td / e_T)
    return RH


def load_era5land(date):
    """
    Load ERA5-Land data for a specific date.
    """
    file_path = os.path.join(era5land_dir, f"ERA5Land_{date}.nc")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERA5-Land file for {date} not found.")
    return xr.open_dataset(file_path)

def load_era5(date):
    """
    Load ERA5-Land data for a specific date.
    """
    file_path = os.path.join(era5_dir, f"ERA5_{date}.nc")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERA5 file for {date} not found.")
    return xr.open_dataset(file_path)

def load_era5pressure(date):
    """
    Load ERA5-Land data for a specific date.
    """
    file_path = os.path.join(era5pressure_dir, f"ERA5_{date}.nc")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERA5 file for {date} not found.")
    return xr.open_dataset(file_path)

# Function to find the nearest valid pixel
def find_nearest_valid_pixel(city_lat, city_lon, Lat, Lon, valid_mask):
    """
    Compute the Euclidean distance (in lat-lon space) from the given city coordinate 
    to all grid pixels, then return the coordinates of the nearest valid pixel 
    (i.e. one that is not NA).
    """
    distances = np.sqrt((Lat - city_lat)**2 + (Lon - city_lon)**2)
    # Exclude invalid pixels by setting their distances to infinity.
    distances[~valid_mask] = np.inf
    idx = np.unravel_index(np.argmin(distances), distances.shape)
    nearest_lat = Lat[idx]
    nearest_lon = Lon[idx]
    return nearest_lat, nearest_lon


def get_tif_data(lat, lon, data_list, transform_list, bounds_list, strict=False):
    """ Get raster value for given lat/lon, checking multiple TIFF files.
        If strict=True, returns None for any invalid/nodata value """
    for data, transform, bounds in zip(data_list, transform_list, bounds_list):
        left, bottom, right, top = bounds

        # Check if coordinates are inside the raster extent
        if left <= lon <= right and bottom <= lat <= top:
            row, col = rasterio.transform.rowcol(transform, lon, lat)
            
            # Ensure indices are within array bounds
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                value = data[row, col]
                
                if strict:
                    # Handle special nodata cases
                    if value == 65535:  # BH nodata
                        return None
                    if np.isnan(value):  # TCD nodata
                        return None
                return value

    return None if strict else np.nan



from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds

def crop_tiff(tiff_file, output_file, bbox_coords, buffer_units=1000, bbox_crs="EPSG:4326"):
    """
    Crop TIFF file using bounding box coordinates given in bbox_crs.
    - bbox_coords: (minx, miny, maxx, maxy) in CRS = bbox_crs
    - buffer_units: buffer amount in units of bbox_crs (e.g. meters for EPSG:3035)
    - output_file: path to write cropped tif
    - The function will transform the bbox to the raster's CRS before computing the window.
    """
    # Unpack bbox in the provided CRS
    minx, miny, maxx, maxy = bbox_coords

    # Apply buffer in bbox_crs units
    if buffer_units:
        minx -= buffer_units
        maxx += buffer_units
        miny -= buffer_units
        maxy += buffer_units

    with rasterio.open(tiff_file) as src:
        src_bounds = src.bounds
        src_crs = src.crs

        # If src has no CRS, assume bbox_crs === src_crs (best-effort)
        if src_crs is None:
            raise RuntimeError(f"Source {tiff_file} has no CRS; cannot transform bbox. src_crs is None.")

        # Transform bbox from bbox_crs -> src_crs (densify helps with curved transformations)
        if bbox_crs is None:
            raise ValueError("bbox_crs must be provided (e.g. 'EPSG:4326' or 'EPSG:3035').")

        if bbox_crs != src_crs.to_string():
            try:
                # transform_bounds returns (left, bottom, right, top) in dst crs
                tb_left, tb_bottom, tb_right, tb_top = transform_bounds(
                    bbox_crs, src_crs, minx, miny, maxx, maxy, densify_pts=21
                )
            except Exception as e:
                raise RuntimeError(f"Error transforming bbox from {bbox_crs} to {src_crs}: {e}")
        else:
            tb_left, tb_bottom, tb_right, tb_top = minx, miny, maxx, maxy

        # Quick check for overlap: if no overlap raise, otherwise clip to raster bounds (to avoid out-of-range windows)
        if (tb_right <= src_bounds.left) or (tb_left >= src_bounds.right) or (tb_top <= src_bounds.bottom) or (tb_bottom >= src_bounds.top):
            raise ValueError("Transformed bounding box is completely outside the TIFF extent")

        # Clip to raster bounds to ensure window is inside or partially inside
        clipped_left = max(tb_left, src_bounds.left)
        clipped_right = min(tb_right, src_bounds.right)
        clipped_bottom = max(tb_bottom, src_bounds.bottom)
        clipped_top = min(tb_top, src_bounds.top)

        # Build a window from the clipped bounds using the raster transform
        window = from_bounds(clipped_left, clipped_bottom, clipped_right, clipped_top, transform=src.transform)
        # round to integer offsets/shape
        window = window.round_offsets().round_shape()

        # Ensure window dimensions are valid integers and inside raster
        col_off = int(max(0, window.col_off))
        row_off = int(max(0, window.row_off))
        win_width = int(min(src.width - col_off, window.width))
        win_height = int(min(src.height - row_off, window.height))

        if win_width <= 0 or win_height <= 0:
            raise RuntimeError("Computed window has zero width or height after clipping")

        read_window = Window(col_off, row_off, win_width, win_height)

        # Read the data (use src.nodata if present, otherwise use np.nan)
        fill_value = src.nodata if src.nodata is not None else np.nan
        subset = src.read(1, window=read_window, boundless=True, fill_value=fill_value)

        # Compute new transform for the window
        new_transform = src.window_transform(read_window)

        # Prepare output metadata - preserve driver, dtype, count
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": src.driver,
            "height": subset.shape[0],
            "width": subset.shape[1],
            "transform": new_transform,
            "crs": src.crs,
            "nodata": src.nodata
        })

    # Write out the cropped file
    with rasterio.open(output_file, "w", **out_meta) as dst:
        dst.write(subset, 1)
        print(f"Cropped {tiff_file} -> {output_file}; shape: {subset.shape}, bounds (in src CRS): {dst.bounds}")



def assign_era5_values(tablex):
    # Load first ERA5 file to establish grid
    first_date = tablex['Date'].iloc[0].strftime('%Y-%m-%d')
    era5_sample = load_era5land(first_date)
    
    # Create coordinate arrays
    era5_lats = era5_sample.latitude.values
    era5_lons = era5_sample.longitude.values
    Lat, Lon = np.meshgrid(era5_lats, era5_lons, indexing='ij')
    
    # Create quick lookup dictionaries
    lat_idx = {lat:i for i,lat in enumerate(era5_lats)}
    lon_idx = {lon:i for i,lon in enumerate(era5_lons)}
    
    # STEP 1: Find nearest ERA5 point for each unique coordinate
    print("Finding nearest ERA5-Land grid point for each location...")
    unique_coords = tablex[['Latitude','Longitude']].drop_duplicates()
    nearest_map = {}  # Will store the nearest point for each coordinate
    
    for _, row in unique_coords.iterrows():
        # Calculate distances to all grid points
        distances = np.sqrt((era5_lats - row['Latitude'])**2 + 
                          (era5_lons[:, None] - row['Longitude'])**2)
        
        # Get index of nearest point
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        lat = era5_lats[min_idx[0]]
        lon = era5_lons[min_idx[1]]
        dist = distances[min_idx[0], min_idx[1]]
        
        nearest_map[(row['Latitude'], row['Longitude'])] = (lat, lon, dist)
    
    # STEP 2: Check validity at first timestep and remap if needed
    print("Checking for invalid points...")
    test_data = era5_sample.isel(valid_time=0)['t2m'].values
    valid_mask = ~np.isnan(test_data)
    remapping = {}
    
    for coords, (lat, lon, dist) in nearest_map.items():
        lat_i, lon_i = lat_idx[lat], lon_idx[lon]
        if not valid_mask[lat_i, lon_i]:
            # Find nearest valid pixel if this one is invalid
            valid_lat, valid_lon = find_nearest_valid_pixel(
                lat, lon, Lat, Lon, valid_mask
            )
            # Recalculate distance from original point to this valid point
            new_dist = np.sqrt((coords[0] - valid_lat)**2 + 
                               (coords[1] - valid_lon)**2)
            remapping[coords] = (valid_lat, valid_lon, new_dist)
    
    # STEP 3: Process all dates with nearest neighbor
    print("Processing all timesteps with nearest neighbor...")
    for date, group in tablex.groupby('Date'):
        era5_data = load_era5land(date.strftime('%Y-%m-%d'))
        print("day ", date.strftime('%Y-%m-%d'))
        
        # Convert ERA5 times to numpy datetime64
        era5_times = era5_data.valid_time.values.astype('datetime64[s]')
        
        # Convert observation times to numpy datetime64
        obs_times = group['Time_UTC'].values.astype('datetime64[s]')
        
        # Vectorized time index calculation
        time_indices = np.array([
            np.abs(era5_times - t).argmin() 
            for t in obs_times
        ], dtype=np.int32)

        # Initialize arrays for results
        t2m_values = np.empty(len(group))
        d2m_values = np.empty(len(group))
        ssrd_values = np.empty(len(group))
        tp_values = np.empty(len(group))
        u10_values = np.empty(len(group))
        v10_values = np.empty(len(group))
        sp_values = np.empty(len(group))
        
        for i, (_, row) in enumerate(group.iterrows()):
            # Get the nearest point (using remapped if available)
            if (row['Latitude'], row['Longitude']) in remapping:
                lat, lon, _ = remapping[(row['Latitude'], row['Longitude'])]
            else:
                lat, lon, _ = nearest_map[(row['Latitude'], row['Longitude'])]
            
            lat_i, lon_i = lat_idx[lat], lon_idx[lon]
            time_i = time_indices[i]
            
            # Get values from nearest point
            t2m_values[i] = era5_data['t2m'].values[time_i, lat_i, lon_i]
            d2m_values[i] = era5_data['d2m'].values[time_i, lat_i, lon_i]
            ssrd_values[i] = era5_data['ssrd'].values[time_i, lat_i, lon_i]
            tp_values[i] = era5_data['tp'].values[time_i, lat_i, lon_i]
            u10_values[i] = era5_data['u10'].values[time_i, lat_i, lon_i]
            v10_values[i] = era5_data['v10'].values[time_i, lat_i, lon_i]
            sp_values[i] = era5_data['sp'].values[time_i, lat_i, lon_i]
        
        # Assign values to dataframe
        tablex.loc[group.index, 't2m'] = t2m_values
        tablex.loc[group.index, 'd2m'] = d2m_values
        tablex.loc[group.index, 'ssrd'] = ssrd_values
        tablex.loc[group.index, 'tp'] = tp_values
        tablex.loc[group.index, 'u10'] = u10_values
        tablex.loc[group.index, 'v10'] = v10_values
        tablex.loc[group.index, 'sp'] = sp_values
    
    return tablex



def assign_era5_and_era5land(tablex):
    # 1. Load first day to get grid info for both datasets
    first_date = tablex['Date'].iloc[0].strftime('%Y-%m-%d')
    era5land_sample = load_era5land(first_date)
    era5_sample     = load_era5(first_date)

    # 2. Extract lat/lon grids
    el_lats = era5land_sample.latitude.values
    el_lons = era5land_sample.longitude.values
    lats2d_el, lons2d_el = np.meshgrid(el_lats, el_lons, indexing='ij')

    e5_lats = era5_sample.latitude.values
    e5_lons = era5_sample.longitude.values
    lats2d_e5, lons2d_e5 = np.meshgrid(e5_lats, e5_lons, indexing='ij')

    # 3. Build index lookups
    el_lat_idx = {lat:i for i, lat in enumerate(el_lats)}
    el_lon_idx = {lon:i for i, lon in enumerate(el_lons)}
    e5_lat_idx = {lat:i for i, lat in enumerate(e5_lats)}
    e5_lon_idx = {lon:i for i, lon in enumerate(e5_lons)}

    # 4. Precompute nearest‐neighbour maps for all unique coords
    print("Finding nearest grid points for ERA5-Land and ERA5...")
    unique = tablex[['Latitude','Longitude']].drop_duplicates().itertuples(index=False)
    nn_maps = {}  # (lat,lon) → dict with keys 'el' and 'e5', each being (lat,lon,dist)
    for lat0, lon0 in unique:
        # ERA5-Land
        dist_el = np.sqrt((lats2d_el - lat0)**2 + (lons2d_el - lon0)**2)
        i_el, j_el = np.unravel_index(np.argmin(dist_el), dist_el.shape)
        nn_el = (el_lats[i_el], el_lons[j_el], dist_el[i_el,j_el])
        # ERA5
        dist_e5 = np.sqrt((lats2d_e5 - lat0)**2 + (lons2d_e5 - lon0)**2)
        i_e5, j_e5 = np.unravel_index(np.argmin(dist_e5), dist_e5.shape)
        nn_e5 = (e5_lats[i_e5], e5_lons[j_e5], dist_e5[i_e5,j_e5])

        nn_maps[(lat0, lon0)] = {'el': nn_el, 'e5': nn_e5}

    # 5. (Optional) For ERA5-Land, recheck first timestep validity and remap invalids
    print("Checking ERA5-Land for invalid gridpoints...")
    valid_mask = ~np.isnan(era5land_sample['t2m'].isel(valid_time=0).values)
    for coords, m in nn_maps.items():
        lat_el, lon_el, _ = m['el']
        i_el, j_el = el_lat_idx[lat_el], el_lon_idx[lon_el]
        if not valid_mask[i_el, j_el]:
            # find_nearest_valid_pixel should return (lat, lon)
            vlat, vlon = find_nearest_valid_pixel(lat_el, lon_el, lats2d_el, lons2d_el, valid_mask)
            new_dist = np.sqrt((coords[0] - vlat)**2 + (coords[1] - vlon)**2)
            m['el'] = (vlat, vlon, new_dist)

    # 6. Loop over each date and assign both sets of variables
    print("Processing all timesteps...")
    for date, group in tablex.groupby('Date'):
        ds_el = load_era5land(date.strftime('%Y-%m-%d'))
        ds_e5 = load_era5(date.strftime('%Y-%m-%d'))

        times_el = ds_el.valid_time.values.astype('datetime64[s]')
        times_e5 = ds_e5.time.values.astype('datetime64[s]')  # or valid_time if same

        obs_times = group['Time_UTC'].values.astype('datetime64[s]')
        idx_el = np.array([np.abs(times_el - t).argmin() for t in obs_times])
        idx_e5 = np.array([np.abs(times_e5 - t).argmin() for t in obs_times])

        # Prepare output arrays
        out = {var: np.empty(len(group)) 
               for var in ['t2m_el','d2m','ssrd','tp','u10','v10','sp',
                           't2m_e5','cape','some_other_e5_var']}
        # (replace 'cape','some_other_e5_var' with your actual ERA5 variable names)

        for i, (_, row) in enumerate(group.iterrows()):
            lat0, lon0 = row['Latitude'], row['Longitude']
            lat_el, lon_el, _ = nn_maps[(lat0,lon0)]['el']
            lat_e5, lon_e5, _ = nn_maps[(lat0,lon0)]['e5']
            i_el, j_el = el_lat_idx[lat_el], el_lon_idx[lon_el]
            i_e5, j_e5 = e5_lat_idx[lat_e5], e5_lon_idx[lon_e5]
            ti_el, ti_e5 = idx_el[i], idx_e5[i]

            # ERA5-Land assignments
            out['t2m'][i] = ds_el['t2m'].values[ti_el, i_el, j_el]
            out['d2m'][i]  = ds_el['d2m'].values[ti_el, i_el, j_el]
            out['ssrd'][i] = ds_el['ssrd'].values[ti_el, i_el, j_el]
            out['tp'][i]   = ds_el['tp'].values[ti_el, i_el, j_el]
            out['u10'][i]  = ds_el['u10'].values[ti_el, i_el, j_el]
            out['v10'][i]  = ds_el['v10'].values[ti_el, i_el, j_el]
            out['sp'][i]   = ds_el['sp'].values[ti_el, i_el, j_el]

            # ERA5 assignments
            out['swvl1'][i]          = ds_e5['swvl1'].values[ti_e5, i_e5, j_e5]
            out['swvl3'][i]            = ds_e5['swvl3'].values[ti_e5, i_e5, j_e5]
            #out['some_other_e5_var'][i] = ds_e5['some_other_e5_var'].values[ti_e5, i_e5, j_e5]

        # Write back into tablex
        for col, arr in out.items():
            tablex.loc[group.index, col] = arr

    return tablex


def assign_era5_and_era5land_xarray(tablex):
    """
    Vectorized pull of nearest-neighbor values for both ERA5-Land (with remapping
    of any water-NaNs to their nearest valid land pixel) and ERA5.
    """
    # Make a copy so we don’t clobber the caller’s DataFrame
    tablex = tablex.copy()

    # 1. Precompute nearest valid ERA5‑Land gridpoint for each unique (lat,lon)
    print("Building remapped nearest‐valid map for ERA5‑Land…")
    first_day = tablex["Date"].iloc[0].strftime("%Y-%m-%d")
    sample_el = load_era5land(first_day)

    lat_el = sample_el.latitude.values
    lon_el = sample_el.longitude.values
    LON_EL2D, LAT_EL2D = np.meshgrid(lon_el, lat_el)  # note order: (y, x)

    valid0 = ~np.isnan(sample_el["t2m"].isel(valid_time=0).values)
    unique = tablex[["Latitude","Longitude"]].drop_duplicates().itertuples(index=False)

    remap_el = {}  # (orig_lat,orig_lon) -> (valid_lat, valid_lon)
    for lat0, lon0 in unique:
        # find nearest in the raw grid
        dist = np.hypot(LAT_EL2D - lat0, LON_EL2D - lon0)
        i0, j0 = np.unravel_index(dist.argmin(), dist.shape)

        if valid0[i0, j0]:
            remap_el[(lat0,lon0)] = (lat_el[i0], lon_el[j0])
        else:
            # water-pixel: find nearest valid
            # (you already have find_nearest_valid_pixel)
            vlat, vlon = find_nearest_valid_pixel(
                lat_el[i0], lon_el[j0],
                LAT_EL2D, LON_EL2D,
                valid0
            )
            remap_el[(lat0,lon0)] = (vlat, vlon)

    # 2. Now loop per day, vectorized .sel on remapped coords for ERA5‑Land
    el_vars = ["t2m","d2m","ssrd","tp","u10","v10","sp"]
    e5_vars = ["swvl1","swvl3"]  
    e5_pressure_vars = ["z", "u", "v"] 
    pressure_levels = [950.0, 850.0, 500.0]

    for date, group in tablex.groupby("Date"):
        print("Processing ", date)
        ds_el = load_era5land(date.strftime("%Y-%m-%d"))
        ds_e5 = load_era5(date.strftime("%Y-%m-%d"))
        ds_e5_pressure = load_era5pressure(date.strftime("%Y-%m-%d"))

        # build obs_times once
        obs_times = xr.DataArray(
            group["Time_UTC"].values.astype("datetime64[ns]"),
            dims="obs",
            name=ds_el.valid_time.name
        )

        # build remapped lat/lon for ERA5‑Land
        remapped = np.array([remap_el[(lat,lon)] for lat,lon in
                             zip(group["Latitude"], group["Longitude"])])
        obs_lats_el = xr.DataArray(remapped[:,0], dims="obs", name="latitude")
        obs_lons_el = xr.DataArray(remapped[:,1], dims="obs", name="longitude")

        # build true lat/lon for ERA5
        obs_lats_e5 = xr.DataArray(group["Latitude"].values,
                                   dims="obs", name="latitude")
        obs_lons_e5 = xr.DataArray(group["Longitude"].values,
                                   dims="obs", name="longitude")

        # vectorized nearest‐neighbor pull
        out_el = ds_el.sel(
            {ds_el.valid_time.name: obs_times,
             "latitude": obs_lats_el,
             "longitude": obs_lons_el},
            method="nearest"
        )
        out_e5 = ds_e5.sel(
            {ds_e5.valid_time.name: obs_times,
             "latitude": obs_lats_e5,
             "longitude": obs_lons_e5},
            method="nearest"
        )
        out_e5_pressure = ds_e5_pressure.sel(
            {ds_e5_pressure.valid_time.name: obs_times,
             "latitude": obs_lats_e5,
             "longitude": obs_lons_e5},
            method="nearest"
        )

        # assign back into DataFrame
        for var in el_vars:
            tablex.loc[group.index, var] = out_el[var].values

        for var in e5_vars:
            tablex.loc[group.index, var] = out_e5[var].values

        for var in e5_pressure_vars:
            for pressure_level in pressure_levels:
                # Create column name like "z_950", "u_850", "v_500", etc.
                col_name = f"{var}_{int(pressure_level)}"
                
                # Select the specific pressure level and assign values
                var_data = out_e5_pressure[var].sel(pressure_level=pressure_level, method="nearest")
                tablex.loc[group.index, col_name] = var_data.values
    return tablex



def assign_era5_single_loc_xarray(tablex, lat0, lon0):
    """
    Pull ERA5‑Land & ERA5 (including pressure levels) at one fixed point (lat0, lon0)
    for every row in tablex.  Use lat0/lon0 from your nominatim() call.
    """
    tablex = tablex.copy()

    # --- 1. find the nearest valid ERA5‑Land pixel for (lat0, lon0) once ---
    first_day = tablex["Date"].iloc[0].strftime("%Y-%m-%d")
    ds0_el = load_era5land(first_day)

    lat_el = ds0_el.latitude.values
    lon_el = ds0_el.longitude.values
    LON2D, LAT2D = np.meshgrid(lon_el, lat_el)

    valid0 = ~np.isnan(ds0_el["t2m"].isel(valid_time=0).values)

    # raw nearest
    dist = np.hypot(LAT2D - lat0, LON2D - lon0)
    i0, j0 = np.unravel_index(dist.argmin(), dist.shape)

    if valid0[i0, j0]:
        pick_lat_el = lat_el[i0]
        pick_lon_el = lon_el[j0]
    else:
        # fallback to nearest valid land pixel
        pick_lat_el, pick_lon_el = find_nearest_valid_pixel(
            lat_el[i0], lon_el[j0],
            LAT2D, LON2D,
            valid0
        )

    # Our “single” ERA5 point is just (lat0, lon0) for the raw ERA5 dataset
    pick_lat_e5 = lat0
    pick_lon_e5 = lon0

    # variable lists
    el_vars = ["t2m","d2m","ssrd","tp","u10","v10","sp"]
    e5_vars = ["swvl1","swvl3"]
    e5_pressure_vars = ["z", "u", "v"]
    pressure_levels = [950.0, 850.0, 500.0]

    # now loop per day
    for date, group in tablex.groupby("Date"):
        print("Processing", date.strftime("%Y-%m-%d"))

        ds_el = load_era5land(date.strftime("%Y-%m-%d"))
        ds_e5 = load_era5(date.strftime("%Y-%m-%d"))
        ds_e5_p = load_era5pressure(date.strftime("%Y-%m-%d"))

        # build obs_times
        obs_times = xr.DataArray(
            group["Time_UTC"].values.astype("datetime64[ns]"),
            dims="obs",
            name=ds_el.valid_time.name
        )

        # build constant lat/lon arrays of shape (n_obs,)
        n_obs = len(group)
        obs_lats_el = xr.DataArray(
            np.full(n_obs, pick_lat_el),
            dims="obs", name="latitude"
        )
        obs_lons_el = xr.DataArray(
            np.full(n_obs, pick_lon_el),
            dims="obs", name="longitude"
        )
        obs_lats_e5 = xr.DataArray(
            np.full(n_obs, pick_lat_e5),
            dims="obs", name="latitude"
        )
        obs_lons_e5 = xr.DataArray(
            np.full(n_obs, pick_lon_e5),
            dims="obs", name="longitude"
        )

        # vectorized NN pull
        out_el = ds_el.sel(
            {ds_el.valid_time.name: obs_times,
             "latitude":      obs_lats_el,
             "longitude":     obs_lons_el},
            method="nearest"
        )
        out_e5 = ds_e5.sel(
            {ds_e5.valid_time.name: obs_times,
             "latitude":      obs_lats_e5,
             "longitude":     obs_lons_e5},
            method="nearest"
        )
        out_p = ds_e5_p.sel(
            {ds_e5_p.valid_time.name: obs_times,
             "latitude":      obs_lats_e5,
             "longitude":     obs_lons_e5},
            method="nearest"
        )

        # write back
        for var in el_vars:
            tablex.loc[group.index, var] = out_el[var].values
        for var in e5_vars:
            tablex.loc[group.index, var] = out_e5[var].values
        for var in e5_pressure_vars:
            for level in pressure_levels:
                col = f"{var}_{int(level)}"
                vals = out_p[var].sel(pressure_level=level, method="nearest").values
                tablex.loc[group.index, col] = vals

    return tablex


ACCUM_VARS = {"ssrd", "tp"}

def deaccumulate_day(vals, prev_h23):
    """
    Deaccumulate a (24,) array of one ERA5-Land accumulated variable.
    
    Per ECMWF convention:
      h00 = h00_raw - prev_day_h23_raw   (needs previous day)
      h01 = h01_raw                       (first step, left as-is)
      h02..h23 = h_n - h_(n-1)
    
    Parameters
    ----------
    vals     : np.ndarray, shape (24,)  raw accumulated values
    prev_h23 : float or None            h23 of the previous day;
                                        if None (first day), h00 is left as-is
    Returns
    -------
    np.ndarray, shape (24,)
    """
    out = vals.copy()
    out[2:] = np.diff(vals)[1:]          # h02..h23: simple diff
    # h01 stays as-is (already step1)
    if prev_h23 is not None:
        out[0] = vals[0] - prev_h23      # h00: cross-day diff
    # if prev_h23 is None we leave h00 as the raw value (best we can do)
    return out


def assign_era5_single_loc_xarray_with_altitude_correction(tablex, lat0, lon0, lapse_rate=0.0065):
    """
    Pull ERA5-Land & ERA5 (including pressure levels) at one fixed point
    (lat0, lon0) for every row in tablex, with:
      - altitude correction for temperature
      - deaccumulation of ssrd and tp (ECMWF hourly convention)
 
    Parameters
    ----------
    tablex    : DataFrame with a 'Date' and 'Time_UTC' column
    lat0, lon0: city-centre coordinates (from Nominatim)
    lapse_rate: K/m  (default 0.0065)
    """
    tablex = tablex.copy()
 
    # ── 1. Find nearest valid ERA5-Land pixel for (lat0, lon0) ───────────────
    first_day = tablex["Date"].iloc[0].strftime("%Y-%m-%d")
    ds0_el = load_era5land(first_day)
 
    lat_el = ds0_el.latitude.values
    lon_el = ds0_el.longitude.values
    LON2D, LAT2D = np.meshgrid(lon_el, lat_el)
 
    valid0 = ~np.isnan(ds0_el["t2m"].isel(valid_time=0).values)
 
    dist = np.hypot(LAT2D - lat0, LON2D - lon0)
    i0, j0 = np.unravel_index(dist.argmin(), dist.shape)
 
    if valid0[i0, j0]:
        pick_lat_el = lat_el[i0]
        pick_lon_el = lon_el[j0]
    else:
        pick_lat_el, pick_lon_el = find_nearest_valid_pixel(
            lat_el[i0], lon_el[j0], LAT2D, LON2D, valid0
        )
 
    pick_lat_e5 = lat0
    pick_lon_e5 = lon0
 
    # ── 2. Variable lists ─────────────────────────────────────────────────────
    el_vars          = ["d2m", "ssrd", "tp", "u10", "v10", "sp"]   # t2m from ERA5
    e5_vars          = ["swvl1", "swvl3", "t2m", "z"]
    e5_pressure_vars = ["z", "u", "v"]
    pressure_levels  = [950.0, 850.0, 500.0]
 
    # ── 3. Deaccumulation cache (one scalar per accum var, carried day-to-day)
    prev_h23 = {v: None for v in ACCUM_VARS}
    prev_date = None
 
    # ── 4. Main loop ──────────────────────────────────────────────────────────
    for date, group in tablex.groupby("Date"):
        print("Processing", date.strftime("%Y-%m-%d"))
        if prev_date is not None and (date - prev_date).days > 1:
            prev_h23 = {v: None for v in ACCUM_VARS}
 
        ds_el  = load_era5land(date.strftime("%Y-%m-%d"))
        ds_e5  = load_era5(date.strftime("%Y-%m-%d"))
        ds_e5_p = load_era5pressure(date.strftime("%Y-%m-%d"))
 
        # ── Deaccumulate ssrd and tp ──────────────────────────────────────────
        # Pull the full 24-h timeseries at the single ERA5-Land point
        raw_accum = {
            v: ds_el[v].sel(
                latitude=pick_lat_el, longitude=pick_lon_el, method="nearest"
            ).values.ravel()                                    # (24,)
            for v in ACCUM_VARS
        }
        deacc = {}
        for v in ACCUM_VARS:
            deacc[v]    = deaccumulate_day(raw_accum[v], prev_h23[v])
            prev_h23[v] = raw_accum[v][-1]     
        prev_date = date                
        # ─────────────────────────────────────────────────────────────────────
 
        # Build xarray selection objects
        obs_times = xr.DataArray(
            group["Time_UTC"].values.astype("datetime64[ns]"),
            dims="obs",
            name=ds_el.valid_time.name
        )
        n_obs = len(group)
 
        obs_lats_el = xr.DataArray(np.full(n_obs, pick_lat_el),
                                   dims="obs", name="latitude")
        obs_lons_el = xr.DataArray(np.full(n_obs, pick_lon_el),
                                   dims="obs", name="longitude")
        obs_lats_e5 = xr.DataArray(np.full(n_obs, pick_lat_e5),
                                   dims="obs", name="latitude")
        obs_lons_e5 = xr.DataArray(np.full(n_obs, pick_lon_e5),
                                   dims="obs", name="longitude")
 
        # Vectorised nearest-neighbour pulls
        out_el = ds_el.sel(
            {ds_el.valid_time.name: obs_times,
             "latitude":  obs_lats_el,
             "longitude": obs_lons_el},
            method="nearest"
        )
        out_e5 = ds_e5.sel(
            {ds_e5.valid_time.name: obs_times,
             "latitude":  obs_lats_e5,
             "longitude": obs_lons_e5},
            method="nearest"
        )
        out_p = ds_e5_p.sel(
            {ds_e5_p.valid_time.name: obs_times,
             "latitude":  obs_lats_e5,
             "longitude": obs_lons_e5},
            method="nearest"
        )
 
        # ── Write ERA5-Land variables ─────────────────────────────────────────
        hours = group["Time_UTC"].dt.hour.values               # (n_obs,)
        for var in el_vars:
            if var in ACCUM_VARS:
                tablex.loc[group.index, var] = deacc[var][hours]
            else:
                tablex.loc[group.index, var] = out_el[var].values
 
        # ── Write ERA5 variables (t2m, z, swvl1, swvl3) ──────────────────────
        for var in e5_vars:
            tablex.loc[group.index, var] = out_e5[var].values
 
        # ── Write pressure-level variables ────────────────────────────────────
        for var in e5_pressure_vars:
            for level in pressure_levels:
                col  = f"{var}_{int(level)}"
                vals = out_p[var].sel(pressure_level=level, method="nearest").values
                tablex.loc[group.index, col] = vals
 
    return tablex


def assign_era5_values_interpolated(tablex):
    # Load first ERA5 file to establish grid
    first_date = tablex['Date'].iloc[0].strftime('%Y-%m-%d')
    era5_sample = load_era5land(first_date)
    
    # Create coordinate arrays
    era5_lats = era5_sample.latitude.values
    era5_lons = era5_sample.longitude.values
    Lat, Lon = np.meshgrid(era5_lats, era5_lons, indexing='ij')
    
    # Create quick lookup dictionaries
    lat_idx = {lat:i for i,lat in enumerate(era5_lats)}
    lon_idx = {lon:i for i,lon in enumerate(era5_lons)}
    
    # STEP 1: Find nearest 3 ERA5 points for each unique coordinate
    print("Finding nearest 3 ERA5 grid points for each location...")
    unique_coords = tablex[['Latitude','Longitude']].drop_duplicates()
    nearest_map = {}  # Will store list of 3 nearest points for each coordinate
    
    for _, row in unique_coords.iterrows():
        # Calculate distances to all grid points
        distances = np.sqrt((era5_lats - row['Latitude'])**2 + 
                          (era5_lons[:, None] - row['Longitude'])**2)
        
        # Get indices of 3 nearest points
        flat_indices = np.argpartition(distances.flatten(), 3)[:3]
        idx_2d = np.unravel_index(flat_indices, distances.shape)
        
        # Store the 3 nearest points and their distances
        nearest_points = []
        for i in range(3):
            lat = era5_lats[idx_2d[0][i]]
            lon = era5_lons[idx_2d[1][i]]
            dist = distances[idx_2d[0][i], idx_2d[1][i]]
            nearest_points.append((lat, lon, dist))
        
        nearest_map[(row['Latitude'], row['Longitude'])] = nearest_points
    
    # STEP 2: Check validity at first timestep and build remapping if needed
    print("Checking for invalid points...")
    test_data = era5_sample.isel(valid_time=0)['t2m'].values
    valid_mask = ~np.isnan(test_data)
    remapping = {}
    
    for coords, nearest_points in nearest_map.items():
        new_points = []
        for lat, lon, dist in nearest_points:
            lat_i, lon_i = lat_idx[lat], lon_idx[lon]
            if not valid_mask[lat_i, lon_i]:
                # Find nearest valid pixel if this one is invalid
                valid_lat, valid_lon = find_nearest_valid_pixel(
                    lat, lon, Lat, Lon, valid_mask
                )
                # Recalculate distance from original point to this valid point
                new_dist = np.sqrt((coords[0] - valid_lat)**2 + 
                                   (coords[1] - valid_lon)**2)
                new_points.append((valid_lat, valid_lon, new_dist))
            else:
                new_points.append((lat, lon, dist))
        
        # Only store if any points were remapped
        if new_points != nearest_points:
            remapping[coords] = new_points
    
    # STEP 3: Process all dates with interpolation
    print("Processing all timesteps with interpolation...")
    for date, group in tablex.groupby('Date'):
        era5_data = load_era5land(date.strftime('%Y-%m-%d'))
        print("day ", date.strftime('%Y-%m-%d'))
        
        # Convert ERA5 times to numpy datetime64
        era5_times = era5_data.valid_time.values.astype('datetime64[s]')
        
        # Convert observation times to numpy datetime64
        obs_times = group['Time_UTC'].values.astype('datetime64[s]')
        
        # Vectorized time index calculation
        time_indices = np.array([
            np.abs(era5_times - t).argmin() 
            for t in obs_times
        ], dtype=np.int32)

        # Initialize arrays for results
        t2m_values = np.empty(len(group))
        d2m_values = np.empty(len(group))
        ssrd_values = np.empty(len(group))
        tp_values = np.empty(len(group))
        u10_values = np.empty(len(group))
        v10_values = np.empty(len(group))
        sp_values = np.empty(len(group))
        
        for i, (_, row) in enumerate(group.iterrows()):
            # Get the 3 nearest points (using remapped if available)
            if (row['Latitude'], row['Longitude']) in remapping:
                points = remapping[(row['Latitude'], row['Longitude'])]
            else:
                points = nearest_map[(row['Latitude'], row['Longitude'])]
            
            # Extract values and weights
            values_t2m = []
            values_d2m = []
            values_ssrd = []
            values_tp = []
            values_u10 = []
            values_v10 = []
            values_sp = []
            weights = []
            total_weight = 0.0
            
            for lat, lon, dist in points:
                lat_i, lon_i = lat_idx[lat], lon_idx[lon]
                time_i = time_indices[i]
                
                # Get values
                values_t2m.append(era5_data['t2m'].values[time_i, lat_i, lon_i])
                values_d2m.append(era5_data['d2m'].values[time_i, lat_i, lon_i])
                values_ssrd.append(era5_data['ssrd'].values[time_i, lat_i, lon_i])
                values_tp.append(era5_data['tp'].values[time_i, lat_i, lon_i])
                values_u10.append(era5_data['u10'].values[time_i, lat_i, lon_i])
                values_v10.append(era5_data['v10'].values[time_i, lat_i, lon_i])
                values_sp.append(era5_data['sp'].values[time_i, lat_i, lon_i])
                
                # Calculate weight (inverse distance)
                if dist == 0:
                    weight = 1e6  # very large weight if exact match
                else:
                    weight = 1.0 / dist
                weights.append(weight)
                total_weight += weight
            
            # Calculate weighted average
            t2m_values[i] = np.sum(np.array(values_t2m) * np.array(weights)) / total_weight
            d2m_values[i] = np.sum(np.array(values_d2m) * np.array(weights)) / total_weight
            ssrd_values[i] = np.sum(np.array(values_ssrd) * np.array(weights)) / total_weight
            tp_values[i] = np.sum(np.array(values_tp) * np.array(weights)) / total_weight
            u10_values[i] = np.sum(np.array(values_u10) * np.array(weights)) / total_weight
            v10_values[i] = np.sum(np.array(values_v10) * np.array(weights)) / total_weight
            sp_values[i] = np.sum(np.array(values_sp) * np.array(weights)) / total_weight
        
        # Assign values to dataframe
        tablex.loc[group.index, 't2m'] = t2m_values
        tablex.loc[group.index, 'd2m'] = d2m_values
        tablex.loc[group.index, 'ssrd'] = ssrd_values
        tablex.loc[group.index, 'tp'] = tp_values
        tablex.loc[group.index, 'u10'] = u10_values
        tablex.loc[group.index, 'v10'] = v10_values
        tablex.loc[group.index, 'sp'] = sp_values
    
    return tablex



def aggregate_feature_importance(importance, features, lcz_count=17):
    """Dynamically aggregate feature importance based on feature names."""
    aggregated = {}
    lcz_start = len(features) - lcz_count
    lcz_importance = 0
    
    for i, feature in enumerate(features):
        # Handle circular features (sin/cos pairs)
        if 'cos_' in feature or 'sin_' in feature:
            base_name = feature.split('_')[1]  # Extract 'lat', 'lon', etc.
            if base_name not in aggregated:
                aggregated[base_name] = 0
            aggregated[base_name] += importance[i]
        # Handle LCZ features
        elif feature.startswith('LCZ_'):
            lcz_importance += importance[i]
        # Handle regular features
        else:
            aggregated[feature] = importance[i]
    
    # Add aggregated LCZ importance
    if lcz_importance > 0:
        aggregated["Local Climate Zone"] = lcz_importance
    
    return aggregated