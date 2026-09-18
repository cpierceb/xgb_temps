#need to be passed tablex, tabley
from xgb_0_functions import *
import pandas as pd
from rasterio.warp import transform

def get_raster_layers(city, country):
    """Single source of truth: (column_name, path, needs_crop) for every sampled raster."""
    return [
        ('BH_100',            f"../tiffs/bh/{city}_100.tif",           False),
        ('BH_100_250rad',     f"../tiffs/bh/{city}_100_250rad.tif",    False),
        ('BH_100_500rad',     f"../tiffs/bh/{city}_100_500rad.tif",    False),
        ('DTM_100',           f"../tiffs/dtm/{city}_100.tif",          False),
        ('LCZ_100',           f"../tiffs/lcz/{city}_100.tif",          False),
        ('LCZ_water_500rad',  f"../tiffs/lcz/{city}_water_500rad.tif", False),
        ('TCD_100',           f"../tiffs/tcd/{city}_100.tif",          False),
        ('TCD_100_250rad',    f"../tiffs/tcd/{city}_100_250rad.tif",   False),
        ('TCD_100_500rad',    f"../tiffs/tcd/{city}_100_500rad.tif",   False),
        ('IMP_100',           f"../tiffs/imp/{city}_100.tif",          False),
        ('IMP_100_250rad',    f"../tiffs/imp/{city}_100_250rad.tif",   False),
        ('IMP_100_500rad',    f"../tiffs/imp/{city}_100_500rad.tif",   False),
    ]


def raster_intersection_3035(city, country):
    """
    Intersection extent (EPSG:3035) of every raster add_geo samples, so each
    grid cell has valid data for all predictors.

    Returns:
      bounds  : (left, bottom, right, top)
      binding : {'left'/'bottom'/'right'/'top': raster name that constrains that side}
      extents : {name: (left, bottom, right, top)} for printing
    """
    from rasterio.warp import transform_bounds
    left, bottom, right, top = -float('inf'), -float('inf'), float('inf'), float('inf')
    binding, extents = {}, {}
    for name, path, _ in get_raster_layers(city, country):
        if name.startswith('BH'):          # ← exclude building-height rasters from extent
            continue
        if path is None:
            continue
        b = raster_valid_bounds_3035(path)
        extents[name] = b
        l, bo, r, t = b
        if l  > left:   left,   binding['left']   = l,  name
        if bo > bottom: bottom, binding['bottom'] = bo, name
        if r  < right:  right,  binding['right']  = r,  name
        if t  < top:    top,    binding['top']    = t,  name
    return (left, bottom, right, top), binding, extents


def add_geo(tablex, city, country, buffer_m=1000):
    """
    Enrich `tablex` with geospatial attributes for a given city (and optional country).

    - Reprojects point coords to EPSG:3035
    - Crops and reads various rasters
    - Samples nearest-neighbor values efficiently
    """
    # 1. Reproject points to EPSG:3035
    xs, ys = transform('EPSG:4326', 'EPSG:3035',
                       tablex['Longitude'].values,
                       tablex['Latitude'].values)
    tablex = tablex.assign(X3035=xs, Y3035=ys)

    # 2. Build bounding box from city-specific BH tif
    bh_path = f"../tiffs/bh/{city}_10.tif"
    with rasterio.open(bh_path) as src:
        bbox = src.bounds
    region_bbox = (bbox.left, bbox.bottom, bbox.right, bbox.top)

    # 3. Define layers to process
    # Each entry: (column_name, path_template, needs_crop)
    layers = get_raster_layers(city, country)
    # layers = [
    #     # City-specific building heights
    #     # ('BH_10', 
    #     # f"../tiffs/bh/{city}_10.tif", 
    #     # False),
    #     ('BH_100', 
    #     f"../tiffs/bh/{city}_100.tif", 
    #     False),
    #     ('BH_100_250rad', 
    #     f"../tiffs/bh/{city}_100_250rad.tif", 
    #     False),
    #     ('BH_100_500rad', 
    #     f"../tiffs/bh/{city}_100_500rad.tif", 
    #     False),
    #     # European DEM to crop + city-specific DEMs
    #     # ('DEM_30', 
    #     # f"../tiffs/dem/{city}_30.tif" if city not in ("bern", "freiburg", "basel", "zurich") else "../tiffs/dem/eu_30.tif",
    #     # False if city not in ("bern", "freiburg", "basel", "zurich") else False),
    #     # ('DEM_90', 
    #     # f"../tiffs/dem/{city}_90.tif", 
    #     # False),
    #     ('DTM_100', 
    #     f"../tiffs/dtm/{city}_100.tif", 
    #     False),
    #     # LCZ (pan-Europe, always crop)
    #     # ('LCZ_100', 
    #     # "../tiffs/lcz_central_eu_meters.tif" if city in ("basel", "freiburg", "zurich", "bern") else f"../tiffs/lcz/{city}_100.tif",
    #     # True if city in ("basel", "freiburg", "zurich", "bern") else False),
    #     ('LCZ_100', 
    #     f"../tiffs/lcz/{city}_100.tif", 
    #     False),       
    #     ('LCZ_water_500rad', 
    #     f"../tiffs/lcz/{city}_water_500rad.tif",
    #     False),  
    #     # TCD: city, country, Europe
    #     # ('TCD_10', 
    #     # f"../tiffs/tcd/{city}_10.tif" if city not in ("bern", "freiburg", "basel", "zurich") else f"../tiffs/tcd/{country}_10.tif",
    #     # False if city not in ("bern", "freiburg", "basel", "zurich") else True),
    #     ('TCD_100', 
    #     f"../tiffs/tcd/{city}_100.tif", 
    #     False),
    #     ('TCD_100_250rad', 
    #     f"../tiffs/tcd/{city}_100_250rad.tif", 
    #     False),
    #     ('TCD_100_500rad', 
    #     f"../tiffs/tcd/{city}_100_500rad.tif", 
    #     False),
    #     # Imperviousness: city, country, Europe
    #     # ('IMP_10',
    #     # f"../tiffs/imp/{country}_10.tif" if country in ("germany", "switzerland") else (f"../tiffs/imp/{city}_10.tif"), 
    #     # False if city not in ("bern", "freiburg", "basel", "zurich") else True),
    #     ('IMP_100', 
    #     f"../tiffs/imp/{city}_100.tif", 
    #     False),
    #     ('IMP_100_250rad', 
    #     f"../tiffs/imp/{city}_100_250rad.tif", 
    #     False),
    #     ('IMP_100_500rad', 
    #     f"../tiffs/imp/{city}_100_500rad.tif", 
    #     False),
    # ]

    # 4. Crop and open each raster, store in dict
    rasters = {}
    for name, template, crop_flag in layers:
        if template is None:
            continue
        in_path = template
        out_path = in_path
        if crop_flag:
            out_path = f"../tiffs/{name}_cropped_{city}.tif"
            #crop_tiff(in_path, out_path, region_bbox, buffer_units=buffer_m)
            crop_tiff(in_path, out_path, region_bbox, buffer_units=buffer_m, bbox_crs="EPSG:3035")

        with rasterio.open(out_path) as src:
            rasters[name] = {
                'data': src.read(1),
                'transform': src.transform,
                'nodata': src.nodata,
            }

    # 5. Sample nearest-neighbor values at each point for each layer
    def sample_nearest(info, x, y):
        row, col = rasterio.transform.rowcol(info['transform'], x, y)
        data = info['data']
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            val = data[row, col]
            return None if val == info['nodata'] else val
        return None

    # 6. Populate columns
    for name, _, _ in layers:
        if name in rasters:
            tablex[name] = [
                sample_nearest(rasters[name], x, y)
                for x, y in zip(tablex['X3035'], tablex['Y3035'])
            ]
        if name.startswith('BH'):
            tablex[name] = tablex[name].fillna(0)
            tablex[name] = tablex[name].replace(65535, 0)

        if name.startswith('TCD'):
            tablex[name] = tablex[name].replace(255, 0)
        
        if name.startswith('IMP'):
            tablex[name] = tablex[name].replace(255, 0)


    # 7. Add 8 directional LCZ features (100m offsets)
    if 'LCZ_100' in rasters:
        offset = 100  # meters
        # Define 8 directions: E, NE, N, NW, W, SW, S, SE
        directions = [
            ('E', offset, 0),
            ('NE', offset, offset),
            ('N', 0, offset),
            ('NW', -offset, offset),
            ('W', -offset, 0),
            ('SW', -offset, -offset),
            ('S', 0, -offset),
            ('SE', offset, -offset)
        ]
        
        for dir_name, dx, dy in directions:
            col_name = f'LCZ_{dir_name}'
            tablex[col_name] = [
                sample_nearest(rasters['LCZ_100'], x + dx, y + dy)
                for x, y in zip(tablex['X3035'], tablex['Y3035'])
            ]

    return tablex

def apply_altitude_correction(tablex, dem_column='DTM_100'):
    """
    Apply altitude correction to temperature based on elevation difference
    between ERA5 geopotential height and local DEM elevation.
    
    Parameters:
    - tablex: DataFrame with ERA5 data (must have 'z' and 't2m' columns)
    - dem_column: column name for local elevation (default 'DEM_90')
    - lapse_rate: temperature lapse rate in K/m (default 0.0065 K/m)
    
    Returns:
    - tablex with additional columns: 'era5_elevation', 't2m_corrected'
    """
    tablex = tablex.copy()
    lapse_rate = 0.0065  # K/m
    
    # Convert ERA5 geopotential to elevation (divide by g = 9.80665 m/s²)
    g = 9.80665
    tablex['era5_elevation'] = tablex['z'] / g
    
    # Calculate elevation difference (positive = local point is higher than ERA5)
    tablex['elevation_diff'] = tablex[dem_column] - tablex['era5_elevation']
    
    # Apply lapse rate correction: T_corrected = T_era5 - lapse_rate * elevation_difference
    tablex['t2m_corrected'] = tablex['t2m'] - (lapse_rate * tablex['elevation_diff'])
    
    # Optional: replace the original t2m with corrected version
    # tablex['t2m'] = tablex['t2m_corrected']
    
    return tablex


def add_meteo(tablex, lat0, lon0, dem_column='DTM_100'):
    """
    Updated add_meteo function that includes altitude correction.
    """
    print("Processing ERA5 data...")
    tablex["Date"] = tablex["Time_UTC"].dt.date
    tablex["Hour"] = tablex["Time_UTC"].dt.hour
    tablex["Day"] = tablex["Time_UTC"].dt.day
    tablex["Month"] = tablex["Time_UTC"].dt.month
    tablex["Year"] = tablex["Time_UTC"].dt.year

    # Get ERA5 data with geopotential
    tablex = assign_era5_single_loc_xarray_with_altitude_correction(tablex, lat0, lon0)
    
    # Apply altitude correction
    tablex = apply_altitude_correction(tablex, dem_column)
    
    # Use corrected temperature for further processing
    tablex['t2m'] = tablex['t2m_corrected']
    
    # Convert temperature units (K to C)
    tablex[["t2m"]] = tablex[["t2m"]] - 273.15  # assuming ktoc = 273.15
    tablex[["d2m"]] = tablex[["d2m"]] - 273.15

    # Apply relative humidity calculation if needed
    if 'rhactivate' in globals() and rhactivate == 1:
        tablex["d2m"] = relative_humidity(tablex["t2m"], tablex["d2m"])
        print("relative humidity calculation all good")

    tablex = add_t2m_lags(tablex, lags=(1, 3, 6, 12))

    return tablex

def add_t2m_lags(tablex, lags=(1, 3, 6, 12)):
    """
    Lagged ERA5 t2m predictors. Single-gridpoint per city => t2m is a pure
    function of Time_UTC, so map (Time_UTC - lag) -> t2m. Gap-safe: missing
    timestamps (series start, data gaps) yield NaN, which XGBoost handles.
    """
    t2m_by_time = (tablex[['Time_UTC', 't2m']]
                   .drop_duplicates('Time_UTC')
                   .set_index('Time_UTC')['t2m'])
    for h in lags:
        shifted = tablex['Time_UTC'] - pd.Timedelta(hours=h)
        tablex[f't2m-{h}'] = shifted.map(t2m_by_time)
    return tablex


def raster_valid_bounds_3035(path):
    """Bounds (EPSG:3035) of the *valid-data* region of a raster —
    i.e. trimming nodata borders, not just the declared extent."""
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as src:
        data   = src.read(1)
        nodata = src.nodata
        valid  = np.ones(data.shape, dtype=bool)
        if nodata is not None:
            valid &= (data != nodata)
        if np.issubdtype(data.dtype, np.floating):
            valid &= ~np.isnan(data)
        if not valid.any():
            return transform_bounds(src.crs, "EPSG:3035", *src.bounds)
        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        left,  top    = src.transform * (cols[0],      rows[0])
        right, bottom = src.transform * (cols[-1] + 1, rows[-1] + 1)
        return transform_bounds(src.crs, "EPSG:3035", left, bottom, right, top)