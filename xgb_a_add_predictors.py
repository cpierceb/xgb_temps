#need to be passed tablex, tabley
from xgb_0_functions import *
import pandas as pd
from rasterio.warp import transform


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
    layers = [
        # City-specific building heights
        # ('BH_10', 
        # f"../tiffs/bh/{city}_10.tif", 
        # False),
        ('BH_100', 
        f"../tiffs/bh/{city}_100.tif", 
        False),
        ('BH_100_250rad', 
        f"../tiffs/bh/{city}_100_250rad.tif", 
        False),
        ('BH_100_500rad', 
        f"../tiffs/bh/{city}_100_500rad.tif", 
        False),
        # European DEM to crop + city-specific DEMs
        # ('DEM_30', 
        # f"../tiffs/dem/{city}_30.tif" if city not in ("bern", "freiburg", "basel", "zurich") else "../tiffs/dem/eu_30.tif",
        # False if city not in ("bern", "freiburg", "basel", "zurich") else False),
        # ('DEM_90', 
        # f"../tiffs/dem/{city}_90.tif", 
        # False),
        ('DTM_100', 
        f"../tiffs/dtm/{city}_100.tif", 
        False),
        # LCZ (pan-Europe, always crop)
        # ('LCZ_100', 
        # "../tiffs/lcz_central_eu_meters.tif" if city in ("basel", "freiburg", "zurich", "bern") else f"../tiffs/lcz/{city}_100.tif",
        # True if city in ("basel", "freiburg", "zurich", "bern") else False),
        ('LCZ_100', 
        f"../tiffs/lcz/{city}_100.tif", 
        False),       
        ('LCZ_water_500rad', 
        f"../tiffs/lcz/{city}_water_500rad.tif",
        False),  
        # TCD: city, country, Europe
        # ('TCD_10', 
        # f"../tiffs/tcd/{city}_10.tif" if city not in ("bern", "freiburg", "basel", "zurich") else f"../tiffs/tcd/{country}_10.tif",
        # False if city not in ("bern", "freiburg", "basel", "zurich") else True),
        ('TCD_100', 
        f"../tiffs/tcd/{city}_100.tif", 
        False),
        ('TCD_100_250rad', 
        f"../tiffs/tcd/{city}_100_250rad.tif", 
        False),
        ('TCD_100_500rad', 
        f"../tiffs/tcd/{city}_100_500rad.tif", 
        False),
        # Imperviousness: city, country, Europe
        # ('IMP_10',
        # f"../tiffs/imp/{country}_10.tif" if country in ("germany", "switzerland") else (f"../tiffs/imp/{city}_10.tif"), 
        # False if city not in ("bern", "freiburg", "basel", "zurich") else True),
        ('IMP_100', 
        f"../tiffs/imp/{city}_100.tif", 
        False),
        ('IMP_100_250rad', 
        f"../tiffs/imp/{city}_100_250rad.tif", 
        False),
        ('IMP_100_500rad', 
        f"../tiffs/imp/{city}_100_500rad.tif", 
        False),
    ]

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

def apply_altitude_correction(tablex, dem_column='DEM_90'):
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


def add_meteo(tablex, lat0, lon0, dem_column='DEM_90'):
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

    return tablex