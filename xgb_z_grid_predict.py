"""
Spatial grid prediction pipeline.

Flow per city:
  1. Build grid + add_geo once → cached to grid_static/{city}_static_geo.pkl
  2. Date range split into chunks sized to keep rows < MAX_ROWS (10M)
     Big cities (>400 km²): ≤3 days/chunk; small (<50 km²): up to 20 days/chunk
  3. For each chunk: load ERA5 days, build hourly DataFrame, predict, save chunk NC
  4. Combine chunk NCs → grid_netcdf/{city}_{year}_combined.nc
  5. Delete chunk NCs and temporal DataFrames as we go

Output per city/year (grid_netcdf/{city}_{year}_combined.nc):
  - t2m    (time, y, x)  — predicted residual + lapse-rate-corrected ERA5 t2m [°C]
  - lcz    (y, x)        — LCZ class, static
"""
import os, gc, time
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from geopy.geocoders import Nominatim
from collections import defaultdict

from xgb_a_add_predictors import add_geo, raster_intersection_3035
from xgb_0_functions import *
from xgb_0_prep_params import rhactivate
# from xgb_0_functions import (
#     load_era5land, load_era5, load_era5pressure,
#     find_nearest_valid_pixel,
# )


# ── Config ─────────────────────────────────────────────────────────────────────
GRID_RES_M   = 100
# MODELS_DIR   = 'geo36_models'
CV_MODE    = '6fold_st'          # 6fold_st 12fold_loco
MODELS_DIR = f'{CV_MODE}_models'
USE_FULL   = False               # ← True: all-cities model; False: a specific fold
FOLD_ID    = 'full' if USE_FULL else 'st6'
CITY     = 'lausanne'                      # city whose GRID we predict

STATIC_DIR   = '../data_processing/grid_static'
OUT_DIR      = '../data_processing/grid_netcdf'
LAPSE_RATE   = 0.0065   # K/m
G            = 9.80665
MAX_ROWS     = 20e6
# Cities that keep the raw (non-square, possibly asymmetric) intersection rectangle.
# Everyone else gets a symmetric square centered on the city.
COASTAL_CITIES = {'amsterdam', 'naples', 'oslo', 'barcelona', 'turku', 'biel', 'zurich', 'lausanne', 'thun'}  
# ── Date range control ─────────────────────────────────────────────────────────
# Default: JJA for each year in JJA_YEARS.
# For any custom period set CUSTOM_DATE_RANGE = ('YYYY-MM-DD', 'YYYY-MM-DD').
CUSTOM_DATE_RANGE = ('2025-06-01', '2025-08-31')       #None            # e.g. ('2020-01-01', '2020-12-31')
JJA_YEARS         = range(2015, 2026)    # 2015 … 2024 inclusive

ERA5LAND_GEOPOT = '../raw_data/era5land/geopotential.nc'   # static, (lat, lon) only

def build_dates():
    if CUSTOM_DATE_RANGE is not None:
        return pd.date_range(CUSTOM_DATE_RANGE[0], CUSTOM_DATE_RANGE[1], freq='D')
    return pd.DatetimeIndex(
        pd.concat([
            pd.Series(pd.date_range(f'{y}-06-01', f'{y}-08-31', freq='D'))
            for y in JJA_YEARS
        ]).values
    )

CITY_URBAN_AREAS = {
    'amsterdam': 308, 'basel':      68, 'berlin':    681, 'bern':   54,
    'biel':       18, 'birmingham': 685, 'freiburg':   50, 'ghent':  94,
    'novisad':    59, 'rennes':      57, 'turku':      23, 'zurich': 187,
    'lyon':      256, 'naples':     618, 'oslo':       244, 'barcelona': 438,
    'lausanne':   75, 'winterthur': 30, 'thun': 26 
}

CITY_COUNTRIES = {
    'amsterdam': 'netherlands', 'basel':     'switzerland',
    'berlin':    'germany',     'bern':       'switzerland',
    'biel':      'switzerland', 'birmingham': 'united kingdom',
    'freiburg':  'germany',     'ghent':      'belgium',
    'novisad':   'serbia',      'rennes':     'france',
    'turku':     'finland',     'zurich':     'switzerland',
    'lyon':      'france',      'naples':     'italy',
    'oslo':      'norway',      'barcelona':  'spain',
    'lausanne':  'switzerland', 'winterthur': 'switzerland',
    'thun':     'switzerland'
}
# ──────────────────────────────────────────────────────────────────────────────
def main():
    model_path = os.path.join(MODELS_DIR, f'xgb_{CV_MODE}_{FOLD_ID}.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"fold model not found: {model_path}")

    cities = [CITY]                         # grid city no longer tied to model existence
    all_dates = build_dates()


    print(f"Cities : {cities}")
    print(f"Dates  : {all_dates[0].date()} → {all_dates[-1].date()}  ({len(all_dates)} days)\n")

    # ERA5-Land sample for pixel lookup
    sample_el = load_era5land(all_dates[0].strftime('%Y-%m-%d'))

    for city in cities:
        print(f"\n{'='*60}\n{city.upper()} — urban area {CITY_URBAN_AREAS[city]} km²")

        # Static grid (load or build once)
        static_grid = build_static_grid(city)
        ny = int(static_grid['_ys_idx'].max()) + 1
        nx = int(static_grid['_xs_idx'].max()) + 1
        n_pts = ny * nx
        print(f"  grid: {ny}×{nx} = {n_pts:,} pts")

        xs = np.load(os.path.join(STATIC_DIR, f"{city}_static_geo_xs.npy"))
        ys = np.load(os.path.join(STATIC_DIR, f"{city}_static_geo_ys.npy"))
        lat0 = static_grid['Latitude'].mean()
        lon0 = static_grid['Longitude'].mean()

        lat_el, lon_el = era5land_pixel(lat0, lon0, sample_el)

        # ── one-off: altitude of the two forcing gridpoints ───────────────
        ds_e5_0 = load_era5(all_dates[0].strftime('%Y-%m-%d'))
        z_e5    = float(ds_e5_0['z'].sel(latitude=lat0, longitude=lon0,
                                         method='nearest').values.ravel()[0])
        ds_e5_0.close()

        ds_gp = xr.open_dataset(ERA5LAND_GEOPOT)
        z_el  = float(ds_gp['z'].sel(latitude=lat_el, longitude=lon_el,
                                     method='nearest').values.ravel()[0])
        ds_gp.close()

        print(f"  ERA5      gridpoint ({lat0:.3f}, {lon0:.3f}): z={z_e5:.0f} m²/s² → {z_e5/G:.1f} m")
        print(f"  ERA5-Land gridpoint ({lat_el:.3f}, {lon_el:.3f}): z={z_el:.0f} m²/s² → {z_el/G:.1f} m")
        print(f"  grid DTM mean: {static_grid['DTM_100'].mean():.1f} m")
        # ─────────────────────────────────────────────────────────────────

        model = xgb.Booster(model_file=model_path)

        chunks          = chunk_days(n_pts, all_dates)
        days_per_chunk  = len(chunks[0])
        rows_per_chunk  = n_pts * 24 * days_per_chunk
        print(f"  chunk size: {days_per_chunk} day(s)  (~{rows_per_chunk/1e6:.1f}M rows/chunk)")

        chunk_paths = []
        total_predict_s = 0.0
        lcz_saved   = False
        chunk_year = None

        prev_h23_accum = {v: None for v in ('ssrd', 'tp')}
        prev_date_accum = None

        for ci, chunk in enumerate(chunks):
            tag       = f"{chunk[0].strftime('%Y%m%d')}_{chunk[-1].strftime('%Y%m%d')}"
            out_chunk = os.path.join(OUT_DIR, f"{city}_{FOLD_ID}_{tag}_chunk.nc")

            if os.path.exists(out_chunk):
                print(f"  chunk {ci+1}/{len(chunks)} exists, skipping")
                chunk_paths.append(out_chunk)
                lcz_saved = True
                continue

            # Load ERA5 for each day in chunk
            era5_days = []
            for date in chunk:
                date_str = date.strftime('%Y-%m-%d')
                # reset cache on non-consecutive dates
                if prev_date_accum is not None and (date - prev_date_accum).days > 1:
                    prev_h23_accum = {v: None for v in ('ssrd', 'tp')}

                ds_el  = load_era5land(date_str)
                ds_e5  = load_era5(date_str)
                ds_e5_p = load_era5pressure(date_str)
                era5, times = extract_day(ds_el, ds_e5, ds_e5_p,
                                        lat_el, lon_el, lat0, lon0,
                                        prev_h23_accum)
                # update cache
                for v in ('ssrd', 'tp'):
                    prev_h23_accum[v] = era5[f'_{v}_raw_h23']  # see extract_day below
                prev_date_accum = date

                era5_days.append((era5, times))
                ds_el.close(); ds_e5.close(); ds_e5_p.close()
                # except FileNotFoundError as e:
                #     print(f"  ✗ missing ERA5: {e}"); ok = False; break

            # if not ok:
            #     continue

            # Build DataFrame, predict, save
            df = build_chunk_df(static_grid, era5_days)
            this_year = chunk[0].year
            is_new_year = (this_year != chunk_year)
            result = predict_chunk(model, df, ny, nx, lcz_grid=None if is_new_year else True)
            total_predict_s += result['predict_s']
            if is_new_year:
                chunk_year = this_year
            
            print(f"  chunk {ci+1}/{len(chunks)}: {tag} ✓  predict {result['predict_s']:.2f}s")
            
            del df; gc.collect()

            save_chunk_nc(result, xs[:nx], ys[:ny], out_chunk)
            del result; gc.collect()
            chunk_paths.append(out_chunk)

        if chunk_paths:
            combine_chunks(city, chunk_paths)

        n_done = sum(1 for _ in chunks)  # or just len(chunks)
        print(f"  total predict time: {total_predict_s:.1f}s "
              f"({total_predict_s/max(1, len(chunks)):.2f}s/chunk avg)")

        del static_grid, model; gc.collect()


os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

tr_fwd = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)
tr_inv = Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
geolocator = Nominatim(user_agent='grid_predict', timeout=30)
# ── Grid + geo ─────────────────────────────────────────────────────────────────

def build_static_grid(city):
    """Build flat DataFrame with Lat/Lon + all geo features. Cache to pkl."""
    static_path = os.path.join(STATIC_DIR, f"{city}_static_geo.pkl")
    if os.path.exists(static_path):
        print(f"  loading cached static grid: {static_path}")
        return pd.read_pickle(static_path)

    loc      = geolocator.geocode(f"{city}, {CITY_COUNTRIES[city]}")
    lat0, lon0 = loc.latitude, loc.longitude


    cx, cy = tr_fwd.transform(lon0, lat0)
    res = GRID_RES_M

    # Clamp grid to intersection of all predictor rasters (kills corner NaNs)
    (left, bottom, right, top), binding, extents = raster_intersection_3035(city, CITY_COUNTRIES[city])

    print("  raster extents (EPSG:3035):")
    for name, (l, bo, r, t) in extents.items():
        print(f"    {name:18s} x[{l:.0f}, {r:.0f}]  y[{bo:.0f}, {t:.0f}]")
    print(f"  intersection       x[{left:.0f}, {right:.0f}]  y[{bottom:.0f}, {top:.0f}]")
    print(f"  binding raster     left={binding['left']}  right={binding['right']}  "
        f"bottom={binding['bottom']}  top={binding['top']}")

    # half-cell inset so every sampled cell centre stays strictly inside every raster
    left += res/2; right -= res/2; bottom += res/2; top -= res/2

    kx_lo = int(np.floor((cx - left)   / res))
    kx_hi = int(np.floor((right - cx)  / res))
    ky_lo = int(np.floor((cy - bottom) / res))
    ky_hi = int(np.floor((top - cy)    / res))

    k_half = int(np.floor(((np.sqrt(CITY_URBAN_AREAS[city]) / 2 + 5) * 1000) / res))
    kx_lo, kx_hi, ky_lo, ky_hi = (min(kx_lo, k_half), min(kx_hi, k_half),
                              min(ky_lo, k_half), min(ky_hi, k_half))

    if min(kx_lo, kx_hi, ky_lo, ky_hi) < 0:
        raise ValueError(f"{city}: city centre lies outside raster intersection; check tiffs")

    if city in COASTAL_CITIES:
        x_neg, x_pos, y_neg, y_pos = kx_lo, kx_hi, ky_lo, ky_hi
        print(f"  {city}: coastal → non-square intersection rectangle")
    else:
        k = min(kx_lo, kx_hi, ky_lo, ky_hi)
        x_neg = x_pos = y_neg = y_pos = k
        print(f"  {city}: square → half-width {k*res/1000:.2f} km centred on ({cx:.0f}, {cy:.0f})")


    half     = (np.sqrt(CITY_URBAN_AREAS[city]) / 2 + 5) * 1000

    xs_half = np.arange(cx - half, cx + half + GRID_RES_M, GRID_RES_M)
    ys_half = np.arange(cy - half, cy + half + GRID_RES_M, GRID_RES_M)

    xs_other = cx + np.arange(-x_neg, x_pos + 1) * res
    ys_other = cy + np.arange(-y_neg, y_pos + 1) * res

    # xs = xs_half
    # ys = ys_half

    xs = xs_other
    ys = ys_other

    # if xs_half[0] < xs[0]:
    #     xs = xs_half.copy()

    # if ys_half[0] < ys[0]:
    #     ys = ys_half.copy()

    # print(f"  half-size logic grid is: {len(xs_half)}×{len(ys_half)} = {len(xs_half)*len(ys_half):,} pts")
    print(f"  final grid: {len(xs)}×{len(ys)} = {len(xs)*len(ys):,} pts")
    XX, YY = np.meshgrid(xs, ys)
    lons, lats = tr_inv.transform(XX.ravel(), YY.ravel())

    df = pd.DataFrame({'Latitude': lats, 'Longitude': lons})
    df = add_geo(df, city, CITY_COUNTRIES[city])
    df['_xs_idx'] = np.tile(np.arange(len(xs)), len(ys))
    df['_ys_idx'] = np.repeat(np.arange(len(ys)), len(xs))
    np.save(static_path.replace('.pkl', '_xs.npy'), xs)
    np.save(static_path.replace('.pkl', '_ys.npy'), ys)
    df.to_pickle(static_path)
    print(f"  saved static grid → {static_path}")
    return df


def get_grid_shape(city):
    half = (np.sqrt(CITY_URBAN_AREAS[city]) / 2 + 5) * 1000
    n    = int(2 * half / GRID_RES_M) + 1
    return n, n   # approx; exact shape recovered from static grid
def chunk_days(n_pts, dates):
    """
    Split dates into chunks that never cross a year boundary.
    Chunk size derived from actual grid n_pts so rows stay under MAX_ROWS.
    """
    rows_per_day   = n_pts * 24
    days_per_chunk = int(max(1, MAX_ROWS // rows_per_day))
    chunks = []
    for year, group in dates.to_series().groupby(dates.year):
        year_dates = pd.DatetimeIndex(group.values)
        chunks += [year_dates[i:i + days_per_chunk]
                   for i in range(0, len(year_dates), days_per_chunk)]
    return chunks
# ── ERA5 extraction ────────────────────────────────────────────────────────────

def era5land_pixel(lat0, lon0, ds0):
    lat_el = ds0.latitude.values
    lon_el = ds0.longitude.values
    LON2D, LAT2D = np.meshgrid(lon_el, lat_el)
    valid0 = ~np.isnan(ds0['t2m'].isel(valid_time=0).values)
    dist   = np.hypot(LAT2D - lat0, LON2D - lon0)
    i0, j0 = np.unravel_index(dist.argmin(), dist.shape)
    if valid0[i0, j0]:
        return lat_el[i0], lon_el[j0]
    return find_nearest_valid_pixel(lat_el[i0], lon_el[j0], LAT2D, LON2D, valid0)



def extract_day(ds_el, ds_e5, ds_e5_p, lat_el, lon_el, lat0, lon0, prev_h23_accum):
    kw_el = dict(latitude=lat_el, longitude=lon_el, method='nearest')
    kw_e5 = dict(latitude=lat0,   longitude=lon0,   method='nearest')
    pt_el = ds_el.sel(**kw_el)
    pt_e5 = ds_e5.sel(**kw_e5)
    pt_p  = ds_e5_p.sel(**kw_e5)

    era5 = {}

    # Non-accumulated ERA5-Land vars
    for v in ['d2m', 'u10', 'v10', 'sp']:
        era5[v] = pt_el[v].values.ravel()

    # Deaccumulate ssrd and tp
    for v in ['ssrd', 'tp']:
        raw = pt_el[v].values.ravel()           # (24,)
        era5[f'_{v}_raw_h23'] = raw[-1]         # stash for next day's cache
        deacc = raw.copy().astype(np.float64)
        deacc[2:] = np.diff(raw)[1:]            # h02..h23
        # h01 stays as-is
        if prev_h23_accum[v] is not None:
            deacc[0] = raw[0] - prev_h23_accum[v]
        else:
            deacc[0] = 0.0                      # no prev day: h00 → 0 (nighttime)
        era5[v] = deacc

    for v in ['t2m', 'z', 'swvl1', 'swvl3']:
        era5[v] = pt_e5[v].values.ravel()
    for v in ['u', 'v', 'z']:
        for lev in [950, 850, 500]:
            era5[f'{v}_{lev}'] = pt_p[v].sel(
                pressure_level=float(lev), method='nearest').values.ravel()

    return era5, ds_el.valid_time.values
# ── Build prediction DataFrame ─────────────────────────────────────────────────

def build_chunk_df(static_grid, era5_days):
    """
    era5_days: list of (era5_dict, times_array) for each day in chunk.
    Returns DataFrame with n_pts * n_hours rows.

    Feature build matches the station-table pipeline (add_meteo) exactly:
      1. lapse-correct t2m, convert to °C
      2. d2m -> °C, then -> RH iff rhactivate == 1 (same relative_humidity fn)
      3. t2m lags on the corrected-°C t2m — PER CELL (the only departure from
         add_t2m_lags, whose drop_duplicates('Time_UTC') assumes one t2m per
         timestamp, true for a single-gridpoint city but false on a grid).
    """
    n_pts  = len(static_grid)
    frames = []

    for era5, times in era5_days:
        for ti, t in enumerate(times):
            df = static_grid.copy()

            for v in ['d2m', 'ssrd', 'tp', 'u10', 'v10', 'sp',
                      'swvl1', 'swvl3',
                      'u_950', 'u_850', 'u_500',
                      'v_950', 'v_850', 'v_500',
                      'z_950', 'z_850', 'z_500']:
                val = era5.get(v, np.nan)
                df[v] = val if np.ndim(val) == 0 else val[ti]

            # 1. lapse correction + K->C, exactly as apply_altitude_correction + add_meteo
            era5_elev            = era5['z'][ti] / G
            df['era5_elevation'] = era5_elev
            df['elevation_diff'] = df['DTM_100'] - era5_elev
            df['t2m']            = (era5['t2m'][ti]
                                    - LAPSE_RATE * (df['DTM_100'] - era5_elev)) - 273.15

            # 2. d2m -> °C, then -> RH under the SAME flag/fn the tables used
            df['d2m'] = df['d2m'] - 273.15
            if rhactivate == 1:
                df['d2m'] = relative_humidity(df['t2m'], df['d2m'])

            df['Time_UTC'] = pd.Timestamp(t)
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # 3. PER-CELL t2m lags — grid analogue of add_t2m_lags.
    #    Each per-timestep frame is static_grid's row order unchanged, so df_all
    #    is time-major / cell-aligned: row (k*n_pts + c) is cell c at timestep k.
    times_sorted = df_all['Time_UTC'].drop_duplicates().sort_values()
    n_t = len(times_sorted)
    assert n_t * n_pts == len(df_all), "df_all not time-major cell-aligned"

    # gap guard: the reshape assumes contiguous hourly steps (add_t2m_lags was
    # gap-safe via map; the reshape is not). Fail loudly rather than misalign.
    gaps = times_sorted.diff().dropna().ne(pd.Timedelta(hours=1))
    if gaps.any():
        raise ValueError("non-hourly gap in chunk timesteps — per-cell lag "
                         "reshape would misalign; split chunk on the gap")

    t2m_mat = df_all['t2m'].to_numpy().reshape(n_t, n_pts)   # (n_t, n_pts)
    for h in (1, 3, 6, 12):
        lagged = np.full_like(t2m_mat, np.nan)
        if h < n_t:
            lagged[h:] = t2m_mat[:-h]           # hourly ⇒ shift = h rows
        df_all[f't2m-{h}'] = lagged.reshape(-1)

    return df_all
# ── Predict + save chunk NC ────────────────────────────────────────────────────

def predict_chunk(model, df, ny, nx, lcz_grid=None):
    """
    Returns:
      t2m_pred (n_times, ny, nx) — residual + era5_corrected_t2m [°C]
      times    pd.DatetimeIndex
      lcz      (ny, nx) int16  — only on first call (lcz_grid=None signals first)
    """
    feats = model.feature_names
    for f in set(feats) - set(df.columns):
        df[f] = np.nan

    t0 = time.perf_counter()
    residuals   = model.predict(xgb.DMatrix(df[feats]))
    predict_s = time.perf_counter() - t0
    t2m_corrected = df['t2m'].values   # lapse-rate-adjusted ERA5, already in °C
    t2m_final   = (residuals + t2m_corrected).reshape(-1, ny, nx).astype('float32')

    times = pd.DatetimeIndex(df['Time_UTC'].unique())
    n_t   = len(times)

    result = {'t2m': t2m_final, 'times': times, 'predict_s': predict_s}

    if lcz_grid is None:
        n_pts = ny * nx
        lcz_vals = df['LCZ_100'].values[:n_pts].reshape(ny, nx)
        lcz_vals = np.where(np.isnan(lcz_vals), 17, lcz_vals).astype('int16')
        result['lcz'] = lcz_vals

        dtm_vals = df['DTM_100'].values[:n_pts].reshape(ny, nx).astype('float32')
        result['dtm'] = dtm_vals

    return result

def save_chunk_nc(result, xs, ys, out_path):
    coords = {'time': result['times'], 'y': ys, 'x': xs}
    data_vars = {
        't2m': (['time', 'y', 'x'], result['t2m']),
    }   
    if 'lcz' in result:
        data_vars['lcz'] = (['y', 'x'], result['lcz'])
    if 'dtm' in result:
        data_vars['dtm'] = (['y', 'x'], result['dtm'])

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update({'crs': 'EPSG:3035', 'resolution_m': GRID_RES_M})
    enc = {
        't2m': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'lcz': {'zlib': True, 'complevel': 4, 'dtype': 'int16'},
        'dtm': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
    }
    ds.to_netcdf(out_path, encoding={k: v for k, v in enc.items() if k in ds.data_vars})
    ds.close()
# ── Combine chunk NCs into yearly files ───────────────────────────────────────

def combine_chunks(city, chunk_paths):
    """Group by year, combine, delete chunks."""
    by_year = defaultdict(list)
    for p in chunk_paths:
        ds = xr.open_dataset(p)
        year = int(pd.Timestamp(ds.time.values[0]).year)
        ds.close()
        by_year[year].append(p)

    for year, paths in sorted(by_year.items()):
        out = os.path.join(OUT_DIR, f"{city}_{FOLD_ID}_{year}_combined.nc")
        print(f"  combining {len(paths)} chunks → {out}")

        ds_list = [xr.open_dataset(p, chunks={'time': 240}) for p in paths]
        combined = xr.concat(
            [ds.drop_vars(['lcz', 'dtm'], errors='ignore') for ds in ds_list],
            dim='time'
        ).sortby('time')

        # Re-attach static fields from first chunk
        ds0 = xr.open_dataset(paths[0])
        if 'lcz' in ds0:
            combined['lcz'] = (('y', 'x'), ds0['lcz'].values)
        if 'dtm' in ds0:
            combined['dtm'] = (('y', 'x'), ds0['dtm'].values)
        ds0.close()

        enc = {
            't2m': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            'lcz': {'zlib': True, 'complevel': 4, 'dtype': 'int16'},
            'dtm': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        }
        combined.to_netcdf(out, encoding={k: v for k, v in enc.items() if k in combined.data_vars},
                           compute=True)
        for ds in ds_list:
            ds.close()
        combined.close()

        for p in paths:
            os.remove(p)
        print(f"  ✓ {os.path.basename(out)}  ({os.path.getsize(out)/1024**2:.0f} MB)")
# ── Main ──────────────────────────────────────────────────────────────────────



    print("\nDone.")
if __name__ == '__main__':
    main()