"""
K-means clustering on combined climate + spatial + meteorological features.

New variables added to the Excel features:
  - mean_wind   : JJA mean wind speed (ERA5 nearest gridpoint, 2015-2024) ^ 0.25
  - mean_ssrd   : JJA mean daily solar radiation (ERA5, last timestep/day)  ^ 0.25
  - mean_tcd_urb: mean Tree Cover Density in urban LCZ cells (LCZ 1-10)
  - mean_imp_urb: mean Imperviousness in urban LCZ cells (LCZ 1-10)
"""

import os
import calendar
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from cmcrameri import cm

# ── Configuration ─────────────────────────────────────────────────────────────

EXCEL_PATH  = "../data_processing/climate_data.xlsx"
ERA5_GLOB   = "../raw_data/era5land/ERA5Land_{date}.nc"
ERA5_YEARS  = range(2015, 2025)
ERA5_MONTHS = [6, 7, 8]

ERA5_CACHE   = "cache_era5.csv"
SPATIAL_CACHE = "cache_spatial.csv"

LCZ_PATH = "../tiffs/lcz/{city}_100.tif"
TCD_PATH = "../tiffs/tcd/{city}_100.tif"
IMP_PATH = "../tiffs/imp/{city}_100.tif"

URBAN_LCZ = set(range(1, 11))

CITIES = {
    "amsterdam":  {"lat": 52.3676, "lon":  4.9041, "area_km2": 308},
    "basel":      {"lat": 47.5596, "lon":  7.5886, "area_km2":  68},
    "berlin":     {"lat": 52.5200, "lon": 13.4050, "area_km2": 681},
    "bern":       {"lat": 46.9480, "lon":  7.4474, "area_km2":  54},
    "biel":       {"lat": 47.1368, "lon":  7.2467, "area_km2":  18},
    "birmingham": {"lat": 52.4862, "lon": -1.8904, "area_km2": 685},
    "freiburg":   {"lat": 47.9990, "lon":  7.8421, "area_km2":  50},
    "ghent":      {"lat": 51.0543, "lon":  3.7174, "area_km2":  94},
    "novisad":    {"lat": 45.2671, "lon": 19.8335, "area_km2":  59},
    "rennes":     {"lat": 48.1173, "lon": -1.6778, "area_km2":  57},
    "turku":      {"lat": 60.4518, "lon": 22.2666, "area_km2":  23},
    "zurich":     {"lat": 47.3769, "lon":  8.5417, "area_km2": 187},
    # "lyon":     {"lat": 45.7640, "lon":  4.8357, "area_km2": 256},
    # "oslo":     {"lat": 59.9139, "lon": 10.7522, "area_km2": 244},
    # "naples":   {"lat": 40.8518, "lon": 14.2681, "area_km2": 618},
    # "barcelona":{"lat": 41.3851, "lon":  2.1734, "area_km2": 438},
}

to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

# ── ERA5: open each file once, extract all cities ─────────────────────────────

def collect_era5_all_cities(city_keys):
    """
    Loop over all JJA daily ERA5 files once.
    For each file, extract wind speed and ssrd for every city in one sel() call.
    Returns dict: {city_key: {"wind": [...], "ssrd": [...]}}
    """
    accum = {k: {"wind": [], "ssrd": []} for k in city_keys}

    # Build full list of dates to process
    all_dates = []
    for year in ERA5_YEARS:
        for month in ERA5_MONTHS:
            ndays = calendar.monthrange(year, month)[1]
            for day in range(1, ndays + 1):
                all_dates.append(f"{year}-{month:02d}-{day:02d}")

    total = len(all_dates)
    found = 0
    missing = 0

    print(f"\n{'='*55}")
    print(f"  ERA5: processing {total} JJA days × {len(city_keys)} cities")
    print(f"{'='*55}")

    for i, date_str in enumerate(all_dates):
        fpath = ERA5_GLOB.format(date=date_str)

        if not os.path.exists(fpath):
            missing += 1
            if missing <= 5:
                print(f"  [ERA5] ⚠ missing: {fpath}")
            elif missing == 6:
                print(f"  [ERA5] ⚠ (further missing files suppressed)")
            continue

        found += 1
        if found % 50 == 1 or found == 1:
            print(f"  [ERA5] day {i+1}/{total}  ({date_str})  ✓ {found} found so far")

        with xr.open_dataset(fpath) as ds:
            for key in city_keys:
                info = CITIES[key]
                pt = ds.sel(latitude=info["lat"], longitude=info["lon"],
                            method="nearest")

                u = pt["u10"].values
                v = pt["v10"].values
                accum[key]["wind"].append(float(np.mean(np.sqrt(u**2 + v**2))))

                ssrd = pt["ssrd"].values
                accum[key]["ssrd"].append(float(ssrd[-1]))

    print(f"\n  [ERA5] Done — {found} files read, {missing} missing")
    return accum


def era5_metrics_from_accum(accum, key):
    """Compute final wind^0.25 and ssrd^0.25 from accumulated lists."""
    wind_vals = accum[key]["wind"]
    ssrd_vals = accum[key]["ssrd"]
    mean_wind = np.mean(wind_vals) ** 0.25 if wind_vals else np.nan
    mean_ssrd = np.mean(ssrd_vals) ** 0.25 if ssrd_vals else np.nan
    return mean_wind, mean_ssrd


# ── Spatial extraction (TCD + IMP in urban cells) ────────────────────────────

def city_box_3035(lat, lon, area_km2):
    cx, cy = to_3035.transform(lon, lat)
    half = (np.sqrt(area_km2) / 2 + 5) * 1000
    return cx - half, cy - half, cx + half, cy + half


def sample_tiff_in_box(tiff_path, box_3035, n_points=500):
    x_min, y_min, x_max, y_max = box_3035
    xs = np.linspace(x_min, x_max, n_points)
    ys = np.linspace(y_min, y_max, n_points)
    xg, yg = np.meshgrid(xs, ys)
    coords = list(zip(xg.ravel(), yg.ravel()))
    with rasterio.open(tiff_path) as src:
        nodata = src.nodata
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if nodata is not None:
            vals[vals == nodata] = np.nan
    return np.array(coords), vals


def get_spatial_metrics(city, lat, lon, area_km2):
    box = city_box_3035(lat, lon, area_km2)

    lcz_path = LCZ_PATH.format(city=city)
    tcd_path = TCD_PATH.format(city=city)
    imp_path = IMP_PATH.format(city=city)

    print(f"  [spatial] LCZ  → {lcz_path}")
    coords, lcz_vals = sample_tiff_in_box(lcz_path, box)
    urban_mask = np.isin(lcz_vals.astype(int), list(URBAN_LCZ)) & ~np.isnan(lcz_vals)
    print(f"  [spatial] urban cells: {urban_mask.sum()} / {len(urban_mask)}")

    if urban_mask.sum() == 0:
        print(f"  [spatial] ⚠ no urban LCZ cells — returning NaN")
        return np.nan, np.nan

    mean_tcd = np.nan
    if os.path.exists(tcd_path):
        print(f"  [spatial] TCD  → {tcd_path}")
        with rasterio.open(tcd_path) as src:
            nodata = src.nodata
            tcd_vals = np.array([v[0] for v in src.sample(coords.tolist())], dtype=float)
            if nodata is not None:
                tcd_vals[tcd_vals == nodata] = np.nan
        mean_tcd = float(np.nanmean(tcd_vals[urban_mask]))
        print(f"  [spatial] TCD mean (urban): {mean_tcd:.2f}")
    else:
        print(f"  [spatial] ⚠ TCD not found: {tcd_path}")

    mean_imp = np.nan
    if os.path.exists(imp_path):
        print(f"  [spatial] IMP  → {imp_path}")
        with rasterio.open(imp_path) as src:
            nodata = src.nodata
            imp_vals = np.array([v[0] for v in src.sample(coords.tolist())], dtype=float)
            if nodata is not None:
                imp_vals[imp_vals == nodata] = np.nan
        mean_imp = float(np.nanmean(imp_vals[urban_mask]))
        print(f"  [spatial] IMP mean (urban): {mean_imp:.2f}")
    else:
        print(f"  [spatial] ⚠ IMP not found: {imp_path}")

    return mean_tcd, mean_imp


# ── Build feature table ───────────────────────────────────────────────────────
def load_or_compute_era5(known_keys):
    """Load ERA5 metrics from cache if available, compute missing cities."""
    # Load existing cache
    if os.path.exists(ERA5_CACHE):
        cache = pd.read_csv(ERA5_CACHE, index_col="city_key")
        print(f"  [ERA5 cache] loaded {len(cache)} cities from {ERA5_CACHE}")
    else:
        cache = pd.DataFrame(columns=["city_key", "mean_wind", "mean_ssrd"]).set_index("city_key")

    missing_keys = [k for k in known_keys if k not in cache.index]

    if missing_keys:
        print(f"  [ERA5 cache] computing {len(missing_keys)} new cities: {missing_keys}")
        era5_accum = collect_era5_all_cities(missing_keys)
        new_rows = []
        for key in missing_keys:
            mw, ms = era5_metrics_from_accum(era5_accum, key)
            new_rows.append({"city_key": key, "mean_wind": mw, "mean_ssrd": ms})
        new_df = pd.DataFrame(new_rows).set_index("city_key")
        cache = pd.concat([cache, new_df])
        cache.to_csv(ERA5_CACHE)
        print(f"  [ERA5 cache] saved → {ERA5_CACHE}")
    else:
        print(f"  [ERA5 cache] all cities already cached — skipping ERA5 loop")

    return cache


def load_or_compute_spatial(known_keys):
    """Load spatial metrics from cache if available, compute missing cities."""
    if os.path.exists(SPATIAL_CACHE):
        cache = pd.read_csv(SPATIAL_CACHE, index_col="city_key")
        print(f"  [spatial cache] loaded {len(cache)} cities from {SPATIAL_CACHE}")
    else:
        cache = pd.DataFrame(columns=["city_key", "mean_tcd_urb", "mean_imp_urb"]).set_index("city_key")

    missing_keys = [k for k in known_keys if k not in cache.index]

    if missing_keys:
        print(f"  [spatial cache] computing {len(missing_keys)} new cities: {missing_keys}")
        new_rows = []
        for key in missing_keys:
            info = CITIES[key]
            print(f"\n  {key}  (lat={info['lat']}, lon={info['lon']}, area={info['area_km2']} km²)")
            mt, mi = get_spatial_metrics(key, info["lat"], info["lon"], info["area_km2"])
            new_rows.append({"city_key": key, "mean_tcd_urb": mt, "mean_imp_urb": mi})
        new_df = pd.DataFrame(new_rows).set_index("city_key")
        cache = pd.concat([cache, new_df])
        cache.to_csv(SPATIAL_CACHE)
        print(f"  [spatial cache] saved → {SPATIAL_CACHE}")
    else:
        print(f"  [spatial cache] all cities already cached — skipping spatial loop")

    return cache

def build_features():
    # --- Excel ---
    print("\n── Loading Excel ──")
    data_raw = pd.read_excel(EXCEL_PATH, index_col=0)
    samples = data_raw.loc["samples"].copy()
    samples.index = samples.index.str.capitalize()
    samples.index = samples.index.str.replace("Novisad", "Novi Sad")

    data_raw = data_raw.drop("samples")
    data = data_raw.T.reset_index().rename(columns={"index": "city"})
    data["city"] = data["city"].str.capitalize().replace("Novisad", "Novi Sad")
    data["samples"] = data["city"].map(samples)
    data["city_key"] = data["city"].str.lower().str.replace(" ", "")
    print(f"  {len(data)} cities loaded from Excel: {list(data['city'])}")

    # Filter to cities we have config for
    known_keys = [k for k in data["city_key"] if k in CITIES]
    unknown    = [k for k in data["city_key"] if k not in CITIES]
    if unknown:
        print(f"  ⚠ no CITIES config for: {unknown} — will fill NaN")

    # --- ERA5: single pass over all files ---
    # --- ERA5 (cached) ---
    era5_cache    = load_or_compute_era5(known_keys)
    spatial_cache = load_or_compute_spatial(known_keys)

    # --- Merge into data ---
    wind_col, ssrd_col, tcd_col, imp_col = [], [], [], []
    for _, row in data.iterrows():
        key = row["city_key"]
        if key not in CITIES or key not in era5_cache.index:
            wind_col.append(np.nan); ssrd_col.append(np.nan)
            tcd_col.append(np.nan);  imp_col.append(np.nan)
            continue
        wind_col.append(era5_cache.loc[key, "mean_wind"])
        ssrd_col.append(era5_cache.loc[key, "mean_ssrd"])
        tcd_col.append(spatial_cache.loc[key, "mean_tcd_urb"] if key in spatial_cache.index else np.nan)
        imp_col.append(spatial_cache.loc[key, "mean_imp_urb"] if key in spatial_cache.index else np.nan)

    data["mean_wind"]    = wind_col
    data["mean_ssrd"]    = ssrd_col
    data["mean_tcd_urb"] = tcd_col
    data["mean_imp_urb"] = imp_col

    return data, samples


# ── Clustering ────────────────────────────────────────────────────────────────

_ALL_CITIES_12 = [
    "Amsterdam", "Basel", "Berlin", "Bern", "Biel", "Birmingham",
    "Freiburg", "Ghent", "Novi Sad", "Rennes", "Turku", "Zürich",
]
_n = len(_ALL_CITIES_12)
COLOR_MAP = {city: cm.batlow(i / (_n - 1)) for i, city in enumerate(_ALL_CITIES_12)}
MARKERS = {
    "Amsterdam": "D", "Basel": "v",      "Berlin": "o",  "Bern": "^",
    "Biel": "s",      "Birmingham": "<", "Freiburg": "p", "Ghent": ">",
    "Novi Sad": "H",  "Rennes": "P",     "Turku": "D",   "Zürich": "X",
}


def run_clustering(data, samples):
    feature_cols = [c for c in data.columns
                    if c not in ("city", "city_key", "samples", "cluster")]
    X = data[feature_cols].astype(float)

    print("\n── Features entering clustering ──")
    print(f"  {list(X.columns)}")
    print(X.describe().round(2).to_string())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    data["cluster"] = kmeans.fit_predict(X_scaled)
    print("\n── Clusters ──")
    print(data.groupby("cluster")["city"].apply(list).to_string())

    cities      = data["city"].values
    dist_matrix = euclidean_distances(X_scaled)

    print("\nCITY_RANKINGS = {")
    for i, city in enumerate(cities):
        dists = dist_matrix[i].copy(); dists[i] = np.inf
        sorted_idx = np.argsort(dists)[:-1]
        neighbors = [cities[j].lower().replace(" ", "") for j in sorted_idx]
        print(f"    '{city.lower().replace(' ', '')}': {neighbors},")
    print("}")

    plot_kmeans_clusters(X_scaled, X, cities, data["cluster"].values,
                         save_path="kmeans_clusters.png")
    return data


def plot_kmeans_clusters(X_scaled, X, cities, cluster_labels,
                         save_path=None, dpi=600):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 12})

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    hull_color = cm.glasgow(0.5)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for c in range(len(np.unique(cluster_labels))):
        pts = coords[cluster_labels == c]
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            ax.add_patch(plt.Polygon(
                pts[hull.vertices], closed=True,
                facecolor=hull_color, alpha=0.10,
                edgecolor=hull_color, linewidth=1.2, zorder=1,
            ))
        elif len(pts) == 2:
            ax.plot(pts[:, 0], pts[:, 1],
                    color=hull_color, linewidth=1.2, alpha=0.4, zorder=1)

    def _norm(s):
        return s.strip().replace("\u00fc", "ü").replace("Zurich", "Zürich")

    for i, city in enumerate(cities):
        color = COLOR_MAP.get(_norm(city), "0.5")
        ax.scatter(coords[i, 0], coords[i, 1],
                   marker=MARKERS.get(city, "o"),
                   color=color, s=70, zorder=3, linewidths=0)

    label_offsets = {
        "Amsterdam": ( 0.05,  0.10), "Basel":      (-0.15,  0.15),
        "Berlin":    (-0.55, -0.20), "Bern":       (-0.45, -0.25),
        "Biel":      ( 0.05,  0.10), "Birmingham": ( 0.05,  0.10),
        "Freiburg":  ( 0.00, -0.30), "Ghent":      (-0.70,  0.05),
        "Novi Sad":  ( 0.10, -0.15), "Rennes":     ( 0.12, -0.01),
        "Turku":     ( 0.05, -0.20), "Zürich":     ( 0.05,  0.12),
    }
    for i, city in enumerate(cities):
        city_n = _norm(city)
        color  = COLOR_MAP.get(city_n, "0.5")
        dx, dy = label_offsets.get(city_n, (0.05, 0.05))
        ax.text(coords[i, 0] + dx, coords[i, 1] + dy,
                city_n, fontsize=10, color=color, zorder=4)

    ax.set_xlabel(f"PC 1  ({var[0]:.1f} % variance)")
    ax.set_ylabel(f"PC 2  ({var[1]:.1f} % variance)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lim = abs(coords).max() * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    return fig, ax


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data, samples = build_features()

    print("\n── Full feature table ──")
    feature_cols = [c for c in data.columns if c not in ("city_key", "samples", "cluster")]
    print(data[feature_cols].to_string(index=False))

    data.to_csv("features_combined.csv", index=False)
    run_clustering(data, samples)
    plt.show()