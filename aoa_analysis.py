"""
Area of Applicability (Meyer & Pebesma 2021) for the XGBoost temperature
downscaling ensemble.

One shared AOA engine, two analyses selected by MODE at the bottom:

  MODE = "stations"  DI for EVERY row of each test city's URS table, then
                     summarised PER STATION: DI min/median/max, #rows, and
                     #rows inside the AOA (DI varies row-to-row with the
                     ERA5 forcing, so the per-station DI *range* is the point).

  MODE = "grid"      DI for EVERY grid cell over ONE day of forcing
                     (default: the hottest ERA5-t2m day in the test period).
                     Mapped as the number of in-AOA hours per cell (0..24)
                     rather than the raw DI.

Why this is faithful: the engine is built from the SAME training X the model
saw (cv_folds.load_cv_data), the feature list IS `features`, and the weights
are the CV-fold-averaged booster importances keyed by those same names. So the
DI feature space == the model's predictor space, LCZ one-hot columns included.
No hand-maintained feature list, no drift.
"""

import os
import glob
import importlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from cmcrameri import cm
import cartopy.crs as ccrs
import contextily as ctx
from pyproj import Transformer

from xgb_0_prep_params import features
from cv_folds import CV_CITIES, TEST_CITIES, get_folds, load_cv_data

# ============================== CONFIG ==================================
CV_MODE      = "6fold_st"          # "12fold_loco" | "6fold_st"
MODELS_DIR   = f"{CV_MODE}_models"
BOOSTER_GLOB = os.path.join(MODELS_DIR, f"xgb_{CV_MODE}_*.json")

DATA_DIR   = "../data_processing/dataframes_ready"
URS_TABLE  = "tablex_{city}.pkl"      # test-city obs table (station × time rows)
OUT_DIR    = "plots_aoa"

GLOBAL_SUB     = 300_000              # train rows for the scoring NN tree + d_bar
FOLD_TRAIN_SUB = 150_000              # per-fold training rows (threshold NN tree)
FOLD_VAL_SUB   =  40_000              # per-fold validation rows queried for threshold
XY_CANDIDATES  = [("x", "y"), ("X3035", "Y3035"), ("X_3035", "Y_3035")]
STATION_KEYS   = ["station", "station_id", "name", "stn", "id"]
# TRIM = 8
# Grid analysis reuses YOUR grid-prediction helpers so the feature build (lapse
# t2m, deaccumulation, t2m lags) is identical to prediction. Rename to your
# grid script's module name (document 3):
GRID_MODULE = "xgb_z_grid_predict"
# ========================================================================

PROJ = ccrs.LambertAzimuthalEqualArea(
    central_longitude=10.0, central_latitude=52.0,
    false_easting=4321000.0, false_northing=3210000.0)


# ------------------------- importance weights --------------------------
def _fold_weights():
    """CV-fold-averaged total_gain importances, normalised, keyed by `features`."""
    paths = sorted(glob.glob(BOOSTER_GLOB))
    if not paths:
        print(f"[weights] WARNING no boosters at {BOOSTER_GLOB} -> uniform weights")
        return np.ones(len(features)) / len(features)
    import xgboost as xgb
    mats = []
    for p in paths:
        b = xgb.Booster(); b.load_model(p)
        sc = b.get_score(importance_type="total_gain")
        v = np.array([sc.get(f, 0.0) for f in features], float)
        s = v.sum()
        mats.append(v / s if s else v)
    w = np.vstack(mats).mean(0)
    s = w.sum()
    if s == 0:
        print("[weights] WARNING booster importances empty -> uniform weights")
        return np.ones(len(features)) / len(features)
    print(f"[weights] averaged {len(paths)} boosters from {BOOSTER_GLOB}")
    return w / s

def _nn_dist(ref, Q, block=4096):
    """Nearest-neighbour distance from each row of Q to the reference set `ref`,
    brute-force in blocks to bound memory. Returns 1-D array of length len(Q)."""
    out = np.empty(len(Q), dtype=np.float64)
    for i in range(0, len(Q), block):
        out[i:i+block] = cdist(Q[i:i+block], ref).min(axis=1)
    return out

# ------------------------------ engine ---------------------------------
class AOA:
    """DI in the weighted, train-standardized predictor space. The threshold is
    Q75 + 1.5*IQR of the DIs of cross-validated hold-out points, where the CV
    splits are the SAME ones the model uses (cv_folds.get_folds / build_fold):

      12fold_loco : val = held city            vs train = other 11 cities
      6fold_st    : val = held cities × held years
                    vs train = other 10 cities × all-but-held years

    Memory-safe: training data is never fully stacked. The scaler is fit
    streaming (partial_fit), and every distance computation runs on per-city
    subsamples drawn from the finite rows only."""

    def __init__(self, seed=0):
        data = load_cv_data()                       # {city: (X, y, year)}
        self.columns = list(features)
        self.weights = _fold_weights()
        self._finite = {c: np.isfinite(data[c][0]).all(1) for c in CV_CITIES}
        for c in CV_CITIES:
            n_bad = int((~self._finite[c]).sum())
            if n_bad:
                print(f"[aoa] {c}: dropping {n_bad:,} rows with NaN/Inf")

        # train-only standardization — exact, streaming, no full stack
        self.scaler = StandardScaler()
        for c in CV_CITIES:
            self.scaler.partial_fit(data[c][0][self._finite[c]])

        rng = np.random.default_rng(seed)

        # global subsample -> d_bar + the tree new points are scored against
        Xw = self._gather(data, CV_CITIES, {c: None for c in CV_CITIES},
                          GLOBAL_SUB, rng)
        a = rng.choice(len(Xw), min(len(Xw), 20_000), replace=False)
        b = rng.choice(len(Xw), min(len(Xw), 20_000), replace=False)
        self.d_bar = np.linalg.norm(Xw[a] - Xw[b], axis=1).mean()
        self._tree = cKDTree(Xw)                    # new points vs ALL training

        # threshold from the actual CV hold-out splits
        self._fit_threshold_cv(data, rng)
        print(f"[aoa] tree_n={len(Xw):,}  d_bar={self.d_bar:.3f}  "
              f"DI* = {self.threshold:.3f}")

    # ---- weighted, standardized space -------------------------------------
    def _scaled(self, X):
        return self.scaler.transform(np.asarray(X, float)) * self.weights

    def _space(self, df):
        return self._scaled(df[self.columns].to_numpy(float))

    # ---- fold-aware subsampler (no full vstack) ---------------------------
    def _gather(self, data, cities, yfilter, cap, rng):
        """Scaled+weighted subsample (<= cap rows) over `cities`, drawn from
        finite rows, with an optional per-city year filter. yfilter maps
        city -> None | ("in", years) | ("out", years)."""
        masks, total = {}, 0
        for c in cities:
            m = self._finite[c].copy()
            yf = yfilter.get(c)
            if yf is not None:
                kind, ys = yf
                inset = np.isin(data[c][2], list(ys))
                m &= inset if kind == "in" else ~inset
            masks[c] = m
            total += int(m.sum())
        if total == 0:
            return np.empty((0, len(self.columns)))
        parts = []
        for c in cities:
            idx = np.flatnonzero(masks[c])
            if len(idx) == 0:
                continue
            take = max(1, int(round(cap * len(idx) / total)))   # size-proportional
            if len(idx) > take:
                idx = rng.choice(idx, take, replace=False)
            parts.append(self._scaled(data[c][0][idx]))
        return np.vstack(parts) if parts else np.empty((0, len(self.columns)))

    # ---- threshold via the real CV train/val partitions -------------------
    def _fit_threshold_cv(self, data, rng):
        di_val = []
        for fold in get_folds(CV_MODE):
            if CV_MODE == "12fold_loco":
                fid          = fold
                val_cities   = [fold]
                train_cities = [c for c in CV_CITIES if c != fold]
                yv = {fold: None}                               # held city, all years
                yt = {c: None for c in train_cities}            # others, all years
            else:
                fid          = fold["id"]
                vc, vy       = set(fold["cities"]), set(fold["years"])
                val_cities   = list(fold["cities"])
                train_cities = [c for c in CV_CITIES if c not in vc]
                yv = {c: ("in",  vy) for c in val_cities}       # held cities × held years
                yt = {c: ("out", vy) for c in train_cities}     # others × all-but-held yrs

            Xt = self._gather(data, train_cities, yt, FOLD_TRAIN_SUB, rng)
            Xv = self._gather(data, val_cities,   yv, FOLD_VAL_SUB,   rng)
            if len(Xt) == 0 or len(Xv) == 0:
                print(f"[aoa] fold {fid}: empty split "
                      f"(train={len(Xt)}, val={len(Xv)}) — skipped")
                continue
            d, _ = cKDTree(Xt).query(Xv, k=1)
            di = d / self.d_bar
            di_val.append(di)
            print(f"[aoa] fold {fid}: train~{len(Xt):,} val~{len(Xv):,} "
                  f"medDI={np.median(di):.3f}")

        di = np.concatenate(di_val)
        q75, q25 = np.percentile(di, [75, 25])
        self.threshold = q75 + 1.5 * (q75 - q25)
        self.di_train  = di

    # def score(self, df):
    #     """Return DI and in_aoa per row; rows with any missing feature -> NaN DI."""
    #     di = np.full(len(df), np.nan)
    #     ok = df[self.columns].notna().all(1).to_numpy()
    #     if ok.any():
    #         d, _ = self._tree.query(self._space(df.loc[ok]), k=1)
    #         di[ok] = d / self.d_bar
    #     return pd.DataFrame({"DI": di, "in_aoa": di <= self.threshold}, index=df.index)


    def score(self, df):
        di = np.full(len(df), np.nan)
        ok = df[self.columns].notna().all(1).to_numpy()
        
        if ok.any():
            d, _ = self._tree.query(self._space(df.loc[ok]), k=1)
            di[ok] = d / self.d_bar
                            

        return pd.DataFrame({"DI": di, "in_aoa": di <= self.threshold}, index=df.index)


    def save(self, path=None):
        path = path or f"aoa_fit_{CV_MODE}.npz"
        np.savez(path, tree_pts=self._tree.data, weights=self.weights,
                 mean=self.scaler.mean_, scale=self.scaler.scale_,
                 d_bar=self.d_bar, threshold=self.threshold,
                 columns=np.array(self.columns))
        print(f"[aoa] saved fit -> {path}")

    @classmethod
    def load(cls, path=None):
        path = path or f"aoa_fit_{CV_MODE}.npz"
        z = np.load(path, allow_pickle=True)
        self = cls.__new__(cls)
        self.columns   = list(z["columns"])
        self.weights   = z["weights"]
        self.d_bar     = float(z["d_bar"])
        self.threshold = float(z["threshold"])
        self.scaler = StandardScaler()
        self.scaler.mean_, self.scaler.scale_ = z["mean"], z["scale"]
        self.scaler.n_features_in_ = len(self.columns)
        # self._ref = z["tree_pts"]              # reference points, brute-force NN
        # self._tree = cKDTree(self._ref)        # keep for score(); station path still uses it
        pts = z["tree_pts"]
        if len(pts) > 40_000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 40_000, replace=False)]
        self._tree = cKDTree(pts)
        # pts = z["tree_pts"]
        # if len(pts) > 40_000:
        #     pts = pts[np.random.default_rng(0).choice(len(pts), 40_000, replace=False)]
        # self._tree = cKDTree(pts)
        # self._tree = cKDTree(z["tree_pts"])
        print(f"[aoa] loaded fit <- {path}  DI* = {self.threshold:.3f}")
        return self
# --------------------------- shared helpers ----------------------------
def _align_features(df, name=""):
    """Ensure every model feature exists as a column; warn on any that don't."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        show = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        print(f"[{name}] WARNING {len(missing)} feature cols missing -> NaN: {show}")
        for f in missing:
            df[f] = np.nan
    return df


def _find_xy(df):
    for xc, yc in XY_CANDIDATES:
        if xc in df.columns and yc in df.columns:
            return xc, yc
    return None, None


def _station_key(df):
    for k in STATION_KEYS:
        if k in df.columns:
            return k
    xc, yc = _find_xy(df)
    if xc is None:
        raise KeyError("No station id column and no coord columns to group by.")
    key = "_station_xy"
    df[key] = (df[xc].round().astype("Int64").astype(str) + "_" +
               df[yc].round().astype("Int64").astype(str))
    return key


def _filter_time(df, date_range):
    if date_range and "Time_UTC" in df.columns:
        lo, hi = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
        df = df[(df["Time_UTC"] >= lo) & (df["Time_UTC"] <= hi)]
        
    return df


# ===================== 1) STATION-BASED ANALYSIS =======================
def analyze_stations(aoa, cities=TEST_CITIES, date_range=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    per_station = []
    for city in cities:
        df = pd.read_pickle(os.path.join(DATA_DIR, URS_TABLE.format(city=city)))
        df["city"] = city
        df = _filter_time(df, date_range)
        df = _align_features(df, city)

        s  = aoa.score(df)
        df = df.assign(DI=s["DI"].values, in_aoa=s["in_aoa"].values)

        key = _station_key(df)
        g = (df.dropna(subset=["DI"])
               .groupby(key)
               .agg(city=("city", "first"),
                    n_rows=("DI", "size"),
                    n_in_aoa=("in_aoa", "sum"),
                    di_min=("DI", "min"),
                    di_med=("DI", "median"),
                    di_max=("DI", "max"))
               .reset_index()
               .rename(columns={key: "station"}))
        g["pct_in_aoa"] = 100 * g["n_in_aoa"] / g["n_rows"]
        per_station.append(g)

        valid   = df["DI"].notna()
        overall = 100 * df.loc[valid, "in_aoa"].mean() if valid.any() else float("nan")
        print(f"{city:<12} rows={int(valid.sum()):>8,}  stations={g.shape[0]:>3}  "
              f"in-AOA={overall:5.1f}%")

    out = pd.concat(per_station, ignore_index=True)
    csv = os.path.join(OUT_DIR, f"aoa_stations_{CV_MODE}.csv")
    out.to_csv(csv, index=False)
    print(f"\nsaved {csv}")
    _plot_station_ranges(out, aoa.threshold)
    return out


def _plot_station_ranges(tbl, thr):
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    tbl = tbl.sort_values(["city", "di_med"]).reset_index(drop=True)
    x = np.arange(len(tbl))

    fig, ax = plt.subplots(figsize=(max(8, 0.16 * len(tbl)), 5))
    ax.vlines(x, tbl["di_min"], tbl["di_max"], color="0.75", lw=1)
    sc = ax.scatter(x, tbl["di_med"], c=tbl["pct_in_aoa"], cmap=cm.batlow,
                    vmin=0, vmax=100, s=22, zorder=3)
    ax.axhline(thr, color="crimson", ls="--", lw=1.2,
               label=f"AOA threshold  DI* = {thr:.2f}")

    # city boundaries
    bounds = tbl.groupby("city").apply(lambda d: (d.index.min(), d.index.max()))
    for city, (i0, i1) in bounds.items():
        ax.text((i0 + i1) / 2, ax.get_ylim()[1], city, ha="center", va="bottom",
                fontsize=9)
        if i0 > 0:
            ax.axvline(i0 - 0.5, color="0.85", lw=0.8)

    ax.set_ylabel("Dissimilarity Index")
    ax.set_xlabel("station (sorted by median DI, grouped by city)")
    ax.set_xticks([])
    ax.legend(loc="upper left", framealpha=0.9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% of rows in AOA")
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"aoa_station_ranges_{CV_MODE}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ======================= 2) GRID-BASED ANALYSIS ========================
def _hottest_day(city, date_range=None):
    df = pd.read_pickle(os.path.join(DATA_DIR, URS_TABLE.format(city=city)))
    df = _filter_time(df, date_range)
    i  = df["t2m"].idxmax()
    d  = pd.Timestamp(df.loc[i, "Time_UTC"]).normalize()
    print(f"[grid] hottest day for {city}: {d.date()}  (t2m={df.loc[i, 't2m']:.1f})")
    return d


def _grid_context(city, first_day):
    """One-time setup: static grid, dims, coords, ERA5 pixel. Reused every day."""
    G = importlib.import_module(GRID_MODULE)
    static = G.build_static_grid(city)                       # loaded ONCE
    ny = int(static["_ys_idx"].max()) + 1
    nx = int(static["_xs_idx"].max()) + 1
    lat0, lon0 = static["Latitude"].mean(), static["Longitude"].mean()
    sample_el = G.load_era5land(pd.Timestamp(first_day).strftime("%Y-%m-%d"))
    lat_el, lon_el = G.era5land_pixel(lat0, lon0, sample_el)
    xs = np.load(os.path.join(G.STATIC_DIR, f"{city}_static_geo_xs.npy"))[:nx]
    ys = np.load(os.path.join(G.STATIC_DIR, f"{city}_static_geo_ys.npy"))[:ny]
    # if TRIM:
    #     static = static[(static["_xs_idx"] >= TRIM) & (static["_xs_idx"] < nx - TRIM) &
    #                     (static["_ys_idx"] >= TRIM) & (static["_ys_idx"] < ny - TRIM)].copy()
    #     static["_xs_idx"] -= TRIM
    #     static["_ys_idx"] -= TRIM
    #     xs, ys = xs[TRIM:-TRIM], ys[TRIM:-TRIM]
    #     nx, ny = nx - 2*TRIM, ny - 2*TRIM
    return dict(G=G, static=static, ny=ny, nx=nx, lat0=lat0, lon0=lon0,
                lat_el=lat_el, lon_el=lon_el, xs=xs, ys=ys)


def _day_forcing(ctx, day):
    """Build one day's grid forcing (loads that day + its predecessor for lags)."""
    G, day = ctx["G"], pd.Timestamp(day)
    era5_days, prev_h23 = [], {v: None for v in ("ssrd", "tp")}
    for d in (day - pd.Timedelta(days=1), day):
        ds_el = G.load_era5land(d.strftime("%Y-%m-%d"))
        ds_e5 = G.load_era5(d.strftime("%Y-%m-%d"))
        ds_p  = G.load_era5pressure(d.strftime("%Y-%m-%d"))
        era5, times = G.extract_day(ds_el, ds_e5, ds_p, ctx["lat_el"], ctx["lon_el"],
                                    ctx["lat0"], ctx["lon0"], prev_h23)
        for v in ("ssrd", "tp"):
            prev_h23[v] = era5[f"_{v}_raw_h23"]
        era5_days.append((era5, times))
        ds_el.close(); ds_e5.close(); ds_p.close()
    df = G.build_chunk_df(ctx["static"], era5_days)
    return df[df["Time_UTC"].dt.normalize() == day].reset_index(drop=True)



def analyze_grid(aoa, city, date_range=("2015-08-01", "2015-08-31"),
                 use_cache=True):
    import time
    os.makedirs(os.path.join(OUT_DIR, city), exist_ok=True)
    cache = os.path.join(OUT_DIR, city, f"aoa_gridcache_{city}_{CV_MODE}.npz")

    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        frac_in, xs, ys = z["frac"], z["xs"], z["ys"]
        n_steps_total = int(z["n_steps"])
        print(f"[grid] loaded cache <- {cache}  "
              f"({n_steps_total}h, mean frac={np.nanmean(frac_in):.2f})")
        _plot_frac_map(city, xs, ys, frac_in,
                   pd.date_range(*date_range, freq="D"),
                   stations=None)#_station_coords(city))
        return frac_in, n_steps_total

    lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    all_days = pd.date_range(lo, hi, freq="D")

    ctx = _grid_context(city, all_days[0])
    ny, nx = ctx["ny"], ctx["nx"]
    cells  = ny * nx
    count_in, n_steps_total = np.zeros((ny, nx), np.int32), 0
    n = len(os.sched_getaffinity(0))

    for day in all_days:
        t0 = time.time()
        df = _day_forcing(ctx, day)
        df = _align_features(df, f"{city}-grid")
        feat = df[aoa.columns].to_numpy(float); del df
        n_steps = len(feat) // cells
        t1 = time.time()

        for t in range(n_steps):
            block = feat[t*cells:(t+1)*cells]
            ok = np.isfinite(block).all(1)
            di = np.full(cells, np.nan)
            if ok.any():
                d, _ = aoa._tree.query(aoa._scaled(block[ok]), k=1, workers=-1)
                di[ok] = d / aoa.d_bar
            count_in += (di <= aoa.threshold).reshape(ny, nx).astype(np.int32)
        n_steps_total += n_steps; del feat
        print(f"[grid] {day.date()}: {n_steps}h  build {t1-t0:.0f}s  "
              f"score {time.time()-t1:.0f}s  (cum {n_steps_total}h)", flush=True)

    frac_in = count_in / n_steps_total
    np.savez(cache, frac=frac_in, xs=ctx["xs"], ys=ctx["ys"],
             n_steps=n_steps_total)
    print(f"[grid] {city}  total={n_steps_total}h  cells={count_in.size:,}  "
          f"mean in-AOA frac={frac_in.mean():.2f}  cached -> {cache}")
    _plot_frac_map(city, ctx["xs"], ctx["ys"], frac_in, all_days,
                   stations=None)#_station_coords(city))
    return frac_in, n_steps_total


def _plot_count_map(city, xs, ys, count, n_hours, date):
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    to4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    xx, yy   = np.meshgrid(xs, ys)
    lon, lat = to4326.transform(xx.ravel(), yy.ravel())
    lon, lat = lon.reshape(xx.shape), lat.reshape(xx.shape)
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": PROJ})
    ax.set_extent([xmin, xmax, ymin, ymax], crs=PROJ)

    norm = BoundaryNorm(np.arange(-0.5, n_hours + 1.5, 1), cm.batlow.N)
    cs = ax.pcolormesh(lon, lat, count, cmap=cm.batlow, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto", alpha=0.6)
    ctx.add_basemap(ax, crs=PROJ)#, source=ctx.providers.CartoDB.Positron)
    # ax.set_title(f"{city.capitalize()} — hours in AOA  ({pd.Timestamp(date).date()})")

    step = max(1, n_hours // 8)
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04,
                      ticks=np.arange(0, n_hours + 1, step))
    cb.set_label(f"hours in AOA (of {n_hours})")

    sl, px, py = 5000, (xmax - xmin) * 0.02, (ymax - ymin) * 0.02
    bx, by, bh = xmax - sl - px, ymin + py, (ymax - ymin) * 0.005
    ax.add_patch(Rectangle((bx, by), sl, bh, facecolor="black",
                           transform=PROJ, zorder=6))
    ax.text(bx + sl / 2, by + bh + py * 0.2, "5 km", ha="center", va="bottom",
            fontsize=10, transform=PROJ, zorder=6)

    city_dir = os.path.join(OUT_DIR, city)
    os.makedirs(city_dir, exist_ok=True)
    out = os.path.join(city_dir, f"aoa_hours_{city}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (+ .pdf)")


def station_cell_aoa(city, frac, xs, ys, table=URS_TABLE):
    """Mean in-AOA fraction of the GRID cells that contain measurement stations.
    Bridges station-level AOA (station's own rows) and grid-mean AOA (all cells):
    same frac object as the grid map, but sampled only at station cells."""
    # unique station coords (lon/lat) -> EPSG:3035, the frac grid's CRS
    pts_ll = _station_coords(city, table)
    to3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    sx, sy = to3035.transform(pts_ll[:, 0], pts_ll[:, 1])

    # nearest cell index on each 1-D axis
    ix = np.abs(xs[None, :] - np.asarray(sx)[:, None]).argmin(1)
    iy = np.abs(ys[None, :] - np.asarray(sy)[:, None]).argmin(1)

    vals = frac[iy, ix]                                    # frac is [ny, nx]
    ok = np.isfinite(vals)
    cells = (pd.DataFrame({"ix": ix[ok], "iy": iy[ok], "frac": vals[ok]})
               .drop_duplicates(["ix", "iy"]))             # collapse stations sharing a cell

    print(f"[stationcell] {city}: {len(pts_ll)} stations, {ok.sum()} matched, "
          f"{len(cells)} unique cells | mean frac = {cells['frac'].mean():.3f} "
          f"(station-weighted = {vals[ok].mean():.3f})")
    return cells
    
    
def _plot_frac_map(city, xs, ys, frac, dates, stations=None):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    to4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = to4326.transform(xx.ravel(), yy.ravel())
    lon, lat = lon.reshape(xx.shape), lat.reshape(xx.shape)
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()

    bounds = np.round(np.arange(0, 1.0001, 0.2), 2)          # 0 .2 .4 .6 .8 1
    cmap   = ListedColormap(cm.batlow(np.linspace(0.05, 0.95, len(bounds) - 1)))
    norm   = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": PROJ})
    ax.set_extent([xmin, xmax, ymin, ymax], crs=PROJ)
    cs = ax.pcolormesh(lon, lat, np.ma.masked_invalid(frac), cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto", alpha=0.6)
    ctx.add_basemap(ax, crs=PROJ)#, source=ctx.providers.CartoDB.Positron)

    if stations is not None and len(stations):
        ax.scatter(stations[:, 0], stations[:, 1], transform=ccrs.PlateCarree(),
                   s=7, marker="s", facecolor="none", edgecolor="crimson",
                   linewidth=0.5, zorder=7, label=f"stations (n={len(stations)})")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04,
                      ticks=bounds, spacing="proportional")

    scale_length = 5000
    padding_x = (xmax - xmin) * 0.02
    padding_y = (ymax - ymin) * 0.02
    bar_x_start = xmax - scale_length - padding_x
    bar_y = ymin + padding_y
    bar_height = (ymax - ymin) * 0.005
    
    scale_bar = Rectangle(
        (bar_x_start, bar_y), scale_length, bar_height,
        facecolor='black', edgecolor='black', linewidth=1,
        transform=PROJ, zorder=6
    )
    ax.add_patch(scale_bar)
    ax.text(bar_x_start + scale_length/2, bar_y + bar_height + padding_y*0.2,
            '5 km', ha='center', va='bottom', fontsize=11,
            transform=PROJ, zorder=6)


    # cb.set_label("fraction of timesteps in AOA")

    city_dir = os.path.join(OUT_DIR, city); os.makedirs(city_dir, exist_ok=True)
    out = os.path.join(city_dir, f"aoa_frac_{city}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (+ .pdf)")
    

LCZ_DIR = "../data_processing/grid_netcdf"
LCZ_NC  = "{city}_lcz.nc"

def _load_lcz_grid(city, want_shape):
    import xarray as xr
    ds = xr.open_dataset(os.path.join(LCZ_DIR, LCZ_NC.format(city=city)))
    cand = [v for v in ds.data_vars if "lcz" in v.lower()]
    var  = cand[0] if cand else list(ds.data_vars)[0]
    arr  = np.asarray(ds[var].values).squeeze()
    ds.close()
    if arr.shape != want_shape:
        raise ValueError(f"{city}: LCZ grid {arr.shape} != frac grid {want_shape} "
                         "— grids not aligned, check orientation before trusting stats.")
    return arr

def lcz_stats(city, frac, water_class=17, save=True):
    lcz = _load_lcz_grid(city, frac.shape)
    f, L = frac.ravel(), lcz.ravel().astype(float)
    ok = np.isfinite(f) & np.isfinite(L)
    df = pd.DataFrame({"lcz": L[ok].astype(int), "frac": f[ok]})

    per = (df.groupby("lcz")["frac"]
             .agg(mean_in_aoa="mean", median="median", n_cells="size")
             .reset_index().sort_values("lcz"))

    overall = df.loc[df["lcz"] != water_class, "frac"].mean()
    print(f"\n[lcz] {city}  mean in-AOA frac (excl water LCZ {water_class}): "
          f"{overall:.3f}")
    print(per.to_string(index=False))

    if save:
        out = os.path.join(OUT_DIR, city, f"aoa_lcz_stats_{city}_{CV_MODE}.csv")
        per.to_csv(out, index=False)
        print(f"[lcz] saved -> {out}")
    return per, overall

def _station_coords(city, table=URS_TABLE):
    """Unique station locations from the city's obs table, as (lon, lat)."""
    df = pd.read_pickle(os.path.join(DATA_DIR, table.format(city=city)))
    xc, yc = _find_xy(df)
    if xc is not None:                                   # EPSG:3035 -> lon/lat
        pts = (df[[xc, yc]].dropna().round().drop_duplicates().to_numpy(float))
        lon, lat = Transformer.from_crs("EPSG:3035", "EPSG:4326",
                                        always_xy=True).transform(pts[:, 0], pts[:, 1])
        return np.column_stack([lon, lat])
    if {"Longitude", "Latitude"}.issubset(df.columns):
        return df[["Longitude", "Latitude"]].dropna().round(5).drop_duplicates().to_numpy(float)
    raise KeyError(f"{city}: no coord columns ({XY_CANDIDATES} or Lon/Lat) in {table}")



def diagnose_station_vs_grid(aoa, city, day, station_row_idx=0):
    """Compare feature vectors from the URS table vs the grid build, for ONE
    station and ONE timestamp, to find which predictor(s) move the DI across
    the AOA threshold. Prints both vectors, their per-feature contribution to
    the DI distance, and the resulting DI / in-AOA verdict for each."""
    day = pd.Timestamp(day, tz="UTC")

    # --- station-table side: pick a station, get its coords + that day's rows -
    dft = pd.read_pickle(os.path.join(DATA_DIR, URS_TABLE.format(city=city)))
    dft = _align_features(dft, f"{city}-tbl")
    xc, yc = _find_xy(dft)                       # station coords in EPSG:3035
    # unique stations, pick one
    stns = dft[[xc, yc]].drop_duplicates().reset_index(drop=True)
    sx, sy = stns.loc[station_row_idx, [xc, yc]]
    print(f"[diag] station #{station_row_idx} at ({sx:.0f}, {sy:.0f}) EPSG:3035")

    # that station's row nearest `day` at, say, 03:00 UTC (night — UHI peak)
    ts = day + pd.Timedelta(hours=3)
    stn_rows = dft[(dft[xc] == sx) & (dft[yc] == sy)]
    # stn_rows = stn_rows.assign(_dt=(stn_rows["Time_UTC"] - ts).abs())
    stn_t = pd.to_datetime(stn_rows["Time_UTC"], utc=True)
    stn_rows = stn_rows.assign(_dt=(stn_t - ts).abs())
    trow = stn_rows.sort_values("_dt").iloc[0]
    ts_actual = trow["Time_UTC"]
    print(f"[diag] station table row at {ts_actual} (target {ts})")

    # --- grid side: build that day's forcing ----------------------------------
    ctx = _grid_context(city, day)
    dfg = _day_forcing(ctx, day.tz_localize(None))   #
    print(f"[diag] dfg rows: {len(dfg)}  (0 means the day filter matched nothing)")
    dfg = _align_features(dfg, f"{city}-grid")
    if dfg["Time_UTC"].dt.tz is None:
        dfg["Time_UTC"] = dfg["Time_UTC"].dt.tz_localize("UTC")
    ts_actual = pd.Timestamp(ts_actual)
    if ts_actual.tz is None:
        ts_actual = ts_actual.tz_localize("UTC")

    # station coords come straight off the station row (same CRS as dfg)
    gxc, gyc = _find_xy(dfg)                       # grid's coord col names
    print(f"[diag] station at ({sx:.0f}, {sy:.0f}) | grid coord cols: {gxc},{gyc}")

    # pick grid timestep nearest the station row, then nearest cell BY COORDS
    gt = pd.to_datetime(dfg["Time_UTC"], utc=True)
    t_target = gt.iloc[(gt - ts_actual).abs().values.argmin()]
    hour = dfg[gt == t_target].copy()
    gxy = hour[[gxc, gyc]].to_numpy(float)
    j = int(np.hypot(gxy[:, 0] - sx, gxy[:, 1] - sy).argmin())
    grow = hour.iloc[j]
    d_cell = float(np.hypot(gxy[j, 0] - sx, gxy[j, 1] - sy))
    print(f"[diag] matched grid cell ({gxy[j,0]:.0f}, {gxy[j,1]:.0f})  "
          f"{d_cell:.0f} m from station  |  grid time {grow['Time_UTC']} "
          f"(station {ts_actual})")
    if d_cell > 100:
        print(f"[diag] WARNING nearest grid cell is {d_cell:.0f} m away — "
              f"station may be outside the grid, or CRS mismatch")

    # --- side-by-side on the model features -----------------------------------
    comp = pd.DataFrame({
        "table": trow[features].astype(float),
        "grid":  grow[features].astype(float),
    })
    comp["diff"] = comp["grid"] - comp["table"]
    # per-feature contribution to the *weighted, standardized* distance
    z_tbl = (trow[features].astype(float).to_numpy() - aoa.scaler.mean_) / aoa.scaler.scale_ * aoa.weights
    z_grd = (grow[features].astype(float).to_numpy() - aoa.scaler.mean_) / aoa.scaler.scale_ * aoa.weights
    comp["contrib_to_gap"] = (z_grd - z_tbl) ** 2      # squared, since DI uses Euclidean

    pd.set_option("display.max_rows", None, "display.width", 140)
    print("\n[diag] features sorted by contribution to the table→grid distance gap:")
    print(comp.reindex(comp["contrib_to_gap"].abs().sort_values(ascending=False).index).round(4))

    # --- DI + verdict for each -------------------------------------------------
    di_tbl = aoa.score(pd.DataFrame([trow[features]]))["DI"].iloc[0]
    di_grd = aoa.score(pd.DataFrame([grow[features]]))["DI"].iloc[0]
    print(f"\n[diag] DI  table = {di_tbl:.3f}  ({'IN ' if di_tbl<=aoa.threshold else 'OUT'} AOA)")
    print(f"[diag] DI  grid  = {di_grd:.3f}  ({'IN ' if di_grd<=aoa.threshold else 'OUT'} AOA)")
    print(f"[diag] threshold DI* = {aoa.threshold:.3f}")
    return comp


# ================================ run ==================================
if __name__ == "__main__":
    MODE       = "grid"        # "stations" | "grid"
    # DATE_RANGE = ("2018-06-01", "2018-08-31")  
    DATE_RANGE = ("2025-06-01", "2025-08-31")# None = full table
    GRID_CITY  = "rennes"      # city whose grid to check (grid mode)

    AOA_CACHE = f"aoa_fit_{CV_MODE}.npz"
    if os.path.exists(AOA_CACHE):
        aoa = AOA.load(AOA_CACHE)
    else:
        aoa = AOA()
        aoa.save(AOA_CACHE)

    # diagnose_station_vs_grid(aoa, "thun", day="2025-07-15", station_row_idx=0)

    if MODE == "stations":
        analyze_stations(aoa, TEST_CITIES, DATE_RANGE)
    else:
        frac, _ = analyze_grid(aoa, GRID_CITY, date_range=DATE_RANGE)
        lcz_stats(GRID_CITY, frac)
        # z = np.load(os.path.join(OUT_DIR, GRID_CITY,
        #             f"aoa_gridcache_{GRID_CITY}_{CV_MODE}.npz"))
        # station_cell_aoa(GRID_CITY, frac, z["xs"], z["ys"])