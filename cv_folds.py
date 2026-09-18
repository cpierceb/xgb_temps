"""
Shared CV fold construction for both tuning and the full run.
CV_MODE switches between 12-fold LOCO and 6-fold spatio-temporal LOCO.
The leakage-critical train/val separation lives ONLY here.
"""
import numpy as np
from xgb_0_prep_params import features
from xgb_d_functions import load_split_xy
from xgb_b_make_model import preprocess

CV_CITIES = ['amsterdam', 'basel', 'berlin', 'bern', 'biel', 'birmingham',
             'freiburg', 'ghent', 'novisad', 'rennes', 'turku', 'zurich']
TEST_CITIES = ['lausanne', 'thun', 'winterthur']

CITY_COUNTRIES = {
    'amsterdam':'netherlands','basel':'switzerland','berlin':'germany','bern':'switzerland',
    'biel':'switzerland','birmingham':'united kingdom','freiburg':'germany','ghent':'belgium',
    'novisad':'serbia','rennes':'france','turku':'finland','zurich':'switzerland',
    'lausanne':'switzerland','thun':'switzerland','winterthur':'switzerland',
}

BUDGET, ES_ROUNDS = 5000, 50          # keep identical in tuner AND runner

# 6-fold spatio-temporal fold definitions (city pair + held-out year block)
ST_FOLDS = [
    {'id': 'st1', 'cities': ['amsterdam', 'novisad'],  'years': [2014, 2015, 2016, 2017]},
    {'id': 'st2', 'cities': ['ghent', 'rennes'],       'years': [2018]},
    {'id': 'st3', 'cities': ['turku', 'zurich'],       'years': [2019, 2020]},
    {'id': 'st4', 'cities': ['basel', 'birmingham'],   'years': [2021]},
    {'id': 'st5', 'cities': ['berlin', 'bern'],        'years': [2022]},
    {'id': 'st6', 'cities': ['freiburg', 'biel'],      'years': [2023]},
]

def get_folds(mode):
    return list(CV_CITIES) if mode == '12fold_loco' else ST_FOLDS

def fold_id(mode, fold):
    return fold if mode == '12fold_loco' else fold['id']


def load_cv_data():
    """Load each CV city ONCE. Returns {city: (X, y, year)} numpy, row-aligned.
    Assumes spatial 'train' == full table (all years) — needed for year filtering."""
    data = {}
    for c in CV_CITIES:
        tx, ty = load_split_xy(c, 'spatial', 'train', target_city=c)
        Xdf, ydf, ref = preprocess([tx], [ty])
        year = ref['Time_UTC'].dt.year.values
        data[c] = (Xdf.to_numpy(), ydf.to_numpy(), year)
        print(f"  loaded {c}: {data[c][0].shape}, years {year.min()}–{year.max()}")
    return data

def build_all(data):
    """Full pooled X, y across all CV_CITIES — all cities, all years, no split.
    For the final descriptive model (in-sample UHI / grid maps)."""
    Xall = np.vstack([data[c][0] for c in CV_CITIES])
    yall = np.vstack([data[c][1] for c in CV_CITIES])
    return Xall, yall

def build_fold(mode, fold, data):
    """Return X_train, y_train, X_val, y_val for one fold. Only place filtering happens."""
    if mode == '12fold_loco':
        held = fold
        Xv, yv, _ = data[held]
        Xt = np.vstack([data[c][0] for c in CV_CITIES if c != held])
        yt = np.vstack([data[c][1] for c in CV_CITIES if c != held])
        return Xt, yt, Xv, yv

    # 6-fold spatio-temporal
    vc, vy = set(fold['cities']), set(fold['years'])
    # VAL = held cities × held years
    Xv, yv = [], []
    for c in fold['cities']:
        X, y, yr = data[c]
        m = np.isin(yr, list(vy))                      # ← held years only
        Xv.append(X[m]); yv.append(y[m])
    Xv, yv = np.vstack(Xv), np.vstack(yv)
    # TRAIN = other 10 cities × all years EXCEPT held years
    Xt, yt = [], []
    for c in CV_CITIES:
        if c in vc:                                     # ← held cities excluded entirely
            continue
        X, y, yr = data[c]
        m = ~np.isin(yr, list(vy))                     # ← held years masked from training
        Xt.append(X[m]); yt.append(y[m])
    Xt, yt = np.vstack(Xt), np.vstack(yt)
    return Xt, yt, Xv, yv