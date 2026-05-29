"""
For each target city: train geo_3_6 model then immediately run xgb_d_check.
Works with jja2021, jja2020, last_year, spatial splits.
"""
import subprocess
import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from xgb_config import ModelConfig
from xgb_0_prep_params import MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES

# ── Configuration ──────────────────────────────────────────────────────────────
SPLIT_TYPE_RUN = 'jja2020'   # jja2021 | jja2020 | last_year | spatial
TARGET_CITIES_TO_RUN = None #['basel']  # None = all, or e.g. ['amsterdam', 'berlin']
N_EVAL   = 3
N_TRAIN  = 6
NO_PREJJA_CITIES = {'biel', 'freiburg'}
NO_LASTYEAR_CITIES = {'biel', 'bern'}
MODELS_DIR = 'geo36_models'  # where to save per-city models
TEST_WINDOWS = {
    'jja2021':   ('2021-07-15', '2021-09-15'),
    'jja2020':   ('2020-07-15', '2020-09-15'),
    'last_year': (None, None),  # xgb_d_check handles via its own cutoff logic
    'spatial':   (None, None),
}
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)

CITY_COUNTRIES = {
    'amsterdam': 'netherlands',  'basel': 'switzerland',
    'berlin': 'germany',         'bern': 'switzerland',
    'biel': 'switzerland',       'birmingham': 'united kingdom',
    'freiburg': 'germany',       'ghent': 'belgium',
    'novisad': 'serbia',         'rennes': 'france',
    'turku': 'finland',          'zurich': 'switzerland',
}


def compute_city_rankings(excel_path='climate_data.xlsx'):
    data_raw = pd.read_excel(excel_path, index_col=0)
    data_raw = data_raw.drop('samples')
    data = data_raw.T.reset_index().rename(columns={'index': 'city'})
    data['city'] = data['city'].str.capitalize().replace('Novisad', 'Novi Sad')
    X_scaled = StandardScaler().fit_transform(data.drop('city', axis=1))
    dist_matrix = euclidean_distances(X_scaled)
    cities = data['city'].values
    rankings = {}
    for i, city in enumerate(cities):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        sorted_idx = np.argsort(dists)[:-1]
        neighbors = [cities[j].lower().replace(' ', '') for j in sorted_idx]
        rankings[city.lower().replace(' ', '')] = neighbors
    return rankings


def select_eval_and_train(ranked, n_eval, n_train, min_samples, split_type, target_city=None):

    def has_eval_data(city):
        if split_type == 'last_year' and city in NO_LASTYEAR_CITIES:
            return False
        if split_type == 'jja2021':
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            tx = pd.read_pickle(path)
            return ((tx['Time_UTC'] >= pd.Timestamp('2021-05-15', tz='UTC')) &
                    (tx['Time_UTC'] <  pd.Timestamp('2021-07-15', tz='UTC'))).sum() >= min_samples
        elif split_type == 'jja2020':
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            tx = pd.read_pickle(path)
            return ((tx['Time_UTC'] >= pd.Timestamp('2020-05-15', tz='UTC')) &
                    (tx['Time_UTC'] <  pd.Timestamp('2020-07-15', tz='UTC'))).sum() >= min_samples
        elif split_type == 'last_year':
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            try:
                with open("../data_processing/city_test_cutoffs.json") as f:
                    cutoffs = json.load(f)
                cutoff = pd.Timestamp(cutoffs[target_city])
            except (FileNotFoundError, KeyError): return False
            cutoff_end = cutoff + pd.DateOffset(years=1)
            tx = pd.read_pickle(path)
            return ((tx['Time_UTC'] >= cutoff) & (tx['Time_UTC'] < cutoff_end)).sum() >= min_samples
        else:
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            return len(pd.read_pickle(path)) >= min_samples

    def has_train_data(city):
        if split_type == 'jja2021':
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            tx = pd.read_pickle(path)
            return (tx['Time_UTC'] < pd.Timestamp('2021-05-15', tz='UTC')).sum() >= min_samples
        elif split_type == 'jja2020':
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            tx = pd.read_pickle(path)
            return (tx['Time_UTC'] < pd.Timestamp('2020-05-15', tz='UTC')).sum() >= min_samples
        elif split_type == 'last_year':
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            try:
                with open("city_test_cutoffs.json") as f:
                    cutoffs = json.load(f)
                cutoff = pd.Timestamp(cutoffs[target_city])
            except (FileNotFoundError, KeyError): return False
            tx = pd.read_pickle(path)
            return (tx['Time_UTC'] < cutoff).sum() >= min_samples
        else:
            path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
            if not os.path.exists(path): return False
            return len(pd.read_pickle(path)) >= min_samples

    eval_cities, train_cities = [], []
    for city in ranked:
        if len(eval_cities) < n_eval and has_eval_data(city):
            eval_cities.append(city)
        elif len(train_cities) < n_train and has_train_data(city):
            train_cities.append(city)
        if len(eval_cities) == n_eval and len(train_cities) == n_train:
            break

    return eval_cities, train_cities


# CITY_RANKINGS = compute_city_rankings()
CITY_RANKINGS = {
    'amsterdam': ['rennes', 'birmingham', 'berlin', 'ghent', 'turku', 'basel', 'freiburg', 'zurich', 'biel', 'bern', 'novisad'],
    'basel': ['freiburg', 'zurich', 'bern', 'biel', 'rennes', 'berlin', 'turku', 'birmingham', 'ghent', 'amsterdam', 'novisad'],
    'berlin': ['birmingham', 'turku', 'rennes', 'ghent', 'basel', 'freiburg', 'amsterdam', 'zurich', 'biel', 'bern', 'novisad'],
    'bern': ['zurich', 'biel', 'freiburg', 'basel', 'turku', 'novisad', 'berlin', 'ghent', 'rennes', 'birmingham', 'amsterdam'],
    'biel': ['bern', 'zurich', 'freiburg', 'basel', 'turku', 'berlin', 'novisad', 'ghent', 'rennes', 'birmingham', 'amsterdam'],
    'birmingham': ['berlin', 'ghent', 'rennes', 'turku', 'amsterdam', 'freiburg', 'basel', 'zurich', 'novisad', 'biel', 'bern'],
    'freiburg': ['basel', 'zurich', 'biel', 'bern', 'berlin', 'turku', 'rennes', 'birmingham', 'ghent', 'novisad', 'amsterdam'],
    'ghent': ['turku', 'birmingham', 'berlin', 'novisad', 'rennes', 'freiburg', 'basel', 'zurich', 'amsterdam', 'biel', 'bern'],
    'novisad': ['turku', 'ghent', 'bern', 'biel', 'zurich', 'berlin', 'birmingham', 'freiburg', 'basel', 'rennes', 'amsterdam'],
    'rennes': ['berlin', 'amsterdam', 'birmingham', 'basel', 'freiburg', 'turku', 'ghent', 'zurich', 'biel', 'bern', 'novisad'],
    'turku': ['ghent', 'berlin', 'birmingham', 'novisad', 'freiburg', 'rennes', 'zurich', 'basel', 'bern', 'biel', 'amsterdam'],
    'zurich': ['bern', 'freiburg', 'basel', 'biel', 'turku', 'berlin', 'rennes', 'ghent', 'birmingham', 'novisad', 'amsterdam'],
}


    # 'basel': ['freiburg', 'zurich', 'biel', 'bern', 'rennes', 'berlin', 'turku', 'birmingham', 'ghent', 'amsterdam', 'novisad'],
    # 'bern': ['zurich', 'biel', 'freiburg', 'basel', 'turku', 'novisad', 'berlin', 'ghent', 'rennes', 'birmingham', 'amsterdam'],
    # 'biel': ['bern', 'zurich', 'freiburg', 'basel', 'turku', 'berlin', 'novisad', 'ghent', 'rennes', 'birmingham', 'amsterdam'],
    # 'zurich': ['bern', 'freiburg', 'basel', 'biel', 'turku', 'berlin', 'rennes', 'ghent', 'birmingham', 'novisad', 'amsterdam'],
    # 'fribourg': ['bern', 'biel', 'novisad', 'zurich', 'turku', 'freiburg', 'ghent', 'basel', 'berlin', 'birmingham', 'rennes', 'amsterdam'],
    # 'geneva':   ['bern', 'biel', 'zurich', 'novisad', 'freiburg', 'turku', 'basel', 'ghent', 'berlin', 'birmingham', 'rennes', 'amsterdam'],
    # 'lausanne': ['bern', 'biel', 'zurich','freiburg', 'basel', 'turku', 'novisad', 'berlin', 'ghent', 'birmingham', 'rennes', 'amsterdam'],
    # 'lugano':   ['bern', 'biel', 'zurich', 'novisad', 'freiburg', 'basel', 'turku', 'ghent', 'berlin', 'birmingham', 'rennes', 'amsterdam'],
    # 'luzern':   ['biel', 'bern', 'zurich', 'freiburg', 'basel', , 'turku', 'berlin', 'novisad', 'ghent', 'rennes', 'birmingham', 'amsterdam'],
    # 'stgallen': ['bern', 'biel', 'zurich', 'freiburg', 'basel', 'turku', 'berlin', 'novisad', 'ghent', 'birmingham', 'rennes', 'amsterdam'],
    # 'thun':     ['bern', 'biel','zurich', 'novisad', 'freiburg', 'turku', 'basel', 'ghent', 'berlin', 'birmingham', 'rennes', 'amsterdam'],
    # 'winterthur': ['bern', 'zurich', 'biel', 'freiburg', 'basel', 'turku', 'berlin', 'ghent', 'novisad', 'birmingham', 'rennes', 'amsterdam'],

def has_enough_data(city):
    path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
    if not os.path.exists(path):
        return False
    return len(pd.read_pickle(path)) >= MIN_TRAIN_SAMPLES


def run_script(script_path, description, env):
    print(f"\n{'='*60}\nRunning: {description}\n{'='*60}\n")
    try:
        subprocess.run([sys.executable, script_path],
                       check=True, env=env)
        print(f"\n✓ {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} (exit {e.returncode})\n")
        return False


def main():
    all_cities = list(CITY_RANKINGS.keys())

    if TARGET_CITIES_TO_RUN is not None:
        target_cities = [c.lower() for c in TARGET_CITIES_TO_RUN]
    else:
        if SPLIT_TYPE_RUN in ('jja2021', 'jja2020'):
            target_cities = [c for c in all_cities if c not in NO_PREJJA_CITIES]
        else:
            # for last_year and spatial, check target city has enough test data
            def has_enough_test_data(city):
                if SPLIT_TYPE_RUN == 'last_year':
                    if city in NO_LASTYEAR_CITIES:
                        return False
                    path = f"../data_processing/dataframes_ready/tablex_{city}_test_lastyear.pkl"
                else:  # spatial, jja2020
                    path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
                if not os.path.exists(path):
                    return False
                return len(pd.read_pickle(path)) >= MIN_TEST_SAMPLES

            target_cities = [c for c in all_cities if has_enough_test_data(c)]

    print(f"Split type : {SPLIT_TYPE_RUN}")
    print(f"Geo split  : {N_EVAL} eval / {N_TRAIN} train")
    print(f"Target cities: {target_cities}\n")

    results = []

    for target_city in target_cities:
        print(f"\n{'#'*70}")
        print(f"# {target_city.upper()}")
        print(f"{'#'*70}")

        # Build ranked cities for this split type
        ranked = CITY_RANKINGS[target_city].copy()
        if SPLIT_TYPE_RUN in ('jja2021', 'jja2020'):
            ranked = [c for c in ranked if c not in NO_PREJJA_CITIES]
        else:
            ranked = [c for c in ranked if has_enough_data(c)]

        if len(ranked) < N_EVAL + N_TRAIN:
            print(f"  ✗ Not enough cities ({len(ranked)} available, need {N_EVAL+N_TRAIN})")
            continue

        eval_cities, train_cities = select_eval_and_train(
            ranked, N_EVAL, N_TRAIN, MIN_TRAIN_SAMPLES, SPLIT_TYPE_RUN,
            target_city=target_city
        )

        if len(eval_cities) < N_EVAL or len(train_cities) < N_TRAIN:
            print(f"  ✗ Could not fill slots: {len(eval_cities)} eval, {len(train_cities)} train")
            continue
        model_path   = os.path.join(MODELS_DIR, f"xgb_geo36_{target_city}.json")

        print(f"  Train cities: {train_cities}")
        print(f"  Eval  cities: {eval_cities}")
        print(f"  Model will be saved to: {model_path}")

        ModelConfig.set_training_cities(train_cities, f'geo36_{target_city}')

        base_env = {
            **os.environ,
            'SPLIT_TYPE':              SPLIT_TYPE_RUN,
            'PIPELINE_TARGET_CITY':    target_city,
            'PIPELINE_TARGET_COUNTRY': CITY_COUNTRIES[target_city],
            'PIPELINE_EVAL_CITIES':    ','.join(eval_cities),
            'PIPELINE_MODEL_NAME':     model_path,
        }

        # ── Train ──────────────────────────────────────────────────────────
        train_ok = run_script('xgb_b_main.py',
                              f"Train geo_3_6 — {target_city}", base_env)
        if not train_ok:
            results.append({'city': target_city, 'train': False, 'test': False})
            continue

        # ── Test (xgb_d_check) ─────────────────────────────────────────────
        # Determine test path to confirm data exists
        test_path = f"../data_processing/dataframes_ready/tablex_{target_city}.pkl"

        if not os.path.exists(test_path):
            print(f"  Skipping test: {test_path} not found")
            results.append({'city': target_city, 'train': True, 'test': None})
            continue


        test_start, test_end = TEST_WINDOWS[SPLIT_TYPE_RUN]

        test_env = {
            **base_env,
            'PIPELINE_USE_JJA':    'true',
            'PIPELINE_MODEL_NAME': model_path,
            'PIPELINE_TEST_START': test_start or '',
            'PIPELINE_TEST_END':   test_end   or '',
        }
        test_ok = run_script('xgb_d_check.py',
                             f"Test geo_3_6 — {target_city}", test_env)

        results.append({'city': target_city, 'train': True, 'test': test_ok})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for r in results:
        t = '✓' if r['train'] else '✗'
        v = '✓' if r['test'] else ('–' if r['test'] is None else '✗')
        print(f"  {r['city']:<15} train={t}  test={v}")

    pd.DataFrame(results).to_csv('geo36_results.csv', index=False)
    print("\nSaved: geo36_results.csv")


if __name__ == "__main__":
    main()