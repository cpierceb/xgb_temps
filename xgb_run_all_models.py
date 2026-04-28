import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
from xgb_config import ModelConfig
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import json
import numpy as np
from xgb_0_prep_params import MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES



# Hardcoded city rankings from AOA analysis
# For each target city, list other cities in order of similarity (most similar first)
# CITY_RANKINGS = {
#     'amsterdam': ['birmingham', 'ghent', 'berlin', 'rennes', 'basel', 'freiburg', 'turku', 'zurich', 'biel', 'novisad', 'bern'],
#     'basel': ['freiburg', 'zurich', 'biel', 'bern', 'rennes', 'berlin', 'novisad', 'amsterdam', 'birmingham', 'ghent', 'turku'],
#     'berlin': ['amsterdam', 'birmingham', 'ghent', 'rennes', 'basel', 'freiburg', 'novisad', 'turku', 'zurich', 'biel', 'bern'],
#     'bern': ['zurich', 'biel', 'basel', 'freiburg', 'rennes', 'novisad', 'berlin', 'birmingham', 'amsterdam', 'ghent', 'turku'],
#     'biel': ['zurich', 'bern', 'basel', 'freiburg', 'rennes', 'berlin', 'novisad', 'birmingham', 'amsterdam', 'turku', 'ghent'],
#     'birmingham': ['amsterdam', 'ghent', 'berlin', 'rennes', 'basel', 'freiburg', 'turku', 'zurich', 'biel', 'novisad', 'bern'],
#     'freiburg': ['basel', 'zurich', 'biel', 'bern', 'rennes', 'berlin', 'amsterdam', 'birmingham', 'novisad', 'ghent', 'turku'],
#     'ghent': ['amsterdam', 'birmingham', 'berlin', 'rennes', 'basel', 'freiburg', 'novisad', 'zurich', 'biel', 'turku', 'bern'],
#     'novisad': ['basel', 'berlin', 'zurich', 'freiburg', 'rennes', 'biel', 'ghent', 'bern', 'birmingham', 'amsterdam', 'turku'],
#     'rennes': ['berlin', 'birmingham', 'amsterdam', 'basel', 'ghent', 'freiburg', 'zurich', 'biel', 'novisad', 'bern', 'turku'],
#     'turku': ['berlin', 'amsterdam', 'birmingham', 'basel', 'freiburg', 'rennes', 'zurich', 'biel', 'ghent', 'bern', 'novisad'],
#     'zurich': ['biel', 'bern', 'basel', 'freiburg', 'rennes', 'novisad', 'berlin', 'birmingham', 'amsterdam', 'ghent', 'turku'],
# }




JJA2021_START = pd.Timestamp('2021-05-15', tz='UTC')
JJA2021_END   = pd.Timestamp('2021-07-15', tz='UTC')
JJA2020_START = pd.Timestamp('2020-05-15', tz='UTC')
JJA2020_END   = pd.Timestamp('2020-07-15', tz='UTC')


def weighted_distance(training_cities, target_city):
    try:
        with open("../data_processing/city_row_counts.json") as f:
            row_counts = json.load(f)
    except FileNotFoundError:
        return None
    
    ti = _city_index[target_city.lower()]
    total_rows, total_weighted = 0, 0.0
    for city in training_cities:
        ci = _city_index.get(city.lower())
        n = row_counts.get(city, 0)
        if ci is None or n == 0:
            continue
        total_weighted += n * _dist_matrix[ci, ti]
        total_rows += n
    return total_weighted / total_rows if total_rows > 0 else None


def compute_city_rankings(excel_path='../data_processing/climate_data.xlsx'):
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
        sorted_idx = np.argsort(dists)[:-1]  # drop self
        neighbors = [cities[j].lower().replace(' ', '') for j in sorted_idx]
        rankings[city.lower().replace(' ',       '')] = neighbors
    return rankings, dist_matrix, cities

CITY_RANKINGS, _dist_matrix, _dist_cities = compute_city_rankings()

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

_city_index = {c.lower().replace(' ',''): i for i, c in enumerate(_dist_cities)}


CITY_COUNTRIES = {
    'amsterdam': 'netherlands',
    'basel': 'switzerland',
    'berlin': 'germany',
    'bern': 'switzerland',
    'biel': 'switzerland',
    'birmingham': 'united kingdom',
    'freiburg': 'germany',
    'ghent': 'belgium',
    'novisad': 'serbia',
    'rennes': 'france',
    'turku': 'finland',
    'zurich': 'switzerland',
}

SPLIT_TYPE_RUN = 'last_year'  # change here only   last_year   jja2021  spatial  jja2020
NO_PREJJA_CITIES = {'biel', 'freiburg'} 

GEO_SPLIT_CONFIGS = [
    {'name': 'geo_1_2', 'n_eval': 1, 'n_train': 2},
    {'name': 'geo_2_4', 'n_eval': 2, 'n_train': 4},
    {'name': 'geo_3_6', 'n_eval': 3, 'n_train': 6},
]

USE_GEO_SPLIT = True  # toggle here

# Add these two functions at module level, after compute_city_rankings()



def _load_base(city):
    path = f"../data_processing/dataframes_ready/tablex_{city}.pkl"
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)



def has_eval_data(city, split_type, target_city=None):
    tx = _load_base(city)
    if tx is None:
        return False
    if split_type == 'jja2021':
        return ((tx['Time_UTC'] >= JJA2021_START) & (tx['Time_UTC'] < JJA2021_END)).sum() >= MIN_TRAIN_SAMPLES
    elif split_type == 'jja2020':
        return ((tx['Time_UTC'] >= JJA2020_START) & (tx['Time_UTC'] < JJA2020_END)).sum() >= MIN_TRAIN_SAMPLES
    elif split_type == 'last_year':
        try:
            with open("../data_processing/city_test_cutoffs.json") as f:
                cutoff = pd.Timestamp(json.load(f)[target_city])
        except (FileNotFoundError, KeyError):
            return False
        val_start = cutoff - pd.DateOffset(years=1)
        return ((tx['Time_UTC'] >= val_start) & (tx['Time_UTC'] < cutoff)).sum() >= MIN_TRAIN_SAMPLES
    else:  # spatial
        return len(tx) >= MIN_TRAIN_SAMPLES

def has_train_data(city, split_type, target_city=None):
    tx = _load_base(city)
    if tx is None:
        return False
    if split_type == 'jja2021':
        return (tx['Time_UTC'] < JJA2021_START).sum() >= MIN_TRAIN_SAMPLES
    elif split_type == 'jja2020':
        return (tx['Time_UTC'] < JJA2020_START).sum() >= MIN_TRAIN_SAMPLES
    elif split_type == 'last_year':
        try:
            with open("../data_processing/city_test_cutoffs.json") as f:
                cutoff = pd.Timestamp(json.load(f)[target_city])
        except (FileNotFoundError, KeyError):
            return False
        val_start = cutoff - pd.DateOffset(years=1)
        return (tx['Time_UTC'] < val_start).sum() >= MIN_TRAIN_SAMPLES
    else:  # spatial
        return len(tx) >= MIN_TRAIN_SAMPLES


def select_eval_train_cities(ranked_cities, n_eval, n_train, split_type, target_city):
    """
    Walk ranked_cities in order. Fill eval slots first (requires eval data),
    then train slots (requires train data). A city can only appear in one slot.
    """
    eval_cities, train_cities = [], []
    for city in ranked_cities:
        if len(eval_cities) < n_eval and has_eval_data(city, split_type, target_city):
            eval_cities.append(city)
        elif len(train_cities) < n_train and has_train_data(city, split_type, target_city):
            train_cities.append(city)
        if len(eval_cities) == n_eval and len(train_cities) == n_train:
            break
    return eval_cities, train_cities


def has_enough_test_data(city):
    tx = _load_base(city)
    if tx is None:
        return False
    if SPLIT_TYPE_RUN == 'jja2021':
        n = ((tx['Time_UTC'] >= JJA2021_START) & (tx['Time_UTC'] < JJA2021_END)).sum()
    elif SPLIT_TYPE_RUN == 'jja2020':
        n = ((tx['Time_UTC'] >= JJA2020_START) & (tx['Time_UTC'] < JJA2020_END)).sum()
    elif SPLIT_TYPE_RUN == 'last_year':
        try:
            with open("../data_processing/city_test_cutoffs.json") as f:
                cutoff = pd.Timestamp(json.load(f)[city])
        except (FileNotFoundError, KeyError):
            return False
        n = (tx['Time_UTC'] >= cutoff).sum()
    else:  # spatial
        n = len(tx)
    return n >= MIN_TEST_SAMPLES
    

def has_enough_data(city):
    tx = _load_base(city)
    if tx is None:
        return False
    if SPLIT_TYPE_RUN == 'jja2021':
        return (tx['Time_UTC'] < JJA2021_START).sum() >= MIN_TRAIN_SAMPLES
    return len(tx) >= MIN_TRAIN_SAMPLES
    
            
def get_city_rankings(target_city):
    """
    Get hardcoded city rankings for a target city
    
    Args:
        target_city: The city to get rankings for (lowercase)
    
    Returns:
        List of cities ranked by similarity (most similar first)
    """
    target_city_lower = target_city.lower()
    
    if target_city_lower not in CITY_RANKINGS:
        raise ValueError(f"No rankings found for city: {target_city}. "
                        f"Available cities: {', '.join(CITY_RANKINGS.keys())}")
    
    return CITY_RANKINGS[target_city_lower].copy()


def run_script(script_path, description, target_city=None, extra_env=None):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}\n")
    
    try:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        if target_city:
            env['PIPELINE_TARGET_CITY'] = target_city
            env['PIPELINE_TARGET_COUNTRY'] = CITY_COUNTRIES[target_city.lower()]
            print(f"Setting PIPELINE_TARGET_CITY={target_city}")
        
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True,
            env=env
        )
        print(f"\n✓ Successfully completed: {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Exit code: {e.returncode}\n")
        return False


def main():

    INCLUDE_TARGET_IN_TRAINING = False  # Toggle: False = exclude target, True = include target as first training city
    
    TARGET_CITIES_TO_RUN = None  # None = all cities, or ['birmingham', 'amsterdam']

    # Get all cities from hardcoded rankings
    all_cities = list(CITY_RANKINGS.keys())


    # Filter to specified cities if configured
    if TARGET_CITIES_TO_RUN is not None:
        # Validate that specified cities exist
        invalid_cities = [c for c in TARGET_CITIES_TO_RUN if c.lower() not in all_cities]
        if invalid_cities:
            print(f"ERROR: The following cities were not found: {invalid_cities}")
            print(f"Available cities: {', '.join(all_cities)}")
            return
        all_target_cities = [c.lower() for c in TARGET_CITIES_TO_RUN]
        print(f"Running for SELECTED cities: {', '.join(all_target_cities)}")
    else:
        all_target_cities = all_cities
        if SPLIT_TYPE_RUN == 'jja2021':
            all_target_cities = [c for c in all_target_cities if c not in NO_PREJJA_CITIES]
        else:
            all_target_cities = [c for c in all_target_cities if has_enough_test_data(c)]
        print(f"Running for ALL {len(all_target_cities)} cities: {', '.join(all_target_cities)}")
    
    
    # Track overall results
    overall_results = []
    
    # Loop over each target city
    for target_city in all_target_cities:
        print(f"\n{'#'*70}")
        print(f"# TARGET CITY: {target_city.upper()}")
        print(f"{'#'*70}\n")
        




        ranked_cities = get_city_rankings(target_city)
        if SPLIT_TYPE_RUN in ('jja2021', 'jja2020'):
            ranked_cities = [c for c in ranked_cities if c not in NO_PREJJA_CITIES]
        # keep city if it has data for either role — selection logic handles the rest
        ranked_cities = [
            c for c in ranked_cities
            if has_eval_data(c, SPLIT_TYPE_RUN, target_city)
            or has_train_data(c, SPLIT_TYPE_RUN, target_city)
        ]
        # else:
        #     ranked_cities = [c for c in ranked_cities if has_enough_data(c)]

        # Add target city as first training city if configured
        if INCLUDE_TARGET_IN_TRAINING:
            ranked_cities = [target_city] + ranked_cities
        
        print(f"Cities ranked by similarity to {target_city}:")
        print(f"  Closest 5: {ranked_cities[:5]}")
        print(f"  Furthest 5: {ranked_cities[-5:]}")
        
        # Create models: 1 closest, 2 closest, 3 closest, ..., all 11
        if USE_GEO_SPLIT:
            models = []
            for cfg in GEO_SPLIT_CONFIGS:
                n_eval  = cfg['n_eval']
                n_train = cfg['n_train']

                # NEW: use period-aware selection instead of blind slicing
                eval_cities, train_cities = select_eval_train_cities(
                    ranked_cities, n_eval, n_train, SPLIT_TYPE_RUN, target_city
                )

                if len(eval_cities) < n_eval or len(train_cities) < n_train:
                    print(f"  Skipping {cfg['name']}: could only fill "
                          f"{len(eval_cities)}/{n_eval} eval, {len(train_cities)}/{n_train} train")
                    continue

                print(f"  {cfg['name']}: eval={eval_cities}, train={train_cities}")
                models.append({
                    'name':         cfg['name'],
                    'description':  f"geo split {n_eval} eval / {n_train} train",
                    'cities':       train_cities,
                    'eval_cities':  eval_cities,
                    'geo_split':    True,
                    'train_script': 'xgb_b_main.py',
                    'check_script': 'xgb_d_check.py',
                })
        else:
            models = []
            for n in range(1, len(ranked_cities) + 1):
                models.append({
                    'name':         f'cumulative_{n}',
                    'description':  f'cumulative {n} {"city" if n == 1 else "cities"}',
                    'cities':       ranked_cities[:n],
                    'eval_cities':  None,
                    'geo_split':    False,
                    'train_script': 'xgb_b_main.py',
                    'check_script': 'xgb_d_check.py',
                })
            for n, city_name in enumerate(ranked_cities, 1):
                models.append({
                    'name':         f'single_{n}',
                    'description':  f'single city {n} ({city_name})',
                    'cities':       [city_name],
                    'eval_cities':  None,
                    'geo_split':    False,
                    'train_script': 'xgb_b_main.py',
                    'check_script': 'xgb_d_check.py',
                })
                
        print(f"\n{'='*70}")
        print(f"Training {len(models)} models for {target_city.upper()}")
        print(f"{'='*70}\n")
        
        extra_env = {'SPLIT_TYPE': SPLIT_TYPE_RUN}
        
        city_results = []
        
        # Train each model configuration
        for i, model in enumerate(models, 1):
            print(f"\n{'*'*60}")
            print(f"* MODEL {i}/{len(models)}: {model['description']}")
            print(f"* Cities: {', '.join(model['cities'][:5])}{'...' if len(model['cities']) > 5 else ''}")
            print(f"{'*'*60}")
            
            # Set configuration
            ModelConfig.set_training_cities(model['cities'], model['name'])



            geo_env = {}
            if model.get('geo_split') and model['eval_cities']:
                geo_env['PIPELINE_EVAL_CITIES'] = ','.join(model['eval_cities'])

            train_success = run_script(
                model['train_script'],
                f"Training model {i} - {model['description']}",
                target_city=target_city,
                extra_env={**extra_env, **geo_env}
            )
            
            if not train_success:
                print(f"\n⚠ Training failed for model {i}. Skipping to next model.")
                city_results.append({
                    'target_city': target_city,
                    'model': model['name'],
                    'description': model['description'],
                    'n_cities': len(model['cities']),
                    'train_success': False,
                    'test_success': False
                })
                continue

            wd = weighted_distance(model['cities'], target_city)
            print(f"  Weighted euclidean distance to {target_city}: {wd:.4f}" if wd else "  Could not compute weighted distance")


            if model.get('geo_split'):
                print(f"  Geo split: eval metrics reported during training, skipping test.")
                test_success = None
            else:
                if SPLIT_TYPE_RUN == 'jja2021':
                    test_suffix = '_jja2021'
                elif SPLIT_TYPE_RUN == 'last_year':
                    test_suffix = '_test_lastyear'
                else:  # spatial, jja2020
                    test_suffix = ''  # full pickle

                test_path = f"../data_processing/dataframes_ready/tablex_{target_city}{test_suffix}.pkl"
                if os.path.exists(test_path) and pd.read_pickle(test_path).shape[0] > 0:
                    test_success = run_script(
                        'xgb_d_check.py',
                        f"Test - {model['description']}",
                        target_city=target_city,
                        extra_env={**extra_env, 'PIPELINE_USE_JJA': 'true'}
                    )
                else:
                    print(f"  Skipping test for {target_city}: no test data")
                    test_success = None
                
                city_results.append({
                    'target_city': target_city,
                    'model': model['name'],
                    'description': model['description'],
                    'n_cities': len(model['cities']),
                    'train_success': train_success,
                    # 'check_success': check_success
                    'test_success': test_success
                })

            
            # if not check_success:
            #     print(f"\n⚠ Evaluation failed for model {i}. Continuing to next model...")
        
        # Summary for this city
        print(f"\n\n{'='*70}")
        print(f"SUMMARY FOR {target_city.upper()}")
        print(f"{'='*70}\n")
        
        for result in city_results:
            status = "✓" if result['train_success'] and result['test_success'] else "✗"
            print(f"{status} {result['description']}: Train={'✓' if result['train_success'] else '✗'}, Test={'✓' if result.get('test_success') else '✗'}")
        
        successes = sum(1 for r in city_results if r['train_success'] and r.get('test_success'))
        print(f"\nCompleted: {successes}/{len(city_results)} models\n")
        
        overall_results.extend(city_results)
    
    # Final overall summary
    print(f"\n\n{'#'*70}")
    print("# FINAL SUMMARY")
    print(f"{'#'*70}\n")
    
    for target_city in all_target_cities:
        city_res = [r for r in overall_results if r['target_city'] == target_city]
        successes = sum(1 for r in city_res if r['train_success'] and r['test_success'])
        print(f"{target_city.upper()}: {successes}/{len(city_res)} models successful")
    
    total_success = sum(1 for r in overall_results if r['train_success'] and r['test_success'])
    print(f"\nOVERALL: {total_success}/{len(overall_results)} models successful")
    
    # Save results to CSV
    results_df = pd.DataFrame(overall_results)
    results_df.to_csv('pipeline_results.csv', index=False)
    print(f"\nDetailed results saved to: pipeline_results.csv\n")


if __name__ == "__main__":
    main()