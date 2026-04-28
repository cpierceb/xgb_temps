import pandas as pd
from xgb_b_make_model import *
from xgb_config import ModelConfig
import time
import json


if __name__ == "__main__":
    model_type = "temperature"
    print("---------1. Loading tables-----------------")
    
    # Get which cities to train on from config
    cities_to_train = ModelConfig.get_training_cities()

    # cities_to_train = [
    #     # 'amsterdam',
    #     'basel',  
    #     'berlin',
    #     # 'bern',
    #     'biel',
    #     # 'birmingham',
    #     'freiburg',
    #     # 'ghent',
    #     # 'novisad',
    #     'rennes',
    #     # 'turku',
    #     'zurich',
    # ] 

    ModelConfig.set_training_cities(cities_to_train, 'my_best_model')

    model_name = ModelConfig.get_model_name()
    
    print(f"\nConfiguration: {model_name}")
    print(f"Training on {len(cities_to_train)} cities: {', '.join(cities_to_train)}")
    
    # Load only the specified cities
    selected_tablex, selected_tabley = [], []

    
    # for city_name in cities_to_train:
    #     if     model_type == "humidity":
    #         tablex = pd.read_pickle(f"dataframes_ready/tablex_{city_name}_rh.pkl")
    #         tabley = pd.read_pickle(f"dataframes_ready/tabley_{city_name}_rh.pkl")
    #     else: 
    #         tablex = pd.read_pickle(f"dataframes_ready/tablex_{city_name}.pkl")
    #         tabley = pd.read_pickle(f"dataframes_ready/tabley_{city_name}.pkl")
    #     selected_tablex.append(tablex)
    #     selected_tabley.append(tabley)
    #     print(f"  Loaded: {city_name}")

    city_row_counts = {}

    JJA2021_START = pd.Timestamp('2021-05-15', tz='UTC')
    JJA2021_END   = pd.Timestamp('2021-07-15', tz='UTC')
    JJA2020_START = pd.Timestamp('2020-05-15', tz='UTC')  # already used as JJA2020_TRAIN_CUTOFF

    split_type = os.environ.get('SPLIT_TYPE', SPLIT_TYPE)
    JJA2020_TRAIN_CUTOFF = pd.Timestamp('2020-05-15', tz='UTC')

    if split_type == 'last_year':
        target_city = os.environ.get('PIPELINE_TARGET_CITY', '').lower()
        with open("city_test_cutoffs.json") as f:
            cutoffs = json.load(f)
        train_cutoff = pd.Timestamp(cutoffs[target_city])
    elif split_type == 'jja2020':
        train_cutoff = JJA2020_TRAIN_CUTOFF

    # Geo-split: separate eval cities passed via env
    eval_cities_env = os.environ.get('PIPELINE_EVAL_CITIES', '')
    eval_cities = [c.strip() for c in eval_cities_env.split(',') if c.strip()]
    geo_split = len(eval_cities) > 0

    def load_city_data(city_name, is_eval=False):
        """Load and temporally slice data for a city based on split_type."""
        if split_type == 'jja2021':
            tx_full = pd.read_pickle(f"dataframes_ready/tablex_{city_name}.pkl")
            ty_full = pd.read_pickle(f"dataframes_ready/tabley_{city_name}.pkl")
            if is_eval:
                mask = (tx_full['Time_UTC'] >= JJA2021_START) & (tx_full['Time_UTC'] < JJA2021_END)
            else:
                mask = tx_full['Time_UTC'] < JJA2021_START
            tx = tx_full[mask].reset_index(drop=True)
            ty = ty_full[mask].reset_index(drop=True)
            period = "JJA2021" if is_eval else f"pre-{JJA2021_START.date()}"
            print(f"  {city_name}: {period} ({len(tx):,} rows)")

        elif split_type == 'jja2020':
            tx_full = pd.read_pickle(f"dataframes_ready/tablex_{city_name}.pkl")
            ty_full = pd.read_pickle(f"dataframes_ready/tabley_{city_name}.pkl")
            if is_eval:
                # Validation: only JJA 2020 (Jun–Aug 2020)
                JJA2020_END = pd.Timestamp('2020-07-15', tz='UTC')
                mask = (tx_full['Time_UTC'] >= JJA2020_TRAIN_CUTOFF) & (tx_full['Time_UTC'] < JJA2020_END)
            else:
                # Training: everything before JJA 2020
                mask = tx_full['Time_UTC'] < JJA2020_TRAIN_CUTOFF
            tx = tx_full[mask].reset_index(drop=True)
            ty = ty_full[mask].reset_index(drop=True)
            period = "JJA2020" if is_eval else f"pre-{JJA2020_TRAIN_CUTOFF.date()}"
            print(f"  {city_name}: {period} ({len(tx):,} rows)")

        elif split_type == 'last_year':
            tx_full = pd.read_pickle(f"dataframes_ready/tablex_{city_name}.pkl")
            ty_full = pd.read_pickle(f"dataframes_ready/tabley_{city_name}.pkl")
            if is_eval:
                cutoff_end = train_cutoff + pd.DateOffset(years=1)
                mask = (tx_full['Time_UTC'] >= train_cutoff) & (tx_full['Time_UTC'] < cutoff_end)
            else:
                mask = tx_full['Time_UTC'] < train_cutoff
            tx = tx_full[mask].reset_index(drop=True)
            ty = ty_full[mask].reset_index(drop=True)
            period = f">= {train_cutoff.date()}" if is_eval else f"< {train_cutoff.date()}"
            print(f"  {city_name}: data {period} ({len(tx):,} rows)")

        else:  # spatial
            tx = pd.read_pickle(f"dataframes_ready/tablex_{city_name}.pkl")
            ty = pd.read_pickle(f"dataframes_ready/tabley_{city_name}.pkl")

        return tx, ty

    # Load training cities
    selected_tablex, selected_tabley = [], []
    for city_name in cities_to_train:
        tablex, tabley = load_city_data(city_name, is_eval=False)
        if tablex.empty or len(tablex) < MIN_TRAIN_SAMPLES:
            print(f"  WARNING: {city_name} insufficient train data, skipping")
            continue
        n = len(tablex)
        city_row_counts[city_name] = n
        print(f"  Loaded train: {city_name} ({n:,} rows)")
        selected_tablex.append(tablex)
        selected_tabley.append(tabley)

    with open("city_row_counts.json", "w") as f:
        json.dump(city_row_counts, f)

    # Load eval cities if geo split
    selected_evalx, selected_evaly = [], []
    if geo_split:
        print(f"\nLoading eval cities: {eval_cities}")
        for city_name in eval_cities:
            ex, ey = load_city_data(city_name, is_eval=True)
            if ex.empty or len(ex) < MIN_TRAIN_SAMPLES:
                print(f"  WARNING: eval city {city_name} insufficient data, skipping")
                continue
            print(f"  Loaded eval:  {city_name} ({len(ex):,} rows)")
            selected_evalx.append(ex)
            selected_evaly.append(ey)

    print("---------2. Preprocessing tables-----------")
    tablex, tabley, tablex_ref = preprocess(selected_tablex, selected_tabley)

    print("---------3. Splitting / training-----------")
    start = time.time()

    if geo_split:
        evalx_combined, evaly_combined, evalx_ref = preprocess(selected_evalx, selected_evaly)

        X_train = tablex.to_numpy()
        y_train = tabley.to_numpy()
        X_val   = evalx_combined.to_numpy()
        y_val   = evaly_combined.to_numpy()

        print(f"Geo split — X_train: {X_train.shape}, X_val: {X_val.shape}")

        print("---------4. Training model-----------------")
        model, y_pred = train(X_train, X_val, y_train, y_val, use_stacking=False)

        end = time.time()
        print(f"Elapsed time: {end - start:.2f}s")

        # Eval metrics on geo eval set
        y_val_abs  = y_val[:, 0]  + evalx_combined['t2m'].values
        y_pred_abs = y_pred        + evalx_combined['t2m'].values
        rmse = np.sqrt(np.mean((y_pred_abs - y_val_abs) ** 2))
        bias = np.mean(y_pred_abs - y_val_abs)
        r2   = 1 - np.sum((y_val_abs - y_pred_abs)**2) / np.sum((y_val_abs - y_val_abs.mean())**2)
        print(f"\nGeo-split eval RMSE: {rmse:.4f} K  |  Bias: {bias:.4f} K  |  R²: {r2:.4f}")

        tablez = evalx_combined.copy()
        tablez['y_obs']    = y_val_abs
        tablez['y_pred']   = y_pred_abs
        tablez['residual'] = tablez['y_pred'] - tablez['y_obs']

    else:
        X_train, X_val, y_train, y_val, tablex, tabley, test_idx = split(tablex, tabley, tablex_ref)

        USE_CAPPING = False
        MAX_OBS_PER_BIN = int(1e4)
        if USE_CAPPING:
            X_train, y_train, keep_mask = cap_observations_per_bin(
                X_train, y_train, tablex_ref, test_idx,
                max_obs_per_bin=MAX_OBS_PER_BIN, random_state=42
            )
            all_indices = np.arange(len(tablex_ref))
            train_mask = np.ones(len(tablex_ref), dtype=bool)
            train_mask[test_idx] = False
            train_t2m_capped = tablex_ref.loc[train_mask, "t2m"].values[keep_mask]
            bin_counts_after = analyze_training_data_distribution(y_train, train_t2m_capped)

        tablez = tablex.iloc[test_idx].copy()
        end = time.time()
        print(f"Elapsed time for splitting: {end - start:.6f} seconds")

        print("---------4. Training model-----------------")
        start = time.time()
        model, y_pred = train(X_train, X_val, y_train, y_val, use_stacking=False)
        end = time.time()
        print(f"Elapsed time for training: {end - start:.6f} seconds")

        tablez['y_obs']    = y_val[:, 0] + tablez["t2m"]
        tablez['y_pred']   = y_pred      + tablez["t2m"]
        tablez['residual'] = tablez['y_pred'] - tablez['y_obs']

    print("---------5. Plotting results---------------")
    importance_comparison(model)
    importance(model)

    density_scatter_hex_norm(tablez)

    print("---------6. Analysis of residuals----------")

    print(f"\n{'='*60}")
    print(f"Model training complete: {model_name}")
    print(f"{'='*60}\n")