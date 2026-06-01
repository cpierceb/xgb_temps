import pandas as pd
from xgb_b_make_model import *
from xgb_config import ModelConfig
import time
import json
from xgb_d_functions import load_split_xy



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
    split_type = os.environ.get('SPLIT_TYPE', SPLIT_TYPE)
    target_city = os.environ.get('PIPELINE_TARGET_CITY', '').lower() or None

    # Geo-split: separate eval cities passed via env
    eval_cities_env = os.environ.get('PIPELINE_EVAL_CITIES', '')
    eval_cities = [c.strip() for c in eval_cities_env.split(',') if c.strip()]
    geo_split = len(eval_cities) > 0

    # Load training cities
    selected_tablex, selected_tabley = [], []
    for city_name in cities_to_train:
        try:
            tx, ty = load_split_xy(city_name, split_type, 'train', target_city=target_city)
            print(f"  Loaded train: {city_name} ({len(tx):,} rows)")
            city_row_counts[city_name] = len(tx)
            selected_tablex.append(tx)
            selected_tabley.append(ty)
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: {city_name} skipped — {e}")

    with open("city_row_counts.json", "w") as f:
        json.dump(city_row_counts, f)

    # Load eval cities
    selected_evalx, selected_evaly = [], []  # ← must be initialized before the if
    if geo_split:
        print(f"\nLoading eval cities: {eval_cities}")
        for city_name in eval_cities:
            try:
                ex, ey = load_split_xy(city_name, split_type, 'val', target_city=target_city)
                print(f"  Loaded eval: {city_name} ({len(ex):,} rows)")
                selected_evalx.append(ex)
                selected_evaly.append(ey)
            except (FileNotFoundError, ValueError) as e:
                print(f"  WARNING: eval {city_name} skipped — {e}")

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

        target_city_env = os.environ.get('PIPELINE_TARGET_CITY', 'unknown')
        val_out = f"../data_processing/dataframes_ready/tablez_val_{target_city_env}.pkl"
        tablez.to_pickle(val_out)
        print(f"  Saved val set to {val_out}")

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