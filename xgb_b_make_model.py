import pandas as pd
from xgb_0_prep_params import *
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor as _XGBRegressor
from sklearn.base import BaseEstimator
import time
import seaborn as sns
from sklearn.utils import shuffle
import cmcrameri.cm as cmcr

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size']= 11





def preprocess(tablex_list,tabley_list):
    #----------------------1. Load arrays, combine & filter---------------------------------------------------
    # Load the arrays
    """
    tablex_list: 
    tabley_list: 
    """
    # if len(tablex_list) != len(tabley_list):
    #     raise ValueError("You must pass the same number of tablex and tabley DataFrames")

    # 1. Concatenate all inputs
    tablex = pd.concat(tablex_list, axis=0, ignore_index=True)
    tabley = pd.concat(tabley_list, axis=0, ignore_index=True)

    print(f"Combined X shape: {tablex.shape}")
    print(f"Combined Y shape: {tabley.shape}")

    tablex["LCZ_100"] = tablex["LCZ_100"].fillna(17)

    # this was commented out on 04.12.2025 because i wanted to run just for humidity and QC should be fine
    # # 1. Build your boolean mask on the DataFrame
    # valid_temp = tabley["Temperature"] >= -30
    # valid_rows = valid_temp.copy()

    # if onlyt == 0:
    #     valid_hum = (tabley["Humidity"] >= 0) & (tabley["Humidity"] <= 100)
    #     valid_rows &= valid_hum

    # print(f"Original number of rows: {tabley.shape[0]}")

    # # 2a. Filter the DataFrame
    # tabley_filtered = tabley[valid_rows].copy()
    # tablex_filtered = tablex[valid_rows].copy()

    # print(f"Number of rows after plausible value check y: {tabley_filtered.shape[0]}")
    # print(f"Number of rows after plausible value check x: {tablex_filtered.shape[0]}")

    # tablex, tabley = tablex_filtered, tabley_filtered

    if tablex['Time_UTC'].dt.tz is not None:
        tablex['Time_UTC'] = tablex['Time_UTC'].dt.tz_localize(None)

    tablex_ref = tablex.copy()
    tablex = tablex[features].copy()

    return tablex,tabley,tablex_ref



def split(tablex,tabley,tablex_ref):
    # We'll use X_full which is the DataFrame before conversion to numpy
    tablex_ref = tablex_ref.sort_values(by=['Latitude', 'Longitude'])
    # Now get the corresponding Y values (need to maintain same order)
    sorted_indices = tablex_ref.index

    tablex = tablex.loc[sorted_indices]
    tabley = tabley.loc[sorted_indices]

    X = tablex.to_numpy()
    Y = tabley.to_numpy()



    # 1. First, extract unique station IDs (assuming each lat/lon pair is a station)
    tablex_ref['station_id'] = tablex_ref.apply(lambda x: f"{x['Latitude']:.4f}_{x['Longitude']:.4f}", axis=1)

    num_stations = tablex_ref['station_id'].nunique()
    num_test = int(num_stations*0.3)
    # 2. Use GroupShuffleSplit to leave out ENTIRE stations
    gss = GroupShuffleSplit(n_splits=1, test_size=num_test, random_state=42)  

    #X_df = X.copy()

    # 3. Split the data
    for train_idx, test_idx in gss.split(X, Y, groups=tablex_ref['station_id']):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = Y[train_idx], Y[test_idx]


    # Verify
    print(f"Train stations: {len(np.unique(tablex_ref.iloc[train_idx]['station_id']))}")
    print(f"Test stations: {len(np.unique(tablex_ref.iloc[test_idx]['station_id']))}")

    # Optional: Save station IDs to inspect later
    held_out_stations = tablex_ref.iloc[test_idx]['station_id'].unique()
    print("Held-out stations:", held_out_stations)

    # Convert to float32 as you had before
    # X_train = X_train.astype(np.float32)
    # X_val = X_val.astype(np.float32)
    # y_train = y_train.astype(np.float32)
    # y_val = y_val.astype(np.float32)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")

    return X_train, X_val, y_train, y_val, tablex, tabley, test_idx

from sklearn.model_selection import train_test_split

def split_by_rows(tablex, tabley, test_size=0.3, random_state=42):
    """
    Splits the data by randomly sampling rows (not stations).
    Returns X_train, X_val, y_train, y_val, tablex_sorted, tabley_sorted, test_idx.
    """
    # 1. (Optional) sort or reindex if you need reproducibility
    tablex_sorted = tablex.sort_index()
    tabley_sorted = tabley.loc[tablex_sorted.index]

    # 2. Convert to numpy arrays
    X = tablex_sorted.to_numpy()
    Y = tabley_sorted.to_numpy()

    # 3. Generate train/test row indices
    all_idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 4. Slice arrays
    X_train, X_val = X[train_idx], X[test_idx]
    y_train, y_val = Y[train_idx], Y[test_idx]

    # 5. (Optional) print shapes
    print(f"Total rows: {len(X)}")
    print(f"Train rows: {len(train_idx)}, Val rows: {len(test_idx)}")
    print(f"Shape of X_train: {X_train.shape}, X_val: {X_val.shape}")

    return X_train, X_val, y_train, y_val, tablex_sorted, tabley_sorted, test_idx


def stratified_split(tablex, tabley, test_frac=0.3, random_state=42):
    # Sort by lat/lon and reset index so .iloc works correctly
    tablex = tablex.sort_values(by=["Latitude", "Longitude"]).reset_index(drop=True)
    tabley = tabley.reset_index(drop=True)

    # Create station_id from lat/lon
    tablex['station_id'] = tablex.apply(
        lambda row: f"{row['Latitude']:.4f}_{row['Longitude']:.4f}", axis=1
    )

    # Build a unique mapping of station -> LCZ
    station_lcz = tablex[['station_id', 'LCZ']].drop_duplicates()

    # Prepare RNG for reproducibility
    rng = np.random.RandomState(random_state)
    test_station_ids = []

    # For each LCZ category, sample stations
    for lcz in station_lcz['LCZ'].unique():
        ids = station_lcz.loc[station_lcz['LCZ'] == lcz, 'station_id'].values
        n_test = int(np.ceil(len(ids) * test_frac))
        chosen = rng.choice(ids, size=n_test, replace=False)
        test_station_ids.extend(chosen)

    # Create boolean mask for test stations
    mask_test = tablex['station_id'].isin(test_station_ids).values
    indices = np.arange(len(tablex))
    test_idx = indices[mask_test]
    train_idx = indices[~mask_test]

    # Split feature/target arrays
    X = tablex.drop(columns=['station_id']).to_numpy()
    Y = tabley.to_numpy()
    X_train, X_val = X[train_idx], X[test_idx]
    y_train, y_val = Y[train_idx], Y[test_idx]

    # Diagnostics using positional indexing
    train_stations = np.unique(tablex.iloc[train_idx]['station_id'])
    test_stations  = np.unique(tablex.iloc[test_idx]['station_id'])
    print(f"Train stations: {len(train_stations)}")
    print(f"Test stations:  {len(test_stations)}")
    print("Held-out stations:", test_stations)

    # LCZ distributions
    full_dist = tablex['LCZ'].value_counts(normalize=True).sort_index()
    test_dist = tablex.iloc[test_idx]['LCZ'].value_counts(normalize=True).sort_index()
    print("LCZ proportions in full data:\n", full_dist)
    print("LCZ proportions in test set:\n", test_dist)

    return X_train, X_val, y_train, y_val, tablex, tabley, test_idx



# def train(X_train, X_val, y_train, y_val):
#     # model = xgb.XGBRegressor(
#     #     objective="reg:squarederror",
#     #     eval_metric="rmse",
#     #     #n_estimators=1000,  
#     #     eta=0.05, 
#     #     #reg_lambda = 1.2,
#     #     #reg_alpha = 0.2,
#     #     max_depth=50,
#     #     # reg_alpha  = 0.2,
#     #     #   scale_pos_weight = 1,
#     #     colsample_bytree = 0.75,
#     #     subsample = 0.95,
#     #     gamma = 0.1,
#     #     min_child_weight = 5,
#     #     early_stopping_rounds=15,
#     # )


#         # In xgb_0_prep_params.py:
#     #alpha = 0.9  
#     params = {
#         "objective": "reg:squarederror",

#         "eval_metric": "rmse",
#         "eta": 0.85,
#         "max_depth": 100,
#         "colsample_bytree": 0.75,
#         "subsample": 0.95,
#         "gamma": 0.1,
#         "min_child_weight": 5,
#     }

#     # DX_train = xgb.DMatrix(X_train, feature_names=features)
#     # Dy_train = xgb.DMatrix(y_train)
#     # DX_val = xgb.DMatrix(X_val, feature_names=features)
#     # Dy_val = xgb.DMatrix(y_val)

#     dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
#     dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=features)

#     watchlist = [(dtrain, "train"), (dval, "valid")]

#     start = time.time()
#     model = xgb.train(
#         params,
#         dtrain,
#         num_boost_round=1000,
#         evals=watchlist,
#         early_stopping_rounds=10,
#         verbose_eval=5,
#     )
#     end = time.time()
#     print(f"Elapsed time for training: {end - start:.6f} seconds")



#     print("done training!")

#     # Step 4: Make predictions on the validation set
#     # y_pred = model.predict(DX_val)
#     # Predict
#     y_pred = model.predict(dval)

#     print("done predicting!")

#     model.save_model("xgb_model_single.json")
#     print("Model saved!")

#     return model, y_pred

def train_quantile_regression(X_train, X_val, y_train, y_val, features, quantiles=[0.05, 0.5, 0.95]):
    """
    Train single XGBoost model that predicts multiple quantiles
    
    Returns:
    --------
    model : xgb.Booster
        Trained model
    predictions : np.ndarray
        Shape (n_samples, n_quantiles) - predictions for each quantile
    """
    
    params = {
        "eta": 0.3,
        "max_depth": 6,
        "colsample_bytree": 0.75,
        "subsample": 0.95,
        "gamma": 0.1,
        "min_child_weight": 5,
        "objective": "reg:quantileerror",
        "quantile_alpha": np.array(quantiles),
        "tree_method": "hist"  # Required for multi-quantile
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=10,
        verbose_eval=50
    )
    
    predictions = model.predict(dval)  # Shape: (n_samples, n_quantiles)

    model.save_model(f"xgb_best_{city}.json")
    # model.save_model(f"xgb_model_single.json")

    
    return model, predictions



def train(X_train, X_val, y_train, y_val, use_stacking=False):
    """
    Train XGBoost model with optional quantile stacking
    """
        
    params = {
        # "eta": 0.3,
        # "max_depth": 6,
        # "colsample_bytree": 0.75,
        # "subsample": 0.95,
        # "gamma": 0.1,
        # "min_child_weight": 5,
        "eta": 0.2,
        "max_depth": 4,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "gamma": 1.0,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    watchlist = [(dtrain, "train"), (dval, "valid")]

    if not use_stacking:
        # Original approach
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse" #"mape"
        start = time.time()
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, 
                         early_stopping_rounds=10, verbose_eval=15)
        end = time.time()
        print(f"Elapsed time for training: {end - start:.6f} seconds")
        print("done training!")

        y_pred = model.predict(dval)
        print("done predicting!")
        
        # model.save_model(f"xgb_model_single.json")
        model_save_name = os.environ.get('PIPELINE_MODEL_NAME', 'xgb_model_single.json')
        model.save_model(model_save_name)
        print(f"Model saved as: {model_save_name}")
        # model.save_model(f"xgb_best_{city}.json")
        # model.save_model(f"xgb_rh_{city}.json")
        # model.save_model(f"xgb_self_{city}.json")
        # model.save_model("xgb_all_cities.json")
        # print(f"Model saved as xgb_best_{city}.json !")
        
    else:
        # Step 1: Train base RMSE model
        print("Training base RMSE model...")
        base_params = params.copy()
        base_params["objective"] = "reg:squarederror"
        
        base_model = xgb.train(base_params, dtrain, num_boost_round=1000, 
                              evals=watchlist, early_stopping_rounds=10, verbose_eval=15)
        
        # Step 2: Get residuals from base model
        print("Training quantile correction model...")
        base_pred_train = base_model.predict(dtrain)

        y_train_flat = np.ravel(y_train)
        base_pred_train_flat = np.ravel(base_pred_train)
        residuals = y_train_flat - base_pred_train_flat
        
        # Step 3: Train quantile model on residuals
        dresidual = xgb.DMatrix(X_train, label=residuals, feature_names=features)
        base_pred_val = base_model.predict(dval)
        # Ensure validation arrays are also 1D
        y_val_flat = np.ravel(y_val)
        base_pred_val_flat = np.ravel(base_pred_val)
        val_residuals = y_val_flat - base_pred_val_flat
        dresidual_val = xgb.DMatrix(X_val, label=val_residuals, feature_names=features)
        
        quantile_params = params.copy()
        quantile_params["objective"] = "reg:quantileerror"
        quantile_params["quantile_alpha"] = 0.9
        quantile_params["eta"] = 0.5
        
        model = xgb.train(quantile_params, dresidual, num_boost_round=500,
                         evals=[(dresidual, "train"), (dresidual_val, "valid")],
                         early_stopping_rounds=10, verbose_eval=15)

        
        # Final predictions = base_pred + quantile_correction
        quantile_correction = model.predict(dval)
        quantile_correction_flat = np.ravel(quantile_correction)
        y_pred = base_pred_val_flat + quantile_correction_flat
        
        # Compare performance
        base_rmse = np.sqrt(np.mean((y_val_flat - base_pred_val_flat) ** 2))
        stacked_rmse = np.sqrt(np.mean((y_val_flat - y_pred) ** 2))
        print(f"Base RMSE: {base_rmse:.4f}, Stacked RMSE: {stacked_rmse:.4f}")



    return model, y_pred



def importance(model):
    # For models trained with xgb.train() and DMatrix with feature_names
    feature_names = model.feature_names
    cmap = cmcr.batlow
    # sample two colors away from extremes for clarity
    predicted_color = cmap(0.7)
        # Get importance scores (you can choose different types)
    importance_dict = model.get_score(importance_type='total_gain')  # or 'gain', 'cover'
    
    # Convert to arrays, ensuring order matches feature_names
    importances = np.array([importance_dict.get(fname, 0.0) for fname in feature_names])

    sorted_indices = np.argsort(importances)
    sorted_importances = importances[sorted_indices]
    sorted_features = [feature_names[i] for i in sorted_indices]

    # Create the horizontal bar plot
    plt.figure(figsize=(12, max(8, len(sorted_features) * 0.3)))
    bars = plt.barh(range(len(sorted_features)), sorted_importances, 
                    color=predicted_color,
                    #  alpha=0.7
                    )

    # Customize the plot
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Total gain')
    plt.ylabel('Features')
    # plt.title('XGBoost Feature Importances')
    plt.grid(axis='x', alpha=0.3)

        # # Add value labels on bars
        # for i, (bar, v) in enumerate(zip(bars, sorted_importances)):
        #     plt.text(v + max(sorted_importances) * 0.01, i, f'{v:.3f}', 
        #             va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("../Plots/Feature_Importances.pdf", dpi=900, bbox_inches='tight')
    plt.show()

    print("\nFeature importances (sorted):")
    for feat, val in zip(sorted_features[::-1], sorted_importances[::-1]):
        print(f"{feat}: {val:.6f}")
    
    # Optional: Print top features
    # print("\nTop 10 most important features:")
    # for i in range(min(10, len(sorted_features))):
    #     idx = len(sorted_features) - 1 - i  # Start from most important
    #     print(f"{sorted_features[idx]}: {sorted_importances[idx]:.4f}")


def importance_comparison(model, importance_types=['weight', 'gain', 'cover','total_gain']):
    """
    Compare different importance types for Booster models
    Only works with models trained using xgb.train()
    """

    feature_names = model.feature_names
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(model.get_score()))]
    
    fig, axes = plt.subplots(1, len(importance_types), figsize=(6*len(importance_types), 8))
    if len(importance_types) == 1:
        axes = [axes]
    
    for idx, imp_type in enumerate(importance_types):
        importance_dict = model.get_score(importance_type=imp_type)
        importances = np.array([importance_dict.get(fname, 0.0) for fname in feature_names])
        
        # Sort features by importance
        sorted_indices = np.argsort(importances)
        sorted_importances = importances[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]
        
        # Plot
        axes[idx].barh(range(len(sorted_features)), sorted_importances, 
                      color='lightcoral', alpha=0.7)
        axes[idx].set_yticks(range(len(sorted_features)))
        axes[idx].set_yticklabels(sorted_features)
        axes[idx].set_xlabel(f'Importance ({imp_type})')
        axes[idx].set_title(f'Feature Importance - {imp_type.title()}')
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"../Plots/Feature_Importances_comparison.png", dpi=600, bbox_inches='tight')
    plt.show()


def scatterplot(tablez):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tablez, x='y_obs', y='y_pred', alpha=0.6)
    plt.plot([tablez['y_obs'].min(), tablez['y_obs'].max()], 
            [tablez['y_obs'].min(), tablez['y_obs'].max()], 
            'r--')  # Perfect prediction line
    plt.xlabel('Observed Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title('Observed vs Predicted Temperature')
    plt.show()
    plt.savefig(f"../Plots/Scatterplot.png", dpi=600, bbox_inches='tight')



def density_scatter_hex_norm(tablez, gridsize=200):
    x = tablez['y_obs']
    y = tablez['y_pred']

    fig, ax = plt.subplots(figsize=(10,6))

    # 1) Initial hexbin, get raw counts per bin
    hb = ax.hexbin(
        x, y,
        gridsize=gridsize,
        cmap='viridis',
        mincnt=1,
        linewidths=0.01,
        edgecolors='grey'
    )
    counts = hb.get_array()       # raw counts

    # 2) Normalize counts to 0–1 by DIVIDING BY MAX
    counts_norm = counts / counts.max()
    hb.set_array(counts_norm)     # overwrite the artist’s array
    hb.set_clim(0, 1)             # fix the color limits to [0,1]

    # 3) Colorbar showing 0→1
    cb = fig.colorbar(hb, ax=ax, label='Density')
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(['0', '0.5', '1'])

    # 4) Perfect‐prediction line
    mn, mx = x.min(), x.max()
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)

    # 5) Labels, title
    ax.set_xlabel('Observed Temperature [°C]')
    ax.set_ylabel('Predicted Temperature [°C]')
    ax.set_title('Observed vs Predicted Temperature')

    # 6) Enforce 1:1 aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # 7) Save & show
    plt.savefig("../Plots/ScatterDensity_hex_norm.png", dpi=600, bbox_inches='tight')
    plt.show()


def density_jointplot(tablez):
    g = sns.jointplot(
        data=tablez,
        x='y_obs', y='y_pred',
        kind='hex',           # or 'kde' for contours
        height=8, ratio=5,
        marginal_ticks=True
    )
    g.ax_joint.plot(
        [tablez['y_obs'].min(), tablez['y_obs'].max()],
        [tablez['y_obs'].min(), tablez['y_obs'].max()],
        'r--'
    )
    g.set_axis_labels('Observed Temperature', 'Predicted Temperature')
    plt.savefig("../Plots/ScatterDensity_joint.png", dpi=600, bbox_inches='tight')



def print_and_write(message, file):
    print(message)
    file.write(str(message) + "\n")


def analysis(tablez):
    with open(f"residual_analysis_report_test.txt", 'w') as f:
    # Define our dual-purpose output function
        def pw(message):
            print(message)
            f.write(str(message) + "\n\n" if isinstance(message, (pd.DataFrame, pd.Series)) else str(message) + "\n")

        residual_stats = tablez['residual'].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95])
        pw("Basic Residual Statistics:")
        pw(residual_stats)

        # Calculate additional statistics
        iqr = residual_stats['90%'] - residual_stats['10%']
        pw(f"\nInterquartile Range (IQR): {iqr:.2f}°C")

        large_residuals = tablez[abs(tablez['residual']) > 2]
        pw(f"Percentage of total predictions with abs(residuals)>2°C: {len(large_residuals)/len(tablez)*100:.2f}%")

        medium_residuals = tablez[abs(tablez['residual']) > 1]
        pw(f"Percentage of total predictions with abs(residuals)>1°C: {len(medium_residuals)/len(tablez)*100:.2f}%")

def cap_observations_per_bin(X_train, y_train, tablex_ref, test_idx, max_obs_per_bin=1000, random_state=42):
    """
    Randomly remove observations from temperature bins that exceed max_obs_per_bin
    
    Args:
        X_train: Training features
        y_train: Training targets (anomalies)
        tablex_ref: Reference DataFrame with t2m
        test_idx: Test set indices (to reconstruct training t2m values)
        max_obs_per_bin: Maximum number of observations to keep per 1°C bin
        random_state: Random seed for reproducibility
    
    Returns:
        X_train_capped, y_train_capped: Reduced training data
        kept_indices: Boolean mask of which training samples were kept
    """
    np.random.seed(random_state)
    
    # Get t2m values for training data directly
    # Create boolean mask for training data
    all_indices = np.arange(len(tablex_ref))
    train_mask = np.ones(len(tablex_ref), dtype=bool)
    train_mask[test_idx] = False
    
    # Extract t2m values for training samples only
    train_t2m = tablex_ref.loc[train_mask, "t2m"].values
    
    # Get absolute temperatures for training data
    train_temps = y_train[:, 0] + train_t2m
    
    # Bin temperatures by 1°C
    temp_bins = np.floor(train_temps)
    
    # Initialize mask to keep all samples
    keep_mask = np.ones(len(X_train), dtype=bool)
    
    print(f"\nCapping observations at {max_obs_per_bin} per temperature bin...")
    print("-"*60)
    
    # For each temperature bin
    unique_bins = np.unique(temp_bins)
    for temp_bin in unique_bins:
        # Find indices in this bin
        bin_mask = (temp_bins == temp_bin)
        bin_indices = np.where(bin_mask)[0]
        
        n_obs = len(bin_indices)
        
        if n_obs > max_obs_per_bin:
            # Randomly select which observations to keep
            keep_these = np.random.choice(bin_indices, size=max_obs_per_bin, replace=False)
            
            # Update mask: keep only selected observations from this bin
            keep_mask[bin_indices] = False
            keep_mask[keep_these] = True
            
            print(f"Bin {temp_bin:.0f}°C: {n_obs} -> {max_obs_per_bin} obs (removed {n_obs - max_obs_per_bin})")
    
    # Apply mask
    X_train_capped = X_train[keep_mask]
    y_train_capped = y_train[keep_mask]
    
    n_removed = len(X_train) - len(X_train_capped)
    pct_removed = (n_removed / len(X_train)) * 100
    
    print("-"*60)
    print(f"Original training samples: {len(X_train)}")
    print(f"Capped training samples: {len(X_train_capped)}")
    print(f"Removed: {n_removed} ({pct_removed:.1f}%)")
    print("="*60 + "\n")
    
    return X_train_capped, y_train_capped, keep_mask


def analyze_training_data_distribution(y_train, train_t2m):
    """
    Analyze and print the distribution of training observations per temperature bin
    
    Args:
        y_train: Training target values (anomalies)
        train_t2m: ERA5 t2m values for training data
    """
    # Get absolute temperatures for training data
    train_temps = y_train[:, 0] + train_t2m
    
    # Bin temperatures by 1°C
    temp_bins = np.floor(train_temps)
    
    # Count observations per bin
    bin_counts = pd.Series(temp_bins).value_counts().sort_index()
    
    print("\n" + "="*60)
    print("TRAINING DATA DISTRIBUTION PER TEMPERATURE BIN (1°C)")
    print("="*60)
    print(f"{'Temperature (°C)':<20} {'Number of Observations':<25}")
    print("-"*60)
    for temp, count in bin_counts.items():
        print(f"{temp:<20.0f} {count:<25}")
    print("-"*60)
    print(f"{'Total':<20} {len(train_temps):<25}")
    print(f"{'Min temp':<20} {train_temps.min():.2f}°C")
    print(f"{'Max temp':<20} {train_temps.max():.2f}°C")
    print(f"{'Mean observations/bin':<20} {bin_counts.mean():.1f}")
    print(f"{'Median observations/bin':<20} {bin_counts.median():.1f}")
    print("="*60 + "\n")
    
    return bin_counts