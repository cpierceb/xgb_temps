import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns
import sklearn
import xarray as xr
import xgboost as xgb
import json

from datetime import datetime, timedelta, timezone

from xgb_0_prep_params import *
from xgb_d_functions import *
from xgb_b_make_model import *

plt.rcParams['font.family'] = 'serif'

print("---------1. Initializing-----------")
# Configuration



timestamps = [start_time + timedelta(hours=i) for i in range(int(24*amount_of_days))]
#end_time = start_time + timedelta(days=amount_of_days)
output_dir = '../Plots/xgb_d_check'
os.makedirs(output_dir, exist_ok=True)


print("---------2. Preprocessing tables-----------")

split_type = os.environ.get('SPLIT_TYPE', SPLIT_TYPE)
use_jja = os.environ.get('PIPELINE_USE_JJA', 'false').lower() == 'true'

if use_jja:
    if split_type == 'jja2021':
        JJA2021_START = pd.Timestamp('2021-07-15', tz='UTC')
        JJA2021_END   = pd.Timestamp('2021-09-15', tz='UTC')
        tx_full = pd.read_pickle(f"../data_processing/dataframes_ready/tablex_{city}.pkl")
        ty_full = pd.read_pickle(f"../data_processing/dataframes_ready/tabley_{city}.pkl")
        mask = (tx_full['Time_UTC'] >= JJA2021_START) & (tx_full['Time_UTC'] < JJA2021_END)
        tablex = tx_full[mask].reset_index(drop=True)
        tabley = ty_full[mask].reset_index(drop=True)
    elif split_type == 'spatial':
        tablex = pd.read_pickle(f"../data_processing/dataframes_ready/tablex_{city}.pkl")
        tabley = pd.read_pickle(f"../data_processing/dataframes_ready/tabley_{city}.pkl")
    elif split_type == 'jja2020': 
        JJA2020_START = pd.Timestamp('2020-07-15', tz='UTC')
        JJA2020_END   = pd.Timestamp('2020-09-15', tz='UTC')
        tx_full = pd.read_pickle(f"../data_processing/dataframes_ready/tablex_{city}.pkl")
        ty_full = pd.read_pickle(f"../data_processing/dataframes_ready/tabley_{city}.pkl")
        mask = (tx_full['Time_UTC'] >= JJA2020_START) & (tx_full['Time_UTC'] < JJA2020_END)
        tablex = tx_full[mask].reset_index(drop=True)
        tabley = ty_full[mask].reset_index(drop=True)
    else:  # last_year
        with open("city_test_cutoffs.json") as f:
            cutoffs = json.load(f)
        test_start = pd.Timestamp(cutoffs[city])
        tx_full = pd.read_pickle(f"../data_processing/dataframes_ready/tablex_{city}.pkl")
        ty_full = pd.read_pickle(f"../data_processing/dataframes_ready/tabley_{city}.pkl")
        mask = tx_full['Time_UTC'] >= test_start
        tablex = tx_full[mask].reset_index(drop=True)
        tabley = ty_full[mask].reset_index(drop=True)
    print(f"Test rows for {city} ({split_type}): {len(tablex):,}")
    tablex, tabley, tablex_ref = preprocess([tablex], [tabley])
else:
    tablex = pd.read_pickle(f"../data_processing/dataframes_ready/tablex_{city}.pkl")
    tabley = pd.read_pickle(f"../data_processing/dataframes_ready/tabley_{city}.pkl")
    tablex, tabley, tablex_ref = preprocess([tablex], [tabley])
    tablex, tabley, tablex_ref = subset_by_timestamps_method2(tablex, tabley, tablex_ref, start_time, amount_of_days)



# tablex = pd.read_pickle(f"dataframes_ready/tablex_{city}.pkl")
# tabley = pd.read_pickle(f"dataframes_ready/tabley_{city}.pkl")



# tablex, tabley, tablex_ref = preprocess([tablex],[tabley])


# tablex, tabley, tablex_ref = subset_by_timestamps_method2(tablex, tabley,tablex_ref, start_time, amount_of_days)

print("this is after taking a subset of the tables:")
print(tablex)
print(tabley)



X = tablex.to_numpy()


# Load model

model = xgb.Booster()

# model.load_model("xgb_best_overall.json")  
# model.load_model(f"xgb_model_single.json")   
model_path = os.environ.get('PIPELINE_MODEL_NAME', 'xgb_model_single.json')
model.load_model(model_path)
print(f"Loaded model: {model_path}")
# model.load_model(f"xgb_best_{city}.json")  
# model.load_model(f"xgb_self_{city}.json")  

# print(f"loaded xgb_best_{city}!")

X_apply = xgb.DMatrix(X, feature_names=features)

#print(X_apply)
print("---------3. Making predictions-----------")

import time 
start = time.time()

y_pred = model.predict(X_apply)
# preds = model.predict(X_apply)
# y_pred = preds[:, 1]

end = time.time()


print(f"predictions for this city took {end-start} seconds")

# print("these are the predictions:")
# print(y_pred.shape, y_pred)
# print("these are the observations:")
# print(tabley.shape, tabley)


y_pred_df = pd.DataFrame(y_pred, columns=['Temperature'], index=tabley.index)

tabley['AbsTemp'] = tabley['Temperature'] + tablex['t2m']  
y_pred_df['AbsTemp'] = y_pred_df['Temperature'] + tablex['t2m']    
y_pred_df['residual'] = y_pred_df['AbsTemp'] - tabley['AbsTemp']
rmse = np.sqrt(np.mean((y_pred_df['AbsTemp'] - tabley['AbsTemp'])**2))
mae = np.mean(np.abs(y_pred_df['AbsTemp'] - tabley['AbsTemp']))


print("---------4. Preparing data for plotting-----------")
# First, create proper datetime objects from tablex columns
datetime_series = pd.to_datetime(tablex_ref[['Year', 'Month', 'Day', 'Hour']].rename(columns={
    'Year': 'year',
    'Month': 'month',
    'Day': 'day',
    'Hour': 'hour'
}))

# Add datetime to both DataFrames (using the same index)
y_pred_df['DateTime'] = datetime_series
tabley['DateTime'] = datetime_series

# Rest of your existing code for merging location info...
y_pred_df = y_pred_df.join(tablex_ref[['Latitude', 'Longitude', 'LCZ_100']])
y_pred_df['Location'] = y_pred_df.apply(lambda row: f"Lat{row['Latitude']:.4f}_Lon{row['Longitude']:.4f}", axis=1)
tabley['Location'] = y_pred_df['Location']



# Get unique locations
unique_locations = y_pred_df['Location'].unique()

# Create directory for location plots
location_plots_dir = os.path.join(output_dir, f'{start_year}_{month}_{city}', 'location_plots')
os.makedirs(location_plots_dir, exist_ok=True)

print("---------5. Analysis of residuals-----------")
y_pred_df['station_id'] = y_pred_df['Location'].copy()
y_pred_df['Hour'] = tablex_ref['Hour'].copy()
y_pred_df['Month'] = tablex_ref['Month'].copy()
print("rmse is: ")
print(rmse)
r2 = sklearn.metrics.r2_score(tabley['AbsTemp'], y_pred_df['AbsTemp'])
print("R² is: ")
print(r2)
# print("mae is: ")
# print(mae)
# per‑LCZ RMSE
grouped = y_pred_df.groupby('LCZ_100')['residual']
rmse_by_lcz = grouped.apply(lambda x: np.sqrt((x**2).mean()))
bias_by_lcz = grouped.mean()

# print("RMSE by LCZ:")
# print(rmse_by_lcz)

# print("Mean bias by LCZ:")
# print(bias_by_lcz)

analysis(y_pred_df)



print("---------6. Creating time series plots per location-----------")
#commented out just to not plot them again but the function is good and works

#location_plots(location_plots_dir,unique_locations,y_pred_df,tabley)

print("---------7. Creating aggregated LCZ plots-----------")
#commented out just to not plot them again but the function is good and works

# Add LCZ information to both DataFrames
y_pred_df['LCZ_100'] = tablex['LCZ_100']
tabley['LCZ_100'] = tablex['LCZ_100']

# Create directory for LCZ plots
lcz_plots_dir = os.path.join(output_dir, f'{start_year}_{month}_{city}/lcz_aggregates')
os.makedirs(lcz_plots_dir, exist_ok=True)

#lcz_aggregates(y_pred_df,tabley,lcz_plots_dir)



print("---------8. Temperature-dependent error analysis-----------")
# Add this after your existing analysis calls
# binned_results = plot_rmse_bias_by_observed_temp(
#     y_pred_df, 
#     tabley, 
#     output_dir, 
#     city, 
#     month, 
#     start_year
# )

# plot_rmse_count_by_observed_temp(
#     y_pred_df, 
#     tabley, 
#     output_dir, 
#     city, 
#     month, 
#     start_year)

# Optional: Print binned results
# print("\nError metrics by temperature bin:")
# print(binned_results)




# ---------------------------
# Example usage (put this after you've constructed y_pred_df, tabley and created location_plots_dir):
# ---------------------------
# results = make_extreme_week_plots(y_pred_df, tabley, location_plots_dir, window_days=3.5, save=True, show=False)
# print("Saved plots:", results['hot']['filename'], results['cold']['filename'], results['combined_filename'])

# Add this after your existing extreme week plots
# print("---------9. LCZ-based extreme week analysis-----------")
lcz_results = make_extreme_week_lcz_plots(
    y_pred_df, 
    tabley, 
    location_plots_dir, 
    window_days=3.5, 
    save=True, 
    show=False
)

# Hot extreme
result_hot = make_single_extreme_lcz_plot(y_pred_df, tabley, location_plots_dir, extreme_type='hot')

# Cold extreme  
result_cold = make_single_extreme_lcz_plot(y_pred_df, tabley, location_plots_dir, extreme_type='cold')
# print(f"Hot plot: {result_hot['filename']}")
# print(f"Cold plot: {result_cold['filename']}")

print(f"LCZ {lcz_results['lcz']} extremes: Hot={lcz_results['hot']['median_temp']:.2f}°C, Cold={lcz_results['cold']['median_temp']:.2f}°C")
# You can inspect the dataframes used for plotting:
# hot_obs_df = results['hot']['obs_df']
# hot_pred_df = results['hot']['pred_df']

print("---------8.1. Extreme analysis-----------")

# Single threshold analysis
# hot_results = calculate_extreme_csi(y_pred_df, tabley, percentile=99, extreme_type='hot')
# cold_results = calculate_extreme_csi(y_pred_df, tabley, percentile=99, extreme_type='cold')

warm_results = calculate_extreme_csi(y_pred_df, tabley, percentile=95, extreme_type='hot')
cool_results = calculate_extreme_csi(y_pred_df, tabley, percentile=95, extreme_type='cold')



print("---------9. Spatial RMSE visualization-----------")
print("---------9. Spatial RMSE visualization-----------")
x_vals_3035 = tablex_ref['X3035'].values
y_vals_3035 = tablex_ref['Y3035'].values




if city == "birmingham":
    lat0 = 52.481983717827056
    lon0 = -1.896175877351571

# Plot RMSE
station_rmse_df = plot_station_metric_points(
    y_pred_df=y_pred_df,
    tabley=tabley,
    x_vals_3035=x_vals_3035,
    y_vals_3035=y_vals_3035,
    output_dir=output_dir,
    city=city,
    month=month,
    center_lon=lon0, 
    center_lat=lat0,
    start_year=start_year,
    metric='rmse',
    vmin=0.0,
    vmax=3.5
)

# Plot R²
station_r2_df = plot_station_metric_points(
    y_pred_df=y_pred_df,
    tabley=tabley,
    x_vals_3035=x_vals_3035,
    y_vals_3035=y_vals_3035,
    output_dir=output_dir,
    city=city,
    month=month,
    center_lon=lon0, 
    center_lat=lat0,
    start_year=start_year,
    metric='r2',
    vmin=0.0,
    vmax=1.0
)

# Plot Bias
station_bias_df = plot_station_metric_points(
    y_pred_df=y_pred_df,
    tabley=tabley,
    x_vals_3035=x_vals_3035,
    y_vals_3035=y_vals_3035,
    output_dir=output_dir,
    city=city,
    month=month,
    center_lon=lon0, 
    center_lat=lat0,
    start_year=start_year,
    metric='bias'
)