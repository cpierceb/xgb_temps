from xgb_0_prep_params import *

import math
import numpy as np
import os
import pandas as pd

import cmcrameri.cm as cmcr
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, LinearSegmentedColormap, CenteredNorm, Normalize, ListedColormap
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import contextily as ctx
from pyproj import Transformer

import seaborn as sns
from scipy.ndimage import gaussian_filter
from datetime import timedelta
from scipy.stats import binned_statistic_2d
from sklearn.metrics import f1_score, precision_score, recall_score



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size']= 11


# Method 2: Using date range filtering - more efficient for large datasets
def subset_by_timestamps_method2(tablex, tabley, tablex_ref, start_time, amount_of_days):
    """
    Subset dataframes using date range filtering with Year/Month/Day/Hour columns
    """
    end_time = start_time + timedelta(days=amount_of_days)
    
    # Create a temporary datetime column for comparison
    tablex_temp = tablex_ref.copy()
    tablex_temp['temp_datetime'] = pd.to_datetime(
        tablex_temp[['Year', 'Month', 'Day', 'Hour']],
        utc=True  
    )    
    # Create boolean mask for the time range
    mask = (tablex_temp['temp_datetime'] >= start_time) & (tablex_temp['temp_datetime'] < end_time)
     
    # Apply mask to both dataframes
    tablex_subset = tablex[mask].copy()
    tabley_subset = tabley[mask].copy()
    tablex_ref_subset = tablex_ref[mask].copy()
    
    return tablex_subset, tabley_subset, tablex_ref_subset

def location_plots(location_plots_dir,unique_locations,y_pred_df,tabley):
    # For each unique location
    for location in unique_locations:
        # Filter data for this location
        loc_pred = y_pred_df[y_pred_df['Location'] == location]
        loc_obs = tabley[tabley['Location'] == location]
        
        # Get location metadata
        lcz = loc_pred['LCZ_100'].iloc[0]
        lat = loc_pred['Latitude'].iloc[0]
        lon = loc_pred['Longitude'].iloc[0]
        
        # Create figure with independent x-axes
        fig, axes = plt.subplots(4, 1, figsize=(15, 15))  # Increased height for clarity
        fig.suptitle(f"Temperature at Location {location}\nLCZ: {lcz}, Lat: {lat:.4f}, Lon: {lon:.4f}")
        
        # Split by date ranges (assuming data is sorted chronologically)
        date_ranges = np.array_split(loc_pred['DateTime'].unique(), 4)
        
        for i, date_range in enumerate(date_ranges):
            ax = axes[i]
            start_date = date_range[0]
            end_date = date_range[-1]
            
            # Filter data for this date range
            mask = (loc_pred['DateTime'] >= start_date) & (loc_pred['DateTime'] <= end_date)
            pred_chunk = loc_pred[mask]
            obs_chunk = loc_obs[mask]
            
            # Plot data
            ax.plot(pred_chunk['DateTime'], pred_chunk['AbsTemp'], 'r-', label='Predicted')
            ax.plot(obs_chunk['DateTime'], obs_chunk['AbsTemp'], 'b-', label='Observed')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0,6,12,18]))
            
            # Set x-axis limits to the date range
            ax.set_xlim(start_date, end_date)
            
            ax.set_ylabel('Temperature (°C)')
            ax.legend()
            ax.grid(True)
            ax.set_title(f"Period {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        plt.tight_layout()
        
        # Save figure
        location_str = location.replace('.', 'p').replace('-', 'n')
        filename = os.path.join(location_plots_dir, f"temp_series_LCZ{lcz}_{location_str}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved location plots to {location_plots_dir}")


def lcz_aggregates(y_pred_df,tabley,lcz_plots_dir):
    # Get unique LCZs
    unique_lczs = sorted(y_pred_df['LCZ_100'].unique())

    # Prepare figure layout
    fig, axes = plt.subplots(4, 1, figsize=(15, 15))
    fig.suptitle("Temperature by LCZ Classification", y=1.02)

    # Split datetime into 4 equal periods
    date_ranges = np.array_split(y_pred_df['DateTime'].unique(), 4)

    for lcz in unique_lczs:
        # Filter data for this LCZ
        lcz_pred = y_pred_df[y_pred_df['LCZ_100'] == lcz]
        lcz_obs = tabley[tabley['LCZ_100'] == lcz]
        
        # Count unique stations in this LCZ
        num_stations = lcz_pred['Location'].nunique()
        
        # Create new figure for each LCZ
        fig, axes = plt.subplots(4, 1, figsize=(15, 15))
        fig.suptitle(f"LCZ {lcz} (n={num_stations} stations)", y=1.02)
        
        for i, date_range in enumerate(date_ranges):
            ax = axes[i]
            start_date = date_range[0]
            end_date = date_range[-1]
            
            # Filter for date range
            mask = (lcz_pred['DateTime'] >= start_date) & (lcz_pred['DateTime'] <= end_date)
            pred_chunk = lcz_pred[mask]
            obs_chunk = lcz_obs[mask]
            
            # Group by datetime and calculate statistics
            pred_stats = pred_chunk.groupby('DateTime')['AbsTemp'].agg(['mean', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
            obs_stats = obs_chunk.groupby('DateTime')['AbsTemp'].agg(['mean', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
            
            # Rename columns
            pred_stats.columns = ['mean', 'p25', 'p75']
            obs_stats.columns = ['mean', 'p25', 'p75']
            
            # Plot mean lines
            ax.plot(pred_stats.index, pred_stats['mean'], 'r-', label='Predicted Mean')
            ax.plot(obs_stats.index, obs_stats['mean'], 'b-', label='Observed Mean')
            
            # Plot shaded areas for percentiles
            ax.fill_between(pred_stats.index, pred_stats['p25'], pred_stats['p75'], 
                            color='red', alpha=0.2, label='Predicted 25-75%')
            ax.fill_between(obs_stats.index, obs_stats['p25'], obs_stats['p75'], 
                            color='blue', alpha=0.2, label='Observed 25-75%')
            
            # Formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.set_xlim(start_date, end_date)
            ax.set_ylabel('Temperature (°C)')
            ax.set_ylim(10,40)
            ax.legend(loc='upper right')
            ax.grid(True)
            ax.set_title(f"Period {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        plt.tight_layout()
        filename = os.path.join(lcz_plots_dir, f"temp_series_LCZ{lcz}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    # --- NEW: boxplot of residuals Overall vs per‑LCZ ---
    # 1. Prepare data
    df = y_pred_df[['LCZ_100', 'residual']].copy()
    df_overall = df.copy()
    df_overall['LCZ_100'] = 'Overall'
    df_plot = pd.concat([df_overall, df], axis=0)

    # 2. Determine ordering: Overall first, then sorted LCZs
    lcz_order = ['Overall'] + sorted(y_pred_df['LCZ_100'].unique())

    # 3. Make the figure
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='LCZ_100',
        y='residual',
        data=df_plot,
        order=lcz_order,
        showfliers=False  # hide outliers if you like
    )
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('LCZ')
    plt.ylabel('Modeled - Measured (°C)')
    plt.title('Error Statistics per LCZ')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 4. Save it
    boxplot_path = os.path.join(lcz_plots_dir, "residuals_boxplot_overall_vs_lcz.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot to {boxplot_path}")

    # … then your print at the end …
    print(f"Saved LCZ aggregated plots to {lcz_plots_dir}")

# 8. Plot loops
def plot_field(field, grid_lons, grid_lats, title, out_path, levels, cmap, norm):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    sm = gaussian_filter(field, sigma=1.3)
    cnt = ax.contourf(grid_lons, grid_lats, sm, levels=levels, cmap=cmap, norm=norm,
                    alpha=0.6, transform=ccrs.PlateCarree())
    lines = ax.contour(grid_lons, grid_lats, sm, levels=levels,
                    colors='lightgrey', linewidths=0.4, transform=ccrs.PlateCarree())
    ax.clabel(lines, inline=True, fontsize=8, fmt='%1.0f°C',colors='black')
    cbar = plt.colorbar(cnt, ax=ax, fraction=0.026, pad=0.04)
    cbar.set_label(title)
    ctx.add_basemap(ax, crs=ccrs.PlateCarree(), source=ctx.providers.CartoDB.Positron)
    ax.set_title(title)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()



def temperature_maps(SAVE_DIR,city,start_time,ds,levels,times_naive,lon_vals,lat_vals,plot_dir,cmap,norm):
    print("---------4. Starting plots...---------------")
    for t in times_naive:
        ds_t = ds.sel(time=t)
        field = ds_t['temperature'].values
        grid_lons, grid_lats = np.meshgrid(lon_vals, lat_vals)
        time_str = t.strftime("%Y-%m-%d_%H-%M")
        title = f"Modelled temperature at {t.strftime('%Y-%m-%d %H:%M UTC')}"
        out_path = os.path.join(plot_dir, f"temp_{time_str}.jpg")
        plot_field(field, grid_lons, grid_lats, title, out_path, levels, cmap, norm)
        print(f"---------4.xx plot for {time_str} done---------------")

    print("--------------5.All plots saved-----------")


def diff_maps(SAVE_DIR,city,start_time,ds,levels,times_naive,lon_vals,lat_vals,diff_dir,cmap_diff,norm_diff):
    print("---------4. Starting plots...---------------")
    for t in times_naive:
        diff = ds.sel(time=t).values
        grid_lons, grid_lats = np.meshgrid(lon_vals, lat_vals)
        time_str = t.strftime("%Y-%m-%d_%H-%M")
        title = f"ΔT with rural reference at {t.strftime('%Y-%m-%d %H:%M UTC')}"
        out = os.path.join(diff_dir, f"diff_{t.strftime('%Y-%m-%d_%H-%M')}.jpg")
        plot_field(
        diff, grid_lons, grid_lats,
        title, out,
        levels, cmap_diff, norm_diff
        )
        print(f"---------4.xx plot for {time_str} done---------------")


    print("--------------5.Diff plots saved-----------")


def ensure_dir(dirpath): os.makedirs(dirpath, exist_ok=True)

def lcz_uhi_plots(SAVE_DIR,city,start_time,tablex,y_pred_df):
    df = tablex[['Time_UTC','Latitude','Longitude','LCZ_100']].copy()
    df['AbsTemp'] = y_pred_df['AbsTemp'].values

    # 2) build outer‐30% mask
    lats = df['Latitude']; lons = df['Longitude']
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    lat_margin = 0.15*(lat_max - lat_min)
    lon_margin = 0.15*(lon_max - lon_min)
    outer_mask = (
        ((lats <  lat_min + lat_margin) | (lats >  lat_max - lat_margin))
        |
        ((lons <  lon_min + lon_margin) | (lons >  lon_max - lon_margin))
    )

    # 3) baseline time‐series
    mask_baseline = (df['LCZ_100'] >= 11) & outer_mask
    baseline_ts = df[mask_baseline].groupby('Time_UTC')['AbsTemp'].mean()

    # 4) mean per LCZ per timestamp
    mean_lcz_ts = df.groupby(['Time_UTC','LCZ_100'])['AbsTemp'] \
                .mean() \
                .unstack(fill_value=np.nan)

    # 5) difference from baseline
    diff_lcz = mean_lcz_ts.sub(baseline_ts, axis=0)


    # 6) aggregate LCZ≤10
    diff_lcz['LCZ_≤10'] = diff_lcz.loc[:, diff_lcz.columns <= 10].mean(axis=1)
    hours = diff_lcz.index.hour
    diff_by_hour = diff_lcz.groupby(hours).mean()  

    # NEW: Print max/min UHI for LCZ≤10
    lcz_leq10 = diff_by_hour['LCZ_≤10']
    max_uhi = lcz_leq10.max()
    min_uhi = lcz_leq10.min()
    hour_max = lcz_leq10.idxmax()
    hour_min = lcz_leq10.idxmin()
    
    print("\n" + "="*60)
    print(f"UHI Statistics for LCZ ≤10 (average across built LCZs)")
    print("="*60)
    print(f"Maximum UHI: {max_uhi:.3f}°C at {hour_max:02d}:00 UTC")
    print(f"Minimum UHI: {min_uhi:.3f}°C at {hour_min:02d}:00 UTC")
    #print(f"Range: {max_uhi - min_uhi:.3f}°C")
    print("="*60 + "\n")


    # 1) Standard LCZ hex colors:
    lcz_colors = {
        1:  "#ff0000", 2:  "#ff0026", 3:  "#ff004d", 4:  "#ff8000",
        5:  "#ff7f00", 6:  "#ff7f7f", 7:  "#ffe6cc", 8:  "#ffff00",
        9:  "#ccff66", 10: "#999999", 11: "#006600", 12: "#339933",
        13: "#33cc33", 14: "#99ff33", 15: "#cccccc", 16: "#0066ff",
        17: "#003399",
    }
    aggregate_color = "#000000"  # or pick any for LCZ≤10

    # 2) Helper to parse & label
    def parse_and_label(col):
        """Return (key, label) where key is int 1–17 or 'LCZ_≤10', label is string."""
        # if it’s the aggregate column:
        if str(col).startswith("LCZ_") and "≤10" in str(col):
            return "LCZ_≤10", "LCZ ≤10"
        # else try to convert to int
        try:
            num = int(col)
            if num <= 10:
                return num, f"LCZ {num}"
            else:
                # 11→A, 12→B, etc
                letter = chr(ord('A') + (num - 11))
                return num, f"LCZ {letter}"
        except:
            # fallback
            return col, str(col)

    # 3) Build lists for plotting
    plot_cols = diff_by_hour.columns.tolist()
    parsed = [parse_and_label(c) for c in plot_cols]
    # parsed is list of (key, label)
    keys, labels = zip(*parsed)
    colors = [ lcz_colors.get(k, aggregate_color) for k in keys ]

    # 4) Make output directory
    out_dir = os.path.join(SAVE_DIR, city, start_time.strftime("%Y-%m-%d"), "uhi_by_hour")
    os.makedirs(out_dir, exist_ok=True)

    # 5) One‐plot‐per‐LCZ
    for col, key, lbl, colr in zip(plot_cols, keys, labels, colors):
        plt.figure(figsize=(8,4))
        plt.plot(diff_by_hour.index, diff_by_hour[col], marker='o', color=colr)
        plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
        plt.xticks(range(24))
        plt.xlabel("Time (Hr UTC)")
        plt.ylabel("Mean UHI intensity (°C)")
        plt.title(f"Mean UHI intensity")
        fname = lbl.replace(" ", "_") + ".png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()

    # 6) Combined plot
    plt.figure(figsize=(10,6))
    for col, lbl, colr in zip(plot_cols, labels, colors):
        plt.plot(diff_by_hour.index, diff_by_hour[col], label=lbl, color=colr)
    plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
    plt.xticks(range(24))
    plt.xlabel("Time (Hr UTC)")
    plt.ylabel("Mean UHI intensity (°C)")
    plt.title("Mean UHI intensity across built LCZs")
    plt.legend(ncol=2, fontsize='small', bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_LCZs_by_hour.png"), dpi=300)
    plt.close()


def plot_rmse_bias_by_observed_temp(y_pred_df, tabley, output_dir, city, month, start_year):
    """
    Plot RMSE and bias as functions of observed temperature in 1°C bins
    
    Args:
        y_pred_df: DataFrame with predictions and residuals
        tabley: DataFrame with observed values
        output_dir: Base output directory
        city: City name (for filename)
        month: Month (for filename)
        start_year: Start year (for filename)
    """
    # Create directory for analysis plots
    analysis_dir = os.path.join(output_dir, f'{start_year}_{month}_{city}', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Prepare data
    data = pd.DataFrame({
        'Observed': tabley['AbsTemp'],
        'Residual': y_pred_df['residual']
    })
    
    # Bin observed temperatures (1°C bins)
    data['Observed_bin'] = np.floor(data['Observed'])
    binned = data.groupby('Observed_bin')['Residual'].agg(
        RMSE=lambda x: np.sqrt(np.mean(x**2)),
        Bias='mean',
        Count='count'
    ).reset_index()

    # # Filter bins with sufficient samples (min 20 observations)
    # binned = binned[binned['Count'] >= 20]
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # RMSE (left axis)
    color = 'tab:red'
    ax1.plot(binned['Observed_bin'], binned['RMSE'], 'o-', color=color, label='RMSE')
    ax1.set_xlabel('Observed Temperature (°C)')
    ax1.set_ylabel('RMSE (°C)', color=color)
    ax1.set_ylim(0, 5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(alpha=0.3)
    
    # Bias (right axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(binned['Observed_bin'], binned['Bias'], 's--', color=color, label='Bias')
    ax2.set_ylabel('Bias (°C)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.7)
    
    # Formatting
    plt.title(f'RMSE and Bias by Observed Temperature | {city} {month} {start_year}')
    fig.tight_layout()
    
    # Save plot
    filename = os.path.join(analysis_dir, 'rmse_bias_vs_observed_temp.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved RMSE/Bias plot to {filename}")
    return binned


def plot_rmse_count_by_observed_temp(y_pred_df, tabley, output_dir, city, month, start_year):
    """
    Plot RMSE and number of observations as functions of observed temperature in 1°C bins
    
    Args:
        y_pred_df: DataFrame with predictions and residuals
        tabley: DataFrame with observed values
        output_dir: Base output directory
        city: City name (for filename)
        month: Month (for filename)
        start_year: Start year (for filename)
    """
    # Create directory for analysis plots
    analysis_dir = os.path.join(output_dir, f'{start_year}_{month}_{city}', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Prepare data
    data = pd.DataFrame({
        'Observed': tabley['AbsTemp'],
        'Residual': y_pred_df['residual']
    })
    
    # Bin observed temperatures (1°C bins)
    data['Observed_bin'] = np.floor(data['Observed'])
    binned = data.groupby('Observed_bin')['Residual'].agg(
        RMSE=lambda x: np.sqrt(np.mean(x**2)),
        Count='count'
    ).reset_index()

    # # Filter bins with sufficient samples (min 20 observations)
    # binned = binned[binned['Count'] >= 20]
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # RMSE (left axis)
    color = 'tab:red'
    ax1.plot(binned['Observed_bin'], binned['RMSE'], 'o-', color=color, label='RMSE')
    ax1.set_xlabel('Observed Temperature (°C)')
    ax1.set_ylabel('RMSE (°C)', color=color)
    ax1.set_ylim(0, 5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(alpha=0.3)
    
    # Number of observations (right axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.bar(binned['Observed_bin'], binned['Count'], alpha=0.3, color=color, label='Count', width=0.8)
    ax2.set_ylabel('Number of Observations', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Formatting
    plt.title(f'RMSE and Observation Count by Observed Temperature | {city} {month} {start_year}')
    fig.tight_layout()
    
    # Save plot
    filename = os.path.join(analysis_dir, 'rmse_count_vs_observed_temp.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved RMSE/Count plot to {filename}")
    return binned

def make_single_extreme_lcz_plot(y_pred_df, tabley, location_plots_dir,
                                extreme_type='hot', window_days=1.5, 
                                save=True, show=False):
    """
    Find the LCZ with the most stations, then plot ±window_days around either:
      - the absolute hottest observed temperature (median across stations in that LCZ)
      - the absolute coldest observed temperature (median across stations in that LCZ)
    Shows median predictions and observations with 25-75% IQR shaded.
    Produces one PNG in location_plots_dir. Sized for 2x2 layout.

    Uses 2 categorical colors sampled from Crameri's 'batlow': one for Observed, one for Predicted.
    Y ticks every 5°C.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import cmcrameri.cm as cmcr
    from datetime import timedelta
    import os
    import math

    cmap = cmcr.batlow
    # sample two colors away from extremes for clarity
    observed_color = cmap(0.1)
    predicted_color = cmap(0.7)

    y_pred_df = y_pred_df.copy()
    tabley = tabley.copy()

    y_pred_df['DateTime'] = pd.to_datetime(y_pred_df['DateTime'])
    tabley['DateTime'] = pd.to_datetime(tabley['DateTime'])

    # Find LCZ with most stations (break ties by smallest LCZ number)
    lcz_station_counts = tabley.groupby('LCZ_100')['Location'].nunique().sort_values(ascending=False)
    max_count = lcz_station_counts.max()
    lcz_with_max = lcz_station_counts[lcz_station_counts == max_count].index
    target_lcz = lcz_with_max.min()

    print(f"Selected LCZ: {target_lcz} with {max_count} stations")

    # Filter data for this LCZ
    obs_lcz = tabley[tabley['LCZ_100'] == target_lcz].copy()
    pred_lcz = y_pred_df[y_pred_df['LCZ_100'] == target_lcz].copy()

    # For each timestamp, compute median and 25/75 percentiles
    def _group_stats(df):
        g = df.groupby('DateTime')['AbsTemp'].agg([
            'median',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75)
        ])
        g.columns = ['median', 'q25', 'q75']
        return g

    obs_grouped = _group_stats(obs_lcz)
    pred_grouped = _group_stats(pred_lcz)

    # Find hottest and coldest median observed temperatures (timestamps)
    if extreme_type.lower() == 'hot':
        extreme_ts = obs_grouped['median'].idxmax()
        extreme_temp = obs_grouped.loc[extreme_ts, 'median']
        extreme_label = "Hottest observed temperature"
    elif extreme_type.lower() == 'cold':
        extreme_ts = obs_grouped['median'].idxmin()
        extreme_temp = obs_grouped.loc[extreme_ts, 'median']
        extreme_label = "Coldest observed temperature"
    else:
        raise ValueError("extreme_type must be 'hot' or 'cold'")

    half_window = timedelta(days=window_days)
    start = extreme_ts - half_window
    end = extreme_ts + half_window

    # Get data for window (per-station then aggregate)
    obs_window = obs_lcz[(obs_lcz['DateTime'] >= start) & (obs_lcz['DateTime'] <= end)]
    pred_window = pred_lcz[(pred_lcz['DateTime'] >= start) & (pred_lcz['DateTime'] <= end)]

    obs_stats = _group_stats(obs_window)
    pred_stats = _group_stats(pred_window)

    # Prepare filename
    safe_lcz = str(target_lcz).replace(" ", "_")
    filename = os.path.join(location_plots_dir, f"LCZ_{safe_lcz}_{extreme_type}_extreme.pdf")

    # ---------- compute y-axis span (multiple of 5) ----------
    def _minmax_from_stats(obs_stats, pred_stats):
        minima = []
        maxima = []
        for df in (obs_stats, pred_stats):
            minima.append(df[['q25', 'median', 'q75']].min().min())
            maxima.append(df[['q25', 'median', 'q75']].max().max())
        return min(minima), max(maxima)

    ymin, ymax = _minmax_from_stats(obs_stats, pred_stats)
    span = ymax - ymin

    # round span up to nearest multiple of 5
    def roundup_to(x, base=5):
        return math.ceil(x / base) * base

    span_rounded = roundup_to(span, 5)
    if span_rounded == 0:
        span_rounded = 5

    # compute center and set y-limits as center ± span_rounded/2
    center = 0.5 * (ymax + ymin)
    half_span = span_rounded / 2.0
    ylim = (center - half_span, center + half_span)

    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Title (no numerical temps—can add annotations if needed)
    ax.set_title(f"{extreme_type.capitalize()} extreme", fontsize=12)

    # Plot medians and IQRs
    ax.plot(obs_stats.index, obs_stats['median'],
            label='Observed', color=observed_color, linewidth=2)
    ax.fill_between(obs_stats.index, obs_stats['q25'], obs_stats['q75'],
                    color=observed_color, alpha=0.2)

    ax.plot(pred_stats.index, pred_stats['median'],
            label='Predicted', color=predicted_color, linewidth=2)
    ax.fill_between(pred_stats.index, pred_stats['q25'], pred_stats['q75'],
                    color=predicted_color, alpha=0.2)

    # Mark the extreme timestamp
    ax.axvline(extreme_ts, color='k', linestyle='--', alpha=0.5, linewidth=1)

    # Set y-limits and ticks
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.grid(True, linestyle=':', alpha=0.6)

    ax.set_xlim(start, end)
    # ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Temperature [°C]', fontsize=11)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


    # Legend
    ax.legend(loc='upper right', fontsize=10)

    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')

    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved LCZ {extreme_type} extreme plot to: {filename}")
    
    if show:
        plt.show()
    
    plt.close(fig)

    result = {
        'lcz': target_lcz,
        'n_stations': max_count,
        'extreme_type': extreme_type,
        'timestamp': extreme_ts,
        'median_temp': extreme_temp,
        'obs_stats': obs_stats,
        'pred_stats': pred_stats,
        'filename': filename
    }

    return result


def make_extreme_week_plots(y_pred_df, tabley, location_plots_dir,
                            window_days=3.5, save=True, show=False):
    """
    Find the station with the most observations and plot ±window_days around:
      - the absolute hottest observed temperature (tabley['AbsTemp'])
      - the absolute coldest observed temperature (tabley['AbsTemp'])
    Produces two separate PNGs and one combined PNG in location_plots_dir.
    Returns a dict with DataFrames used for the hot and cold windows.

    Parameters
    ----------
    y_pred_df : pd.DataFrame
        DataFrame with predicted results. Must contain columns: 'AbsTemp', 'DateTime', 'Location'.
    tabley : pd.DataFrame
        Observations DataFrame. Must contain columns: 'AbsTemp', 'DateTime', 'Location'.
    location_plots_dir : str
        Directory path to save plots.
    window_days : float (default 3.5)
        Number of days before and after the extreme to include.
    save : bool
        Whether to save PNG files.
    show : bool
        Whether to call plt.show() for the figures (useful in interactive sessions).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timedelta
    import os

    # Ensure DateTime columns are datetimes
    if 'DateTime' not in y_pred_df.columns or 'DateTime' not in tabley.columns:
        raise ValueError("Both y_pred_df and tabley must have a 'DateTime' column.")

    y_pred_df = y_pred_df.copy()
    tabley = tabley.copy()

    y_pred_df['DateTime'] = pd.to_datetime(y_pred_df['DateTime'])
    tabley['DateTime'] = pd.to_datetime(tabley['DateTime'])

    # Make sure 'Location' exists
    if 'Location' not in tabley.columns or 'Location' not in y_pred_df.columns:
        raise ValueError("Both y_pred_df and tabley must have a 'Location' column.")

    # Pick station with most observations (tie defaults to the first returned)
    counts = tabley['Location'].value_counts()
    if len(counts) == 0:
        raise ValueError("No locations found in tabley.")
    station = counts.index[0]

    # Subset for this station
    obs_station = tabley[tabley['Location'] == station].sort_values('DateTime')
    pred_station = y_pred_df[y_pred_df['Location'] == station].sort_values('DateTime')

    if obs_station.empty or pred_station.empty:
        raise ValueError(f"No data for selected station: {station}")

    # Find timestamps of hottest and coldest observed AbsTemp
    hottest_idx = obs_station['AbsTemp'].idxmax()
    coldest_idx = obs_station['AbsTemp'].idxmin()
    hottest_ts = obs_station.loc[hottest_idx, 'DateTime']
    coldest_ts = obs_station.loc[coldest_idx, 'DateTime']

    half_window = timedelta(days=window_days)

    hot_start = hottest_ts - half_window
    hot_end = hottest_ts + half_window
    cold_start = coldest_ts - half_window
    cold_end = coldest_ts + half_window

    hot_obs = obs_station[(obs_station['DateTime'] >= hot_start) & (obs_station['DateTime'] <= hot_end)]
    hot_pred = pred_station[(pred_station['DateTime'] >= hot_start) & (pred_station['DateTime'] <= hot_end)]

    cold_obs = obs_station[(obs_station['DateTime'] >= cold_start) & (obs_station['DateTime'] <= cold_end)]
    cold_pred = pred_station[(pred_station['DateTime'] >= cold_start) & (pred_station['DateTime'] <= cold_end)]

    # Prepare filenames
    safe_station_name = station.replace(" ", "_").replace(":", "_")
    hot_fn = os.path.join(location_plots_dir, f"{safe_station_name}_hottest_week.png")
    cold_fn = os.path.join(location_plots_dir, f"{safe_station_name}_coldest_week.png")
    combined_fn = os.path.join(location_plots_dir, f"{safe_station_name}_extremes_combined.png")

    # Helper to plot a window
    def _plot_window(obs_df, pred_df, start, end, title, out_fn):
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        # Plot observed and predicted AbsTemp
        ax.plot(obs_df['DateTime'], obs_df['AbsTemp'], linestyle='-', label='Observed', color = 'b')
        ax.plot(pred_df['DateTime'], pred_df['AbsTemp'], linestyle='-', label='Predicted', color = 'r')

        # Formatting
        ax.set_xlim(start, end)
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Absolute Temperature (°C)')
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        if save:
            plt.savefig(out_fn, dpi=200)
        if show:
            plt.show()
        plt.close(fig)

    # Plot hot window
    hot_title = (f"Hottest obs {obs_station.loc[hottest_idx,'AbsTemp']:.2f}°C "
                 f"at {hottest_ts}")
    _plot_window(hot_obs, hot_pred, hot_start, hot_end, hot_title, hot_fn)

    # Plot cold window
    cold_title = (f"Coldest obs {obs_station.loc[coldest_idx,'AbsTemp']:.2f}°C "
                  f"at {coldest_ts}")
    _plot_window(cold_obs, cold_pred, cold_start, cold_end, cold_title, cold_fn)

    # Combined figure (two rows)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=False)
    # Hot subplot
    ax1.plot(hot_obs['DateTime'], hot_obs['AbsTemp'], label='Observed', color = 'b')
    ax1.plot(hot_pred['DateTime'], hot_pred['AbsTemp'], label='Predicted', color = 'r')
    ax1.set_xlim(hot_start, hot_end)
    ax1.set_ylabel('Abs Temp (°C)')
    ax1.set_title(hot_title)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))

    # Cold subplot
    ax2.plot(cold_obs['DateTime'], cold_obs['AbsTemp'], label='Observed', color = 'b')
    ax2.plot(cold_pred['DateTime'], cold_pred['AbsTemp'], label='Predicted', color = 'r')
    ax2.set_xlim(cold_start, cold_end)
    ax2.set_ylabel('Abs Temp (°C)')
    ax2.set_title(cold_title)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))

    for ax in (ax1, ax2):
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    plt.tight_layout()
    if save:
        plt.savefig(combined_fn, dpi=200)
    if show:
        plt.show()
    plt.close(fig)

    result = {
        'station': station,
        'hot': {
            'timestamp': hottest_ts,
            'obs_df': hot_obs,
            'pred_df': hot_pred,
            'filename': hot_fn
        },
        'cold': {
            'timestamp': coldest_ts,
            'obs_df': cold_obs,
            'pred_df': cold_pred,
            'filename': cold_fn
        },
        'combined_filename': combined_fn
    }

    return result


def make_extreme_week_lcz_plots(y_pred_df, tabley, location_plots_dir,
                                window_days=3.5, save=True, show=False):
    """
    Find the LCZ with the most stations, then plot ±window_days around:
      - the absolute hottest observed temperature (median across stations in that LCZ)
      - the absolute coldest observed temperature (median across stations in that LCZ)
    Shows median predictions and observations with 25-75% IQR shaded.
    Produces one combined PNG in location_plots_dir.

    Modifications:
    - Use 2 categorical colors sampled from Crameri's 'batlow': one for Observed, one for Predicted.
    - Annotate hottest/coldest observed and predicted median values directly on the plot at the
      corresponding timestamp and value (not in the title).
    - Legend only on the hottest plot, and it only lists Observed and Predicted (IQRs not in legend).
    - Y ticks every 5°C and both plots share the same y-span (rounded to multiples of 5).
    """




    cmap = cmcr.batlow
    # sample two colors away from extremes for clarity
    observed_color = cmap(0.1)
    predicted_color = cmap(0.7)

    y_pred_df = y_pred_df.copy()
    tabley = tabley.copy()

    y_pred_df['DateTime'] = pd.to_datetime(y_pred_df['DateTime'])
    tabley['DateTime'] = pd.to_datetime(tabley['DateTime'])

    # Find LCZ with most stations (break ties by smallest LCZ number)
    lcz_station_counts = tabley.groupby('LCZ_100')['Location'].nunique().sort_values(ascending=False)
    max_count = lcz_station_counts.max()
    lcz_with_max = lcz_station_counts[lcz_station_counts == max_count].index
    target_lcz = lcz_with_max.min()

    print(f"Selected LCZ: {target_lcz} with {max_count} stations")

    # Filter data for this LCZ
    obs_lcz = tabley[tabley['LCZ_100'] == target_lcz].copy()
    pred_lcz = y_pred_df[y_pred_df['LCZ_100'] == target_lcz].copy()


    # For each timestamp, compute median and 25/75 percentiles
    def _group_stats(df):
        g = df.groupby('DateTime')['AbsTemp'].agg([
            'median',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75)
        ])
        g.columns = ['median', 'q25', 'q75']
        return g

    obs_grouped = _group_stats(obs_lcz)
    pred_grouped = _group_stats(pred_lcz)

    # Find hottest and coldest median observed temperatures (timestamps)
    hottest_ts = obs_grouped['median'].idxmax()
    coldest_ts = obs_grouped['median'].idxmin()
    hottest_temp = obs_grouped.loc[hottest_ts, 'median']
    coldest_temp = obs_grouped.loc[coldest_ts, 'median']

    # safe retrieval of predicted medians at those timestamps (may be missing)
    def _get_pred_value(ts, series):
        return series.loc[ts, 'median']


    # Not strictly necessary but keep original approach
    hottest_pred = _get_pred_value(hottest_ts, pred_grouped)
    coldest_pred = _get_pred_value(coldest_ts, pred_grouped)

    half_window = timedelta(days=window_days)

    hot_start = hottest_ts - half_window
    hot_end = hottest_ts + half_window
    cold_start = coldest_ts - half_window
    cold_end = coldest_ts + half_window

    # Get data for hot window (per-station then aggregate)
    hot_obs_window = obs_lcz[(obs_lcz['DateTime'] >= hot_start) & (obs_lcz['DateTime'] <= hot_end)]
    hot_pred_window = pred_lcz[(pred_lcz['DateTime'] >= hot_start) & (pred_lcz['DateTime'] <= hot_end)]

    hot_obs_stats = _group_stats(hot_obs_window)
    hot_pred_stats = _group_stats(hot_pred_window)

    # Get data for cold window
    cold_obs_window = obs_lcz[(obs_lcz['DateTime'] >= cold_start) & (obs_lcz['DateTime'] <= cold_end)]
    cold_pred_window = pred_lcz[(pred_lcz['DateTime'] >= cold_start) & (pred_lcz['DateTime'] <= cold_end)]

    cold_obs_stats = _group_stats(cold_obs_window)
    cold_pred_stats = _group_stats(cold_pred_window)

    # Prepare filename
    combined_fn = os.path.join(location_plots_dir, f"LCZ_{target_lcz}_extremes_combined.png")

    # ---------- compute y-axis spans so both subplots have same span (multiple of 5) ----------
    def _minmax_from_stats(obs_stats, pred_stats):
        # consider q25/median/q75 and medians (covers spread)
        minima = []
        maxima = []
        for df in (obs_stats, pred_stats):
            minima.append(df[['q25', 'median', 'q75']].min().min())
            maxima.append(df[['q25', 'median', 'q75']].max().max())
            return min(minima), max(maxima)


    hot_min, hot_max = _minmax_from_stats(hot_obs_stats, hot_pred_stats)
    cold_min, cold_max = _minmax_from_stats(cold_obs_stats, cold_pred_stats)


    span_hot = hot_max - hot_min
    span_cold = cold_max - cold_min
    span = max(span_hot, span_cold)

    # round span up to nearest multiple of 5
    def roundup_to(x, base=5):
        return math.ceil(x / base) * base

    span_rounded = roundup_to(span, 5)
    if span_rounded == 0:
        span_rounded = 5

    # compute centers for each subplot, then set y-limits as center ± span_rounded/2
    hot_center = 0.5 * (hot_max + hot_min)
    cold_center = 0.5 * (cold_max + cold_min)
    half_span = span_rounded / 2.0

    hot_ylim = (hot_center - half_span, hot_center + half_span)
    cold_ylim = (cold_center - half_span, cold_center + half_span)

    # ---------- plotting ----------
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=False)

    # latex_textwidth_in_inches = 6.5  
    # per_fig_width = 0.48 * latex_textwidth_in_inches

    # fig_height = 4.5  

    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(per_fig_width, fig_height), sharex=False, dpi=300)

    # Titles (do not include numerical temps—annotations will show numbers)
    ax1.set_title(f"Hottest observed temperature")
    ax2.set_title(f"Coldest observed temperature")

    # Hot subplot: medians and IQRs (IQRs have no legend entry)
    ax1.plot(hot_obs_stats.index, hot_obs_stats['median'],
             label='Observed', color=observed_color, linewidth=2)
    ax1.fill_between(hot_obs_stats.index, hot_obs_stats['q25'], hot_obs_stats['q75'],
                     color=observed_color, alpha=0.2)

    ax1.plot(hot_pred_stats.index, hot_pred_stats['median'],
             label='Predicted', color=predicted_color, linewidth=2)
    ax1.fill_between(hot_pred_stats.index, hot_pred_stats['q25'], hot_pred_stats['q75'],
                     color=predicted_color, alpha=0.2)

    # mark the hottest timestamp lines
    ax1.axvline(hottest_ts, color='k', linestyle='--', alpha=0.5, linewidth=1)

    # Annotate hottest observed & predicted at hottest_ts (if present)
    # Observed
    obs_val_hot = hot_obs_stats.loc[hottest_ts, 'median']

    # Predicted (may be missing)
    pred_val_hot = hot_pred_stats['median'].loc[hottest_ts] if hottest_ts in hot_pred_stats.index else np.nan

    # Plot markers and annotate (only if not NaN)
    def _annotate_point(ax, x, y, label, color, va='bottom'):
        ax.plot([x], [y], marker='o', color=color, markersize=6)
        # place text a little above/below
        y_off = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        if va == 'bottom':
            txt_y = y + y_off
        else:
            txt_y = y - y_off
        ax.annotate(f"{label}: {y:.2f}°C", xy=(x, y), xytext=(0, 0),
                    textcoords='offset points', xycoords='data',
                    fontsize=9, ha='center', va=va,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # set y-limits for ax1 before annotations to get reasonable offset
    ax1.set_ylim(hot_ylim)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax1.grid(True, linestyle=':', alpha=0.6)

    # _annotate_point(ax1, hottest_ts, obs_val_hot, "Obs", observed_color, va='bottom')
    # _annotate_point(ax1, hottest_ts, pred_val_hot, "Pred", predicted_color, va='top')

    ax1.set_xlim(hot_start, hot_end)
    ax1.set_ylabel('Temperature [°C]')
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Legend only here (no IQR entries because fills have no label)
    ax1.legend(loc='best')

    # Cold subplot
    ax2.plot(cold_obs_stats.index, cold_obs_stats['median'], label='Observed', color=observed_color, linewidth=2)
    ax2.fill_between(cold_obs_stats.index, cold_obs_stats['q25'], cold_obs_stats['q75'], color=observed_color, alpha=0.2)

    ax2.plot(cold_pred_stats.index, cold_pred_stats['median'], label='Predicted', color=predicted_color, linewidth=2)
    ax2.fill_between(cold_pred_stats.index, cold_pred_stats['q25'], cold_pred_stats['q75'], color=predicted_color, alpha=0.2)

    ax2.axvline(coldest_ts, color='k', linestyle='--', alpha=0.5, linewidth=1)

    # compute observed/predicted values for cold timestamp (may be missing)
    obs_val_cold = cold_obs_stats.loc[coldest_ts, 'median']

    pred_val_cold = cold_pred_stats['median'].loc[coldest_ts] if coldest_ts in cold_pred_stats.index else np.nan

    # set y-limits and ticks
    ax2.set_ylim(cold_ylim)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax2.grid(True, linestyle=':', alpha=0.6)

    # _annotate_point(ax2, coldest_ts, obs_val_cold, "Obs", observed_color, va='top')
    # _annotate_point(ax2, coldest_ts, pred_val_cold, "Pred", predicted_color, va='bottom')

    ax2.set_xlim(cold_start, cold_end)
    ax2.set_ylabel('Temperature [°C]')
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))

    # rotate x labels
    for ax in (ax1, ax2):
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

    plt.tight_layout()
    if save:
        # save as vector PDF that will not be rescaled when included at width=0.48\textwidth
        combined_fn_pdf = os.path.splitext(combined_fn)[0] + ".pdf"
        plt.savefig(combined_fn_pdf, bbox_inches='tight')
        # optional: also save a PNG for quick preview
        combined_fn_png = os.path.splitext(combined_fn)[0] + ".png"
        plt.savefig(combined_fn_png, dpi=300, bbox_inches='tight')
        print(f"Saved LCZ extreme weeks plot to: {combined_fn_pdf} and {combined_fn_png}")
    if show:
        plt.show()
    plt.close(fig)

    result = {
        'lcz': target_lcz,
        'n_stations': max_count,
        'hot': {
            'timestamp': hottest_ts,
            'median_temp': hottest_temp,
            'obs_stats': hot_obs_stats,
            'pred_stats': hot_pred_stats
        },
        'cold': {
            'timestamp': coldest_ts,
            'median_temp': coldest_temp,
            'obs_stats': cold_obs_stats,
            'pred_stats': cold_pred_stats
        },
        'combined_filename': combined_fn
    }

    return result



def calculate_extreme_csi(y_pred_df, tabley, percentile=99, extreme_type='hot', verbose=True):
    """
    Calculate Critical Success Index (CSI) and False Alarm Rate (FAR) for extreme temperature events.
    
    Parameters:
    -----------
    y_pred_df : pd.DataFrame
        DataFrame containing predicted temperatures with 'AbsTemp' column
    tabley : pd.DataFrame
        DataFrame containing observed temperatures with 'AbsTemp' column
    percentile : float
        Percentile threshold for defining extremes (default: 99)
        For hot extremes: uses the percentile value (e.g., 99th)
        For cold extremes: uses (100 - percentile) (e.g., 1st percentile for 99)
    extreme_type : str
        'hot' for hot extremes or 'cold' for cold extremes
    verbose : bool
        If True, print detailed results
    
    Returns:
    --------
    dict : Dictionary containing CSI, FAR, and all confusion matrix elements
    """
    
    obs_temps = tabley['AbsTemp'].values
    pred_temps = y_pred_df['AbsTemp'].values
    
    # Calculate threshold based on observed temperatures
    if extreme_type.lower() == 'hot':
        threshold = np.percentile(obs_temps, percentile)
        obs_extreme = obs_temps >= threshold
        pred_extreme = pred_temps >= threshold
        extreme_label = f"{percentile}th percentile (hottest)"
    elif extreme_type.lower() == 'cold':
        threshold = np.percentile(obs_temps, 100 - percentile)
        obs_extreme = obs_temps <= threshold
        pred_extreme = pred_temps <= threshold
        extreme_label = f"{100-percentile}th percentile (coldest)"
    else:
        raise ValueError("extreme_type must be 'hot' or 'cold'")
    
    # Calculate confusion matrix elements
    T = np.sum(obs_extreme & pred_extreme)    # True Positives
    F = np.sum(~obs_extreme & pred_extreme)   # False Positives
    R = np.sum(obs_extreme & ~pred_extreme)   # False Negatives (Misses)
    TN = np.sum(~obs_extreme & ~pred_extreme) # True Negatives (for reference)
    
    # Calculate metrics
    CSI = T / (T + F + R) if (T + F + R) > 0 else 0
    FAR = F / (F + T) if (F + T) > 0 else 0
    POD = T / (T + R) if (T + R) > 0 else 0  # Probability of Detection (bonus metric)
    
    F1 = f1_score(obs_extreme, pred_extreme)
    Precision = precision_score(obs_extreme, pred_extreme, zero_division=0)
    Recall = recall_score(obs_extreme, pred_extreme, zero_division=0)

    results = {
        'extreme_type': extreme_type,
        'percentile': percentile,
        'threshold': threshold,
        'true_positives': int(T),
        'false_positives': int(F),
        'false_negatives': int(R),
        'true_negatives': int(TN),
        'CSI': CSI,
        'FAR': FAR,
        'POD': POD,
        'F1': F1,              
        'Precision': Precision, 
        'Recall': Recall,      
        'total_extreme_obs': int(T + R),
        'total_extreme_pred': int(T + F)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Extreme Event Analysis: {extreme_type.upper()} extremes")
        print(f"Threshold: {extreme_label} = {threshold:.2f}°C")
        print(f"{'='*60}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (Hits):        {T:6d}")
        print(f"  False Positives (False Alarms): {F:6d}")
        print(f"  False Negatives (Misses):     {R:6d}")
        print(f"  True Negatives:               {TN:6d}")
        print(f"\nPerformance Metrics:")
        print(f"  Critical Success Index (CSI): {CSI:.4f}")
        print(f"  False Alarm Rate (FAR):       {FAR:.4f}")
        print(f"  Probability of Detection (POD): {POD:.4f}")
        print(f"  F1 Score:                     {F1:.4f}")      # 
        print(f"  Precision:                    {Precision:.4f}") #
        print(f"  Recall (=POD):                {Recall:.4f}")    #
        print(f"\nExtreme Event Counts:")
        print(f"  Observed extreme events:      {T + R:6d}")
        print(f"  Predicted extreme events:     {T + F:6d}")


        print(f"{'='*60}\n")
    
    return results



def calculate_rmse(predictions, observations):
    """Calculate RMSE between predictions and observations"""
    return np.sqrt(np.mean((predictions - observations)**2))

def plot_sparse_rmse_map(y_pred_df, tabley, x_vals_3035, y_vals_3035,
                         output_dir, city, month, start_year,
                         grid_resolution=100, vmin=0, vmax=3):
    """
    Create a pcolormesh map showing RMSE at sparse station locations.

    Parameters
    ----------
    grid_resolution : int or float
        Cell size in meters (e.g. 100 means 100 m x 100 m cells).
    """
    # Extract coordinates and temperatures
    lats = y_pred_df['Latitude'].values
    lons = y_pred_df['Longitude'].values
    pred_temps = y_pred_df['AbsTemp'].values
    obs_temps = tabley['AbsTemp'].values

    # residuals squared for RMSE
    residuals = pred_temps - obs_temps

    # Convert station lat/lon to EPSG:3035
    transformer_to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_stations, y_stations = transformer_to_3035.transform(lons, lats)

    # Use existing grid extent from x_vals_3035 and y_vals_3035 (assumed in EPSG:3035 meters)
    x_min, x_max = float(np.min(x_vals_3035)), float(np.max(x_vals_3035))
    y_min, y_max = float(np.min(y_vals_3035)), float(np.max(y_vals_3035))

    # Guard against zero extent
    if x_max <= x_min:
        raise ValueError("x_vals_3035 has zero or negative extent")
    if y_max <= y_min:
        raise ValueError("y_vals_3035 has zero or negative extent")

    # grid_resolution is cell size in meters
    cell_size = float(grid_resolution)

    # compute number of cells in each direction
    n_x = int(np.ceil((x_max - x_min) / cell_size))
    n_y = int(np.ceil((y_max - y_min) / cell_size))

    # safety: avoid ridiculously large arrays
    max_cells = 2000 * 2000  # adjust if you know your machine can handle more
    if n_x * n_y > max_cells:
        raise MemoryError(f"Requested grid would have {n_x*n_y} cells (> {max_cells}). "
                          "Increase cell size or reduce extent.")

    # Build edges such that grid covers the full extent
    x_edges = np.linspace(x_min, x_min + n_x * cell_size, n_x + 1)
    y_edges = np.linspace(y_min, y_min + n_y * cell_size, n_y + 1)

    # Calculate RMSE per grid cell using binned_statistic_2d
    rmse_mean_sq, _, _, _ = binned_statistic_2d(
        x_stations, y_stations, residuals**2,
        statistic='mean',
        bins=[x_edges, y_edges]
    )

    # Take square root to get RMSE and transpose to match plotting orientation
    rmse_grid = np.sqrt(rmse_mean_sq).T

    # Count number of observations per cell
    count_grid, _, _, _ = binned_statistic_2d(
        x_stations, y_stations, residuals,
        statistic='count',
        bins=[x_edges, y_edges]
    )
    count_grid = count_grid.T

    # Mask cells with no data
    rmse_grid = np.ma.masked_where(count_grid == 0, rmse_grid)

    # Create grid centers for pcolormesh (centers, then transformed to lon/lat)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx_centers, yy_centers = np.meshgrid(x_centers, y_centers)

    # Transform grid centers to lat/lon for plotting
    transformer_to_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons_plot, lats_plot = transformer_to_4326.transform(
        xx_centers.ravel(), yy_centers.ravel()
    )
    lons_plot = lons_plot.reshape(xx_centers.shape)
    lats_plot = lats_plot.reshape(yy_centers.shape)

    # Setup projection
    proj_3035 = ccrs.LambertAzimuthalEqualArea(
        central_longitude=10.0, central_latitude=52.0,
        false_easting=4321000.0, false_northing=3210000.0
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj_3035})
    ax.set_extent([x_min, x_max, y_min, y_max], crs=proj_3035)

    # Create colormap and normalization
    # cmap = cmcr.bilbao_r
    # norm = Normalize(vmin=vmin, vmax=vmax)

    # create 0.0,0.5,...,3.0 boundaries (7 edges → 6 bins)
    boundaries = np.arange(0.0, 3.0 + 1e-8, 0.5)  # [0.0,0.5,...,3.0]

    # sample the continuous colormap to create a discrete 6-color cmap
    base_cmap = cmcr.acton_r
    cmap_discrete = ListedColormap(base_cmap(np.linspace(0, 1, len(boundaries)-1)))

    # discrete normalization using the boundaries
    norm = BoundaryNorm(boundaries, ncolors=cmap_discrete.N, clip=True)

    # use cmap_discrete below in pcolormesh
    cmap = cmap_discrete

    # Plot RMSE with pcolormesh
    cs = ax.pcolormesh(
        lons_plot, lats_plot, rmse_grid,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading='auto',
        alpha=0.6
    )

    # # Add station markers
    # ax.scatter(
    #     lons, lats,
    #     c='black',
    #     s=10,
    #     alpha=0.7,
    #     transform=ccrs.PlateCarree(),
    #     zorder=10,
    #     edgecolors='white',
    #     linewidths=0.5
    # )

    # Add basemap
    ctx.add_basemap(ax, crs=proj_3035, source=ctx.providers.CartoDB.Positron)

    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('RMSE [K]')

    # Add title
    ax.set_title(f'Station-based RMSE')

    # Add scale bar
    scale_length = 5000
    padding_x = (x_max - x_min) * 0.02
    padding_y = (y_max - y_min) * 0.02
    bar_x_start = x_max - scale_length - padding_x
    bar_y = y_min + padding_y
    bar_height = (y_max - y_min) * 0.005

    scale_bar = Rectangle(
        (bar_x_start, bar_y),
        scale_length,
        bar_height,
        facecolor='black',
        edgecolor='black',
        linewidth=1,
        transform=proj_3035,
        zorder=6
    )
    ax.add_patch(scale_bar)

    ax.text(
        bar_x_start + scale_length/2,
        bar_y + bar_height + padding_y*0.2,
        '5 km',
        ha='center',
        va='bottom',
        fontsize=10,
        transform=proj_3035,
        zorder=6
    )

    plt.tight_layout()

    # Save
    analysis_dir = os.path.join(output_dir, f'{start_year}_{month}_{city}', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    filename = os.path.join(analysis_dir, 'rmse_spatial_map.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate statistics
    overall_rmse = calculate_rmse(pred_temps, obs_temps)
    n_stations = len(np.unique(y_pred_df["Location"]))

    print(f"Saved RMSE map to {filename}")
    print(f"Overall RMSE: {overall_rmse:.2f}°C")
    print(f"Stations: {n_stations}")
    print(f"Grid cells with data: {np.sum(count_grid > 0)}/{n_x*n_y}")
    print(f"Average observations per cell: {np.mean(count_grid[count_grid > 0]):.1f}")

    return rmse_grid, count_grid


# add imports if not already present
import numpy as np
from pyproj import Transformer

def compute_square_extent(center_lon, center_lat,
                          station_lons=None, station_lats=None,
                          station_x=None, station_y=None,
                          padding=0.05,    # fraction of half-size to add as margin
                          min_half_size=5000.0):
    """
    Return a square extent in EPSG:3035 coordinates centered on (center_lon, center_lat)
    that is large enough to include the furthest station.
    
    Parameters
    ----------
    center_lon, center_lat : floats
        center in degrees (EPSG:4326).
    station_lons, station_lats : arrays or None
        arrays of station longitudes/latitudes (deg). Used if station_x/y not provided.
    station_x, station_y : arrays or None
        arrays of station coordinates in EPSG:3035 (metres). If provided, these are used directly.
    padding : float
        fraction (e.g. 0.05 -> 5%) added to the half-size to avoid points sitting on the edge.
    min_half_size : float
        minimum half-side length in metres to avoid a tiny box for tightly clustered stations.
    
    Returns
    -------
    extent : tuple (xmin, xmax, ymin, ymax)  # all in EPSG:3035 metres
    center_proj : tuple (center_x, center_y, half_size)
    """
    # transformer to EPSG:3035 (always_xy -> lon,lat order)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)

    if (station_x is not None) and (station_y is not None):
        x = np.asarray(station_x, dtype=float)
        y = np.asarray(station_y, dtype=float)
    elif (station_lons is not None) and (station_lats is not None):
        lons = np.asarray(station_lons, dtype=float)
        lats = np.asarray(station_lats, dtype=float)
        x, y = transformer.transform(lons, lats)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    else:
        raise ValueError("Provide either station_x & station_y (EPSG:3035) or station_lons & station_lats (deg).")

    # distances (metres)
    dists = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    maxd = float(np.nanmax(dists)) if dists.size else 0.0

    # half-size = max distance, but at least min_half_size
    half_size = max(maxd, float(min_half_size))
    half_size *= (1.0 + float(padding))   # add margin

    xmin = center_x - half_size
    xmax = center_x + half_size
    ymin = center_y - half_size
    ymax = center_y + half_size

    return (xmin, xmax, ymin, ymax), (center_x, center_y, half_size)


def plot_station_metric_points(y_pred_df, tabley, x_vals_3035, y_vals_3035,
                               output_dir, city, month, start_year,
                               center_lon, center_lat, metric='rmse',
                               vmin=None, vmax=None):
    """
    Plot station metrics (RMSE, R², or Bias) as colored points.
    
    Parameters:
    -----------
    metric : str
        'rmse', 'r2', or 'bias'
    """
    from sklearn.metrics import r2_score
    
    # Calculate metric per station
    stations = y_pred_df['Location'].unique()
    station_data = []
    
    for station in stations:
        mask = y_pred_df['Location'] == station
        pred = y_pred_df[mask]['AbsTemp'].values
        obs = tabley[mask]['AbsTemp'].values
        
        if metric == 'rmse':
            value = calculate_rmse(pred, obs)
        elif metric == 'r2':
            value = r2_score(obs, pred)
        elif metric == 'bias':
            value = np.mean(pred - obs)  # Bias = mean(predicted - observed)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        lat = y_pred_df[mask]['Latitude'].iloc[0]
        lon = y_pred_df[mask]['Longitude'].iloc[0]
        
        station_data.append({
            'lat': lat,
            'lon': lon,
            'value': value,
            'n_obs': len(pred)
        })
    
    station_df = pd.DataFrame(station_data)
    
    # Setup projection
    proj_3035 = ccrs.LambertAzimuthalEqualArea(
        central_longitude=10.0, central_latitude=52.0,
        false_easting=4321000.0, false_northing=3210000.0
    )
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj_3035})
    
    # Compute extent
    extent, (cx, cy, half) = compute_square_extent(
        center_lon=center_lon, center_lat=center_lat,
        station_x=x_vals_3035, station_y=y_vals_3035,
        padding=0.05, min_half_size=5000.0
    )
    
    x_min, x_max = extent[0], extent[1]
    y_min, y_max = extent[2], extent[3]
    ax.set_extent([x_min, x_max, y_min, y_max], crs=proj_3035)
    
    # Add basemap
    ctx.add_basemap(ax, crs=proj_3035, source=ctx.providers.CartoDB.Positron)
    
    # Configure colormap for stations based on metric
    if metric == 'rmse':
        if vmin is None: vmin = 0.0
        if vmax is None: vmax = 5.0
        boundaries = np.arange(vmin, vmax + 0.1, 0.5)
        base_cmap = cmcr.acton_r
        label = 'RMSE [K]'
        title = 'Station RMSE'
    elif metric == 'r2':
        if vmin is None: vmin = 0.8
        if vmax is None: vmax = 1.0
        boundaries = np.linspace(vmin, vmax, 9)
        base_cmap = cmcr.bamako
        label = 'R²'
        title = 'Station R²'
    elif metric == 'bias':
        if vmin is None and vmax is None:
            abs_max = np.ceil(np.abs(station_df['value']).max() * 2) / 2
            vmin, vmax = -abs_max, abs_max
        boundaries = np.linspace(vmin, vmax, 9)
        base_cmap = cmcr.vik
        label = 'Bias [K]'
        title = 'Station Bias'
    
    # Create discrete colormap
    cmap_discrete = ListedColormap(base_cmap(np.linspace(0, 1, len(boundaries)-1)))
    norm = BoundaryNorm(boundaries, ncolors=cmap_discrete.N, clip=True)
    
    # Plot stations as scatter points
    scatter = ax.scatter(
        station_df['lon'], station_df['lat'],
        c=station_df['value'],
        s=150,
        cmap=cmap_discrete,
        norm=norm,
        edgecolors='black',
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
        zorder=10
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    
    # Scale bar
    scale_length = 5000
    padding_x = (x_max - x_min) * 0.02
    padding_y = (y_max - y_min) * 0.02
    bar_x_start = x_max - scale_length - padding_x
    bar_y = y_min + padding_y
    bar_height = (y_max - y_min) * 0.005
    
    scale_bar = Rectangle(
        (bar_x_start, bar_y), scale_length, bar_height,
        facecolor='black', edgecolor='black', linewidth=1,
        transform=proj_3035, zorder=11
    )
    ax.add_patch(scale_bar)
    ax.text(
        bar_x_start + scale_length/2, bar_y + bar_height + padding_y*0.2,
        '5 km', ha='center', va='bottom', fontsize=10,
        transform=proj_3035, zorder=11
    )
    
    plt.tight_layout()
    
    # Save
    analysis_dir = os.path.join(output_dir, f'{start_year}_{month}_{city}', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    filename = os.path.join(analysis_dir, f'{metric}_station_points.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nSaved {metric.upper()} station map to {filename}")
    print(f"Stations: {len(stations)}")
    print(f"{metric.upper()} range: {station_df['value'].min():.3f} - {station_df['value'].max():.3f}")
    print(f"Mean {metric.upper()}: {station_df['value'].mean():.3f}")
    
    return station_df