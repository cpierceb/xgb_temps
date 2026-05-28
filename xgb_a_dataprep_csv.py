import pandas as pd
import numpy as np
import glob

CITIES = ['bern', 'biel', 'lausanne', 'thun', 'winterthur', 'zurich']
BASE_PATH = '../raw_data/temperature/urs/'


def read_logger_csv(path, logger_coords):
    df = pd.read_csv(path, sep=None, engine='python', na_values=['NA'])

    # Parse time column
    time_col = df.columns[0]
    df['Time_UTC'] = pd.to_datetime(df[time_col], format='mixed', dayfirst=True, utc=True)
    df = df.drop(columns=time_col)
    
    # Wide → long
    df = df.melt(id_vars='Time_UTC', var_name='logger', value_name='Temperature')
    df = df.dropna(subset=['Temperature'])
    
    # Keep only full hours
    df = df[df['Time_UTC'].dt.minute == 0]
    
    # Add coordinates from lookup
    df['Latitude']  = df['logger'].map(lambda s: logger_coords.get(s, (np.nan, np.nan))[0])
    df['Longitude'] = df['logger'].map(lambda s: logger_coords.get(s, (np.nan, np.nan))[1])
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    df = df.sort_values('Time_UTC').reset_index(drop=True)
    
    tablex = df[['Time_UTC', 'Latitude', 'Longitude']]
    tabley = df[['Temperature']]

    print(tablex)
    print(tabley)
    
    return tablex, tabley

for city in CITIES:
    print(f"working on {city}")
    meta_path = glob.glob(BASE_PATH + f'{city}/*metadata.csv')[0]
    data_path = glob.glob(BASE_PATH + f'{city}/*_pcd.csv')[0]

    meta = pd.read_csv(meta_path, sep=None, engine='python', index_col='Nr', encoding='latin-1')
    meta.columns = meta.columns.str.strip()
    lat_col = 'Latitude' if 'Latitude' in meta.columns else 'Lat'
    lon_col = 'Longitude' if 'Longitude' in meta.columns else 'Lon'
    logger_coords = {nr: (row[lat_col], row[lon_col]) for nr, row in meta.iterrows()}
        
    tablex, tabley = read_logger_csv(data_path, logger_coords)

    tablex.to_pickle(f'../data_processing/dataframes/tablex_urs_{city}.pkl')
    tabley.to_pickle(f'../data_processing/dataframes/tabley_urs_{city}.pkl')
