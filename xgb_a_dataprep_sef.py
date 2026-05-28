import os
import glob
import pandas as pd
from xgb_0_prep_params import *
import matplotlib.pyplot as plt

def read_tsv_with_metadata(path):
    """
    Read a TSV file with metadata header lines until the line starting with 'Year'.
    Returns the data DataFrame and metadata dict.
    """
    metadata = {}
    header_line = None
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t', 1)
            if parts and parts[0] == 'Year':
                header_line = i
                break
            if len(parts) == 2:
                key, val = parts
                metadata[key] = val
    if header_line is None:
        raise ValueError(f"No data header found in {path}")

    # Read data into DataFrame, treat 'NA' as missing
    df = pd.read_csv(
        path,
        sep='\t',
        header=header_line,
        dtype={
            'Year': 'Int64',      # Changed from int
            'Month': 'Int64',     # Changed from int
            'Day': 'Int64',       # Changed from int
            'Hour': 'Int64',      # Changed from int
            'Minute': 'Int64',    # Changed from int
            'Period': str,
            'Meta': str
        },
        na_values=['NA', '   NA'],  # Added '   NA' for the spaces in your data
        keep_default_na=True,
        low_memory=False
    )
    return df, metadata

def process_folder(folder):
    """
    Process all TSV files in a folder, keeping only hourly (minute == 0) non-null measurements from 2018 onwards.
    Filters out data with QC flags and keeps only data from 2018 and later.
    Returns combined tablex (Time_UTC, Latitude, Longitude) and tabley (Temperature).
    """
    all_records = []
    for path in glob.glob(os.path.join(folder, '*.tsv')):
        #print(f"Processing: {os.path.basename(path)}")
        df, meta = read_tsv_with_metadata(path)
        
        # Filter full hours and drop missing values in Value (NOW after conversion)
        df = df[df['Minute'] == 0].dropna(subset=['Value'])

        # Filter to keep only data from june 2014 onwards
        df = df[
            (df['Year'] > 2014) | 
            ((df['Year'] == 2014) & (df['Month'] >= 6))
        ]
        
        # Convert Value to numeric FIRST, coerce non-numeric (including 'NA') to NaN
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])
        
        # Filter out rows with QC flags in Meta column
        # Keep only rows where Meta is NaN, empty string, or just whitespace
        df = df[
            df['Meta'].isna() | 
            (df['Meta'].str.strip() == '') | 
            df['Meta'].str.strip().str.startswith('orig.time', na=False)
        ]
        
        # # Amsterdam-specific: filter out 0.00 values
        # if city.lower() == 'amsterdam':
        #     df = df[df['Value'] != 0.00]
        

        
        if df.empty:
            print(f"  No valid data found in {os.path.basename(path)} for 2014+")
            continue
            
        # Construct UTC datetime
        df['Time_UTC'] = pd.to_datetime(
            df[['Year','Month','Day','Hour','Minute']]
                .assign(Second=0),
            utc=True
        )
        
        # Add metadata coords
        lat = float(meta.get('Lat', 'nan'))
        lon = float(meta.get('Lon', 'nan'))
        df['Latitude'] = lat
        df['Longitude'] = lon
        
        # Standardize column names
        df = df.rename(columns={'Value': 'Temperature'})
        
        print(f"  Added {len(df)} records from {os.path.basename(path)}")
        all_records.append(df)

    if not all_records:
        print("Warning: No valid data found in any files!")
        return pd.DataFrame(columns=['Time_UTC', 'Latitude', 'Longitude']), pd.DataFrame(columns=['Temperature'])
    
    combined = pd.concat(all_records, ignore_index=True)
    combined = combined.sort_values('Time_UTC').reset_index(drop=True)
    
    # Create tables
    tablex = combined[['Time_UTC', 'Latitude', 'Longitude']]
    tabley = combined[['Temperature']]
    
    return tablex, tabley

if __name__ == '__main__':
    city = "zurich"
    folder = f"../raw_data/temperature/{city}"  # Updated folder name
    tablex, tabley = process_folder(folder)
    print(f"\nResult shapes: tablex={tablex.shape}, tabley={tabley.shape}")
    
    if not tablex.empty:
        print(f"Date range: {tablex['Time_UTC'].min()} to {tablex['Time_UTC'].max()}")
        print(f"Temperature range: {tabley['Temperature'].min():.2f}°C to {tabley['Temperature'].max():.2f}°C")
        print("\nFirst few rows of tablex:")
        print(tablex.head())
        print("\nFirst few rows of tabley:")
        print(tabley.head())

        # Save to pickle
        os.makedirs('dataframes', exist_ok=True)
        tablex.to_pickle(f"dataframes/tablex_{city}.pkl")  # Updated filename
        tabley.to_pickle(f"dataframes/tabley_{city}.pkl")  # Updated filename
        print(f"\nSaved to dataframes/tablex_{city}.pkl and dataframes/tabley_{city}.pkl")
    else:
        print("No data to save!")


    if not tablex.empty:
        monthly_counts = (
            tablex.assign(YearMonth=tablex['Time_UTC'].dt.to_period('M'))
            .groupby('YearMonth')
            .size()
        )

        fig, ax = plt.subplots(figsize=(14, 4))
        monthly_counts.plot(kind='bar', ax=ax, width=0.8, color='steelblue')
        ax.set_title(f'Observations per month – {city.title()}')
        ax.set_xlabel('')
        ax.set_ylabel('N observations')
        # Show only every 6th tick label to avoid clutter
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[::6])
        ax.set_xticklabels([str(monthly_counts.index[i]) for i in ticks[::6]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'obs_per_month_{city}.png', dpi=150)
        plt.show()

    yearly_counts = tablex.groupby(tablex['Time_UTC'].dt.year).size()
    print("\nObservations per year:")
    for year, count in yearly_counts.items():
        print(f"  {year}: {count:,}")