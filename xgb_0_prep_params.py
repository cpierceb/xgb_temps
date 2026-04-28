from datetime import datetime, timedelta, timezone
from geopy.geocoders import Nominatim, Photon
import os
import pandas as pd

#Directories and strings
era5land_dir = "../raw_data/era5land"
era5_dir = "../raw_data/era5"
era5pressure_dir = "../raw_data/era5pressure"
ktoc = 273.15


# Check if running from pipeline
if 'PIPELINE_TARGET_CITY' in os.environ:
    city = os.environ['PIPELINE_TARGET_CITY']
    country = os.environ['PIPELINE_TARGET_COUNTRY']
    # print(f"Using pipeline target: {city}")#, {country}")
    print("===============================new start====================================")
    print(f"Using pipeline target: {city}, {country}")
else:
    city = "novisad"
    country = "serbia" 

TRAIN_CUT_DATE = pd.Timestamp('2020-05-15', tz='UTC')
JJA_END_DATE   = pd.Timestamp('2020-07-15', tz='UTC')
SPLIT_TYPE = 'jja2020'  # or 'last_year' 'jja2021' 'spatial' 'jja2020'
MIN_TRAIN_SAMPLES = 10000
MIN_TEST_SAMPLES = 10000


# geolocator = Nominatim(user_agent="MyApp",timeout=10)
# if city == "birmingham":
#     lat0, lon0 = 52.481526356041286, -1.8916838972710635
# else:
geolocator = Photon(user_agent="MyApp",timeout=10)
location = geolocator.geocode(f"{city}, {country}")
lat0, lon0 = location.latitude, location.longitude

    

#Application of the model (xgb_d)

#needs the utc for check but not for apply
start_year = 2014
month = 6
start_day = 1
start_time = datetime(start_year, month, start_day, 0, 0, tzinfo=timezone.utc)  
amount_of_days = 365*7 + 366*3 + 30
# amount_of_days = 15
grid_size = None

#Options for Data processing
wanna_scale = 0
onlyt = 1
rhactivate = 1 #1 for relative humidity, 0 for dewpoint temperature
temp_diff = 1 #1 for predicting temperature difference from era5, 0 for predicting absolute
combination = 1



#for xgb_2 and xgb_3
features = [
    # "Latitude" , "Longitude", # 0 1
    # "X3035", "Y3035", #2 3 
    # "BH_10", 
    "BH_100", 
    "BH_100_250rad",
    "DEM_30", 
    "DEM_90", 
    "LCZ_100", 
    "LCZ_water_500rad",
    # "LCZ_E", "LCZ_NE", "LCZ_N", "LCZ_NW", "LCZ_W", "LCZ_SW", "LCZ_S", "LCZ_SE",
    # "TCD_10", 
    "TCD_100",
    "TCD_100_250rad",
    # "IMP_10", 
    "IMP_100",
    "IMP_100_250rad",
    "t2m", "d2m", "ssrd" , "tp",#7 8 9 10
    "u10", "v10", "sp" , #11 12 13
    "swvl1", "swvl3", #14 15 
    "z_950", "z_850", "z_500", "u_950", "u_850", "u_500", "v_950", "v_850", "v_500", #16 17 18 19 20 21 22 23 24
    # "Hour","Day", "Month", "Year" #25 26 27 28
]



#Plotting xgb_4
discrete_plotting = 1
beginning_temp = 15 #start of temperature scale
temp_inc = 20 #how much you add to beginning
level_jump = 1 #change color every level_jump degrees