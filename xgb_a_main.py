import pandas as pd
from xgb_0_prep_params import *
#from xgb_0_functions import *
from xgb_a_add_predictors import *

if __name__ == "__main__":
    tablex = pd.read_pickle(f"dataframes/tablex_{city}.pkl")
    tabley = pd.read_pickle(f"dataframes/tabley_{city}.pkl")

    print(f"predictor table before NaN treatment for {city}:",tablex)
    print(f"predictand table before NaN treatment for {city}:",tabley)

    mask = tabley.notna().all(axis=1) & tablex.notna().all(axis=1)

    tabley = tabley[mask].reset_index(drop=True)
    tablex = tablex[mask].reset_index(drop=True)

    # print("predictor table before NaN treatment:",tablex)
    # tablex = tablex.dropna()
    print(f"predictor table after NaN treatment for {city}:",tablex)
    
    # print("predictand table before NaN treatment:",tabley)
    # tabley = tabley.iloc[tablex.index]
    print(f"predictand table after NaN treatment for {city}:",tabley)
    

    tablex = add_geo(tablex,city,country)
    print(tablex)
    tablex = add_meteo(tablex, lat0, lon0)


    tabley["Temperature"] = tabley["Temperature"] - tablex["t2m"]
    


    print("predictor table after NaN removal:")
    print(tablex)
    print("predictand table after NaN removal:")
    print(tabley)



    for name in ["tablex", "tabley"]:
        df = globals()[name]
        df.to_pickle(f"dataframes_ready/{name}_{city}.pkl")