# Code repository for XGBoost prediction of air temperature in European Cities

Observed temperature data is obtained from the FAIRUrbTemp dataset: https://boris-portal.unibe.ch/entities/product/1b11be9a-c79c-4045-82bb-90fde3ca6189 
with the acompanying paper: 

Amini, S., Huerta, A., Franke, J., Brugnara, Y., Caluwaerts, S., Anet, J., Savic, S., Gubler, M.,
Steeneveld, G.-J., Chapman, L., Meier, F., Dubreuil, V., Christen, A., Zeeman, M.,
Lalic, B., Schlögl, S., Käyhkö, J., Azadfar, A., & Brönnimann, S. (2026). 
Comprehensive compilation and quality assessment of street-level urban air temperature measurements across European networks. _Scientific Data_ _2026_.
https://doi.org/10.1038/s41597-026-06804-4

This codebase was used for the following manuscript:
Pierce, C., Burger, M. & Brönnimann, S. (2026)
Investigating Generalization Capabilities of an XGBoost-Based Model for Urban Air Temperature Prediction in European Cities. _Machine Learning: Earth_.

Short description of files:

xgb_0_prep_params.py serves as a namelist, it controls most inputs/outputs (cities, split type, features etc...)

xgb_0_functions.py contains base functions for the scripts and is hence fully imported.

xgb_b_main.py is the caller script for the XGBoost model which is defined in xgb_b_make_model.py.

xgb_d_check.py is for the error metrics of predicted vs observed.

xgb_d_train_and_check.py is the wrapper that iteratively calls xgb_b_main.py (for training) and xgb_d_check.py (for predicting) for cities defined in xgb_0_prep_params.py


