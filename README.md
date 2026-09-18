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

aoa_analysis.py : for the AOA analysis once models are already trained. Gridding functions from xgb_grid_predict.py necessary to run

climate_cluster.py is obsolete and not used for the publication or workflows

cv_folds.py holds the structure for the different CV folds, necessary for run_cv.py

run_cv.py trains and evaluates models according to the folds in cv_folds.py

xgb_0_functions.py contains base functions for the scripts and is hence fully imported.

xgb_0_prep_params.py serves as a namelist, it controls most inputs/outputs (cities, split type, features etc...)

xgb_a_add_predictors.py holds functions for dataframe and gridding constructions

xgb_a_dataprep_csv.py for processing CSV data

xgb_a_dataprep_sef.py for processing SEF data (FAIRUrbTemp data)

xgb_a_main.py runner script for preprocessing

xgb_a_runthemain.py wrapper script for xgb_a_main.py to process multiple cities at once

xgb_b_main.py is the caller script for the XGBoost model which is defined in xgb_b_make_model.py.

xgb_b_make_model.py holds the model parameters

xgb_d_check.py is for the error metrics of predicted vs observed.

xgb_d_functions.py holds mainly plotting functions

xgb_run_nested_cv.py runs a nested spatio-temporal CV, including tuning on inner CV set

xgb_tune.py for tuning XGBoost hyperparameters

xgb_z_grid_predict.py for gridding and predicting on the grid

xgb_z_plot.py for simple plots and representations
