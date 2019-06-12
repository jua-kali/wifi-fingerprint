#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0_data_preprocessing.py must be run first to generate the preprocessed csv.

Status: IN PROGRESS
Purpose: Predict Wifi location in test competition set
    + Import pre-processed data from 0data.py, 
    + Autotune models

    
Created on June 06 2019
@author: Laura Stupin

"""

# %% Parameters for running models --------------------------------------------------------------------

RANDOM_SEED = 42                   # Set random seed
CORES = 3                  # Number of computer cores to use

# Setup --------------------------------------------------------------------

# Import standard libraries
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, r2_score


# Load data -------------------------------------------------------

# This data has already been cleaned/processed by 0_data_preprocessing.py
waps_all = pd.read_csv('data/preprocessed/all_waps.csv')
waps_fixed = waps_all[waps_all['location_method'] == 'fixed']
waps_random = waps_all[waps_all['location_method'] == 'random']

# Build list of names of all the columns that start with 'WAP'
wap_names = [col for col in waps_all if col.startswith('WAP')]


# %% Setup test/train sampled datasets -----------------------------------------

# For now we assume that we will train on the fixed locations and test with 
# the random locations. These are two completely different data sets.
train = waps_fixed
test = waps_random

X_train = train[wap_names].copy()
X_test = test[wap_names].copy()

# Predict the building -------------------------------------------------------

y_train = train['BUILDINGID']
y_test = test['BUILDINGID']

rf_building = RandomForestClassifier(n_estimators=300)
rf_building.fit(X_train, y_train)
y_pred_build = rf_building.predict(X_test)

print('BUILDING')
print('Accuracy:', rf_building.score(X_test, y_test))
print('Kappa:', cohen_kappa_score(y_pred_build, y_test))
print('')

#Accuracy: 0.9981998199819982
#Kappa: 0.9971547547096364


# %% Predict Floor -----------------------------------------------------------

# Add building prediction to the features
X_train['pred_building'] = rf_building.predict(X_train)
X_test['pred_building'] = rf_building.predict(X_test)

y_train = train['FLOOR']
y_test = test['FLOOR']

rf_floor = RandomForestClassifier(n_estimators=500, n_jobs = CORES)
rf_floor.fit(X_train, y_train)
y_pred_floor = rf_floor.predict(X_test)

print('FLOOR')
print('Accuracy:', rf_floor.score(X_test, y_test))
print('Kappa:', cohen_kappa_score(y_pred_floor, y_test))


# With Building
#Accuracy: 0.9144914491449145
#Kappa: 0.8801900703455525


#X_train['pred_floor'] = rf_floor.predict(X_train)
#X_test['pred_floor'] = rf_floor.predict(X_test)


# %% Predict Longitude

y_train = train['LONGITUDE']
y_test = test['LONGITUDE']

rf_long = RandomForestRegressor(n_estimators = 500, n_jobs = CORES)

rf_long.fit(X_train, y_train)
y_pred_long = rf_long.predict(X_test)

print('')
print('LONGITUDE')
print('MAE:', mean_absolute_error(y_test, y_pred_long))
print('R2:', r2_score(y_test, y_pred_long))

# LONGITUDE, split
#MAE: 5.9912873654059
#R2: 0.9935665302798705

#LONGITUDE, together
#MAE: 6.03646695674289
#R2: 0.9934772161275138

#LONGITUDE, no Floor
#MAE: 6.093362827569391
#R2: 0.993093258237803

##LONGITUDE, no Building
#MAE: 7.629769132508398
#R2: 0.9885843918213203

#X_train['pred_longitude'] = rf_long.predict(X_train)
#X_test['pred_longitude'] = rf_long.predict(X_test)

# %% Predict Latitude --------------------------------------------------------

y_train = train['LATITUDE']
y_test = test['LATITUDE']

rf_lat = RandomForestRegressor(n_estimators = 500, n_jobs = CORES)

rf_lat.fit(X_train, y_train)
y_pred_lat = rf_lat.predict(X_test)

print('')
print('LATITUDE')
print('MAE:', mean_absolute_error(y_test, y_pred_lat))
print('R2:', r2_score(y_test, y_pred_lat))

#Latitude, no Longitude
#MAE: 6.1746875078498675
#R2: 0.9785037009176718

#LATITUDE, using Longitude
#MAE: 6.917217421125785
#R2: 0.9746902434308652

# %% Final error calculation

# Create arrays for the error in coordinates, building, floor
dist = np.sqrt((test['LATITUDE'].values - y_pred_lat)**2 + (test['LONGITUDE'].values - y_pred_long)**2 )

building_error = np.absolute(test['BUILDINGID'].values - y_pred_build)
unique, build_counts = np.unique(building_error, return_counts=True)
wrong_buildings = sum(build_counts[1:])
building_hit_rate = (len(building_error) - wrong_buildings)/len(building_error)

floor_error = np.absolute(test['FLOOR'].values - y_pred_floor)
unique, floor_counts = np.unique(floor_error, return_counts=True)
floor_summary = dict(zip(unique, floor_counts))
wrong_floors = sum(floor_counts[1:])
floor_hit_rate = (len(floor_error)-wrong_floors)/len(floor_error)

total_error = dist + 50*building_error + 4*floor_error
print('Mean total distance error:', total_error.mean().round(1), 'm')

error_summary = pd.DataFrame(dict(
        Baseline = [8.46, 3.39, 6.5, 11.72, 21.41, 73.3, 1, .8534],
        HFTS = [8.49, 3.69, 6.99, 11.6, 19.93, 40.7, 1, .9625],
        MOSAIC = [11.64, 3.26, 6.72, 12.12, 21.54, 313.33, .9865, .9386],
        RTLS = [6.2, 2.51, 4.57, 8.34, 15.81, 52.27, 1, .9374],
        ICSL = [7.67, 3.1, 5.88, 10.87, 19.68, 39.14, 1, .8693],
        Ensemble = [6.1, 2.51, 4.56, 8.24, 15.41, 52.27, 1, .9643]
        ),
        index = ['Mean Error', '25th percentile', '50th percentile', 
          '75th percentile', '95th percentile', '100th percentile',
          'Building Hit Rate', 'Floor Hit Rate'],
       )

error_summary.loc['Mean Error', 'My Model'] = total_error.mean().round(2)
error_summary.loc['25th percentile', 'My Model'] = np.percentile(total_error, 25).round(2)
error_summary.loc['50th percentile', 'My Model'] = np.percentile(total_error, 50).round(2)
error_summary.loc['75th percentile', 'My Model'] = np.percentile(total_error, 75).round(2)
error_summary.loc['95th percentile', 'My Model'] = np.percentile(total_error, 95).round(2)
error_summary.loc['100th percentile', 'My Model'] = np.percentile(total_error, 100).round(2)
error_summary.loc['Building Hit Rate', 'My Model'] = building_hit_rate.round(4)
error_summary.loc['Floor Hit Rate', 'My Model'] = floor_hit_rate.round(4)



import plotly.graph_objs as go
from plotly.offline import  plot, iplot
import cufflinks as cf
# You may also need
cf.go_offline()

plot(error_summary.iplot(asFigure=True,
                   kind='bar',
#                   xTitle='Dates',
#                   yTitle='Quantity',
#                   title=('Quantities for ' + shop_label)
                   )
    )


counts
