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
TEST_SIZE = .25

# Setup --------------------------------------------------------------------

# Import custom functions for this task
import custom_functions as cf

# Import standard libraries
import pandas as pd
import numpy as np

from plotly.offline import plot
import plotly.graph_objs as go

from pprint import pprint
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, r2_score
from sklearn.externals import joblib


# Load data -------------------------------------------------------


waps_all = pd.read_csv('data/preprocessed/all_waps.csv')
waps_fixed = waps_all[waps_all['location_method'] == 'fixed']
waps_random = waps_all[waps_all['location_method'] == 'random']



wap_names = [col for col in waps_all if col.startswith('WAP')]



''' Setup test/train sampled datasets -----------------------------------------
It's important to sample data from each floor and building for the test and 
train sets.  Otherwise, random splits might leave us with a lot of data from
one floor/building, and none from another.
'''

train = waps_fixed
test = waps_random

X_train = train[wap_names].copy()
X_test = test[wap_names].copy()

y_train = train['BUILDINGID']
y_test = test['BUILDINGID']

rf_building = RandomForestClassifier(n_estimators=300)
rf_building.fit(X_train, y_train)
y_pred = rf_building.predict(X_test)

print('BUILDING')
print('Accuracy:', rf_building.score(X_test, y_test))
print('Kappa:', cohen_kappa_score(y_pred, y_test))
print('')

#Accuracy: 0.9981998199819982
#Kappa: 0.9971547547096364


# %% Predict Floor -----------------------------------------------------------

# Add building prediction to the features
X_train['pred_building'] = rf_building.predict(X_train)
X_test['pred_building'] = rf_building.predict(X_test)

y_train = train['FLOOR']
y_test = test['FLOOR']

rf_floor = RandomForestClassifier(n_estimators=500)
rf_floor.fit(X_train, y_train)
y_pred = rf_floor.predict(X_test)

print('FLOOR')
print('Accuracy:', rf_floor.score(X_test, y_test))
print('Kappa:', cohen_kappa_score(y_pred, y_test))



# With Building
#Accuracy: 0.9144914491449145
#Kappa: 0.8801900703455525


X_train['pred_floor'] = rf_floor.predict(X_train)
X_test['pred_floor'] = rf_floor.predict(X_test)

# %% Split datasets by building





# %% Predict Longitude

rf_long = RandomForestRegressor(n_estimators = 500)

y_train = train['LONGITUDE']
y_test = test['LONGITUDE']

rf_long.fit(X_train, y_train)
y_pred = rf_long.predict(X_test)

print('')
print('LONGITUDE')
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))

# All predicted the same
#LONGITUDE
#MAE: 6.127759205383107
#R2: 0.9926204605263867


# %% Predict Latitude --------------------------------------------------------

rf_lat = RandomForestRegressior(n_estimators=500)

y_train = train['LATITUDE']
y_test = test['LATITUDE']


# %% Sandbox -----------------------------------------------------------------

X_train_build0 = X_train[X_train['pred_building'] == 0]
y_train_build0 = train.loc[X_train_build0.index, 'FLOOR' ]

len(X_train_build0.index.unique())

len(X_train.index.unique())
len(X_train.index)

X_test_build0 = X_test[X_test.pred_building == 0]
y_test_build0 = test.loc[X_test_build0.index, 'FLOOR' ]

rf_classifier = RandomForestClassifier(n_estimators=800)
rf_classifier.fit(X_train_build0, y_train_build0)
y_pred = rf_classifier.predict(X_test_build0)

print('BUILDING 0')
print('Accuracy:', rf_classifier.score(X_test_build0, y_test_build0))
print('Kappa:', cohen_kappa_score(y_pred, y_test_build0))

#Accuracy: 0.9981060606060606
#Kappa: 0.9973224337454486

X_train['pred_floor'] = rf.classifier




# %% Boost building score to a full 1 with only 2 observations per building/floor

waps_random_train = (waps_random.groupby(['BUILDINGID', 'FLOOR'])
            .apply(lambda x: x.sample(n=5, random_state = RANDOM_SEED))
            .droplevel(level = ['BUILDINGID', 'FLOOR'])
           )

train = pd.concat([waps_fixed, waps_random_train]) #.reset_index(drop=True)
test = waps_random.drop(waps_random_train.index) #.reset_index(drop=True)

X_train = train[wap_names].copy()
X_test = test[wap_names].copy()


y_train = train['BUILDINGID']
y_test = test['BUILDINGID']

rf_building = RandomForestClassifier(n_estimators=300)
rf_building.fit(X_train, y_train)
y_pred = rf_building.predict(X_test)

print('BUILDINGID')
print('Accuracy:', rf_building.score(X_test, y_test))
print('Kappa:', cohen_kappa_score(y_pred, y_test))

# %% Old train/test



# Select test sample with 390 observations from each floor, building 
# This is roughly 25% of the fixed location data
waps_fixed_test = (waps_fixed.groupby(['BUILDINGID', 'FLOOR'])
                   .apply(lambda x: x.sample(n = 390, random_state = RANDOM_SEED))
                   .droplevel(level = ['BUILDINGID', 'FLOOR'])
                   )

# Select test sample with 21 observations from each floor, building 
# This is roughly 25% of the random location data
waps_random_test = (waps_random.groupby(['BUILDINGID', 'FLOOR'])
            .apply(lambda x: x.sample(n=21, random_state = RANDOM_SEED))
            .droplevel(level = ['BUILDINGID', 'FLOOR'])
           )

# Combine both datasets into the test set
test = pd.concat([waps_fixed_test, waps_random_test])

# The train set is all the data not in the test set
train = waps_all.drop(test.index)

# Plot to check spacing of test/train
#trace0 = go.Scatter3d(
#        x=train['LONGITUDE'], 
#        y=train['LATITUDE'],
#        z=train['FLOOR'],
#        mode='markers',
#        name = 'Train'
#        )
#
#trace1 = go.Scatter3d(
#        x=test['LONGITUDE'], 
#        y=test['LATITUDE'],
#        z=test['FLOOR'] +,
#        mode='markers',
#        name = 'Test'
#        )
#
#data = [trace0, trace1]
#plot(data)


X_train = train[wap_names].copy()
X_test = test[wap_names].copy()

# %% Predict Building ---------------------------------------------------------

y_train = train['BUILDINGID']
y_test = test['BUILDINGID']


rf_classifier = RandomForestClassifier(n_estimators=200)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

print('Accuracy:', rf_classifier.score(X_test, y_test))
print('Kappa:', cohen_kappa_score(y_pred, y_test))







def rf_cascade():
    