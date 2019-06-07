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
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
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


X_train = train[wap_names]
X_test = test[wap_names]

# %% Predict Building ---------------------------------------------------------

y_train = train['BUILDINGID']
y_test = test['BUILDINGID']


rf_classifier = RandomForestClassifier(n_estimators=200)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

rf_classifier.score(X_test, y_test)
cohen_kappa_score(y_pred, y_test),


