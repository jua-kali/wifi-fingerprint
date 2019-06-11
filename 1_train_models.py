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



# %% Setup test/train sampled datasets -----------------------------------------


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

rf_floor = RandomForestClassifier(n_estimators=500, n_jobs = CORES)
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


X_train = train[wap_names].copy()
X_test = test[wap_names].copy()
X_train['pred_building'] = rf_building.predict(X_train)
X_test['pred_building'] = rf_building.predict(X_test)
#X_train['pred_floor'] = rf_floor.predict(X_train)
#X_test['pred_floor'] = rf_floor.predict(X_test)

target = 'LONGITUDE'



def predict_by_building(target, train, X_train, X_test):
    
    all_predictions = pd.DataFrame()
    
    for i in [0, 1, 2]:
        X_train_temp = X_train[X_train['pred_building'] == i]
        y_train_temp = train.loc[X_train_temp.index, target]
        
    
        X_test_temp = X_test[X_test.pred_building == i]
        y_test_temp = test.loc[X_test_temp.index, target]
        
        rf_regressor = RandomForestRegressor(n_estimators=500, n_jobs = CORES)
        rf_regressor.fit(X_train_temp, y_train_temp)
        y_pred = rf_regressor.predict(X_test_temp)
        
        temp= pd.DataFrame({target: y_test_temp,
                            'predicted': y_pred,
                            'BUILDINGID': i})
        
        all_predictions = pd.concat([all_predictions, temp])
        
        print('')
        print('LONGITUDE, BUILDING', i)
        print('MAE:', mean_absolute_error(y_test_temp, y_pred))
        print('R2:', r2_score(y_test_temp, y_pred))
        
    return(all_predictions)
    
   
all_predictions = predict_by_building(target, train, X_train, X_test)

split

print('')
print('LONGITUDE')
print('MAE:', mean_absolute_error(all_predictions['LONGITUDE'], all_predictions['predicted']))
print('R2:', r2_score(all_predictions['LONGITUDE'], all_predictions['predicted']))


#LONGITUDE, BUILDING 0
#MAE: 4.703652458498485
#R2: 0.9199451652416797
#
#LONGITUDE, BUILDING 1
#MAE: 7.186656707225823
#R2: 0.9326206446104566
#
#LONGITUDE, BUILDING 2
#MAE: 7.769591434167459
#R2: 0.828342858348318

# All together
#LONGITUDE
#MAE: 6.129353583465702
#R2: 0.9926150163208298


# %% Predict Longitude

rf_long = RandomForestRegressor(n_estimators = 500, n_jobs = CORES)

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


# %% Try without Floor

y_train = train['LONGITUDE']
y_test = test['LONGITUDE']

rf_long = RandomForestRegressor(n_estimators = 500, n_jobs = CORES)

rf_long.fit(X_train, y_train)
y_pred = rf_long.predict(X_test)

print('')
print('LONGITUDE')
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))

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
y_pred = rf_lat.predict(X_test)

print('')
print('LATITUDE')
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))

#Latitude, no Longitude
#MAE: 6.1746875078498675
#R2: 0.9785037009176718

#Latitude when you include longitude
#LATITUDE
#MAE: 6.917217421125785
#R2: 0.9746902434308652


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
    