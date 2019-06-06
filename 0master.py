#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prerequisites: Run 0data.py first to generate pre-processed data.

Status: IN PROGRESS
Purpose: Master script for models and analysis

To do: Change variable names to fixed, rand

Created on Mon Mar 18 15:18:21 2019
@author: Laura Stupin
"""

# %% Parameters for running models --------------------------------------------------------------------

rand = 42                   # Set random seed
n_jobs = 3                  # Number of computer cores to use
lon_lat_rand_search = 8     # Number of lat/lon model tuning combinations to try
floor_rand_search = 6       # Number of floor model tuning combinations to try


# Setup --------------------------------------------------------------------

# Import custom functions for this task
import custom_functions as cf

# Import standard libraries
import pandas as pd
import numpy as np

from pprint import pprint
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.externals import joblib

#from scipy.spatial import distance
#import plotly.graph_objs as go
#from plotly.offline import plot

# Load data -------------------------------------------------------

# 0data.py must be run first to generate the preprocessed csv.

df_all = pd.read_csv('data/preprocessed/df.csv')
df_tr = df_all[df_all['dataset'] == 'train']
df_val = df_all[df_all['dataset'] == 'val']
df_test = df_all[df_all['dataset'] == 'test']

wap_names = [col for col in df_all if col.startswith('WAP')]

# Empty dataframe to hold final predictions
df_pred = pd.DataFrame(
        index = range(0,len(df_test)),
        columns = ['FLOOR', 'LATITUDE', 'LONGITUDE'])


''' Setup test/train sampled datasets -----------------------------------------
I experimented with two strategies: 
'5_1_fixed_rand'     5-to-1 ratio  of data from the fixed and random data sets 
'20_1_fixed_rand'   20-to-1 ratio (all available data) from fixed/random data sets  
'''

test, train, train_final = cf.test_train('20_1_fixed_rand', 
                                      rand, df_val, df_tr, df_all)

