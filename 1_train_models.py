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


# Load data -------------------------------------------------------


waps_all = pd.read_csv('data/preprocessed/all_waps.csv')
waps_train = waps_all[waps_all['dataset'] == 'train']
waps_val = waps_all[waps_all['dataset'] == 'val']
waps_test = waps_all[waps_all['dataset'] == 'test']

wap_names = [col for col in waps_all if col.startswith('WAP')]



''' Setup test/train sampled datasets -----------------------------------------
I experimented with two strategies: 
'5_1_fixed_rand'     5-to-1 ratio  of data from the fixed and random data sets 
'20_1_fixed_rand'   20-to-1 ratio (all available data) from fixed/random data sets  
'''

test, train, train_final = cf.test_train('20_1_fixed_rand', 
                                      rand, waps_val, waps_train, waps_all)
