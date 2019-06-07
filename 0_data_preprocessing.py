#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Pre-process data, save as csv that can be opened by a Python or R script

Created on Wed Feb 20 16:45:16 2019
@author: Laura Stupin

"""

# %% Model and Data Assumptions -------------------------------------------------

x100 = -110                 # Replace 100s with this value
drop_null_waps = True       # Drop WAPs not included in both fixed and random set
drop_na_rows = True         # If no WAPs recorded, drop row
drop_duplicate_rows = False 

# 76 na rows in train, but none in validation or test
#

# %% Setup --------------------------------------------------------------------

import pandas as pd
import numpy as np

#%% Load data -----------------------------------------------------------------


# Test data
fixed_location = pd.read_csv('data/raw/fixedLocation.csv')
fixed_location['location_method'] = 'fixed'

# Validation data
rand_location = pd.read_csv('data/raw/randomLocation.csv')
rand_location['location_method'] = 'random'

# %% Find the null columns ----------------------------------------------------
dfs = [fixed_location, rand_location] 
names = ['fixed', 'random']

nulls = dict()
i = 0

# Create a dictionary that has a list of null waps for each dataset
for df1 in dfs:
    na_cols = df1.replace(100, np.nan).isna().sum()
    null = na_cols[na_cols == len(df1)].index.tolist()
    nulls[names[i]] = null
    i = i+1



# Combine datasets so identical pre-processing will happen to all
df = (pd.concat([fixed_location, rand_location])
       .reset_index()
       .rename(columns ={'index': 'orig_index'})
       )


# Drop null WAPs
all_nulls = nulls['fixed'] + nulls['random']
if drop_null_waps: df = df.drop(all_nulls, axis=1)


# Collect valid WAP names
wap_names = [col for col in df if col.startswith('WAP')]


''' In the original dataset, 100 was used as a placeholder to indicate the signal
for that WAP was not detected for that observation.  In other words, 100s are
actually NaNs. For the purposes of training, though, it makes sense to replace 
them with a very distant signal, for example -110 dBm
'''
# Switch 100 dB values to NaNs temporarily
df = df.replace(100, np.nan)

# Calculate the number of signals observed for each row
df['sig_count'] = len(wap_names) - df[wap_names].isnull().sum(axis=1)


if drop_na_rows: df = df[df['sig_count'] != 0]
if drop_duplicate_rows: df = df.drop_duplicates()

# Replace NaN's with the selected number
df = df.replace(np.nan, x100)
df.to_csv('data/preprocessed/all_waps.csv', index=False)
