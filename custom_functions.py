#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:24:12 2019

@author: chief
"""

import pandas as pd

from pprint import pprint
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

# Prepare test/train samples given different datasets



def test_train(sample, rand, df_val, df_tr, df_all):
    if sample == '5_1_fixed_rand':

        # Build a random sample of val data, 25 from each floor/building
        test_val = df_val.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(n=20, random_state=rand))
        test_val = test_val.droplevel(level = ['BUILDINGID', 'FLOOR'])
        # Old way, random sampling all data
        # test_val = df_val.sample(n = 250, random_state = rand)
        # The rest is training
        train_val = df_val.drop(test_val.index)
        
        # Build a random sample with 400 observations from each floor, building 
        tr_samp = df_tr.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(n = 400, random_state = rand))
        # Reduce multi-index to single level index
        tr_samp = tr_samp.droplevel(level = ['BUILDINGID', 'FLOOR'])
        test_tr = tr_samp.sample(n=round(.25*len(tr_samp)), random_state = rand)
        
        train_tr = tr_samp.drop(test_tr.index)
        
        # Build the final test/train sets from both
        test = pd.concat([test_val, test_tr])
        train = pd.concat([train_val, train_tr])
        train_final = pd.concat([df_val, tr_samp])

    if sample == '20_1_fixed_rand':
        # Build a random sample of val data alone
        test_val = df_val.sample(n = 250, random_state = rand)
        
        # Build a random test sample with 400 observations from each floor, building 
        test = df_tr.groupby(['BUILDINGID', 'FLOOR']).apply(lambda x: x.sample(n = 400, random_state = rand))
        # Reduce multi-index to single level index
        test = test.droplevel(level = ['BUILDINGID', 'FLOOR'])
        
        # Put both random samples into the main test set
        test = pd.concat([test, test_val])
        # Training is all observations not in random test sample or provided test set
        train = df_all[df_all['dataset'] != 'test'].drop(test.index)
        
        # Use all available data for final prediction
        train_final = pd.concat([df_tr, df_val])

    return(test, train, train_final)

# Calculate and display error metrics for Random Forest Classification
def rfc_pred(X_test, y_test, model):
    clfpred = model.predict(X_test)
    print(pd.crosstab(y_test, clfpred, rownames=['Actual'], colnames=['Predicted']))
    print('Accuracy:', model.score(X_test, y_test))
    print('Kappa:', cohen_kappa_score(clfpred, y_test))

# Set up y for each target - Floor, Latitude, Longitude
def set_y(target, train, test, test_val, train_final):
    y_train = train[target]   
    y_test = test[target]

    # A more difficult test
    y_test2 = test_val[target]
    y_train_final = train_final[target]
    return(y_train, y_test, y_test2, y_train_final)
    
# Define an accuracy report for Random Forest classification models
def acc_report(model, tag, is_search, X_test, y_test, X_test2, y_test2):
    accuracy = dict(test_acc = model.score(X_test, y_test),
                    test_kappa = cohen_kappa_score(model.predict(X_test), y_test),
                    test2_acc = model.score(X_test2, y_test2),
                    test2_kappa = cohen_kappa_score(model.predict(X_test2), y_test2))
    if is_search:
        print(tag, 'BEST PARAMETERS')
        pprint(model.best_params_)
    print('\n', tag, 'FULL TEST')
    print(pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted']))
    print('\n', tag, ' TEST WITH VALIDATION ONLY')
    print(pd.crosstab(y_test2, model.predict(X_test2), rownames=['Actual'], colnames=['Predicted']))
    pprint(accuracy)

def mae_report(model, is_search, X_test, y_test, X_test2, y_test2):
    
    if is_search: 
        best_score = model.best_score_
        best_params = model.best_params_
        print("Cross Validation Scores:", model.cv_results_['mean_test_score'])
        print("Best score: {}".format(best_score))
        print("Best params: ")
        for param_name in sorted(best_params.keys()):
            print('%s: %r' % (param_name, best_params[param_name]))
    
    mae = dict(test_mae = mean_absolute_error(model.predict(X_test), y_test),
               test2_mae = mean_absolute_error(model.predict(X_test2), y_test2)
               )      
    pprint(mae)
