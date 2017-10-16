# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 01:34:26 2017

@author: Yunfei
"""

#
# import libariry
#

import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from tqdm import tqdm
import gc

#
# version 36 -> 6436
#
#version 29 -> LB:0.6446
#   add more feature
#
#version 28 -> LB:0.6445
#   model params 'n_estimators' -> 100
#
# version 26 -> LB:0.6443
#   model params 'n_estimators' -> 50
#

def load_data():
    train_2016 = pd.read_csv('../input/train_2016_v2.csv')
    train_2017 = pd.read_csv('../input/train_2017.csv')
    
    train = pd.concat([train_2016, train_2017], ignore_index=True)
    properties_2016 = pd.read_csv('../input/properties_2017.csv')
    properties_2017 = pd.read_csv('../input/properties_2017.csv')
    properties = pd.concat([properties_2016, properties_2017], ignore_index=True)
    sample = pd.read_csv('../input/sample_submission.csv')
    del train_2016, train_2017, properties_2016, properties_2017
    gc.collect();
    print("Preprocessing...")
    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)
            
    print("Set train/test data...")
    
            
    #
    # Add Feature
    #
    # life of property
    properties['N-life'] = 2018 - properties['yearbuilt']

    properties['A-calculatedfinishedsquarefeet'] = properties['finishedsquarefeet12'] + properties['finishedsquarefeet15']

    # error in calculation of the finished living area of home
    properties['N-LivingAreaError'] = properties['calculatedfinishedsquarefeet'] / properties['finishedsquarefeet12']

    # proportion of living area
    properties['N-LivingAreaProp'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']
    properties['N-LivingAreaProp2'] = properties['finishedsquarefeet12'] / properties['finishedsquarefeet15']

    # Amout of extra space
    properties['N-ExtraSpace'] = properties['lotsizesquarefeet'] - properties['calculatedfinishedsquarefeet']
    properties['N-ExtraSpace-2'] = properties['finishedsquarefeet15'] - properties['finishedsquarefeet12']

    # Total number of rooms
    properties['N-TotalRooms'] = properties['bathroomcnt'] + properties['bedroomcnt']

    # Average room size
    #properties['N-AvRoomSize'] = properties['calculatedfinishedsquarefeet'] / properties['roomcnt']

    # Number of Extra rooms
    properties['N-ExtraRooms'] = properties['roomcnt'] - properties['N-TotalRooms']

    # Ratio of the built structure value to land area
    properties['N-ValueProp'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']

    # Does property have a garage, pool or hot tub and AC?
    #properties['N-GarPoolAC'] = ((properties['garagecarcnt'] > 0) & (properties['pooltypeid10'] > 0) & (properties['airconditioningtypeid'] != 5)) * 1

    properties["N-location"] = properties["latitude"] + properties["longitude"]
    properties["N-location-2"] = properties["latitude"] * properties["longitude"]
    #properties["N-location-2round"] = properties["N-location-2"].round(-4)

    # Ratio of tax of property over parcel
    properties['N-ValueRatio'] = properties['taxvaluedollarcnt'] / properties['taxamount']

    # TotalTaxScore
    properties['N-TaxScore'] = properties['taxvaluedollarcnt'] * properties['taxamount']

    # polnomials of tax delinquency year
    properties["N-taxdelinquencyyear-2"] = properties["taxdelinquencyyear"] ** 2
    properties["N-taxdelinquencyyear-3"] = properties["taxdelinquencyyear"] ** 3

    # Length of time since unpaid taxes
    properties['N-live'] = 2018 - properties['taxdelinquencyyear']

    # Number of properties in the zip
    zip_count = properties['regionidzip'].value_counts().to_dict()
    properties['N-zip_count'] = properties['regionidzip'].map(zip_count)

    # Number of properties in the city
    city_count = properties['regionidcity'].value_counts().to_dict()
    properties['N-city_count'] = properties['regionidcity'].map(city_count)

    # Number of properties in the city
    region_count = properties['regionidcounty'].value_counts().to_dict()
    properties['N-county_count'] = properties['regionidcounty'].map(region_count)


    id_feature = ['heatingorsystemtypeid','propertylandusetypeid', 'storytypeid', 'airconditioningtypeid',
        'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'typeconstructiontypeid']
    for c in properties.columns:
        properties[c]=properties[c].fillna(-1)
        if properties[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
        if c in id_feature:
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
            dum_df = pd.get_dummies(properties[c])
            dum_df = dum_df.rename(columns=lambda x:c+str(x))
            properties = pd.concat([properties,dum_df],axis=1)
            properties = properties.drop([c], axis=1)
            #print np.get_dummies(properties[c])
    
    #
    # Make train and test dataframe
    #
    train = train.merge(properties, on='parcelid', how='left')
    sample['parcelid'] = sample['ParcelId']
    test = sample.merge(properties, on='parcelid', how='left')

    # drop out ouliers
    train = train[train.logerror > -0.4]
    train = train[train.logerror < 0.418]

    train["transactiondate"] = pd.to_datetime(train["transactiondate"])
    train["Month"] = train["transactiondate"].dt.month
    train["quarter"] = train["transactiondate"].dt.quarter
    
    test["Month"] = 10
    test['quarter'] = 4

    x_train = train.drop(['parcelid', 'logerror','transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = train["logerror"].values
    
    x_test = test[x_train.columns]
    del test, train, properties
    gc.collect();
    print(x_train.shape, y_train.shape, x_test.shape)
    
    return x_train, y_train, x_test

x_train, y_train, x_test = load_data()

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in tqdm(enumerate(self.base_models)):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in tqdm(enumerate(folds)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
#                y_mean = np.mean(y_train)
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]                
                
                mae_baseline = mean_absolute_error(y_holdout, y_pred)
                print("Current fold MAE is {:.8f}".format(mae_baseline))
                
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()
        
        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

# rf params
rf_params = {}
rf_params['n_estimators'] = 500
rf_params['max_depth'] = 8
rf_params['max_features'] = 'sqrt'
rf_params['min_samples_split'] = 100
rf_params['min_samples_leaf'] = 30

# xgb params
xgb_params = {}
xgb_params['n_estimators'] = 3000
xgb_params['min_child_weight'] = 12
xgb_params['learning_rate'] = 0.03
xgb_params['max_depth'] = 5
xgb_params['subsample'] = 0.77

xgb_params['reg_lambda'] = 0.8
xgb_params['reg_alpha'] = 0.4
y_mean = 0
xgb_params['base_score'] = 0
#xgb_params['seed'] = 400
xgb_params['silent'] = 1
#xgb_params['tree_method'] = 'hist'
xgb_params['n_jobs'] = 8




# lgb params
lgb_params = {}
lgb_params['n_estimators'] = 1000
lgb_params['max_bin'] = 63
lgb_params['learning_rate'] = 0.037 # shrinkage_rate
lgb_params['metric'] = 'l1'          # or 'mae'
lgb_params['sub_feature'] = 0.35    
lgb_params['bagging_fraction'] = 0.85 # sub_row
lgb_params['bagging_freq'] = 40
lgb_params['num_leaves'] = 256        # num_leaf
lgb_params['min_data'] = 500         # min_data_in_leaf
lgb_params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params['verbose'] = 0
lgb_params['feature_fraction_seed'] = 2
lgb_params['bagging_seed'] = 3
lgb_params['device'] = 'gpu'

# catboost params
cat_params= {}
cat_params['iterations'] = 3000
cat_params['learning_rate'] = 0.01
cat_params['depth'] = 6
cat_params['l2_leaf_reg'] = 10
cat_params['loss_function'] = 'MAE'
cat_params['eval_metric'] = 'MAE'


# CatBoost
cat_model = CatBoostRegressor(**cat_params)

# XGB model
xgb_model = XGBRegressor(**xgb_params)

# lgb model
lgb_model = LGBMRegressor(**lgb_params)

# RF model
rf_model = RandomForestRegressor(**rf_params)

# ET model
et_model = ExtraTreesRegressor()

# SVR model
# SVM is too slow in more then 10000 set
#svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.05)

# DecsionTree model
dt_model = DecisionTreeRegressor()

# AdaBoost model
ada_model = AdaBoostRegressor()






stack = Ensemble(n_splits=5,
        stacker=LinearRegression(),
        base_models=(xgb_model,lgb_model, cat_model, rf_model, et_model, ada_model))

y_test = stack.fit_predict(x_train, y_train, x_test)

from datetime import datetime
print("submit...")
pre = y_test
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = pre
submit_file = '../output/{}.csv'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
sub.to_csv(submit_file, index=False,  float_format='%.4f')