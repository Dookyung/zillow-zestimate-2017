# -*- coding: utf-8 -*-
"""
initial test
Created on Mon May 29 00:50:20 2017

@author: Yunfei
"""


import numpy as np
import pandas as pd
import xgboost as xgb
import gc

import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train_2016.csv')
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../output/sample_submission.csv')
res_dir = '../output/'


print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')
df_train['mon'] = pd.to_datetime(df_train['transactiondate']).dt.month

enc = OneHotEncoder(n_values= 13)
train_mon_enc = enc.fit_transform(df_train['mon'].values.reshape(-1, 1))
cols = [ 'mon_' + str(j) for j in range(0, int(enc.n_values_))]
train_mon_enc_df = pd.DataFrame(train_mon_enc.toarray(),columns=cols)
train_mon_enc_df.drop('mon_0', axis = 1, inplace = True)
cols.pop(0)
df_train[cols] =  train_mon_enc_df[cols]
df_train.drop('mon', axis = 1, inplace = True)

x_train = df_train.drop(['parcelid', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

#del df_train; gc.collect()


#df_train.reindex(np.random.permutation(df_train.index))

#split = 80000
#x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
X_train, X_test, Y_train, Y_test =  train_test_split(x_train, y_train, test_size=0.2)

X_train = X_train.copy()
X_test = X_test.copy()
Y_train = Y_train.copy()
X_train.query('logerror < 0.2 & logerror > -0.2', inplace = True)
Y_train = Y_train[(Y_train < 0.2) & (Y_train >-0.2)]

X_train.drop( 'logerror', axis = 1, inplace = True)
X_test.drop( 'logerror', axis = 1, inplace = True)


print('Building DMatrix...')


d_train = xgb.DMatrix(X_train, label=Y_train)
d_valid = xgb.DMatrix(X_test, label=Y_test)

#del X_train, X_test; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.01
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 0

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

#################################################################################
print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

#df_test[cols] = 0

#del prop; gc.collect()

#df_test.drop( 'logerror', axis = 1, inplace = True)
#x_test = df_test[train_columns]
#for c in x_test.dtypes[x_test.dtypes == object].index.values:
#    x_test[c] = (x_test[c] == True)
#
#d_test = xgb.DMatrix(x_test)




print('Building test set ...')
for i in range(1, 13):
    df_test['mon_' + str(i)] = 0
    
train_columns = X_train.columns
df_test = df_test[train_columns]
for c in df_test.dtypes[df_test.dtypes == object].index.values:
    df_test[c] = (df_test[c] == True)

    
df_test.iloc[:, 66] = 1     
d_test = xgb.DMatrix(df_test)
print('Predicting on test ...')
p_test12 = clf.predict(d_test)

df_test.iloc[:, 66] = 0
df_test.iloc[:, 65] = 1
d_test = xgb.DMatrix(df_test)
print('Predicting on test ...')
p_test11 = clf.predict(d_test)


df_test.iloc[:, 65] = 0
df_test.iloc[:, 64] = 1
d_test = xgb.DMatrix(df_test)
print('Predicting on test ...')
p_test10 = clf.predict(d_test)


sub = sample.rename(columns = {'parcelid' : 'ParcelId'})
sub['201610'] = p_test10
sub['201710'] = p_test10

sub['201611'] = p_test11
sub['201711'] = p_test11


sub['201612'] = p_test12
sub['201712'] = p_test12

#sub.drop(7, axis = 1, inplace = True)
sub.to_csv(res_dir +  str(datetime.date.today()) +'_'+ 'xgb.csv', index=False, float_format='%.4f')