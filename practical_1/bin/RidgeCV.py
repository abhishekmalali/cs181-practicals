# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import train_test_split
from harness import RMSE, train_test
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import inspect

# Pull in input data
"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv('../data/train_new_features.csv')
df_test = pd.read_csv("../data/test_new_features.csv")

def elem_counts(df):
    df['len_smiles'] = df['smiles'].str.len()
    elements = ['nH', 'n', 'c', 'c1', 'Si', 'SiH2', '=', '-', 'CC', 'ncc', 'C1', 'C', 'H', 'cc', 'ccc', 'cccc', 'cc1',\
           '(C1)', '(c1)', '(o1)', '(s1)', 'nc', 'c12', 'c2', 'c1cc', '(cc1)', 'c2C', 'cc3', 'oc', 'ncc', 'C1=C',\
                'C=c', 'C=C', 'ccn', 'c3', '[se]', '=CCC=', 'c21', 'c1c', 'cn', 'c4c', 'c3c', 'coc',\
               'ccccc', '[SiH2]C', 'cc4']
    for elem in elements:
        col_name = 'count_' + elem
        df[col_name] = df['smiles'].str.count(elem)
    return df

# Engineer new features
df_train = elem_counts(df_train)
# Drop columns and split data to train and test
df_train = df_train.drop(['smiles'], axis=1)
df_train = df_train.drop(["Unnamed: 0"], axis =1)
#store gap values
Y = df_train.gap.values
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(df_train, Y, test_size=0.10, random_state=1)


print "Train features:", X_train.shape
print "Train gap:", Y_train.shape
print "Test features:", X_test.shape
print "Test gap:", Y_test.shape

# Train on linear models

LR = LinearRegression(normalize=True)
LR.fit(X_train, Y_train)
LR_pred = LR.predict(X_test)

RR = Ridge(normalize=True)
RR.fit(X_train, Y_train)
RR_pred = RR.predict(X_test)

RRCV = RidgeCV(alphas = [2, 0.000000001, 0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100], normalize=True, store_cv_values = True)
RRCV.fit(X_train, Y_train)
RRCV_pred = RRCV.predict(X_test)

