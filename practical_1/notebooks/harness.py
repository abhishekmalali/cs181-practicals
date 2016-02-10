from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import numpy as np 
import pandas as pd

def train_test(filename = "../data/train.csv", split=0.1, seed=1):
    print "Loading Training Data..."
    df_train = pd.read_csv(filename)
    print "Splitting Data..."
    train, test = train_test_split(df_train, test_size = split, random_state=seed)
    ytrain = train['gap']
    ytest = test['gap']
    xtrain = train.drop(['gap'], axis=1)
    xtest = test.drop(['gap'], axis=1)
    return xtrain, xtest, ytrain, ytest

def RMSE(ytest, predictions):
    rms = np.sqrt(mean_squared_error(ytest, predictions))
    print "RMSE = ", rms
    return rms

