import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from harness import RMSE
from createFeatures import elem_counts, generate_rdk_features

##Loading the Training dataset
train_df = pd.read_csv('../data/train.csv')
##Generating the features for the training data
train_df = elem_counts(train_df)
train_df = pd.concat([train_df,\
    train_df.smiles.apply(lambda s: pd.Series(generate_rdk_features(s)))], axis=1)
##Splitting the dataset into training and test dataset
train_cols = train_df.columns
train_cols = train_cols.difference(['smiles', 'gap'])
X = train_df[train_cols]
y = train_df['gap']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,\
        random_state=1)

##Setting RandomForestRegressor settings
RF = RandomForestRegressor(n_estimators=10)
RF.fit(X_train, y_train)

##Checking RMSE for test set
pred = RF.predict(X_test)
print(RMSE(y_test, pred))

##Loading the test data
test_df = pd.read_csv('../data/test.csv')
##Generating the features for the test set
test_df = elem_counts(test_df)
test_df = pd.concat([test_df,\
    test_df.smiles.apply(lambda s: pd.Series(generate_rdk_features(s)))], axis=1)
##Omitting the 'smiles' columns
test_data = test_df[train_cols]

##Predicting through Random Forests
test_pred = RF.predict(test_data)

##Writing output to file
out_df = pd.DataFrame({'Id':np.array(test_df.index), 'Prediction': test_pred})
out_df['Id'] = out_df['Id'] + 1
out_df = out_df.set_index('Id')
out_df.Prediction = out_df.Prediction.astype(float)
out_df.to_csv('../output/random_forest_submission.csv')
