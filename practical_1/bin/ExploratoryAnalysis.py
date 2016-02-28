import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from harness import RMSE, train_test, write_to_file

# Use defined function from harness file to make training/validation split
xtrain, xtest, ytrain, ytest=train_test()

# Collect smiles strings.  
xtest_smiles = xtest['smiles']
xtest = xtest.drop(['smiles'], axis=1)
xtrain_smiles = xtrain['smiles']
xtrain = xtrain.drop(['smiles'], axis=1)

# Baseline model performance on validation set
LR1 = LinearRegression()
LR1.fit(xtrain, ytrain)
LR_pred1 = LR1.predict(xtest)
rmse1 = RMSE(LR_pred1, ytest)

RF1 = RandomForestRegressor()
RF1.fit(xtrain, ytrain)
RF_pred1 = RF1.predict(xtest)
rmse2 = RMSE(RF_pred1, ytest)

## We will use these error values as baselines for our model tweaks.
def LRimprovement(prediction, test=ytest):
    imp = rmse1 - RMSE(prediction, test)
    print "Improvement on original Linear Regression: ", imp
    return imp

def RFimprovement(prediction, test=ytest):
    imp = rmse2 - RMSE(prediction, test)
    print "Improvement on original Random Forest: ", imp
    return imp

## To beat the baseline, we will want to do some exploratory analysis to get some intuition on the features.  
print "Maximum gap in training set: ", np.max(ytrain)
print "Maximum gap in test set:     ", np.max(ytest)
print "Minimum gap in training set: ", np.min(ytrain)
print "Minimum gap in test set:     ", np.min(ytest)
print "Mean gap value in test set:  ", np.mean(ytest)

## Notice the anomalous value in the training set.  Let's look at the distribution of gap values in each set.  
plt.hist(ytest, bins=np.linspace(-2,4,20))
plt.title("Distribution of Gap Values")
plt.xlabel("Gap Value")
plt.ylabel("Count")
plt.xlim([-2,4])
plt.hist(ytrain, bins=np.linspace(-2,4,20))
plt.xlim([-2,4])

## The distributions are the same shape, but there are three values in the test set that are clearly erroneous:
print "Number of implausible negative gap values: ", np.sum(ytrain<0)

## Let's drop those three rows out, because they can't be helping our models:
xtrain = xtrain[ytrain>0]
ytrain = ytrain[ytrain>0]

#Retrain baselines with these values removed
LR = LinearRegression()
LR.fit(xtrain, ytrain)
LR_pred = LR.predict(xtest)
LRimprovement(LR_pred)

RF = RandomForestRegressor()
RF.fit(xtrain, ytrain)
RF_pred = RF.predict(xtest)
RFimprovement(RF_pred)

## This didn't make much difference, but there is no reason to keep the rows with implausible gap values in our 
#      training set, so we will stick with this.  

## Next, let's take a look at the features themselves.  
train_descrip = xtrain.describe()
## Notice that some features have 0 values for all observations.  These columns contain no information, so perhaps
##     we should remove these from the model.  We could remove these columns from the training set, train the model, 
##     then remove the same columns from the test set before making the predictions.  
train_descrip

## In fact, 225 features have zero mean
sum(train_descrip.loc['mean']==0)

## But none have a mean of 1
sum(train_descrip.loc['mean']==1)

## Names of all the features that are all zeroes in the training set
zeros = train_descrip.columns.values[np.array(train_descrip.loc['mean']==0)]

# Drop the columns that are filled with all zeroes
xtrain = xtrain.drop(zeros, axis=1)
xtest = xtest.drop(zeros, axis=1)

# Fit a new linear regression on just these columns.  Note that the error is nearly identical.  This makes sense
#    because presumably the coefficients for those columns are zero, or the values for those columns in the 
#    validation set are still zero.  
LR = LinearRegression()
LR.fit(xtrain, ytrain)
LR_pred = LR.predict(xtest)
LRimprovement(LR_pred)

## Let's see if this pattern holds for the Random Forest Regression.  
RF = RandomForestRegressor()
RF.fit(xtrain, ytrain)
RF_pred = RF.predict(xtest)
RFimprovement(RF_pred)

## This didn't significantly improve the RMSE value of either model, but there is still no reason to keep those 
##    columns around as they are not informing any model that we develop.  

## Look at the regression coefficients.  Notice that 6 of them have massive absolute values, on the order of 1e08-1e11.  
##    Next, notice that the element in the 3rd and 28th position are exactly the opposite, as are the 13th and 25th,
#     and 22nd and 23rd.  
LR.coef_

## We can show that these features have identical values, and the regression simply assigns coefficients to cancel
##    each other out.  
np.sum(xtrain.iloc[:,21] - xtrain.iloc[:,22])
np.sum(xtrain.iloc[:,27] - xtrain.iloc[:,2])
np.sum(xtrain.iloc[:,12] - xtrain.iloc[:,24])

## These are the names of the columns that have LR coefficients less than one.  This includes the features that are
###    assigned low importance and the redundant features.
dropcols = xtrain.columns[[22,2,24]]
xtrain_reduced = xtrain.drop(dropcols, axis=1)

## Now we do the same column removal to the validation set and test its effect on the models.  
xtest_reduced = xtest.drop(dropcols, axis=1)
LR = LinearRegression()
LR.fit(xtrain_reduced, ytrain)
LR_pred = LR.predict(xtest_reduced)
LRimprovement(LR_pred)

RF = RandomForestRegressor()
RF.fit(xtrain_reduced, ytrain)
RF_pred = RF.predict(xtest_reduced)
RFimprovement(RF_pred)

## Next, we look at how PCA changes the performance of the models:
# Make two lists to store improvement values of the models on pca-transformed data
lr_pca = [10] # 0th element has nonsense value
rf_pca = [10]
for i in range(1,32):
    pca = PCA(n_components=i)
    x_pca = pca.fit_transform(xtrain)
    xtest_pca = pca.transform(xtest)
    
    LR = LinearRegression()
    LR.fit(x_pca, ytrain)
    LR_pred = LR.predict(xtest_pca)
    lr_pca.append(LRimprovement(LR_pred))
    
    RF = RandomForestRegressor()
    RF.fit(x_pca, ytrain)
    RF_pred = RF.predict(xtest_pca)
    rf_pca.append(RFimprovement(RF_pred))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
for ax, j, title in zip(axes.ravel(), [lr_pca, rf_pca], ['RMSE Improvement for LR', 'RMSE Improvement for RF']):
    ax.plot(range(1,32),j[1:])
    ax.set_title(title);
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('RMSE Improvement over Baseline')
    ax.set_xlim((1,31))





