from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

def genScore(RF, X_valid, y_valid):
    #generating the predictions
    pred = RF.predict(X_valid)
    #calculating classification accuracy
    return np.sum(pred == y_valid)/float(len(y_valid))


train = pd.read_csv('../outputs/features_v1.csv')
y = train['class']
train_ids = train['id']
X = train[train.columns[1:-2]]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, \
                                test_size=0.10, random_state=1)
RF = RandomForestClassifier(n_estimators=10)
RF.fit(X_train, y_train)


#Loading the test data
test = pd.read_csv('../outputs/features_test_v1.csv')
test_ids = test['Id']
X_test = test[test.columns[1:-1]]
pred = RF.predict(X_test)
out_df = pd.DataFrame(test_ids, columns=['Id'])
out_df['Prediction'] = pred
out_df = out_df.set_index('Id')
out_df.to_csv('../outputs/RF_prediction.csv')




#Code for Logistic regression
X_train1, X_test_train, Y_train, Y_test = train_test_split(X_train, t_train, test_size=0.10, random_state=1)


LR = LogisticRegression(penalty='l1', multi_class='ovr', 
                       n_jobs = -1, warm_start = True, solver = 'liblinear') 
LR.fit(X_train1, Y_train)
LR_pred = LR.predict(X_test_train)

print(accuracy_score(Y_test, LR_pred))

LR_pred = LR.predict(X_test_real)

out_df = pd.DataFrame(test_ids, columns=['Id'])
out_df['Prediction'] = LR_pred
out_df = out_df.set_index('Id')

out_df.to_csv('../outputs/LR_prediction_V_LR.csv')
