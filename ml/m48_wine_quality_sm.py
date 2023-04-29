import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Load data
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# Remove rows with single class label
single_class_label = train_csv['quality'].nunique() == 1
if single_class_label:
    train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# Label encode 'type'
le = LabelEncoder()
train_csv['type'] = le.fit_transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# Split data
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=850, train_size=0.7, stratify=y)

# # One-hot encode 'y'
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# Scale data
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# parameters = {'n_estimators': 1000,  
#               'learning_rate': 0.3, 
#               'max_depth': 3,
#               'boosting_type': 'gbdt',        
#               'min_child_weight': 1,  
#               'subsample': 0.5, 
#               'colsample_bytree': 1,
#               'colsample_bynode': 1,
#               'reg_alpha': 1,        
#               'reg_lambda': 1,
#               'early_stopping_rounds': 100
#               }
params = {
    'boosting_type': 'dart',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 9,
    'num_leaves': 350,
    'learning_rate': 0.005,
    # 'feature_fraction': 0.7,
    # 'bagging_fraction': 0.5,
    # 'bagging_freq': 5,
    'verbose': -1,
    'num_iterations' : 2000
}
model = LGBMClassifier(**params)
# model.set_params(#**parameters, 
# **params
#                  )
model.fit(x_train, y_train, 
        #early_stopping_rounds=100,
        #,eval_set=[x_test, y_test]
        #eval_set=[(x_test, y_test)]
        ) 

# Evaluate model
results = model.score(x_test, y_test)
print("최종점수:", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc 는", acc)

y_pred = model.predict(test_csv)

submission = pd.read_csv('./_data/dacon_wine/sample_submission.csv', index_col=0)

submission['quality'] = y_pred

submission.to_csv('./_data/dacon_wine/sub.csv')