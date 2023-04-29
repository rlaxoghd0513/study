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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
poly = PolynomialFeatures(degree= 2 ,include_bias= False)

# Load data
path = './_data/wine/'
path_save = './_save/wine/'
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

# # Scale data
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

# Add polynomial features
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
test_csv_poly = poly.transform(test_csv)

print(x_train_poly.shape,x_test_poly.shape)

# params = {
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 3,
#     'num_leaves': 31,
#     'learning_rate': 0.1,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': -1,
#     'num_boost_round' : 3000
# }
model = make_pipeline(PCA(n_components=60),
                      MinMaxScaler(), 
                      LGBMClassifier(objective='multiclass', 
                                     num_class=3, metric='multi_logloss', 
                                     num_leaves=31, learning_rate=0.1, 
                                     feature_fraction=0.9, 
                                     bagging_fraction=0.8, 
                                     bagging_freq=5, 
                                     verbose=-1, 
                                     num_boost_round=3000))
 #11
# model.set_params(
# **params
#                  )
model.fit(x_train_poly, y_train, 
        #early_stopping_rounds=100,
        #,eval_set=[x_test, y_test]
        #eval_set=[(x_test, y_test)]
        ) 

# Evaluate model
results = model.score(x_test_poly, y_test)
print("최종점수:", results)

y_predict = model.predict(x_test_poly)
acc = accuracy_score(y_test, y_predict)
print("acc 는", acc)

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
y_submit = model.predict(test_csv_poly)

y_submit = pd.DataFrame(y_submit)
y_submit = np.array(y_submit)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['quality'] = y_submit
submission.to_csv(path_save + 'wine_' + date + '.csv')