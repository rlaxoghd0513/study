import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import GridSearchCV, StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBClassifier
import time
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']-1

print(np.unique(y, return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), 
# array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size = 0.8,
    stratify=y
)

print(pd.Series(y_train).value_counts())




# print('=======================SMOTE 적용 후======================')
# smote = SMOTE(random_state=13, k_neighbors= 5)
# x_train, y_train = smote.fit_resample(x_train, y_train)

# save_path = './_save/npy/'
# np.save(save_path + 'keras56_x_train.npy', arr=x_train)          #수치화된 데이터를 np형태로 저장
# np.save(save_path + 'keras56_y_train.npy', arr=y_train)    
# print(pd.Series(y_train).value_counts())


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {'n_estimators': [100],
              'learning_rate' : [0.1],
              'max_depth' : [3],
              'gamma': [0],
              'min_child_weight': [0],
              'subsample' : [0.2],
              'colsample_bytree' : [0],
              'colsample_bylevel' : [0],
              'colsample_bynode' : [0],
              'reg_alpha' : [0],
              'reg_lambda' : [1],
              }  

#2. 모델
xgb = XGBClassifier(random_state=123,
                    )
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
#4. 평가, 예측

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)    
print('acc : ', score)
print('f1_score : ', f1_score(y_test, y_predict, average='macro'))


#####################스모트 적용 전###########################

# acc :  0.5963787509788904
# f1_score :  0.18647735185755066

#####################스모트 적용 후###########################
# acc :  0.6026436494754869 