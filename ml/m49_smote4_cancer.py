import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)
# print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)
print(pd.Series(y_train).value_counts())
# 1    285
# 0    170

print("#========================== SMOTE 적용 후 ============================ ")
smote = SMOTE(random_state=123, k_neighbors=3)  
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 1    285
# 0    285

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {'n_estimators': [500],
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
print('f1_score : ', f1_score(y_test, y_predict))



#SMOTE 적용 전
# acc :  0.9649122807017544
# f1_score :  0.9726027397260274

#SMOTE 적용 후
# acc :  0.9912280701754386
# f1_score :  0.993006993006993