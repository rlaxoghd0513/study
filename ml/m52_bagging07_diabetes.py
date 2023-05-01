import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


#1데이터
data_list = [load_diabetes, fetch_california_housing]
model_list = [XGBRegressor, RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor]
data_list_name = ['디아뱃','캘리포니아']
model_list_name = ['XGB', 'RF','DT','GB']

for i,value in enumerate(data_list):
    x,y = value(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123, train_size=0.8, shuffle=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    for j,value1 in enumerate(model_list):
        aaa = value1()
        model = BaggingRegressor(aaa,
                                  n_estimators=11,
                                  random_state = 333,
                                  bootstrap=True)
        model.fit(x_train, y_train)
        print(data_list_name[i])
        print(model_list_name[j])
        print('model.score:',model.score(x_test, y_test))
        y_predict = model.predict(x_test)
        print('r2:',r2_score(y_test, y_predict))

# 디아뱃
# XGB
# model.score: 0.5623151991194182
# r2: 0.5623151991194182
# 디아뱃
# RF
# model.score: 0.5522912265789272
# r2: 0.5522912265789272
# 디아뱃
# DT
# model.score: 0.4999119834076299
# r2: 0.4999119834076299
# 디아뱃
# GB
# model.score: 0.589871778224852
# r2: 0.589871778224852
# 캘리포니아
# XGB
# model.score: 0.8493363627127936
# r2: 0.8493363627127936
# 캘리포니아
# RF
# model.score: 0.8095790974124395
# r2: 0.8095790974124395
# 캘리포니아
# DT
# model.score: 0.8009815696385079
# r2: 0.8009815696385079
# 캘리포니아
# GB
# model.score: 0.795648972932579
# r2: 0.795648972932579