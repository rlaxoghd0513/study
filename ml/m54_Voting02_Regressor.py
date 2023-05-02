import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor #투표


#3대장.
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor  #연산할 필요 없는 것들을 빼버림, 잘나오는 곳 한쪽으로만 감.
from catboost import CatBoostRegressor

data_list = [load_diabetes, fetch_california_housing]
data_list_name = ['디아벳','캘리포니아']

#1. 데이터
for i,value in enumerate(data_list):
    x,y = value(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True, train_size=0.8, random_state = 1030)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    xgb = XGBRegressor()
    lg = LGBMRegressor()
    cat = CatBoostRegressor(verbose=0)
    model = VotingRegressor(
        estimators = [('XGB',xgb),('LG',lg),('CAT',cat)])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(data_list_name[i])
    print('model.score : ', model.score(x_test,y_test))
    print("Voting.r2 : ", r2_score(y_test,y_pred))
    Regressors = [xgb, lg, cat]
    li = []
    for model2 in Regressors:
        
        model2.fit(x_train, y_train)
    
    # 모델 예측
        y_predict = model2.predict(x_test)
    
    # 모델 성능 평가
        score2 = r2_score(y_test,y_predict)
    
        class_name = model2.__class__.__name__ 
        print("{0} R2 : {1:4f}".format(class_name, score2))
        li.append(score2)
    print(li)

# 디아벳
# model.score :  0.32668887363050436
# Voting.r2 :  0.32668887363050436
# XGBRegressor R2 : 0.210424
# LGBMRegressor R2 : 0.335687
# CatBoostRegressor R2 : 0.323708
# [0.2104239542910319, 0.3356871311484668, 0.3237078721406743]

# 캘리포니아
# model.score :  0.8526566162322068
# Voting.r2 :  0.8526566162322068
# XGBRegressor R2 : 0.835095
# LGBMRegressor R2 : 0.839618
# CatBoostRegressor R2 : 0.855336
# [0.8350945020148811, 0.8396184794295622, 0.8553363257900071]
    

