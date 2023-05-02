#스태킹 최대 문제점
# 프리딕한걸 또 프리딕한다
# 결국 x_test로 또 훈련을 하는거니까 과적합의 문제가 생길 수도 있다
# 스태킹할땐 항상 과적합문제 신경써야한다 -> train을 두개로 나누었다?

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#1. 데이터
data_list = [load_diabetes, fetch_california_housing]
data_list_name = ['디아벳','캘리포니아']
for i,value in enumerate(data_list):
    x,y  = value(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,y, shuffle= True, train_size=0.8, random_state=1030
    )

    scaler = StandardScaler()
    x_train =  scaler.fit_transform(x_train)
    x_test =  scaler.fit_transform(x_test)

#2. 모델
    knn = KNeighborsRegressor(n_neighbors=8)
    xgb = XGBRegressor()
    cat = CatBoostRegressor(verbose=0)

# model = VotingClassifier(
    model = StackingRegressor(
        estimators=[('knn', knn), ('xgb', xgb), ('cat', cat)],#voting안먹힘
    # final_estimator=LogisticRegression(), #디폴트는 logisticregression #predict한걸 훈련할 모델 
    # final_estimator=KNeighborsClassifier(),
        final_estimator=RandomForestRegressor(),
    # final_estimator=VotingClassifier('주저리주저리') 넣을순 있는데 성능이 좋을지는 모른다
    ) 

#3. 훈련
    model.fit(x_train,y_train)

#4. 평가, 예측
    y_pred = model.predict(x_test)
    print(data_list_name[i])
    print('model.score : ', model.score(x_test,y_test))
    print("Stacking.r2 : ", r2_score(y_test,y_pred))

    Classifiers = [knn,xgb,cat]

    for model2 in Classifiers:
        model2.fit(x_train,y_train)
        y_pred = model2.predict(x_test)
        score2 = r2_score(y_test,y_pred)
        class_name = model2.__class__.__name__ 
        print("{0}정확도 : {1:4f}".format(class_name, score2))

# 디아벳
# model.score :  0.14439618312607805
# Stacking.r2 :  0.14439618312607805
# KNeighborsRegressor정확도 : 0.308391
# XGBRegressor정확도 : 0.211415
# CatBoostRegressor정확도 : 0.333354
# 캘리포니아
# model.score :  0.6715642927985698
# Stacking.r2 :  0.6715642927985698
# KNeighborsRegressor정확도 : 0.716945
# XGBRegressor정확도 : 0.642591
# CatBoostRegressor정확도 : 0.712629