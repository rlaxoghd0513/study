import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,BaggingClassifier
from sklearn.ensemble import VotingClassifier

#1. 데이터#1. 데이터
data_list = [load_diabetes, fetch_california_housing]
data_list_name = ['디아벳','캘리포니아']

for i,value in enumerate(data_list):
    x,y  = value(return_X_y=True)
     
    poly = PolynomialFeatures()
    x_poly = poly.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_poly,y, shuffle= True, train_size=0.8, random_state=1030
    )

    scaler = StandardScaler()
    x_train =  scaler.fit_transform(x_train)
    x_test =  scaler.fit_transform(x_test)

#2. 모델

    model = RandomForestRegressor()

#3. 훈련
    model.fit(x_train,y_train)

#4. 평가, 예측
    y_pred = model.predict(x_test)
    print(data_list_name[i])
    print('model.score : ', model.score(x_test,y_test))
    
# 디아벳
# model.score :  0.43468608492531724
