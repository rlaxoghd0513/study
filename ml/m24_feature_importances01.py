#트리계열만. 훈련을 돌린 후 feature_importances 보기

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier       #설치해야됨 xgboost
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline 


#1 데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 칼럼에서 필요없는 칼럼을 찾아내보자

#2 모델
model_list = [DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier]
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()
for i,value in enumerate(model_list):
    model = value()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print('====================')
    print('model.score:', result)
    print('acc:', acc)
    if i !=3:                #3번째 모델이 아니면 이렇게 뽑고(당연히 0번째 모델부터 시작)
        print(model, ':' , model.feature_importances_)
    else:                    #3번째 모델이면 이렇게 뽑아라
        print('XGBClassifier() : ', model.feature_importances_)
        
#이 네 모델의 공통점 tree구조다 트리계열이다

# #3 훈련
# model.fit(x_train,y_train)

# #4 평가예측
# result = model.score(x_test, y_test)
# print('model.score:', result)

# y_predict = model.predict(x_test)
# print('acc', accuracy_score(y_test, y_predict))

# print('=====================================================')
# print(model, ':', model.feature_importances_)   #트리계열에만 있다 feature_importances

# DecisionTreeClassifier()     : [0.01253395 0.01253395 0.5618817  0.4130504 ] 
# RandomForestClassifier()     : [0.08775716 0.02174753 0.45285768 0.43763763]
# GradientBoostingClassifier() : [0.0010226  0.02412132 0.76674882 0.20810725]
# XGBclassifier()              : [0.0089478  0.01652037 0.7527313  0.22180054]
# 공통적으로 세번째 feature가 가장 중요하다고 나온다

