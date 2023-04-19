import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
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