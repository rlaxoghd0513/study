import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings(action = 'ignore')  #터미널에 뜨는 warning을 필터링해서 무시한다
from sklearn.metrics import accuracy_score, r2_score

#1 데이터
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True), 
             load_wine(return_X_y=True), 
             load_digits(return_X_y=True), 
             fetch_covtype(return_X_y=True)]
model_list = [LinearSVC, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier]

data_name_list = ['아이리스 :',
                  '브리스트 캔서:',
                  '와인',
                  '디깃츠',
                  '코브타입']

model_name_list = ['LinearSVC', 'DecisionTreeClassifier', 'LogisticRegression', 'RandomForestClassifier']

#분류는 acc 회귀는 r2가 디폴트다

for i,value in enumerate(data_list):     
    x,y = value
    # print(x.shape, y.shape)
    print('=====================')
    print(data_name_list[i])
        #2 모델구성
    for j,values in enumerate(model_list):
         model = values()
         #3 컴파일 훈련
         model.fit(x,y)
         #4 평가 예측
         results = model.score(x,y)
         print(model_name_list[j],'model.score:', results)
         y_predict = model.predict(x)
         acc= accuracy_score(y, y_predict)
         print(model_name_list[j], 'accuracy_score:', acc)

    