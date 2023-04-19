import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype, load_diabetes, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
import pandas as pd

#1 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['casual','registered','count'], axis=1)
y= train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

########################################################################################
#랜덤포레스트 파라미터와 파이프라인 파라미터는 다르다 
#랜덤포레스트의 파라미터를 파이프라인 파라미터 형태로 바꿔준다
#파라미터들 앞에 파이프라인에서 내가 지은 모델 이름과 아래작대기두개__를 붙여준다
parameters = [
    {'rfc__n_estimators':[100,200], 'rfc__max_depth':[6,10,12]},
    {'rfc__max_depth':[6,8,10,10], 'rfc__min_samples_leaf':[3,5,7,10]},
    {'rfc__min_samples_leaf':[3,5,7,10]}
]

#2 모델
pipe = Pipeline([("std",StandardScaler()), ("rfc",RandomForestRegressor())]) 
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)



#3 훈련
model.fit(x_train,y_train)

#4 평가예측
result = model.score(x_test, y_test)
print('model.score:', result)

y_predict = model.predict(x_test)
print('r2', r2_score(y_test, y_predict))

# model.score: 0.34976768407398107
# r2 0.34976768407398107