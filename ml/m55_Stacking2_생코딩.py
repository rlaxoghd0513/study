#스태킹 최대 문제점
# 프리딕한걸 또 프리딕한다
# 결국 x_test로 또 훈련을 하는거니까 과적합의 문제가 생길 수도 있다
# 스태킹할땐 항상 과적합문제 신경써야한다 -> train을 두개로 나누었다?

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

#####3대장####
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#1. 데이터
x,y  = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, train_size=0.8, random_state=1030
)

scaler = StandardScaler()
x_train =  scaler.fit_transform(x_train)
x_test =  scaler.fit_transform(x_test)

#2. 모델
cat = CatBoostClassifier(verbose=0)
lb = LGBMClassifier()
xgb = XGBClassifier()
li = []
models = [xgb,lb,cat]

for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # print(y_predict.shape) #(114,)
    y_predict = y_predict.reshape(y_predict.shape[0],1)
    li.append(y_predict)
    
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print('{0}acc:{1:4f}'.format(class_name,score))
print(li)
#model.predict의 형태는 넘파이형태
#li는 벡터형태 1차원인데 행렬형태로 컨캣해줄라면 2차원으로 만들어줘야된다
# y_predict가 (114,)이니까 2차원으로 만들기 위해 열을 줄거다
#그래서 넘파이로 바꿔서 합칠거다 concat한다

y_stacking_predict = np.concatenate(li, axis=1)
print(y_stacking_predict)
print(y_stacking_predict.shape)  #(114, 3)

model = CatBoostClassifier(verbose=0)
model.fit(y_stacking_predict,y_test)

score = model.score(y_stacking_predict, y_test)
print('스태킹결과:', score) #스태킹결과: 0.956140350877193





