import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier #Decision모여있는게 RandomForest
from sklearn.linear_model import LogisticRegression

#1데이터
data_list = [load_iris, load_breast_cancer, load_wine, load_digits]
model_list = [RandomForestClassifier, LogisticRegression,DecisionTreeClassifier]
data_list_name = ['아이리스', '캔서', '와인', '디깃']
model_list_name = ['RandomForestClassifier', 'LogisticRegression','DecisionTreeClassifier']

for i,value in enumerate(data_list):
    x,y = value(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123, train_size=0.8, shuffle=True, stratify=y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    for j,value1 in enumerate(model_list):
        aaa = value1()
        model = BaggingClassifier(aaa,
                                  n_estimators=11,
                                  random_state = 333,
                                  bootstrap=True)
        model.fit(x_train, y_train)
        print(data_list_name[i])
        print(model_list_name[j])
        print('model.score:',model.score(x_test, y_test))
        y_predict = model.predict(x_test)
        print('acc:',accuracy_score(y_test, y_predict))
        
# 아이리스
# RandomForestClassifier
# model.score: 0.9333333333333333
# acc: 0.9333333333333333
# 아이리스
# LogisticRegression
# model.score: 0.9333333333333333
# acc: 0.9333333333333333
# 아이리스
# DecisionTreeClassifier
# model.score: 0.9333333333333333
# acc: 0.9333333333333333
# 캔서
# RandomForestClassifier
# model.score: 0.956140350877193
# acc: 0.956140350877193
# 캔서
# LogisticRegression
# model.score: 0.9736842105263158
# acc: 0.9736842105263158
# 캔서
# DecisionTreeClassifier
# model.score: 0.956140350877193
# acc: 0.956140350877193
# 와인
# RandomForestClassifier
# model.score: 0.9722222222222222
# acc: 0.9722222222222222
# 와인
# LogisticRegression
# model.score: 1.0
# acc: 1.0
# 와인
# DecisionTreeClassifier
# model.score: 0.9166666666666666
# acc: 0.9166666666666666
# 디깃
# RandomForestClassifier
# model.score: 0.9916666666666667
# acc: 0.9916666666666667
# 디깃
# LogisticRegression
# model.score: 0.9861111111111112
# acc: 0.9861111111111112
# 디깃
# DecisionTreeClassifier
# model.score: 0.9527777777777777
# acc: 0.9527777777777777
