# for문으로 만들기
# aaa는 리스트  리스트는 이터레이터(반복함수 있는놈)에 포함된다

aaa = ['a','b','c',4]  #문자와 숫자 섞어서는 리스트에서는 가능하지만 넘파이에서는 안됨 
for i in aaa:
    print(i)
# a
# b
# c    
for index, value in enumerate(aaa): #index순서 value 값
    print(index, value)
# 0 a
# 1 b
# 2 c

import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings(action = 'ignore')


data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True), 
             load_wine(return_X_y=True), 
             load_digits(return_X_y=True), 
             fetch_covtype(return_X_y=True)]

model_list = [LinearSVC, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier]

for i in data_list:     
    x,y = i
    print('=====================')
    
    for j in model_list:
         model = j()
         model.fit(x,y)
         results = model.score(x,y)
         print(results)